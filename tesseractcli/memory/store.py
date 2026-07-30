"""
tesseractcli/memory/store.py

SQLite-backed persistence for chat messages, keyed per workspace.

Scope (deliberately minimal): every message gets written to disk as soon
as it's appended to the in-memory `messages` list, so a crash/restart
never loses what was already said. Reading history back in on startup
("resume this session") is intentionally NOT implemented here - this
schema is built so that feature doesn't need a migration when it lands.

Layout on disk:
    <CONFIG_DIR>/db_store/<workspace_id>/conversation.db

`workspace_id` is a stable hash of the workspace's resolved absolute
path - it's the on-disk partition key only (one db file per workspace),
it is NOT a column in any table below.

Four tables, split by responsibility (table-per-role-subtype - every
message gets exactly one `messages` row, plus at most one extension row
in whichever of `model_meta`/`tools` matches its role):

- `sessions`   - one row per app open/bind. `session_id` is a fresh
                 uuid4 every time (NOT reused across restarts, unlike
                 the old single-persistent-id design), so a future
                 `resume <session_id>` can group by it. `workspace_path`
                 is intentionally duplicated per row rather than
                 pulled into its own table - there's exactly one
                 workspace per db file, so this just saves a join.
- `messages`   - core, role-agnostic: id, session_id (FK), role,
                 content, timestamp. Nothing role-specific lives here.
- `model_meta` - one row per assistant message (`message_id` is both PK
                 and FK - strict 1:1). model_name/pack_name/temperature/
                 max_tokens come from `llm/dispatcher.py`'s attribution
                 stamp on `response_metadata`; input_tokens/output_tokens
                 come straight off LangChain's standard
                 `AIMessage.usage_metadata` when a provider reports it.
                 `session_id` is deliberately NOT duplicated here - it's
                 one join away via `message_id -> messages.session_id`.
- `tools`      - one row per executed tool call, keyed by `tool_call_id`
                 itself (already a unique natural key from the model -
                 no synthetic id needed). `message_id` points at the
                 ToolMessage row. `args` is the JSON the model actually
                 sent the tool; `success` is 0/1 pulled from the same
                 `ToolResult` `agent/loop.py` already has in hand at
                 that point (rather than re-parsing an "ERROR:" prefix
                 out of the content string later).

Known gap: if the inner loop stops before a requested tool call ever
gets dispatched (e.g. agent.max_iterations hit mid-call), that
one request's args are never persisted - there's no ToolMessage row
for it to hang off of. Accepted for now given how rare/edge-case it is.
"""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage

from tesseractcli.config.global_config.manager import CONFIG_DIR
from tesseractcli.config.logger import logger

DB_STORE_DIR = CONFIG_DIR / "db_store"

# Same convention as config/global_config/manager.py's _FILE_MODE/_DIR_MODE -
# chat history can contain file contents read via tools (code, configs,
# possibly secrets the sandbox didn't catch), so it's treated as sensitive
# from day one. Unlike that module, a chmod failure here just logs a
# warning rather than raising - persistence problems must never crash the chat.
_DB_FILE_MODE = 0o600
_DB_DIR_MODE = 0o700


def _secure_dir(path: Path) -> None:
    if os.name != "posix":
        return
    try:
        path.chmod(_DB_DIR_MODE)
    except OSError:
        logger.warning("failed to set secure permissions on %s", path)


def _secure_file(path: Path) -> None:
    if os.name != "posix":
        return
    try:
        path.chmod(_DB_FILE_MODE)
    except OSError:
        logger.warning("failed to set secure permissions on %s", path)


_SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    session_id TEXT PRIMARY KEY,
    workspace_path TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL REFERENCES sessions(session_id),
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    timestamp TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_messages_session_id ON messages(session_id);

CREATE TABLE IF NOT EXISTS model_meta (
    message_id INTEGER PRIMARY KEY REFERENCES messages(id),
    model_name TEXT,
    pack_name TEXT,
    temperature REAL,
    max_tokens INTEGER,
    input_tokens INTEGER,
    output_tokens INTEGER
);

CREATE TABLE IF NOT EXISTS tools (
    tool_call_id TEXT PRIMARY KEY,
    message_id INTEGER NOT NULL REFERENCES messages(id),
    name TEXT,
    args TEXT,
    success INTEGER
);
CREATE INDEX IF NOT EXISTS idx_tools_message_id ON tools(message_id);
"""


def workspace_id(workspace_root: Path) -> str:
    """Stable, filesystem-safe id for a workspace: sha256 of the
    resolved absolute path, truncated to 16 hex chars. Deterministic -
    the same workspace_root always maps to the same id, across restarts,
    regardless of what characters are in the actual folder name. This
    is the on-disk folder key only - it is not stored in any table."""
    resolved = str(Path(workspace_root).expanduser().resolve())
    return hashlib.sha256(resolved.encode("utf-8")).hexdigest()[:16]


def _role_and_content(message: BaseMessage) -> tuple[str, str]:
    if isinstance(message, HumanMessage):
        return "user", str(message.content)
    if isinstance(message, AIMessage):
        return "assistant", str(message.content)
    if isinstance(message, ToolMessage):
        return "tool", str(message.content)
    return message.type, str(message.content)  # e.g. SystemMessage - still persisted


def _model_meta_row(message: AIMessage) -> dict:
    """model_name/pack_name/temperature/max_tokens from the dispatcher's
    `response_metadata` stamp; input/output tokens from LangChain's own
    `usage_metadata` attribute (only populated when a provider reports
    it, so all four may be None)."""
    meta = message.response_metadata or {}
    provider = meta.get("tesseract_provider")
    model = meta.get("tesseract_model")
    model_name = (
        f"{provider}:{model}"
        if provider and model
        else (meta.get("model_name") or meta.get("model"))
    )

    usage: Mapping[str, Any] = message.usage_metadata or {}
    return {
        "model_name": model_name,
        "pack_name": meta.get("tesseract_pack_name"),
        "temperature": meta.get("tesseract_temperature"),
        "max_tokens": meta.get("tesseract_max_tokens"),
        "input_tokens": usage.get("input_tokens"),
        "output_tokens": usage.get("output_tokens"),
    }

def _tool_row(message: ToolMessage) -> dict:
    """name/args/success come from `additional_kwargs`, set by
    `agent/loop.py` when it builds the ToolMessage (it already has the
    original call's name/args and the ToolResult's success flag in
    hand at that point - nothing here is re-derived from `content`)."""
    kwargs = message.additional_kwargs or {}
    args = kwargs.get("tool_args")
    return {
        "name": kwargs.get("tool_name"),
        "args": json.dumps(args) if args is not None else None,
        "success": int(kwargs["tool_success"]) if "tool_success" in kwargs else None,
    }


class ConversationStore:
    """One SQLite connection bound to a single workspace's conversation.db."""

    def __init__(self, workspace_root: Path):
        self.workspace_root = Path(workspace_root).expanduser().resolve()
        self.workspace_id = workspace_id(
            self.workspace_root
        )  # on-disk partition key, stable
        self.session_id = str(
            uuid.uuid4()
        )  # fresh every time a workspace is opened/bound
        self.db_path = DB_STORE_DIR / self.workspace_id / "conversation.db"
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        _secure_dir(self.db_path.parent)

        # check_same_thread=False: tool approval runs the terminal
        # prompt via asyncio.to_thread, so this connection may be
        # touched from more than one OS thread over the app's life.
        # Writes here are short and serial, so that's safe. WAL mode
        # keeps readers from blocking on that occasional cross-thread
        # write, and is the standard recommendation for exactly this
        # "mostly one writer, maybe read from elsewhere" shape.
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode = WAL")
        self._conn.execute("PRAGMA foreign_keys = ON")
        self._conn.executescript(_SCHEMA)
        _secure_file(self.db_path)
        self._conn.execute(
            "INSERT INTO sessions (session_id, workspace_path, created_at) VALUES (?, ?, ?)",
            (
                self.session_id,
                str(self.workspace_root),
                datetime.now(timezone.utc).isoformat(),
            ),
        )
        self._conn.commit()

    def save_message(self, message: BaseMessage) -> None:
        role, content = _role_and_content(message)
        timestamp = datetime.now(timezone.utc).isoformat()
        try:
            cursor = self._conn.execute(
                "INSERT INTO messages (session_id, role, content, timestamp) VALUES (?, ?, ?, ?)",
                (self.session_id, role, content, timestamp),
            )
            message_id = cursor.lastrowid

            if isinstance(message, AIMessage):
                meta = _model_meta_row(message)
                self._conn.execute(
                    "INSERT INTO model_meta "
                    "(message_id, model_name, pack_name, temperature, max_tokens, input_tokens, output_tokens) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (
                        message_id,
                        meta["model_name"],
                        meta["pack_name"],
                        meta["temperature"],
                        meta["max_tokens"],
                        meta["input_tokens"],
                        meta["output_tokens"],
                    ),
                )

            elif isinstance(message, ToolMessage):
                tool = _tool_row(message)
                self._conn.execute(
                    "INSERT INTO tools (tool_call_id, message_id, name, args, success) "
                    "VALUES (?, ?, ?, ?, ?)",
                    (
                        message.tool_call_id,
                        message_id,
                        tool["name"],
                        tool["args"],
                        tool["success"],
                    ),
                )

            self._conn.commit()
        except sqlite3.Error:
            # Persistence must never take the chat down - log and move
            # on, the message still exists in the in-memory list either way.
            logger.exception("failed to persist message to %s", self.db_path)

    def close(self) -> None:
        self._conn.close()


class PersistentMessageList(list):
    """Drop-in replacement for `list[BaseMessage]`: behaves exactly like
    a plain list, so every existing `messages.append(...)` call site in
    `agent/loop.py` needs zero changes - but it also writes every
    appended message to a bound `ConversationStore`.

    Unbound (`_store is None`) it's just a normal list - this lets the
    UI create it in `on_mount` before the workspace is known yet, then
    call `bind_store(...)` once the workspace is picked (or changed
    later via the `workspace` command).
    """

    def __init__(self, *args):
        super().__init__(*args)
        self._store: ConversationStore | None = None

    def bind_store(self, store: ConversationStore) -> None:
        old = self._store
        self._store = store
        if old is not None:
            old.close()

    def append(self, message: BaseMessage) -> None:
        super().append(message)
        if self._store is not None:
            self._store.save_message(message)

    def close(self) -> None:
        """Closes the bound store's sqlite connection, if any. Not
        required for data safety (every `save_message` already commits
        immediately), but avoids leaving the connection open until the
        OS reclaims it on process exit. Safe to call when unbound."""
        if self._store is not None:
            self._store.close()
