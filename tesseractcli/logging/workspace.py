"""tesseractcli/logging/workspace.py

Per-workspace, per-session log files layered on top of the base
`logger` configured in `tesseractcli.config.logger`.

Layout on disk::

    ~/.tesseract/logs/<workspace_name>_<workspace_hash>/session_<timestamp>.log

`<workspace_hash>` reuses `tesseractcli.memory.store.workspace_id` -
the exact same stable hash of the workspace's *resolved* absolute path
already used to key ``db_store/<workspace_id>/conversation.db``. A
given workspace gets one identifier everywhere in the app this way,
instead of a second, independently-computed hash here that could
silently drift out of sync with the db partition key. Two folders
that happen to share a name (``D:\\Work\\MyProject`` vs.
``E:\\Clients\\MyProject``) still land in different directories,
because the hash is over the resolved path, not the bare name.

Call `init_workspace_logging(workspace_root)` once a workspace is
picked or changed. It starts a new session file for that workspace and
swaps it in as the active sink, removing whichever workspace sink (if
any) was active before it - so switching workspaces mid-run
(`workspace` / `-ws`) doesn't keep writing into the old workspace's
directory. The bootstrap sink from `config/logger.py` is left running
throughout and is never touched here.
"""

from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path

from tesseractcli.config.logger import IS_DEV, LOGS_ROOT, logger
from tesseractcli.config.settings import get_settings

__all__ = ["init_workspace_logging", "workspace_log_dir_name"]

settings = get_settings()

# Guards against characters that are illegal in Windows filenames (and
# awkward everywhere else) turning up in a workspace folder name.
_UNSAFE_CHARS = re.compile(r'[<>:"/\\|?*\x00-\x1f]')

# Handler id of the currently-active workspace/session sink, so that
# binding a new workspace can remove the previous one first. None until
# the first workspace is bound.
_active_sink_id: int | None = None


def _sanitize(name: str) -> str:
    cleaned = _UNSAFE_CHARS.sub("_", name).strip().strip(".")
    return cleaned or "workspace"


def workspace_log_dir_name(workspace_root: str | Path) -> str:
    """`<folder_name>_<8-char hash>`, e.g. ``ProjectA_81ac29fd``.

    The hash is the first 8 hex characters of the same
    `memory.store.workspace_id()` used to key `db_store/` - see the
    module docstring for why this isn't computed independently here.
    """
    # Imported lazily: `config.logger` (which this module sits on top
    # of) loads very early during startup, before the rest of the app -
    # `memory.store` pulls in sqlite3/langchain, which is comparatively
    # heavy and unnecessary until a workspace is actually bound.
    from tesseractcli.memory.store import workspace_id

    resolved = Path(workspace_root).expanduser().resolve()
    name = _sanitize(resolved.name)
    short_hash = workspace_id(resolved)[:8]
    return f"{name}_{short_hash}"


def init_workspace_logging(workspace_root: str | Path) -> Path:
    """Bind logging to `workspace_root`.

    Creates (or reuses) that workspace's log directory under
    `~/.tesseract/logs/`, opens a fresh ``session_<timestamp>.log``
    file in it, and installs it as the active sink - replacing
    whichever workspace sink was active before (the workspace-agnostic
    bootstrap sink is untouched). Safe to call repeatedly, including
    with the same workspace, since every call starts a brand-new
    session file rather than reopening an old one.

    Returns:
        Path to the newly-created session log file.
    """
    global _active_sink_id

    workspace_dir = LOGS_ROOT / workspace_log_dir_name(workspace_root)
    workspace_dir.mkdir(parents=True, exist_ok=True)

    session_name = f"session_{datetime.now():%Y-%m-%d_%H-%M-%S}.log"
    session_path = workspace_dir / session_name

    if _active_sink_id is not None:
        logger.remove(_active_sink_id)
        _active_sink_id = None

    _active_sink_id = logger.add(
        session_path,
        level=settings.LOG_LEVEL,
        format=(
            "{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | "
            "{name}:{function}:{line} - {message}"
        ),
        # One file per session, closed for good when the process exits
        # or the workspace changes - no rotation/retention needed here,
        # that's for the long-lived bootstrap/global sinks only.
        serialize=not IS_DEV,  # JSON in prod, plain text in dev
        backtrace=IS_DEV,
        diagnose=IS_DEV,  # never leak local variable values in prod
        encoding="utf-8",
    )

    logger.debug(f"workspace logging bound: {workspace_root} -> {session_path}")
    return session_path
