"""
tesseractcli/commands/parser.py

Pure function: `parse(text, stage) -> Command`. No Textual import, no
widget access, no I/O - this is what makes the entire command grammar
(GLOBAL_ALIASES, NAV_COMMANDS, RESET_HEADS/RELOAD_HEADS, compound
shorthand, the free-text-stage carve-out) unit-testable with plain
strings, independent of a running App.

This module owns every constant the old `ui/app.py` module level used
to define (NAV_COMMANDS, NAV_ALIASES, GLOBAL_ALIASES, RESET_/RELOAD_
HEADS+SCOPES, _FREE_TEXT_STAGES, _SHORTHAND_KIND_FLAGS) plus the
`_translate_compound_shorthand` helper - none of that is UI, all of it
is grammar, so all of it lives here now.

One deliberate exception: `settings_commands.ALIASES` is still imported
from the existing `ui/views/settings_commands` module for first-word
alias resolution (e.g. `p` -> `packs`, `rm` -> `remove`). That's a
dependency on another pure lookup table, not on Textual or any live
state, so it doesn't compromise testability.
"""

from __future__ import annotations

from tesseractcli.ui.views import settings_commands

from .types import (
    ApprovalInput,
    ChatMessage,
    Command,
    ConfirmInput,
    InlineModelSwitch,
    InlineWorkspaceSwitch,
    MetaCommand,
    NavCommand,
    NavTarget,
    ReloadCommand,
    ResetCommand,
    SettingsCommand,
    StartWizard,
    UnknownInput,
    WizardFreeText,
    WorkspacePathInput,
)

# Bare-word navigation commands, recognized only on an exact (stripped,
# case-insensitive) match against the whole input - see the original
# module docstring in ui/app.py for the full rationale (kept verbatim
# in spirit: bare words, not slash-commands, was the explicit UX ask).
NAV_COMMANDS = {"settings", "chat", "home", "model", "exit", "quit"}

# Single-letter shortcuts resolved before NAV_COMMANDS matching.
NAV_ALIASES = {"q": "exit"}

# Meta/utility shortcuts available from (almost) any stage. Dash
# convention: single "-" = short flag, "--" = long form (getopt/argparse
# style, matching typer).
GLOBAL_ALIASES = {
    "-h": "help",
    "--help": "help",
    "?": "help",
    "-cfg": "settings",
    "--config": "settings",
    "-cnf": "settings",  # soft-deprecated synonym of -cfg, kept for compatibility
    "-q": "exit",
    "--quit": "exit",
    "--chat": "chat",
    "--home": "home",
    "--model": "model",
    "-c": "copy",
    "copy": "copy",
    "-e": "expand",
    "--expand": "expand",
    "expand": "expand",
    "-cls": "clear",
    "--clear": "clear",
    "clear": "clear",
    "cls": "clear",
    "-p": "packs",
    "--packs": "packs",
    "packs": "packs",
    "-ws": "workspace",
    "--workspace": "workspace",
    "workspace": "workspace",
}

# "reset"/"reload" take an optional second word instead of being flat
# GLOBAL_ALIASES entries, so a future scope is a new dict entry, not a
# rewrite. "-r"/"-rm"/"-rn" are already taken inside the settings-stage
# grammar (remove/rename) - these two use distinct heads on purpose.
RESET_HEADS = {"-rst", "--reset", "reset"}
RESET_SCOPES = {"temp": "temp", "context": "temp", "memory": "temp"}
DEFAULT_RESET_SCOPE = "temp"

RELOAD_HEADS = {"-rl", "--reload", "reload"}
RELOAD_SCOPES = {"cfg": "config", "config": "config", "settings": "config"}
DEFAULT_RELOAD_SCOPE = "config"

# Second-token flags recognized right after a "-rm"/"remove"/"-a"/"add"
# head, so "-rm -p mypack" / "add -md mypack groq llama" is understood
# without spelling "pack"/"model" out.
_SHORTHAND_KIND_FLAGS = {"-p": "pack", "pack": "pack", "-md": "model", "model": "model"}

# Stages where free text is being captured verbatim on purpose - the
# whole GLOBAL_ALIASES/nav grammar is deliberately NOT applied here, or
# "help"/"-h" could never be typed as e.g. a literal model id.
_FREE_TEXT_STAGES = {
    "awaiting_approval",
    "wiz_model_custom",
    "wiz_pack_new",
    "awaiting_reset_confirm",
    "awaiting_remove_pack_confirm",
    "awaiting_activate_confirm",
}

_CONFIRM_STAGES = {
    "awaiting_reset_confirm",
    "awaiting_remove_pack_confirm",
    "awaiting_activate_confirm",
}


def translate_compound_shorthand(words: list[str]) -> str | None:
    """Rewrites compound shorthand ('-rm -p <n>', 'add -md <pack>
    <provider> <model>') into the canonical 'remove pack <n>' / 'add
    model ...' line `settings_commands.handle()` already understands.
    Returns None (falls through to normal handling) if `words` doesn't
    match this shape - also matches the already-canonical spelling, so
    it's a strict superset rather than a second competing grammar."""
    if len(words) < 2:
        return None
    head = settings_commands.ALIASES.get(words[0].lower(), words[0].lower())
    if head not in ("add", "remove"):
        return None
    kind = _SHORTHAND_KIND_FLAGS.get(words[1].lower())
    if kind is None:
        return None
    return " ".join([head, kind, *words[2:]])


def _is_confirm(text: str) -> bool:
    return text.strip().lower() in ("y", "yes")


def parse(text: str, stage: str) -> Command:  # noqa: PLR0911, PLR0912 - grammar dispatch
    """The single entry point. `text` is the raw (already-stripped by
    the caller is NOT assumed - parse() strips internally) input;
    `stage` is `SessionController.stage` at the moment of submission.
    Returns exactly one Command; never raises for malformed input (an
    unparseable line becomes UnknownInput, never an exception) and never
    touches global/app state - every branch below is a pure function of
    its two arguments."""
    stripped = text.strip()

    # ---- stage-scoped free-text capture takes priority over grammar ----
    if stage in _CONFIRM_STAGES:
        return ConfirmInput(confirmed=_is_confirm(stripped), text=stripped, stage=stage)  # type: ignore[arg-type]

    if stage == "awaiting_approval":
        return ApprovalInput(approved=_is_confirm(stripped), text=stripped)

    if stage == "wiz_model_custom":
        return WizardFreeText(text=stripped, field="model")

    if stage == "wiz_pack_new":
        return WizardFreeText(text=stripped, field="pack_new")

    if stage == "workspace":
        return WorkspacePathInput(text=stripped)

    if stage == "workspace_edit":
        return WorkspacePathInput(text=stripped, is_cancel=stripped.lower() == "cancel")

    # ---- global grammar (nav words, aliases, reset/reload, shorthand) ----
    words_probe = stripped.split()
    head_probe = words_probe[0].lower() if words_probe else ""

    if head_probe in RESET_HEADS and len(words_probe) <= 2:
        return ResetCommand(
            scope=words_probe[1].lower() if len(words_probe) == 2 else None, text=stripped
        )

    if head_probe in RELOAD_HEADS and len(words_probe) <= 2:
        return ReloadCommand(
            scope=words_probe[1].lower() if len(words_probe) == 2 else None, text=stripped
        )

    global_cmd = GLOBAL_ALIASES.get(stripped.lower())
    if global_cmd in ("help", "copy", "expand", "clear", "packs"):
        return MetaCommand(kind=global_cmd)  # type: ignore[arg-type]
    if global_cmd and stage in {"chat", "settings", "home"}:
        return NavCommand(target=global_cmd)  # type: ignore[arg-type]

    if stage in {"chat", "home"}:
        words = stripped.split()
        first = words[0].lower() if words else ""

        if first in ("-cfg", "--config") and len(words) >= 2:
            rest = words[1:]
            head = rest[0].lower()
            if head == "model":
                return InlineModelSwitch(args=tuple(rest[1:]), text=stripped)
            if head in ("-ws", "--workspace", "workspace"):
                return InlineWorkspaceSwitch(args=tuple(rest[1:]), text=stripped)
            if head in RESET_HEADS and len(rest) <= 2:
                return ResetCommand(
                    scope=rest[1].lower() if len(rest) == 2 else None, text=stripped
                )
            if head in RELOAD_HEADS and len(rest) <= 2:
                return ReloadCommand(
                    scope=rest[1].lower() if len(rest) == 2 else None, text=stripped
                )
            translated = translate_compound_shorthand(rest)
            if translated is not None:
                return SettingsCommand(raw_text=translated, original_text=stripped)
            resolved_head = settings_commands.ALIASES.get(head, head)
            if resolved_head == "suggest" and len(rest) == 1:
                return StartWizard()
            resolved_text = " ".join([resolved_head, *rest[1:]])
            return SettingsCommand(raw_text=resolved_text, original_text=stripped)

        translated = translate_compound_shorthand(words)
        if translated is not None:
            return SettingsCommand(raw_text=translated, original_text=stripped)

    normalized = NAV_ALIASES.get(stripped.lower(), stripped.lower())
    if normalized in NAV_COMMANDS and stage in {"chat", "settings", "home"}:
        return NavCommand(target=normalized)  # type: ignore[arg-type]

    if stage == "settings":
        words = stripped.split()
        if not words:
            return UnknownInput(text=stripped, stage=stage)
        translated = translate_compound_shorthand(words)
        if translated is not None:
            return SettingsCommand(raw_text=translated, original_text=stripped)
        first_word = words[0].lower()
        resolved_cmd = settings_commands.ALIASES.get(first_word, first_word)
        if resolved_cmd == "suggest" and len(words) == 1:
            return StartWizard()
        resolved_text = " ".join([resolved_cmd, *words[1:]])
        return SettingsCommand(raw_text=resolved_text, original_text=stripped)

    if stage == "home":
        return UnknownInput(text=stripped, stage=stage)

    # stage == "chat" (default): a real message for the agent
    return ChatMessage(text=stripped)
