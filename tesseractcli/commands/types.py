"""
tesseractcli/commands/types.py

The contract between the View (TesseractApp) and the SessionController.
Every user text submission is turned into exactly one of these by
`commands.parser.parse()` before the Controller ever sees it - the View
never inspects `text`/`stage` itself past that call, and the Controller
never inspects raw strings. This is what makes both sides independently
testable: the parser with plain strings in / Command out, the controller
with Command objects in / RenderInstruction objects out, neither one
needing a running Textual App.

OptionList selections (`on_option_list_option_selected`) are NOT text and
don't go through the parser - they're routed straight to
`SessionController.handle_option(list_id, option_id)` as a plain
(str, str) pair, keeping that path Textual-event-free too without forcing
an artificial Command wrapper around something that was never text.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

NavTarget = Literal["chat", "settings", "home", "model", "workspace", "exit"]
MetaKind = Literal["help", "copy", "expand", "clear", "packs"]


@dataclass(frozen=True)
class ChatMessage:
    """A real message for the agent - the chat stage's default case."""

    text: str


@dataclass(frozen=True)
class NavCommand:
    """Bare nav word (chat/settings/home/model/workspace/exit/quit) or
    its GLOBAL_ALIASES/NAV_ALIASES-resolved form (q -> exit, --chat ->
    chat, ...)."""

    target: NavTarget


@dataclass(frozen=True)
class SettingsCommand:
    """A settings-grammar command line, already alias-resolved and
    already stripped of any compound-shorthand rewriting
    (`-rm -p X` -> `remove pack X`) - forwarded to
    `settings_commands.handle()` verbatim by the Controller. Carries the
    resolved text, not the raw text the user typed, since the parser is
    the one place that knows the full alias/shorthand grammar.
    `original_text` is kept separately so the echo line still shows
    exactly what the user typed (matching the original UX), not the
    resolved form."""

    raw_text: str
    original_text: str


@dataclass(frozen=True)
class InlineModelSwitch:
    """`-cfg model [pack]` / bare `model` when args are present -
    one-shot pack switch without opening the picker. Empty `args` means
    "open the picker instead" and is handled by the Controller, not
    encoded as a separate Command."""

    args: tuple[str, ...]
    text: str


@dataclass(frozen=True)
class InlineWorkspaceSwitch:
    """`-cfg -ws <path>` / `-cfg workspace <path>` - one-shot workspace
    switch without leaving the current stage. Empty `args` -> Controller
    falls back to the interactive `workspace_edit` stage."""

    args: tuple[str, ...]
    text: str


@dataclass(frozen=True)
class ResetCommand:
    """`reset`/`-rst`/`--reset`, optionally with a scope word
    (`reset temp`). `scope` is the raw second word, unresolved -
    RESET_SCOPES lookup happens in the Controller since "unknown scope"
    is a domain error message, not a parsing concern. `text` is the
    original typed line, for the echo."""

    scope: str | None
    text: str


@dataclass(frozen=True)
class ReloadCommand:
    """`reload`/`-rl`/`--reload`, optionally with a scope word."""

    scope: str | None
    text: str


@dataclass(frozen=True)
class MetaCommand:
    """help / copy / expand / clear / packs - global utility commands
    available from (almost) any stage."""

    kind: MetaKind


@dataclass(frozen=True)
class WorkspacePathInput:
    """Free-text path typed during the `workspace` (first-run) or
    `workspace_edit` (later change) stages. `is_cancel` is True only for
    the literal word 'cancel' typed during `workspace_edit` - the
    first-run `workspace` stage has no cancel option."""

    text: str
    is_cancel: bool = False


@dataclass(frozen=True)
class WizardFreeText:
    """Free-text step inside the suggest wizard - either a hand-typed
    model id (`wiz_model_custom`) or a new pack name (`wiz_pack_new`).
    `field` tells the Controller/PackWizard which WizardState field to
    fill; an OptionList can't take arbitrary text, so these two steps
    fall through to plain text submission instead."""

    text: str
    field: Literal["model", "pack_new"]


@dataclass(frozen=True)
class ConfirmInput:
    """Any of the y/n confirm stages (reset, remove-pack, activate-pack)
    resolved to a bool here, once, instead of every call site re-parsing
    `text.strip().lower() in ('y', 'yes')` itself. `stage` is carried
    through so the Controller knows which pending confirmation this
    answers. `text` is the raw typed line, kept only so the echo line
    can show exactly what was typed (matches the original UX, which
    echoed the literal input, not a normalized 'y'/'n')."""

    confirmed: bool
    text: str
    stage: Literal[
        "awaiting_reset_confirm",
        "awaiting_remove_pack_confirm",
        "awaiting_activate_confirm",
    ]


@dataclass(frozen=True)
class ApprovalInput:
    """y/n answer to a pending tool-approval request."""

    approved: bool
    text: str


@dataclass(frozen=True)
class StartWizard:
    """The 'suggest' command (bare, in `settings`, or via `-cfg suggest`
    from chat/home) - starts the provider->model->pack->target wizard.
    Distinct from `NavCommand(target='model')`, which opens the plain
    pack picker instead - the two are different flows in the original
    grammar and stay different Commands here."""


@dataclass(frozen=True)
class UnknownInput:
    """Parser couldn't classify this text for the current stage (e.g.
    plain text typed while `home`). The Controller decides what
    'unknown' means per-stage - the parser doesn't guess."""

    text: str
    stage: str


Command = (
    ChatMessage
    | NavCommand
    | SettingsCommand
    | InlineModelSwitch
    | InlineWorkspaceSwitch
    | ResetCommand
    | ReloadCommand
    | MetaCommand
    | WorkspacePathInput
    | WizardFreeText
    | ConfirmInput
    | ApprovalInput
    | StartWizard
    | UnknownInput
)


# ---------------------------------------------------------------------
# RenderInstruction - the Controller -> View direction
# ---------------------------------------------------------------------

RenderKind = Literal[
    "echo",  # "you typed this" line - content: str (already _echo_text-formatted)
    "log",  # a plain markup line - content: str
    "box",  # a bordered render_box - content: dict(title, body, style)
    "divider",  # a nav divider - content: str (label)
    "clear",  # wipe the scrollback and reprint the banner - content: None
    "status",  # status line text - content: str
    "mode_line",  # refresh the ws/pack/stage strip - content: None
    "mount_options",  # show an inline OptionList - content: dict(options, prompt, list_id)
    "restore_input",  # bring back #main-input - content: None
    "set_input_value",  # pre-fill #main-input - content: str
    "remove_echo",  # remove a previously-echoed line by id - content: str (id)
    "error",  # rendered error box - content: dict(title, exc, user_text)
]


@dataclass(frozen=True)
class RenderInstruction:
    kind: RenderKind
    content: Any = None
    id: str | None = None
