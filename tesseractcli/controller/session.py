"""
tesseractcli/controller/session.py

Owns every piece of state that used to live directly on `TesseractApp`
(stage, workspace_root, selected_pack, messages, wizard state, pending
confirms) and the orchestration logic that used to be ~25 methods on
that class. Never touches a Textual widget - `SessionController.handle()`
takes a `Command` (built by `commands.parser.parse`) and returns a
`list[RenderInstruction]` for the View to draw; `handle_chat_turn()` is
the one async/streaming exception, since an agent turn genuinely takes
time and the View needs to show "thinking" immediately.

`TesseractApp` (the new, thin `ui/app.py`) is the only caller of this
class from outside tests.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

from tesseractcli.agent.loop import run_inner_loop
from tesseractcli.commands.parser import (
    DEFAULT_RELOAD_SCOPE,
    DEFAULT_RESET_SCOPE,
    RELOAD_SCOPES,
    RESET_SCOPES,
)
from tesseractcli.commands.types import (
    ApprovalInput,
    ChatMessage,
    Command,
    ConfirmInput,
    InlineModelSwitch,
    InlineWorkspaceSwitch,
    MetaCommand,
    NavCommand,
    ReloadCommand,
    RenderInstruction,
    ResetCommand,
    SettingsCommand,
    StartWizard,
    UnknownInput,
    WizardFreeText,
    WorkspacePathInput,
)
from tesseractcli.config.global_config.manager import ConfigManager
from tesseractcli.llm.dispatcher import LLMDispatcher
from tesseractcli.logging.workspace import init_workspace_logging
from tesseractcli.memory.store import ConversationStore, PersistentMessageList
from tesseractcli.models.exceptions import ConfigError
from tesseractcli.services.approval import ApprovalService
from tesseractcli.tools.registry import ToolRegistry
from tesseractcli.ui.views import settings_commands
from tesseractcli.ui.views.approval_view import ToolCallInfo, render_approval_preview
from tesseractcli.ui.views.model_picker import PackChoice, load_pack_choices

from .pack_wizard import PackWizard, WizardPrompt, WizardState


def _echo_instr(text: str, *, style: str = "dim") -> RenderInstruction:
    return RenderInstruction(kind="echo", content={"text": text, "style": style})


def _box(title: str, body: str, style: str) -> RenderInstruction:
    return RenderInstruction(
        kind="box", content={"title": title, "body": body, "style": style}
    )


class SessionController:
    def __init__(
        self,
        *,
        config_manager: ConfigManager,
        dispatcher: LLMDispatcher,
        tool_registry: ToolRegistry,
        approval_service: ApprovalService | None = None,
    ) -> None:
        self.config_manager = config_manager
        self.dispatcher = dispatcher
        self.tool_registry = tool_registry
        self.approval = approval_service or ApprovalService()

        self.stage: str = "workspace"
        self.workspace_root: Path | None = None
        self.selected_pack: str | None = None
        self.messages = PersistentMessageList()

        self._pack_return_stage: str = "chat"
        self._workspace_return_stage: str = "chat"
        self._reset_return_stage: str = "chat"
        self._remove_pack_return_stage: str = "chat"
        self._activate_return_stage: str = "chat"
        self._pending_reset_scope: str = DEFAULT_RESET_SCOPE
        self._pending_remove_pack: str | None = None
        self._pending_activate_pack: str | None = None
        self._pack_choices: list[PackChoice] = []

        self._wizard = PackWizard(config_manager)
        self._wizard_state: WizardState | None = None

        self._last_agent_reply: str = ""
        self._last_full_input: str | None = None
        self._turn_counter: int = 0

    # ------------------------------------------------------------------
    # echo / truncation - business logic (what counts as "too long"),
    # not presentation, so it stays here rather than in the View.
    # ------------------------------------------------------------------

    def echo_text(
        self, text: str, *, limit_lines: int = 8, limit_chars: int = 600
    ) -> str:
        lines = text.splitlines() or [text]
        shown = text
        truncated = False
        if len(lines) > limit_lines:
            shown = "\n".join(lines[:limit_lines])
            truncated = True
        if len(shown) > limit_chars:
            shown = shown[:limit_chars]
            truncated = True
        if not truncated:
            return text
        self._last_full_input = text
        hidden_chars = len(text) - len(shown)
        hidden_lines = max(0, len(lines) - limit_lines)
        extra = f", {hidden_lines} more line(s)" if hidden_lines else ""
        return (
            f"{shown}\n"
            f"[dim]… truncated ({hidden_chars} more char(s){extra}) - "
            "type 'expand' to see the full text.[/dim]"
        )

    # ------------------------------------------------------------------
    # main sync entry point
    # ------------------------------------------------------------------

    def handle(self, command: Command) -> list[RenderInstruction]:
        if isinstance(command, NavCommand):
            return self._handle_nav(command)
        if isinstance(command, SettingsCommand):
            return [
                _echo_instr(self.echo_text(command.original_text))
            ] + self._dispatch_settings(command.raw_text)
        if isinstance(command, InlineModelSwitch):
            return self._handle_inline_model_switch(command)
        if isinstance(command, InlineWorkspaceSwitch):
            return self._handle_inline_workspace_switch(command)
        if isinstance(command, ResetCommand):
            return [
                _echo_instr(self.echo_text(command.text))
            ] + self._start_reset_confirm(command.scope)
        if isinstance(command, ReloadCommand):
            return [_echo_instr(self.echo_text(command.text))] + self._reload_config(
                command.scope
            )
        if isinstance(command, MetaCommand):
            return self._handle_meta(command)
        if isinstance(command, WorkspacePathInput):
            return self._handle_workspace_path(command)
        if isinstance(command, WizardFreeText):
            return self._handle_wizard_free_text(command)
        if isinstance(command, ConfirmInput):
            return self._handle_confirm(command)
        if isinstance(command, ApprovalInput):
            self.approval.resolve(command.approved)
            return [_echo_instr(self.echo_text(command.text))]
        if isinstance(command, StartWizard):
            self._pack_return_stage = self.stage
            return self.start_wizard()
        if isinstance(command, UnknownInput):
            return self._handle_unknown(command)
        if isinstance(command, ChatMessage):
            raise TypeError(
                "ChatMessage must go through handle_chat_turn(), not handle()"
            )
        raise AssertionError(f"unreachable: {command!r}")

    # ------------------------------------------------------------------
    # navigation
    # ------------------------------------------------------------------

    def _handle_nav(self, command: NavCommand) -> list[RenderInstruction]:
        target = command.target
        if target == "chat":
            self.stage = "chat"
            return [RenderInstruction(kind="divider", content="chat")]
        if target == "settings":
            self.stage = "settings"
            return [
                RenderInstruction(kind="divider", content="settings"),
                RenderInstruction(kind="render_settings"),
            ]
        if target == "home":
            self.stage = "home"
            return [
                RenderInstruction(kind="divider", content="home"),
                RenderInstruction(kind="render_home"),
            ]
        if target == "model":
            self._pack_return_stage = self.stage
            return self.start_model_picker()
        if target == "workspace":
            self._workspace_return_stage = self.stage
            self.stage = "workspace_edit"
            return [
                RenderInstruction(kind="divider", content="workspace"),
                RenderInstruction(
                    kind="log",
                    content=f"[bold]Current workspace:[/bold] {self.workspace_root}",
                ),
                RenderInstruction(
                    kind="log",
                    content="[bold]New workspace folder:[/bold] (type a path, or 'cancel')",
                ),
            ]
        if target in ("exit", "quit"):
            return [RenderInstruction(kind="exit")]
        raise AssertionError(f"unreachable nav target: {target!r}")

    # ------------------------------------------------------------------
    # settings dispatch
    # ------------------------------------------------------------------

    def _dispatch_settings(self, cmd_text: str) -> list[RenderInstruction]:
        parts = cmd_text.strip().split()
        if (
            len(parts) == 3
            and parts[0].lower() == "remove"
            and parts[1].lower() == "pack"
        ):
            return self._start_remove_pack_confirm(parts[2])
        result = settings_commands.handle(self.config_manager, cmd_text)
        return [
            RenderInstruction(kind="log", content=result)
        ] + self._after_settings_command(cmd_text)

    def _after_settings_command(self, cmd_text: str) -> list[RenderInstruction]:
        words = cmd_text.strip().split()
        if len(words) < 3 or words[1].lower() != "pack":
            return []
        head = words[0].lower()
        providers = self.config_manager.config.providers

        if head == "add":
            name = words[2]
            if name in providers:
                self.selected_pack = name
                return [
                    _box_active_pack(name),
                    RenderInstruction(kind="mode_line"),
                ]
            return []

        if head == "remove":
            name = words[2]
            if name not in providers and self.selected_pack == name:
                self.selected_pack = None
                self._pack_return_stage = self.stage
                return [
                    RenderInstruction(
                        kind="log",
                        content="[yellow]active pack was removed - pick a new one:[/yellow]",
                    ),
                    *self.start_model_picker(),
                ]
            return []

        if head == "rename" and len(words) >= 4:
            old_name, new_name = words[2], words[3]
            if (
                self.selected_pack == old_name
                and new_name in providers
                and old_name not in providers
            ):
                self.selected_pack = new_name
                return [RenderInstruction(kind="mode_line")]
        return []

    def _handle_inline_model_switch(
        self, command: InlineModelSwitch
    ) -> list[RenderInstruction]:
        instr = [_echo_instr(self.echo_text(command.text))]
        if not command.args:
            self._pack_return_stage = self.stage
            return instr + self.start_model_picker()
        name = command.args[0]
        if name in self.config_manager.config.providers:
            self.selected_pack = name
            return instr + [_box_active_pack(name), RenderInstruction(kind="mode_line")]
        return instr + [
            RenderInstruction(
                kind="log",
                content=f"[red]no such pack: '{name}'.[/red] Try [bold]-p[/bold] to list packs.",
            )
        ]

    def _handle_inline_workspace_switch(
        self, command: InlineWorkspaceSwitch
    ) -> list[RenderInstruction]:
        instr = [_echo_instr(self.echo_text(command.text))]
        if not command.args:
            self._workspace_return_stage = self.stage
            self.stage = "workspace_edit"
            return instr + [
                RenderInstruction(kind="divider", content="workspace"),
                RenderInstruction(
                    kind="log",
                    content=f"[bold]Current workspace:[/bold] {self.workspace_root}",
                ),
                RenderInstruction(
                    kind="log",
                    content="[bold]New workspace folder:[/bold] (type a path, or 'cancel')",
                ),
                RenderInstruction(kind="mode_line"),
            ]
        path = Path(" ".join(command.args)).expanduser().resolve()
        if not path.exists() or not path.is_dir():
            return instr + [
                RenderInstruction(
                    kind="log", content=f"[red]not a valid directory: {path}[/red]"
                )
            ]
        self.workspace_root = path
        self.messages.bind_store(ConversationStore(path))
        return instr + [
            RenderInstruction(
                kind="log", content=f"[green]✓[/green] workspace set: {path}"
            ),
            RenderInstruction(kind="mode_line"),
        ]

    # ------------------------------------------------------------------
    # reset / reload
    # ------------------------------------------------------------------

    def _start_reset_confirm(self, scope: str | None) -> list[RenderInstruction]:
        resolved = DEFAULT_RESET_SCOPE if scope is None else RESET_SCOPES.get(scope)
        if resolved is None:
            return [
                _box(
                    "Unknown reset target",
                    f"'{scope}' isn't something I can reset. Available: "
                    f"{', '.join(sorted(set(RESET_SCOPES.values())))}.",
                    "#C4374F",
                )
            ]
        n = len(self.messages)
        self._reset_return_stage = self.stage
        self._pending_reset_scope = resolved
        self.stage = "awaiting_reset_confirm"
        return [
            RenderInstruction(kind="divider", content="reset"),
            _box(
                "Reset temp memory?",
                f"This clears the [bold]{n}[/bold] message(s) currently in "
                "context for the model. Nothing is deleted from the saved "
                "conversation history on disk.\n\n"
                "Type [bold]y[/bold] to confirm, anything else to cancel.",
                "#C4374F",
            ),
        ]

    def _reload_config(self, scope: str | None) -> list[RenderInstruction]:
        resolved = DEFAULT_RELOAD_SCOPE if scope is None else RELOAD_SCOPES.get(scope)
        if resolved is None:
            return [
                _box(
                    "Unknown reload target",
                    f"'{scope}' isn't something I can reload. Available: "
                    f"{', '.join(sorted(set(RELOAD_SCOPES.values())))}.",
                    "#C4374F",
                )
            ]
        try:
            self.config_manager.reload()
        except ConfigError as exc:
            return self._error_instructions("Config reload failed", exc) + [
                RenderInstruction(
                    kind="log",
                    content="[dim]Kept the previously loaded configuration.[/dim]",
                )
            ]
        self._pack_choices = load_pack_choices(self.config_manager)
        return [
            RenderInstruction(
                kind="log", content="[green]global_config.yaml reloaded.[/green]"
            ),
            RenderInstruction(kind="mode_line"),
        ]

    # ------------------------------------------------------------------
    # meta commands
    # ------------------------------------------------------------------

    def _handle_meta(self, command: MetaCommand) -> list[RenderInstruction]:
        if command.kind == "help":
            return [
                RenderInstruction(kind="log", content=""),
                RenderInstruction(kind="render_help"),
            ]
        if command.kind == "copy":
            if not self._last_agent_reply:
                return [
                    RenderInstruction(
                        kind="log",
                        content="[yellow]nothing to copy yet - no agent reply in this session.[/yellow]",
                    )
                ]
            return [
                RenderInstruction(
                    kind="copy_to_clipboard", content=self._last_agent_reply
                )
            ]
        if command.kind == "expand":
            if self._last_full_input is None:
                return [
                    RenderInstruction(
                        kind="log",
                        content="[dim]Nothing truncated to expand - the last input was shown in full.[/dim]",
                    )
                ]
            return [_box("Full input", self._last_full_input, "#7c8bff")]
        if command.kind == "clear":
            return [
                RenderInstruction(kind="clear"),
                RenderInstruction(kind="mode_line"),
            ]
        if command.kind == "packs":
            return [
                RenderInstruction(
                    kind="log",
                    content=settings_commands.render_packs_overview(
                        self.config_manager
                    ),
                )
            ]
        raise AssertionError(f"unreachable meta kind: {command.kind!r}")

    def _handle_unknown(self, command: UnknownInput) -> list[RenderInstruction]:
        if command.stage == "home":
            return [
                _echo_instr(self.echo_text(command.text)),
                RenderInstruction(
                    kind="log",
                    content="[dim]type 'chat' to start chatting, or 'settings' for configuration[/dim]",
                ),
            ]
        return [_echo_instr(self.echo_text(command.text))]

    # ------------------------------------------------------------------
    # workspace stage
    # ------------------------------------------------------------------

    def _handle_workspace_path(
        self, command: WorkspacePathInput
    ) -> list[RenderInstruction]:
        instr = [_echo_instr(self.echo_text(command.text))]

        if command.is_cancel:
            self.stage = self._workspace_return_stage
            return instr + [
                RenderInstruction(
                    kind="log", content="[dim]— workspace unchanged —[/dim]"
                ),
                RenderInstruction(kind="mode_line"),
            ]

        path = Path(command.text).expanduser().resolve()
        if not path.exists() or not path.is_dir():
            return instr + [
                RenderInstruction(
                    kind="log", content=f"[red]not a valid directory: {path}[/red]"
                )
            ]

        self.workspace_root = path
        init_workspace_logging(path)
        self.messages.bind_store(ConversationStore(path))

        if self.stage == "workspace":
            # First-run flow cascades straight into the model picker.
            self._pack_return_stage = "chat"
            return instr + [
                RenderInstruction(
                    kind="log",
                    content=f"[green]✓[/green] workspace set: [#236f9b]{path}[/#236f9b]",
                ),
                RenderInstruction(kind="mode_line"),
                *self.start_model_picker(),
            ]

        # workspace_edit: return to wherever `workspace` was invoked from.
        self.stage = self._workspace_return_stage
        return instr + [
            RenderInstruction(
                kind="log", content=f"[green]✓[/green] workspace set: {path}"
            ),
            RenderInstruction(kind="mode_line"),
        ]

    # ------------------------------------------------------------------
    # confirm stages (reset / remove-pack / activate-pack)
    # ------------------------------------------------------------------

    def _handle_confirm(self, command: ConfirmInput) -> list[RenderInstruction]:
        instr = [_echo_instr(self.echo_text(command.text))]
        if command.stage == "awaiting_reset_confirm":
            return instr + self._resolve_reset_confirm(command.confirmed)
        if command.stage == "awaiting_remove_pack_confirm":
            return instr + self._resolve_remove_pack_confirm(command.confirmed)
        if command.stage == "awaiting_activate_confirm":
            return instr + self._resolve_activate_confirm(command.confirmed)
        raise AssertionError(f"unreachable confirm stage: {command.stage!r}")

    def _resolve_reset_confirm(self, confirmed: bool) -> list[RenderInstruction]:
        out: list[RenderInstruction] = []
        if confirmed and self._pending_reset_scope == "temp":
            n = len(self.messages)
            self.messages.clear()
            out.append(
                RenderInstruction(
                    kind="log",
                    content=f"[green]Temp memory cleared[/green] ({n} message(s) dropped from context).",
                )
            )
        else:
            out.append(
                RenderInstruction(kind="log", content="[dim]Reset cancelled.[/dim]")
            )
        self.stage = self._reset_return_stage
        out.append(RenderInstruction(kind="mode_line"))
        return out

    def _start_remove_pack_confirm(self, name: str) -> list[RenderInstruction]:
        if name not in self.config_manager.packs.list_packs():
            return [_box("Unknown pack", f"'{name}' isn't a known pack.", "#C4374F")]
        self._remove_pack_return_stage = self.stage
        self._pending_remove_pack = name
        self.stage = "awaiting_remove_pack_confirm"
        return [
            _box(
                "Confirm delete",
                f"This permanently deletes pack '[bold]{name}[/bold]' and every "
                "model inside it. A backup is taken automatically before the "
                "delete.\n\nType [bold]y[/bold] to confirm, anything else to cancel.",
                "#C4374F",
            )
        ]

    def _resolve_remove_pack_confirm(self, confirmed: bool) -> list[RenderInstruction]:
        name = self._pending_remove_pack
        out: list[RenderInstruction] = []
        if confirmed:
            assert name is not None, (
                "remove-pack-confirm entered without a pending pack"
            )
            try:
                self.config_manager.backup()
                self.config_manager.packs.remove_pack(name)
                self.config_manager.save()
                self._pack_choices = load_pack_choices(self.config_manager)
                out.append(
                    RenderInstruction(
                        kind="log",
                        content=(
                            f"[green]✓[/green] removed pack '{name}'.\n"
                            "[dim]A backup was taken first - run 'restore' if this was a mistake.[/dim]"
                        ),
                    )
                )
            except ConfigError as exc:
                out.append(RenderInstruction(kind="log", content=f"[red]{exc}[/red]"))
        else:
            out.append(
                RenderInstruction(kind="log", content="[dim]Delete cancelled.[/dim]")
            )
        self.stage = self._remove_pack_return_stage
        out.append(RenderInstruction(kind="mode_line"))
        return out

    def _resolve_activate_confirm(self, confirmed: bool) -> list[RenderInstruction]:
        pack = self._pending_activate_pack
        out: list[RenderInstruction] = []
        if confirmed:
            assert pack is not None, "activate-confirm entered without a pending pack"
            self.selected_pack = pack
            out.append(_box_active_pack(pack))
        else:
            out.append(
                RenderInstruction(
                    kind="log", content="[dim]Kept the current active pack.[/dim]"
                )
            )
        self.stage = self._activate_return_stage
        out.append(RenderInstruction(kind="restore_input"))
        out.append(RenderInstruction(kind="mode_line"))
        return out

    # ------------------------------------------------------------------
    # model / pack picker
    # ------------------------------------------------------------------

    def start_model_picker(self) -> list[RenderInstruction]:
        self._pack_choices = load_pack_choices(self.config_manager)
        self.stage = "model_pick"
        options = [{"label": c.label, "value": c.name} for c in self._pack_choices]
        options.append({"label": "+ Add new pack", "value": "__add_pack__"})
        options.append({"label": "Cancel", "value": "__cancel__"})
        return [
            RenderInstruction(
                kind="mount_options",
                content={
                    "options": options,
                    "prompt": "Select a model pack",
                    "list_id": "pack-options",
                },
            )
        ]

    def handle_option(self, list_id: str, option_id: str) -> list[RenderInstruction]:
        """Routes an OptionList selection. Not a `Command` - selections
        aren't text and don't go through the parser (see commands/types.py)."""
        if self.stage == "model_pick" and list_id == "pack-options":
            return self._resolve_pack_pick(option_id)
        if list_id == "wiz-options":
            return self._advance_wizard_option(option_id)
        return []

    def _resolve_pack_pick(self, chosen_id: str) -> list[RenderInstruction]:
        if chosen_id == "__cancel__":
            self.stage = self._pack_return_stage
            return [
                RenderInstruction(kind="restore_input"),
                RenderInstruction(
                    kind="log", content="[dim]— cancelled, pack unchanged —[/dim]"
                ),
                RenderInstruction(kind="mode_line"),
            ]
        if chosen_id == "__add_pack__":
            return self.start_wizard(
                activate=True, return_stage=self._pack_return_stage
            )
        self.selected_pack = chosen_id
        self.stage = self._pack_return_stage
        return [
            RenderInstruction(kind="restore_input"),
            _box_active_pack(chosen_id),
            RenderInstruction(kind="mode_line"),
            RenderInstruction(
                kind="log",
                content="\n[#7F849C]────────────────Chat───────────────────[/#7F849C]\n",
            ),
        ]

    # ------------------------------------------------------------------
    # suggest wizard
    # ------------------------------------------------------------------

    def start_wizard(
        self, *, activate: bool = False, return_stage: str = "settings"
    ) -> list[RenderInstruction]:
        prompt = self._wizard.start(activate=activate, return_stage=return_stage)
        return self._wizard_prompt_instructions(prompt, stage="wiz_provider")

    def _advance_wizard_option(self, option_id: str) -> list[RenderInstruction]:
        state = self._wizard_state
        assert state is not None
        if state.step == "provider":
            prompt = self._wizard.choose_provider(state, option_id)
            return self._wizard_prompt_instructions(prompt, stage="wiz_model")
        if state.step == "model":
            prompt = self._wizard.choose_model(state, option_id)
            next_stage = "wiz_model_custom" if option_id == "__custom__" else "wiz_pack"
            return self._wizard_prompt_instructions(prompt, stage=next_stage)
        if state.step == "pack":
            prompt = self._wizard.choose_pack(state, option_id)
            next_stage = "wiz_pack_new" if option_id == "__new__" else "wiz_target"
            return self._wizard_prompt_instructions(prompt, stage=next_stage)
        if state.step == "target":
            final_state = self._wizard.choose_target(state, option_id)
            return self._commit_wizard(final_state)
        return []

    def _handle_wizard_free_text(
        self, command: WizardFreeText
    ) -> list[RenderInstruction]:
        instr = [_echo_instr(self.echo_text(command.text))]
        state = self._wizard_state
        assert state is not None
        if command.field == "model":
            prompt = self._wizard.submit_custom_model(state, command.text)
            return instr + self._wizard_prompt_instructions(prompt, stage="wiz_pack")
        prompt = self._wizard.submit_new_pack_name(state, command.text)
        return instr + self._wizard_prompt_instructions(prompt, stage="wiz_target")

    def _wizard_prompt_instructions(
        self, prompt: WizardPrompt, *, stage: str
    ) -> list[RenderInstruction]:
        self._wizard_state = prompt.state
        self.stage = stage
        if prompt.choices is None:
            return [
                RenderInstruction(kind="restore_input"),
                RenderInstruction(kind="log", content=f"[bold]{prompt.prompt}[/bold]:"),
                RenderInstruction(kind="mode_line"),
            ]
        options = [{"label": c.label, "value": c.value} for c in prompt.choices]
        return [
            RenderInstruction(kind="mode_line"),
            RenderInstruction(
                kind="mount_options",
                content={
                    "options": options,
                    "prompt": prompt.prompt,
                    "list_id": "wiz-options",
                },
            ),
        ]

    def _commit_wizard(self, state: WizardState) -> list[RenderInstruction]:
        result = self._wizard.commit(state)
        self._wizard_state = None

        if not result.ok:
            self.stage = result.return_stage
            return [
                RenderInstruction(kind="log", content=f"[red]{result.error}[/red]"),
                RenderInstruction(kind="restore_input"),
                RenderInstruction(kind="mode_line"),
            ]

        out = [
            RenderInstruction(
                kind="log",
                content=(
                    f"[green]✓[/green] added {result.provider}/{result.model} "
                    f"to '{result.pack}' ({result.target})."
                ),
            )
        ]

        if result.activate:
            self._pending_activate_pack = result.pack
            self._activate_return_stage = result.return_stage
            self.stage = "awaiting_activate_confirm"
            out.append(
                _box(
                    "Activate this pack?",
                    f"Set '[bold]{result.pack}[/bold]' as the active pack now?\n\n"
                    "Type [bold]y[/bold] to confirm, anything else to keep "
                    "the current active pack.",
                    "#A6E3A1",
                )
            )
            out.append(RenderInstruction(kind="restore_input"))
            out.append(RenderInstruction(kind="mode_line"))
            return out

        self.stage = result.return_stage
        out.append(RenderInstruction(kind="restore_input"))
        out.append(RenderInstruction(kind="mode_line"))
        return out

    # ------------------------------------------------------------------
    # chat turn (the one async/streaming path)
    # ------------------------------------------------------------------

    async def handle_chat_turn(
        self, command: ChatMessage
    ) -> AsyncIterator[RenderInstruction]:
        """Async generator: the View iterates this and draws each
        instruction as it arrives, instead of `handle()`'s
        all-at-once list - an agent turn genuinely takes time (LLM
        call, possibly a tool-approval round trip) and the View needs
        to show the echo + 'thinking' status immediately, not after
        the whole turn resolves."""
        self._turn_counter += 1
        echo_id = f"turn-echo-{self._turn_counter}"
        user_text = command.text
        yield RenderInstruction(
            kind="echo",
            content={"text": self.echo_text(user_text), "style": "bold cyan"},
            id=echo_id,
        )
        yield RenderInstruction(kind="status", content="[dim]● thinking…[/dim]")

        try:
            assert self.workspace_root is not None, "Workspace root must be set"
            reply = await run_inner_loop(
                user_input=user_text,
                messages=self.messages,
                registry=self.tool_registry,
                dispatcher=self.dispatcher,
                workspace_root=self.workspace_root,
                name_pack=self.selected_pack,
                approve_fn=self._approve_via_service,
            )
        except Exception as exc:  # noqa: BLE001 - agent-turn error boundary
            yield RenderInstruction(kind="status", content="")
            yield RenderInstruction(kind="remove_echo", content=echo_id)
            for instr in self._error_instructions(
                "Agent turn failed", exc, user_text=user_text
            ):
                yield instr
            return

        yield RenderInstruction(kind="status", content="")
        self._last_agent_reply = reply
        yield RenderInstruction(kind="remove_echo", content=echo_id)
        body = f"[bold cyan]›[/bold cyan] {self.echo_text(user_text)}\n\n{reply}"
        yield _box("Agent", body, "#b98cff")

    async def _approve_via_service(
        self, tool_name: str, args: dict[str, Any], workspace_root: Path
    ) -> bool:
        """The `approve_fn` injected into `run_inner_loop`. Writing the
        approval-preview box is the View's job (it needs `write_log`),
        so this only does the state transition + wait; the caller
        (`ui/app.py`'s bridge) is responsible for rendering the
        `render_approval_preview` box before this resolves and restoring
        the previous stage after."""
        previous_stage = self.stage
        self.stage = "awaiting_approval"
        approved = await self.approval.request(tool_name, args, workspace_root)
        self.stage = previous_stage
        return approved

    def pending_approval_preview(self) -> str | None:
        """Lets the View render the approval box right when a request
        starts (it can't `await` inside a sync render call, so it polls
        this once `_approve_via_service` has set `self.stage`)."""
        if not self.approval.awaiting or self.approval.pending_tool is None:
            return None
        call = ToolCallInfo(
            tool_name=self.approval.pending_tool,
            args=self.approval.pending_args or {},
            workspace_root=self.workspace_root or Path.cwd(),
        )
        return render_approval_preview(call)

    # ------------------------------------------------------------------
    # error rendering (verbose.errors is config state, so the decision
    # of "show a traceback or not" is the Controller's; the box itself
    # is content the View draws unmodified)
    # ------------------------------------------------------------------

    def _error_instructions(
        self, title: str, exc: Exception, *, user_text: str | None = None
    ) -> list[RenderInstruction]:
        prefix = (
            f"[bold cyan]›[/bold cyan] {user_text}\n\n" if user_text is not None else ""
        )
        message = f"{prefix}[red]{type(exc).__name__}: {exc}[/red]"

        verbose_errors = False
        try:
            verbose_errors = bool(self.config_manager.config.verbose.errors)
        except AttributeError:
            pass

        if verbose_errors:
            import traceback

            tb = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
            message += f"\n\n[dim]{tb.strip()}[/dim]"
        else:
            message += (
                "\n\n[dim]set verbose.errors true to see the full traceback.[/dim]"
            )

        return [_box(title, message, "red")]

    def report_error(self, title: str, exc: Exception) -> list[RenderInstruction]:
        """Public wrapper for the View's top-level catch-all boundary
        (`on_chat_text_area_submitted`'s try/except)."""
        return self._error_instructions(title, exc)


def _box_active_pack(name: str) -> RenderInstruction:
    return RenderInstruction(
        kind="log", content=f"[green]✓[/green] active pack: [#4ad851]{name}[/#4ad851]"
    )
