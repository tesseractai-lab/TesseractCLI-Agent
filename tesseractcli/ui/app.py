"""
tesseractcli/ui/app.py

The View layer, post-refactor. `TesseractApp` now does exactly four
things: (1) `compose()`/CSS/widget mounting, (2) turn a Textual event
into a `Command` via `commands.parser.parse` (or a plain (list_id,
option_id) pair for OptionList picks), (3) call `SessionController`,
(4) draw whatever `list[RenderInstruction]` comes back. It holds no
business state of its own beyond widget handles - `stage`,
`workspace_root`, `selected_pack`, wizard/confirm state all live on
`self.controller` now (see `controller/session.py`).

Was ~1480 lines before this refactor; the whole command grammar,
wizard state machine, and settings/reset/reload/approval orchestration
moved out to `commands/parser.py`, `controller/session.py`, and
`controller/pack_wizard.py`. See ARCHITECTURE.md for the full map.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, ClassVar

from textual import work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.css.query import NoMatches
from textual.widgets import OptionList, Static, TextArea
from textual.widgets.option_list import Option

from tesseractcli.commands.parser import NAV_COMMANDS, parse
from tesseractcli.commands.types import ChatMessage, RenderInstruction
from tesseractcli.config.global_config.manager import ConfigManager
from tesseractcli.controller.session import SessionController
from tesseractcli.llm.dispatcher import LLMDispatcher
from tesseractcli.llm.routing import RoutingResolver
from tesseractcli.models.exceptions import ConfigError
from tesseractcli.tools.registry_builder import build_registry
from tesseractcli.ui.views import settings_commands
from tesseractcli.ui.views.banner import build_banner_panel
from tesseractcli.ui.views.box import render_box
from tesseractcli.ui.views.help_view import render_help
from tesseractcli.ui.views.home_view import render_home
from tesseractcli.ui.views.settings_view import render_settings
from tesseractcli.ui.widgets.chat_input import ChatTextArea


def _truncate_path_display(path: Path | None, *, max_parts: int = 2) -> str:
    """Display a shortened workspace path.

    Examples:
        G:\\foo\\bar\\Project\\tests -> ~\\Project\\tests
        G:\\foo\\bar\\Project        -> ~\\Project
    """
    if path is None:
        return "(not set)"
    parts = path.parts
    tail = parts[-max_parts:] if len(parts) >= max_parts else parts
    return "~\\" + "\\".join(tail)


class SelectableStatic(Static):
    """`Static`, but explicit about wanting Textual's built-in
    click-drag text selection turned on. See the original module
    docstring (preserved in ARCHITECTURE.md) for why `RichLog` can't
    support this and every scrollback entry is one of these instead."""

    ALLOW_SELECT = True


class TesseractApp(App):
    TITLE = "TesseractCLI"

    BINDINGS: ClassVar[list[Binding | tuple[str, str] | tuple[str, str, str]]] = [
        Binding("ctrl+c", "quit", "Exit"),
    ]

    CSS = """
    Screen { layout: vertical; }
    #scrollback { height: 1fr; width: 1fr; padding: 0 1; }
    #status-line { height: 1; width: 1fr; padding: 0 1; color: $text-muted; }
    #input-area {
        height: auto; width: 1fr; margin: 0 1 1 1; padding: 0 1;
        border: round #3b3f51;
    }
    #prompt-row { height: auto; width: 1fr; }
    #prompt-glyph { width: auto; padding: 0 1 0 0; color: #4dd8ff; text-style: bold; }
    #main-input { width: 1fr; border: none; background: transparent; padding: 0; height: 1; }
    #main-input:focus { border: none; }
    #mode-line { height: 1; width: 1fr; padding: 0; color: #7c8bff; }
    """

    def __init__(self) -> None:
        super().__init__()
        self._pending_config_error: tuple[str, str] | None = None

    # ------------------------------------------------------------------
    # compose / lifecycle
    # ------------------------------------------------------------------

    def compose(self) -> ComposeResult:
        yield VerticalScroll(id="scrollback")
        yield Static("", id="status-line")
        with Vertical(id="input-area"):
            with Horizontal(id="prompt-row"):
                yield Static("›", id="prompt-glyph")
                yield ChatTextArea(id="main-input", placeholder="")
            yield Static("", id="mode-line")

    def on_text_area_changed(self, event: TextArea.Changed) -> None:
        if event.text_area.id != "main-input":
            return
        lines = event.text_area.document.line_count
        event.text_area.styles.height = max(1, min(lines, 8))

    def on_mount(self) -> None:
        tool_registry = build_registry()
        config_manager = ConfigManager()
        dispatcher = LLMDispatcher(RoutingResolver(config_manager))
        self._pending_config_error = self._load_config_safely(config_manager)

        self.controller = SessionController(
            config_manager=config_manager,
            dispatcher=dispatcher,
            tool_registry=tool_registry,
        )

        self._print_banner()
        self.write_log("")
        if self._pending_config_error is not None:
            exc_name, exc_msg = self._pending_config_error
            self.write_log(
                render_box(
                    "Config reset to defaults",
                    f"[red]{exc_name}: {exc_msg}[/red]\n\n"
                    "Your global_config.yaml couldn't be loaded, so it was backed up "
                    "and defaults were restored. Run [bold]settings[/bold] then "
                    "[bold]restore[/bold] to see and roll back to a previous backup.",
                    style="red",
                )
            )
            self.write_log("")
        self.query_one("#main-input", ChatTextArea).value = str(Path.cwd())
        self.query_one("#main-input", ChatTextArea).focus()
        self._refresh_command_choices()
        self._refresh_mode_line()

    def on_unmount(self) -> None:
        self.controller.messages.close()

    # ------------------------------------------------------------------
    # read-only proxies onto SessionController state - kept ONLY because
    # settings_view.render_settings / home_view.render_home /
    # help_view.render_help still expect a `TesseractApp`-shaped object
    # (`app.stage`, `app.workspace_root`, `app.selected_pack`,
    # `app.config_manager`) rather than the `SessionController` those
    # values actually live on now. Confirmed via `mypy tesseractcli`
    # against the real project (those three view modules type-hint
    # their parameter as `TesseractApp`, not `SessionController`).
    #
    # Deliberately NOT touched: settings_view.py / home_view.py /
    # help_view.py themselves - those weren't part of this refactor's
    # scope, so the compatibility shim lives here instead of forcing a
    # signature change on three files this refactor never reviewed.
    # If/when those three get updated to take `SessionController`
    # directly, delete this block and go back to passing
    # `self.controller` at the three call sites in `_draw_one`.
    # ------------------------------------------------------------------

    @property
    def stage(self) -> str:
        return self.controller.stage

    @property
    def workspace_root(self) -> Path | None:
        return self.controller.workspace_root

    @property
    def selected_pack(self) -> str | None:
        return self.controller.selected_pack

    @property
    def config_manager(self) -> ConfigManager:
        return self.controller.config_manager

    @staticmethod
    def _load_config_safely(config_manager: ConfigManager) -> tuple[str, str] | None:
        try:
            config_manager.load()
        except ConfigError as exc:
            config_manager.reset(keep_backup=True)
            return (type(exc).__name__, str(exc))
        return None

    # ------------------------------------------------------------------
    # log / status / mode-line - pure widget mechanics
    # ------------------------------------------------------------------

    def write_log(self, renderable: Any, *, id: str | None = None) -> SelectableStatic:
        widget = SelectableStatic(renderable, id=id)
        container = self.query_one("#scrollback", VerticalScroll)
        container.mount(widget)
        container.scroll_end(animate=False)
        return widget

    def set_status(self, text: str) -> None:
        self.query_one("#status-line", Static).update(text)

    def _refresh_mode_line(self) -> None:
        ws_display = _truncate_path_display(self.controller.workspace_root)
        pack = self.controller.selected_pack or "(not set)"
        self.query_one("#mode-line", Static).update(
            f"[dim]ws:[/dim] [#236f9b]{ws_display}[/#236f9b] •  "
            f"[dim][bold][#4ad851]{pack}[/#4ad851][/bold][/dim]  •  "
            f"[dim][yellow] {self.controller.stage}[/yellow][/dim]"
        )

    def _refresh_command_choices(self) -> None:
        choices: list[str] = []
        stage = self.controller.stage
        if stage in {"chat", "settings", "home"}:
            choices.extend(NAV_COMMANDS)
        if stage == "settings":
            choices.extend(settings_commands.COMMAND_CHOICES)
        try:
            self.query_one("#main-input", ChatTextArea).command_choices = choices
        except NoMatches:
            pass

    def _print_banner(self) -> None:
        self.write_log(build_banner_panel(self.size.width))

    def _clear_scrollback(self) -> None:
        self.query_one("#scrollback", VerticalScroll).remove_children()
        self._print_banner()

    def _mount_options(
        self, options: list[Option], prompt: str, *, list_id: str
    ) -> None:
        self.write_log("")
        self.write_log(f"[bold]{prompt}[/bold] (↑/↓ then Enter):")
        input_widget = self.query_one("#main-input", ChatTextArea)
        input_widget.display = False
        option_list = OptionList(*options, id=list_id)
        self.query_one("#input-area", Vertical).mount(option_list)
        option_list.focus()

    def _restore_input(self) -> None:
        input_widget = self.query_one("#main-input", ChatTextArea)
        input_widget.display = True
        input_widget.value = ""
        input_widget.focus()

    # ------------------------------------------------------------------
    # RenderInstruction -> widget calls - the ONLY place that interprets
    # controller output. Nothing else in this class inspects controller
    # internals beyond `.stage`/`.workspace_root`/`.selected_pack` for
    # the mode-line/autocomplete, both read-only.
    # ------------------------------------------------------------------

    def _draw(self, instructions: list[RenderInstruction]) -> None:
        for instr in instructions:
            self._draw_one(instr)
        self._refresh_command_choices()

    def _draw_one(self, instr: RenderInstruction) -> None:
        kind, content = instr.kind, instr.content
        if kind == "echo":
            style = content.get("style", "dim")
            glyph = "[bold cyan]›[/bold cyan]" if "cyan" in style else "[dim]›[/dim]"
            self.write_log(f"{glyph} {content['text']}", id=instr.id)
        elif kind == "log":
            self.write_log(content)
        elif kind == "box":
            self.write_log(
                render_box(content["title"], content["body"], style=content["style"])
            )
        elif kind == "divider":
            self.write_log(
                f"\n[#6c7086]──────────────────── {content} ────────────────────[/#6c7086]\n"
            )
        elif kind == "clear":
            self._clear_scrollback()
        elif kind == "status":
            self.set_status(content)
        elif kind == "mode_line":
            self._refresh_mode_line()
        elif kind == "mount_options":
            options = [Option(o["label"], id=o["value"]) for o in content["options"]]
            self._mount_options(options, content["prompt"], list_id=content["list_id"])
        elif kind == "restore_input":
            self._restore_input()
        elif kind == "set_input_value":
            self.query_one("#main-input", ChatTextArea).value = content
        elif kind == "remove_echo":
            try:
                self.query_one(f"#{content}", SelectableStatic).remove()
            except NoMatches:
                pass
        elif kind == "render_settings":
            self.write_log(render_settings(self))
        elif kind == "render_home":
            self.write_log(render_home(self))
        elif kind == "render_help":
            self.write_log(render_help(self))
        elif kind == "copy_to_clipboard":
            try:
                self.copy_to_clipboard(content)
                self.write_log("[green]✓[/green] last reply copied to clipboard.")
            except Exception as exc:  # noqa: BLE001 - clipboard support varies by terminal
                self.write_log(
                    f"[yellow]couldn't reach the system clipboard ({exc}).[/yellow] Try "
                    "holding Shift while you drag-select text - most terminals let you "
                    "select natively that way, bypassing the app."
                )
        elif kind == "exit":
            self.exit()
        else:
            raise AssertionError(f"unhandled RenderInstruction kind: {kind!r}")

    # ------------------------------------------------------------------
    # input routing - the single choke point (unchanged boundary from
    # before), now just: parse -> controller.handle -> draw
    # ------------------------------------------------------------------

    def on_chat_text_area_submitted(self, event: ChatTextArea.Submitted) -> None:
        try:
            self._route_input(event)
        except Exception as exc:  # noqa: BLE001 - intentional catch-all boundary
            self._draw(self.controller.report_error("Internal error", exc))

    def _route_input(self, event: ChatTextArea.Submitted) -> None:
        text = event.value
        input_widget = self.query_one("#main-input", ChatTextArea)
        input_widget.value = ""

        if not text.strip():
            return

        command = parse(text, self.controller.stage)

        if isinstance(command, ChatMessage):
            self._run_chat_turn(command)
            return

        self._draw(self.controller.handle(command))

    @work(exclusive=True)
    async def _run_chat_turn(self, command: ChatMessage) -> None:
        """`@work` keeps this off the main event-loop turn so the UI
        stays responsive while `run_inner_loop` awaits the LLM/tools -
        same role as the old `run_agent_turn`. A pending tool-approval
        request surfaces mid-stream: once the Controller flips
        `stage` to `awaiting_approval`, this draws the approval preview
        box itself (the Controller can't - it has no `write_log`)."""
        async for instr in self.controller.handle_chat_turn(command):
            self._draw_one(instr)
            if self.controller.stage == "awaiting_approval":
                preview = self.controller.pending_approval_preview()
                if preview is not None:
                    self.write_log("")
                    self.write_log(render_box("Tool approval", preview, style="yellow"))
            self._refresh_command_choices()

    async def on_option_list_option_selected(
        self, event: OptionList.OptionSelected
    ) -> None:
        try:
            await self._route_option_selected(event)
        except Exception as exc:  # noqa: BLE001 - same boundary as text submission
            self._draw(self.controller.report_error("Internal error", exc))
            self._restore_input()

    async def _route_option_selected(self, event: OptionList.OptionSelected) -> None:
        list_id = event.option_list.id or ""
        option_id = str(event.option.id)
        # await the removal first - Widget.remove() only *schedules*
        # removal, so mounting the next OptionList with the same id
        # before this completes raises DuplicateIds (see the original
        # module docstring, preserved in ARCHITECTURE.md).
        await event.option_list.remove()
        self._draw(self.controller.handle_option(list_id, option_id))


def run() -> None:
    from tesseractcli.observability import configure_tracing

    configure_tracing()
    TesseractApp().run()
