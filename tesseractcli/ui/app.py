"""
tesseractcli/ui/app.py

Full rewrite. The old version was a Textual screen-stack app:
`App.on_mount` pushed `WelcomeScreen`, which on a keypress caused the App
to pop it and push `WorkspaceSelectorScreen`, then `ModelPickerScreen`,
then `ChatScreen` - four separate full-screen `Screen`s, each with its
own `Header`/`Footer`, replacing the one before it.

This version has exactly one screen (the App itself, no `Screen`
subclasses at all): a persistent `RichLog` scrollback that nothing ever
clears, and one `Input` fixed at the bottom that is reused for every
stage (workspace path, model pack, chat messages, tool approval, and the
`settings`/`chat`/`home`/`model` navigation commands). What used to be a
screen transition is now just "write some text into the log and change
`self.stage`".

Known simplification worth flagging: `RichLog` can only append lines, it
can't remove a specific line once written (unlike a `Screen` you can pop
and discard). So the transient "thinking" status while the agent is
running is shown on a separate one-line `Static` above the input, not in
the log itself - that line gets overwritten/cleared, the log never does.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from textual import work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.css.query import NoMatches
from textual.widgets import OptionList, RichLog, Static, TextArea
from textual.widgets.option_list import Option

from tesseractcli.agent.loop import run_inner_loop
from tesseractcli.config.global_config.manager import ConfigManager
from tesseractcli.config.provider_catalog import PROVIDER_CATALOG
from tesseractcli.llm.dispatcher import LLMDispatcher
from tesseractcli.models.exceptions import ConfigError
from tesseractcli.tools.registry_builder import build_registry
from tesseractcli.ui.views import settings_commands
from tesseractcli.ui.views.approval_view import ToolCallInfo, render_approval_preview
from tesseractcli.ui.views.banner import build_banner_panel, build_tagline
from tesseractcli.ui.views.box import render_box
from tesseractcli.ui.views.help_view import render_help
from tesseractcli.ui.views.home_view import render_home
from tesseractcli.ui.views.model_picker import PackChoice, load_pack_choices
from tesseractcli.ui.views.settings_view import render_settings
from tesseractcli.ui.widgets.chat_input import ChatTextArea

# Bare-word navigation commands, recognized only on an exact (stripped,
# case-insensitive) match against the whole input - not a slash-command
# parser. This was an explicit tradeoff: it means you can't send a chat
# message that is literally just the word "chat"/"settings"/"home"/
# "model"/"exit"/"quit" while in the chat stage, since it will be read
# as navigation instead. Accepted because that's the exact UX asked for
# (bare words, not "/settings"); flagging it here in case it becomes a
# real problem once real usage shows people actually want to send those
# single words.
NAV_COMMANDS = {"settings", "chat", "home", "model", "exit", "quit"}

# Single-letter shortcuts resolved before NAV_COMMANDS matching, so "q"
# closes the app the same way typing "exit" does.
NAV_ALIASES = {"q": "exit"}

# Meta/utility shortcuts available from (almost) any stage - not gated
# behind NAV_COMMANDS' {"chat","settings","home"} restriction, since
# "how do I even use this" and "get me out of here" need to work
# mid-workspace-setup or mid-wizard too, not just once you've already
# reached chat.
#
# Dash convention (decided): a single leading "-" is a short flag-style
# shortcut (1-4 letters: -h, -cfg, -q, -c), and a leading "--" spells
# the same command out in full (--help, --config, --quit) - the
# classic getopt/argparse short-option/long-option shape, which is
# already what typer (this project's CLI framework) uses. Picked that
# over the reverse (long form single-dash, short form double-dash)
# specifically because it's the convention everyone already knows -
# nothing new to learn on top of a CLI tool.
GLOBAL_ALIASES = {
    "-h": "help", "--help": "help", "?": "help",
    "-cfg": "settings", "--config": "settings",
    "-cnf": "settings",  # soft-deprecated synonym of -cfg, kept for compatibility
    "-q": "exit", "--quit": "exit",
    "--chat": "chat",
    "--home": "home",
    "--model": "model",
    "-c": "copy", "copy": "copy",
    "-cls": "clear", "--clear": "clear", "clear": "clear", "cls": "clear",
    "-p": "packs", "--packs": "packs", "packs": "packs",
}

# Stages where free text is being captured *verbatim on purpose* (a
# tool-approval y/n, a hand-typed model id, a hand-typed new pack
# name) - GLOBAL_ALIASES is deliberately not intercepted here, or
# "help"/"-h" could never actually be typed as, say, a literal model
# id. Same tradeoff already documented above for NAV_COMMANDS.
_FREE_TEXT_STAGES = {"awaiting_approval", "wiz_model_custom", "wiz_pack_new"}


class TesseractApp(App):
    TITLE = "TesseractCLI"

    BINDINGS = [
        Binding("ctrl+c", "quit", "Exit"),
    ]

    # Layout uses 1fr/auto (not fixed cells) throughout so the whole app
    # reflows with the terminal instead of just the logo - the log grows
    # to fill available height, and prompt/status rows size to content.
    #
    # The input itself is deliberately NOT a boxed field: `border: none`
    # plus a transparent background makes it blend into the scrollback,
    # and the actual "you can type here" affordance is the colored `›`
    # glyph in `#prompt-glyph` sitting to its left - closer to a real
    # terminal/REPL prompt than a form input.
    CSS = """
    Screen { layout: vertical; }
    #scrollback { height: 1fr; width: 1fr; padding: 0 1; }
    #status-line { height: 1; width: 1fr; padding: 0 1; color: $text-muted; }
    #mode-line { height: 1; width: 1fr; padding: 0 1; color: #7c8bff; }
    #input-area { height: auto; width: 1fr; padding: 0 1 1 1; }
    #prompt-row { height: auto; width: 1fr; }
    #prompt-glyph { width: auto; padding: 0 1 0 0; color: #4dd8ff; text-style: bold; }
    #main-input { width: 1fr; border: none; background: transparent; padding: 0; height: 1; }
    #main-input:focus { border: none; }
    """

    def on_text_area_changed(self, event: TextArea.Changed) -> None:
        """`#main-input` starts at `height: 1` (see CSS) since almost
        every stage - workspace path, settings commands, wizard steps,
        approval y/n - is a one-liner; this is what makes it actually
        grow for the one stage (chat) where someone might paste a
        multi-line code block. Clamped to 8 lines so a huge paste
        scrolls inside the box instead of pushing the input off-screen."""
        if event.text_area.id != "main-input":
            return
        lines = event.text_area.document.line_count
        event.text_area.styles.height = max(1, min(lines, 8))

    def compose(self) -> ComposeResult:
        yield RichLog(id="scrollback", markup=True, wrap=True, highlight=False)
        yield Static("", id="status-line")
        yield Static("", id="mode-line")
        with Vertical(id="input-area"):
            with Horizontal(id="prompt-row"):
                yield Static("›", id="prompt-glyph")
                yield ChatTextArea(id="main-input", placeholder="")

    # ------------------------------------------------------------------
    # stage property
    # ------------------------------------------------------------------
    #
    # A plain `self.stage = "..."` attribute used to be set directly at
    # ~10 call sites (navigation, workspace/model-pick, every wizard
    # step, tool approval). Adding autocomplete meant `#main-input`'s
    # `command_choices` needs to stay in sync with whatever stage is
    # active - rather than adding a second call next to all ~10 of
    # those assignments (and inevitably missing one later), `stage` is
    # a property so every existing `self.stage = "..."` assignment
    # keeps working unchanged and automatically refreshes autocomplete.

    @property
    def stage(self) -> str:
        return self._stage

    @stage.setter
    def stage(self, value: str) -> None:
        self._stage = value
        self._refresh_command_choices()

    def _refresh_command_choices(self) -> None:
        """Ghost-text autocomplete choices for `#main-input`: the bare
        nav words (chat/settings/home/model/exit/quit) wherever they're
        valid, plus the full settings command grammar while actually in
        `settings` - empty everywhere else (wizard steps, approval y/n,
        workspace path, and plain chat text) so it never suggests
        something irrelevant over a real message to the agent."""
        choices: list[str] = []
        if self.stage in {"chat", "settings", "home"}:
            choices.extend(NAV_COMMANDS)
        if self.stage == "settings":
            choices.extend(settings_commands.COMMAND_CHOICES)
        try:
            self.query_one("#main-input", ChatTextArea).command_choices = choices
        except NoMatches:
            pass  # stage set before compose() has run yet

    def on_mount(self) -> None:
        # Shared state, built once - same as the old on_mount, just no
        # longer paired with `self.push_screen(WelcomeScreen())`.
        self.tool_registry = build_registry()
        self.dispatcher = LLMDispatcher()
        # The single source of truth for global_config.yaml (packs,
        # models, paths, agent defaults). `render_settings`,
        # `settings_commands.handle`, and `load_pack_choices` all read
        # and write through this one instance - nothing in the UI talks
        # to the YAML file or `config/settings.py` directly.
        self.config_manager = ConfigManager()
        self._load_config_safely()
        self.workspace_root: Path | None = None
        self.selected_pack: str | None = None
        self.messages: list = []  # BaseMessage list, mutated in place by run_inner_loop
        self._last_agent_reply: str = ""  # backs the 'copy'/-c command

        self.stage: str = "workspace"
        self._pack_return_stage: str = "chat"
        self._pack_choices: list[PackChoice] = []
        self._approval_event: asyncio.Event | None = None
        self._approval_result: bool = False
        self._wiz: dict[str, Any] = {}  # scratch state for the interactive 'suggest' wizard

        self._print_banner()
        self.write_log("[dim]Working on:[/dim] " + str(Path.cwd()))
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
        self.write_log("[bold]Workspace folder:[/bold] (press Enter to accept, or type a path)")
        self.query_one("#main-input", ChatTextArea).value = str(Path.cwd())
        self.query_one("#main-input", ChatTextArea).focus()
        self._refresh_mode_line()

    def _load_config_safely(self) -> None:
        """`config_manager.load()` raises `InvalidConfigError` on a
        corrupt/schema-mismatched `global_config.yaml` and
        `ConfigFileNotFoundError` if it's missing and can't be created -
        both are `ConfigError` subclasses. Previously this call was
        unguarded, so either case would crash the app before a single
        widget could render a helpful message. Now it's caught, the
        broken config is backed up automatically (`reset()`'s own
        `keep_backup=True`, the same backup mechanism `settings restore`
        will read from), and the app still starts on working defaults -
        the user sees exactly what went wrong instead of a bare
        traceback and can fix or restore their config from `settings`."""
        try:
            self.config_manager.load()
        except ConfigError as exc:
            self._pending_config_error = (type(exc).__name__, str(exc))
            self.config_manager.reset(keep_backup=True)
        else:
            self._pending_config_error = None

    # ------------------------------------------------------------------
    # log / status helpers
    # ------------------------------------------------------------------

    def write_log(self, renderable: Any) -> None:
        self.query_one("#scrollback", RichLog).write(renderable)

    def set_status(self, text: str) -> None:
        self.query_one("#status-line", Static).update(text)

    def _refresh_mode_line(self) -> None:
        """Always-visible workspace/pack/stage strip, distinct in color
        (#7c8bff, the purple end of the logo gradient) from both the
        cyan input prompt and the plain scrollback text - a persistent
        status bar so this context isn't only visible on the 'home'/
        'settings' screens."""
        workspace = str(self.workspace_root) if self.workspace_root else "(not set)"
        pack = self.selected_pack or "(not set)"
        self.query_one("#mode-line", Static).update(
            f"[dim]{workspace}[/dim]  •  pack: [bold]{pack}[/bold]  •  {self.stage}"
        )

    def _print_banner(self) -> None:
        self.write_log(build_banner_panel(self.size.width))
        self.write_log(build_tagline())

    # ------------------------------------------------------------------
    # input routing - the whole replacement for the old push_screen chain
    # ------------------------------------------------------------------

    def on_chat_text_area_submitted(self, event: ChatTextArea.Submitted) -> None:
        """Thin wrapper: every input path funnels through here, so this
        is the single choke point where an unhandled exception anywhere
        below (a bad command, a bug in a view module, a config error we
        forgot to catch) would otherwise bubble up to Textual's default
        handler and crash the whole app to a traceback screen. Instead
        it's caught here and rendered as a normal error panel in the
        scrollback - the app keeps running and the user can just try
        again, the same way a real terminal survives a bad command."""
        try:
            self._route_input(event)
        except Exception as exc:  # noqa: BLE001 - intentional catch-all boundary
            self._report_error("Internal error", exc)

    def _report_error(self, title: str, exc: Exception) -> None:
        import traceback

        tb = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
        self.write_log(
            render_box(
                title,
                f"[red]{type(exc).__name__}: {exc}[/red]\n\n[dim]{tb.strip()}[/dim]",
                style="red",
            )
        )

    def _route_input(self, event: ChatTextArea.Submitted) -> None:
        text = event.value
        input_widget = self.query_one("#main-input", ChatTextArea)
        input_widget.value = ""

        stripped = text.strip()
        if not stripped:
            return

        if self.stage not in _FREE_TEXT_STAGES:
            global_cmd = GLOBAL_ALIASES.get(stripped.lower())
            if global_cmd == "help":
                self.write_log(f"[dim]›[/dim] {text}")
                self.write_log("")
                self.write_log(render_help(self))
                return
            if global_cmd == "copy":
                self.write_log(f"[dim]›[/dim] {text}")
                self._copy_last_reply()
                return
            if global_cmd == "clear":
                self._clear_scrollback()
                return
            if global_cmd == "packs":
                self.write_log(f"[dim]›[/dim] {text}")
                self.write_log(settings_commands.render_packs_overview(self.config_manager))
                return
            if global_cmd and self.stage in {"chat", "settings", "home"}:
                self._navigate(global_cmd)
                return

        if self.stage == "workspace":
            self._handle_workspace_input(stripped)
            return

        if self.stage == "awaiting_approval":
            self._handle_approval_input(stripped)
            return

        if self.stage == "wiz_model_custom":
            self.write_log(f"[dim]›[/dim] {text}")
            self._wiz["model"] = stripped
            self._wiz_show_pack_step()
            return

        if self.stage == "wiz_pack_new":
            self.write_log(f"[dim]›[/dim] {text}")
            self._wiz["pack"] = stripped
            self._wiz["pack_is_new"] = True
            self._wiz_show_target_step()
            return

        normalized = NAV_ALIASES.get(stripped.lower(), stripped.lower())
        if normalized in NAV_COMMANDS and self.stage in {"chat", "settings", "home"}:
            self._navigate(normalized)
            return

        if self.stage == "settings":
            self.write_log(f"[dim]›[/dim] {text}")
            words = stripped.split()
            first_word = words[0].lower()
            resolved_cmd = settings_commands.ALIASES.get(first_word, first_word)
            if resolved_cmd == "suggest" and len(words) == 1:
                self.start_suggest_wizard()
                return
            # Previously `handle()` was called with the raw `text`, so
            # aliases (p/ls/h/?/a/rm/r) only ever affected the "suggest"
            # equality check above and were silently ignored for every
            # other command - `handle()` re-splits the raw string itself
            # and never consulted ALIASES. Rebuild the command line with
            # the resolved first word so aliases actually take effect.
            resolved_text = " ".join([resolved_cmd, *words[1:]])
            self.write_log(settings_commands.handle(self.config_manager, resolved_text))
            return

        if self.stage == "home":
            self.write_log(f"[dim]›[/dim] {text}")
            self.write_log("[dim]type 'chat' to start chatting, or 'settings' for configuration[/dim]")
            return

        # stage == "chat": a real message for the agent
        self.write_log(f"[bold cyan]›[/bold cyan] {text}")
        self.run_agent_turn(text)

    def _navigate(self, command: str) -> None:
        if command == "chat":
            self.stage = "chat"
            self.write_log("[dim]— back to chat —[/dim]")
        elif command == "settings":
            self.stage = "settings"
            self.write_log("")
            self.write_log(render_settings(self))
        elif command == "home":
            self.stage = "home"
            self.write_log("")
            self.write_log(render_home(self))
        elif command == "model":
            self._pack_return_stage = self.stage
            self.show_model_picker()
        elif command in ("exit", "quit"):
            self.exit()
        self._refresh_mode_line()

    # ------------------------------------------------------------------
    # workspace stage (replaces WorkspaceSelectorScreen)
    # ------------------------------------------------------------------

    def _handle_workspace_input(self, text: str) -> None:
        path = Path(text).expanduser().resolve()
        self.write_log(f"[dim]›[/dim] {text}")

        if not path.exists() or not path.is_dir():
            self.write_log(f"[red]not a valid directory: {path}[/red]")
            return

        self.workspace_root = path
        self.write_log(f"[green]✓[/green] workspace set: {path}")
        self._pack_return_stage = "chat"
        self._refresh_mode_line()
        self.show_model_picker()

    # ------------------------------------------------------------------
    # model pack picker (replaces ModelPickerScreen)
    # ------------------------------------------------------------------

    def _mount_options(self, options: list[Option], prompt: str, *, list_id: str) -> None:
        """Shared helper: mount an inline OptionList over the Input with
        a heading, used by both the pack picker and every step of the
        suggest wizard below."""
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

    def _clear_scrollback(self) -> None:
        """Backs the 'clear'/-cl/cls command. `RichLog` (unlike a real
        terminal buffer) has no concept of "scroll back up past this
        point" once cleared - `.clear()` wipes it outright, same as a
        real terminal's `clear`/`cls` would. Re-prints the banner
        after, purely so clearing doesn't leave a totally blank screen
        with no sense of where you are."""
        self.query_one("#scrollback", RichLog).clear()
        self._print_banner()
        self._refresh_mode_line()

    def _copy_last_reply(self) -> None:
        """Backs the 'copy'/-c command. `RichLog` scrollback text can't
        be mouse-selected the way a normal terminal buffer can - that's
        a Textual limitation (the app owns mouse input for its
        widgets), not a bug here. Two ways around it: (1) most terminal
        emulators (Windows Terminal, iTerm2, kitty, Alacritty, WezTerm,
        GNOME Terminal) let you hold Shift while dragging to select
        text natively, bypassing the app entirely - no code involved;
        (2) this command, which pushes the last agent reply to the
        system clipboard over OSC 52 via Textual's own
        `App.copy_to_clipboard`, which works even over SSH."""
        if not self._last_agent_reply:
            self.write_log("[yellow]nothing to copy yet - no agent reply in this session.[/yellow]")
            return
        try:
            self.copy_to_clipboard(self._last_agent_reply)
            self.write_log("[green]✓[/green] last reply copied to clipboard.")
        except Exception as exc:  # noqa: BLE001 - clipboard support varies by terminal
            self.write_log(
                f"[yellow]couldn't reach the system clipboard ({exc}).[/yellow] Try "
                "holding Shift while you drag-select text - most terminals let you "
                "select natively that way, bypassing the app."
            )

    def show_model_picker(self) -> None:
        """Mounts an inline, arrow-key navigable OptionList in place of
        the Input - not a separate screen. Removed again the moment a
        choice is made (see `on_option_list_option_selected`)."""
        self._pack_choices = load_pack_choices(self.config_manager)

        if not self._pack_choices:
            self.write_log("")
            self.write_log(
                "[red]no packs configured yet.[/red] Go to 'settings' and run "
                "'add pack <name>' then 'add model <name> <provider> <model>' first."
            )
            self.stage = self._pack_return_stage
            return

        self.stage = "model_pick"
        self._mount_options(
            [Option(choice.label, id=choice.name) for choice in self._pack_choices],
            "Select a model pack",
            list_id="pack-options",
        )

    async def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        try:
            await self._route_option_selected(event)
        except Exception as exc:  # noqa: BLE001 - same boundary as on_input_submitted
            self._report_error("Internal error", exc)
            # Safety net: if the exception happened after `main-input`
            # was hidden (see `_mount_options`) but before whichever
            # branch below reached its own `_restore_input()`, the
            # input would otherwise stay hidden forever with no way to
            # type anything else - same failure mode the DuplicateIds
            # bug below used to trigger.
            self._restore_input()

    async def _route_option_selected(self, event: OptionList.OptionSelected) -> None:
        """Root cause of the old `DuplicateIds` crash: `Widget.remove()`
        only *schedules* removal (it returns an awaitable, it doesn't
        remove synchronously) - so the old code's `event.option_list.
        remove()` followed immediately by e.g. `_wiz_show_model_step()`
        (which mounts a fresh `OptionList` with the *same* `id="wiz-
        options"`) could run the new mount before the old widget had
        actually left the DOM, and Textual refuses to insert a second
        widget with an ID that's still in use. Every branch below now
        `await`s the removal before mounting whatever comes next, so
        the old one is guaranteed gone first."""
        if self.stage == "model_pick" and event.option_list.id == "pack-options":
            chosen_name = event.option.id
            await event.option_list.remove()
            self._restore_input()

            self.selected_pack = chosen_name
            self.write_log(f"[green]✓[/green] active pack: {chosen_name}")
            self.stage = self._pack_return_stage
            self._refresh_mode_line()
            return

        if self.stage == "wiz_provider" and event.option_list.id == "wiz-options":
            await event.option_list.remove()
            self._wiz["provider"] = event.option.id
            self._wiz_show_model_step()
            return

        if self.stage == "wiz_model" and event.option_list.id == "wiz-options":
            await event.option_list.remove()
            if event.option.id == "__custom__":
                self.stage = "wiz_model_custom"
                self._restore_input()
                self.write_log("[bold]Type the model id[/bold] (e.g. gpt-4o-mini):")
                return
            self._wiz["model"] = event.option.id
            self._wiz_show_pack_step()
            return

        if self.stage == "wiz_pack" and event.option_list.id == "wiz-options":
            await event.option_list.remove()
            if event.option.id == "__new__":
                self.stage = "wiz_pack_new"
                self._restore_input()
                self.write_log("[bold]Type the new pack's name[/bold]:")
                return
            self._wiz["pack"] = event.option.id
            self._wiz["pack_is_new"] = False
            self._wiz_show_target_step()
            return

        if self.stage == "wiz_target" and event.option_list.id == "wiz-options":
            await event.option_list.remove()
            self._wiz["target"] = event.option.id
            self._wiz_commit()
            return

    # ------------------------------------------------------------------
    # interactive "suggest" wizard: provider -> model -> pack -> pool/fallback
    # ------------------------------------------------------------------
    #
    # Replaces the old plain-text 'suggest' output with an actual picker,
    # so choosing a model also wires it straight into a pack in one flow
    # instead of reading a suggestion list and then typing
    # 'add model ...' by hand afterwards. The two free-text steps
    # (custom model id, new pack name) are handled back in
    # `on_input_submitted` via the `wiz_model_custom`/`wiz_pack_new`
    # stages, since an OptionList can't take arbitrary text.

    def start_suggest_wizard(self) -> None:
        self._wiz = {}
        self.stage = "wiz_provider"
        self._refresh_mode_line()
        options = [
            Option(f"{e.provider:<14} {e.label} — e.g. {', '.join(e.example_models)}", id=e.provider)
            for e in PROVIDER_CATALOG
        ]
        self._mount_options(options, "Pick a provider", list_id="wiz-options")

    def _wiz_show_model_step(self) -> None:
        self.stage = "wiz_model"
        self._refresh_mode_line()
        entry = next(e for e in PROVIDER_CATALOG if e.provider == self._wiz["provider"])
        options = [Option(model, id=model) for model in entry.example_models]
        options.append(Option("type a different model id…", id="__custom__"))
        self._mount_options(options, f"Pick a model ({entry.label})", list_id="wiz-options")

    def _wiz_show_pack_step(self) -> None:
        self.stage = "wiz_pack"
        self._refresh_mode_line()
        pack_names = self.config_manager.packs.list_packs()
        options = [Option(name, id=name) for name in pack_names]
        options.append(Option("+ create a new pack…", id="__new__"))
        self._mount_options(options, "Add to which pack?", list_id="wiz-options")

    def _wiz_show_target_step(self) -> None:
        self.stage = "wiz_target"
        self._refresh_mode_line()
        options = [
            Option("pool (primary)", id="pool"),
            Option("fallback", id="fallback"),
        ]
        self._mount_options(options, f"Pool or fallback for '{self._wiz['pack']}'?", list_id="wiz-options")

    def _wiz_commit(self) -> None:
        provider = self._wiz["provider"]
        model = self._wiz["model"]
        pack = self._wiz["pack"]
        target = self._wiz["target"]

        try:
            if self._wiz.get("pack_is_new"):
                self.config_manager.packs.add_pack(pack)
            self.config_manager.packs.add_model(pack, provider, model, target=target)
            self.config_manager.save()
            self.write_log(f"[green]✓[/green] added {provider}/{model} to '{pack}' ({target}).")
        except ConfigError as exc:
            self.write_log(f"[red]{exc}[/red]")

        self._wiz = {}
        self.stage = "settings"
        self._restore_input()
        self._refresh_mode_line()

    # ------------------------------------------------------------------
    # chat stage (replaces ChatScreen)
    # ------------------------------------------------------------------

    @work(exclusive=True)
    async def run_agent_turn(self, user_text: str) -> None:
        """Previously a raised exception here (provider error, hitting
        INNER_LOOP_MAX_ITERATIONS, a rate-limit that survived the
        dispatcher's own fallback/retry, a tool crash) would propagate
        out of this `@work` coroutine as a failed Worker, which Textual
        surfaces as an unhandled app-level exception - i.e. the whole
        app exits instead of just this one turn failing. Caught here
        now so a bad turn just prints an error and the user can keep
        chatting or fix their pack/config and retry, exactly the same
        as any other command failure."""
        self.set_status("[dim]● thinking…[/dim]")
        try:
            reply = await run_inner_loop(
                user_input=user_text,
                messages=self.messages,
                registry=self.tool_registry,
                dispatcher=self.dispatcher,
                workspace_root=self.workspace_root,
                name_pack=self.selected_pack,
                approve_fn=self.approve_via_ui,
            )
        except Exception as exc:  # noqa: BLE001 - agent-turn error boundary
            self.set_status("")
            self._report_error("Agent turn failed", exc)
            return
        self.set_status("")
        self._last_agent_reply = reply
        self.write_log(render_box("Agent", reply, style="#b98cff"))

    # ------------------------------------------------------------------
    # tool approval (replaces ApprovalModal)
    # ------------------------------------------------------------------

    async def approve_via_ui(self, tool_name: str, args: dict[str, Any], workspace_root: Path) -> bool:
        """The `approve_fn` injected into `run_inner_loop`. Same role as
        the old modal-backed version: block (from the agent loop's point
        of view) on an `asyncio.Event` so the Textual event loop stays
        responsive - the difference is the preview and y/n prompt are
        written straight into the log/input instead of a popup screen."""
        call = ToolCallInfo(tool_name=tool_name, args=args, workspace_root=workspace_root)

        self.write_log("")
        self.write_log(render_approval_preview(call))

        previous_stage = self.stage
        self.stage = "awaiting_approval"
        self._approval_event = asyncio.Event()
        self._approval_result = False

        await self._approval_event.wait()

        self.stage = previous_stage
        return self._approval_result

    def _handle_approval_input(self, text: str) -> None:
        approved = text.strip().lower() in {"y", "yes"}
        self.write_log(f"[dim]›[/dim] {text}")
        self._approval_result = approved
        if self._approval_event is not None:
            self._approval_event.set()


def run() -> None:
    TesseractApp().run()
