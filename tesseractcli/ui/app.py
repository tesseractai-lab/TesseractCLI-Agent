"""
tesseractcli/ui/app.py

Full rewrite. The old version was a Textual screen-stack app:
`App.on_mount` pushed `WelcomeScreen`, which on a keypress caused the App
to pop it and push `WorkspaceSelectorScreen`, then `ModelPickerScreen`,
then `ChatScreen` - four separate full-screen `Screen`s, each with its
own `Header`/`Footer`, replacing the one before it.

This version has exactly one screen (the App itself, no `Screen`
subclasses at all): a scrollback that nothing ever clears on its own,
and one `Input` fixed at the bottom that is reused for every stage
(workspace path, model pack, chat messages, tool approval, and the
`settings`/`chat`/`home`/`model` navigation commands). What used to be a
screen transition is now just "write some text into the log and change
`self.stage`".

The scrollback was originally a `RichLog` (append-only, no mouse text
selection, can't remove a specific line once written). It's now a
`VerticalScroll` (`#scrollback`) holding one `SelectableStatic` widget
per entry, mounted via `write_log()` - this is what makes real
click-drag text selection work, and as a side effect also makes a
specific entry removable by id (see `_remove_echo`, used to swap the
"you typed this" line for the finished, bordered turn box once an
agent reply or error comes back)."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, ClassVar

from textual import work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.css.query import NoMatches
from textual.widgets import OptionList, Static, TextArea
from textual.widgets.option_list import Option

from tesseractcli.agent.loop import run_inner_loop
from tesseractcli.config.global_config.manager import ConfigManager
from tesseractcli.config.provider_catalog import PROVIDER_CATALOG
from tesseractcli.llm.dispatcher import LLMDispatcher
from tesseractcli.llm.routing import RoutingResolver
from tesseractcli.logging.workspace import init_workspace_logging
from tesseractcli.memory.store import ConversationStore, PersistentMessageList
from tesseractcli.models.exceptions import ConfigError
from tesseractcli.tools.registry_builder import build_registry
from tesseractcli.ui.views import settings_commands
from tesseractcli.ui.views.approval_view import ToolCallInfo, render_approval_preview
from tesseractcli.ui.views.banner import build_banner_panel
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

# "reset"/"reload" take an optional second word ("reset temp",
# "reload cfg") instead of being flat GLOBAL_ALIASES entries, so a
# future scope (e.g. a "session" reset, once sessions exist) can be
# added to *_SCOPES without changing the command's shape. No scope
# given falls back to DEFAULT_*_SCOPE. "-r"/"-rm"/"-rn" are already
# taken inside the settings-stage grammar (settings_commands.ALIASES:
# remove/rename) - these two use distinct heads on purpose so they
# never shadow that meaning while inside `settings`.
RESET_HEADS = {"-rst", "--reset", "reset"}
RESET_SCOPES = {"temp": "temp", "context": "temp", "memory": "temp"}
DEFAULT_RESET_SCOPE = "temp"

RELOAD_HEADS = {"-rl", "--reload", "reload"}
RELOAD_SCOPES = {"cfg": "config", "config": "config", "settings": "config"}
DEFAULT_RELOAD_SCOPE = "config"

# Second-token flags recognized right after a "-rm"/"remove"/"-a"/"add"
# head, so shorthand like "-rm -p mypack" or "add -md mypack groq llama"
# is understood as "remove pack mypack" / "add model mypack groq llama"
# without having to spell the word "pack"/"model" out. Kept intentionally
# tiny (just the two nouns the grammar actually has) rather than a
# general flag parser.
_SHORTHAND_KIND_FLAGS = {"-p": "pack", "pack": "pack", "-md": "model", "model": "model"}


def _translate_compound_shorthand(words: list[str]) -> str | None:
    """Rewrites compound shorthand ('-rm -p <name>', 'add -md <pack>
    <provider> <model>') into the canonical 'remove pack <name>' / 'add
    model ...' line `settings_commands.handle()` already understands.
    Returns None (falls through to normal handling) if `words` doesn't
    match this shape - also matches the already-canonical
    'add pack <n>'/'remove model ...' spelling, so it's a strict
    superset rather than a second competing grammar."""
    if len(words) < 2:
        return None
    head = settings_commands.ALIASES.get(words[0].lower(), words[0].lower())
    if head not in ("add", "remove"):
        return None
    kind = _SHORTHAND_KIND_FLAGS.get(words[1].lower())
    if kind is None:
        return None
    return " ".join([head, kind, *words[2:]])


# Stages where free text is being captured *verbatim on purpose* (a
# tool-approval y/n, a hand-typed model id, a hand-typed new pack
# name) - GLOBAL_ALIASES is deliberately not intercepted here, or
# "help"/"-h" could never actually be typed as, say, a literal model
# id. Same tradeoff already documented above for NAV_COMMANDS.
_FREE_TEXT_STAGES = {
    "awaiting_approval",
    "wiz_model_custom",
    "wiz_pack_new",
    "awaiting_reset_confirm",
    "awaiting_remove_pack_confirm",
    "awaiting_activate_confirm",
}


def _truncate_path_display(path: Path | None, *, max_parts: int = 2) -> str:
    """Display a shortened workspace path.

    Examples:
        G:\foo\bar\\Project\tests -> ~\\Project\tests
        G:\foo\bar\\Project       -> ~\\Project
    """
    if path is None:
        return "(not set)"

    parts = path.parts

    # Keep the last `max_parts` path components.
    tail = parts[-max_parts:] if len(parts) >= max_parts else parts

    return "~\\" + "\\".join(tail)


class SelectableStatic(Static):
    """`Static`, but explicit about wanting Textual's built-in
    click-drag text selection turned on. `RichLog` (the old scrollback
    widget) fundamentally can't support this - Textual only tracks
    selectable spans for widgets that keep their rendered content
    around, and `RichLog` deliberately doesn't (it's an append-only
    write-once buffer). Every scrollback entry is now one of these,
    mounted into the `#scrollback` `VerticalScroll` instead of written
    into a `RichLog` - see `TesseractApp.write_log`."""

    ALLOW_SELECT = True


class TesseractApp(App):
    TITLE = "TesseractCLI"

    BINDINGS: ClassVar[list[Binding]] = [
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
    /* Item 8: the input is now one bordered container (like Claude
       Code's prompt box) instead of a borderless line with a separate
       mode-line strip floating above it. `height: auto` lets it grow
       with #main-input (still 1-8 lines, see on_text_area_changed)
       without a fixed cell size fighting the content. */
    #input-area {
        height: auto; width: 1fr; margin: 0 1 1 1; padding: 0 1;
        border: round #3b3f51;
    }
    #prompt-row { height: auto; width: 1fr; }
    #prompt-glyph { width: auto; padding: 0 1 0 0; color: #4dd8ff; text-style: bold; }
    #main-input { width: 1fr; border: none; background: transparent; padding: 0; height: 1; }
    #main-input:focus { border: none; }
    /* Status row now lives INSIDE the bordered box, under the prompt
       row - see compose(). Still #mode-line so _refresh_mode_line
       doesn't need to change which widget it targets. */
    #mode-line { height: 1; width: 1fr; padding: 0; color: #7c8bff; }
    """

    def __init__(self) -> None:
        super().__init__()
        self.workspace_root: Path | None = None
        self._last_workspace: tuple[str, str] | None = None
        self._pending_config_error: tuple[str, str] | None = None

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
        yield VerticalScroll(id="scrollback")
        yield Static("", id="status-line")
        with Vertical(id="input-area"):
            with Horizontal(id="prompt-row"):
                yield Static("›", id="prompt-glyph")
                yield ChatTextArea(id="main-input", placeholder="")
            yield Static("", id="mode-line")

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
        # The single source of truth for global_config.yaml (packs,
        # models, paths, agent defaults). `render_settings`,
        # `settings_commands.handle`, and `load_pack_choices` all read
        # and write through this one instance - nothing in the UI talks
        # to the YAML file or `config/settings.py` directly.
        #
        # Built BEFORE the dispatcher, and handed to it explicitly via
        # RoutingResolver(self.config_manager) - NOT LLMDispatcher()
        # with no args, which would call get_routing_resolver() and
        # construct a SECOND, separate ConfigManager() of its own.
        # ConfigManager.config caches in memory rather than re-reading
        # the file on every access, so two instances silently diverge:
        # `set agent.max_context_messages ...` in the settings screen
        # would mutate this instance while the dispatcher kept reading
        # its own stale copy forever. One shared instance is what makes
        # the "live config" behavior actually true.
        self.config_manager = ConfigManager()
        self.dispatcher = LLMDispatcher(RoutingResolver(self.config_manager))
        self._load_config_safely()
        self.workspace_root = None
        self.selected_pack: str | None = None
        self.messages = (
            PersistentMessageList()
        )  # BaseMessage list, mutated in place by run_inner_loop
        # unbound until a workspace is picked (see _handle_workspace_input /
        # _handle_workspace_edit_input) - every append() persists once bound
        self._last_agent_reply: str = ""  # backs the 'copy'/-c command
        self._last_full_input: str | None = None  # backs the 'expand'/-e command

        self.stage: str = "workspace"
        self._pack_return_stage: str = "chat"
        self._workspace_return_stage: str = "chat"
        self._reset_return_stage: str = "chat"
        self._pending_reset_scope: str = DEFAULT_RESET_SCOPE
        self._pack_choices: list[PackChoice] = []
        self._approval_event: asyncio.Event | None = None
        self._approval_result: bool = False
        self._wiz: dict[
            str, Any
        ] = {}  # scratch state for the interactive 'suggest' wizard
        self._turn_counter: int = 0  # backs each chat turn's unique echo-widget id

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
        # self.write_log(
        #     f"[bold]Workspace:[/bold] [dim]{Path.cwd()}[/dim] "
        #     "(press Enter to accept, or type a different path)"
        # )
        self.query_one("#main-input", ChatTextArea).value = str(Path.cwd())
        self.query_one("#main-input", ChatTextArea).focus()
        self._refresh_mode_line()

    def on_unmount(self) -> None:
        """Textual lifecycle hook, runs once on app exit (normal quit,
        not a crash). Closes the bound ConversationStore's sqlite
        connection cleanly - not required for data safety (every saved
        message already commits immediately) but avoids leaving the fd
        open until the OS reclaims it. Safe no-op if a workspace was
        never picked (self.messages is unbound)."""
        self.messages.close()

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

    def write_log(self, renderable: Any, *, id: str | None = None) -> SelectableStatic:
        """Mounts one new `SelectableStatic` per call instead of writing
        into a shared `RichLog`. This is what makes mouse text-selection
        work at all (see `SelectableStatic`), and as a side effect it
        also means an entry can be found again by `id` and removed
        later - which `run_agent_turn` uses to replace the "you typed
        this" echo line with the final combined turn box once the
        reply lands, instead of both staying in the log permanently."""
        widget = SelectableStatic(renderable, id=id)
        container = self.query_one("#scrollback", VerticalScroll)
        container.mount(widget)
        container.scroll_end(animate=False)
        return widget

    def _echo_text(
        self, text: str, *, limit_lines: int = 8, limit_chars: int = 600
    ) -> str:
        """Formats raw user-typed text for the "you typed this" echo
        lines. Long pastes/messages (many lines, or just a huge single
        line) get cut short with a `(...) type 'expand' to see the
        full text` note instead of flooding the scrollback - the full
        original is stashed in `_last_full_input` so 'expand'/-e can
        still show it on demand. Short input passes through unchanged."""
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

    def _expand_last_input(self) -> None:
        if self._last_full_input is None:
            self.write_log(
                "[dim]Nothing truncated to expand - the last input was shown in full.[/dim]"
            )
            return
        self.write_log(render_box("Full input", self._last_full_input, style="#7c8bff"))

    def set_status(self, text: str) -> None:
        self.query_one("#status-line", Static).update(text)

    def _refresh_mode_line(self) -> None:
        """Always-visible workspace/pack/stage row, now inside the
        bordered input box itself (item 8) rather than a separate
        strip above it. Distinct in color (#7c8bff, the purple end of
        the logo gradient) from both the cyan input prompt and the
        grey nav-divider lines (item 7, see `_write_nav_divider`) so
        the two don't get visually confused even though both mark
        "where you are" - the workspace path is truncated for display
        only (`_truncate_path_display`); `self.workspace_root` itself
        is never touched."""
        ws_display = _truncate_path_display(self.workspace_root)
        pack = self.selected_pack or "(not set)"
        self.query_one("#mode-line", Static).update(
            f"[dim]ws:[/dim] [#236f9b]{ws_display}[/#236f9b] •  [dim][bold][#4ad851]{pack}[/#4ad851][/bold][/dim]  •  [dim][yellow] {self.stage}[/yellow][/dim]"
        )

    def _write_nav_divider(self, label: str) -> None:
        self.write_log(
            f"\n[#6c7086]──────────────────── {label} ────────────────────[/#6c7086]\n"
        )

    def _print_banner(self) -> None:
        self.write_log(build_banner_panel(self.size.width))

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

    def _report_error(
        self, title: str, exc: Exception, *, user_text: str | None = None
    ) -> None:
        prefix = (
            f"[bold cyan]›[/bold cyan] {user_text}\n\n" if user_text is not None else ""
        )
        message = f"{prefix}[red]{type(exc).__name__}: {exc}[/red]"

        # verbose.errors (item 10): off by default - a config error, a
        # tool crash, etc. always shows type+message, but the full
        # traceback (often huge) only prints if the user opted in via
        # 'set verbose.errors true'. Reading through config_manager
        # directly (not a cached bool) so flipping the setting takes
        # effect on the very next error, no restart needed.
        verbose_errors = False
        try:
            verbose_errors = bool(self.config_manager.config.verbose.errors)
        except AttributeError:
            pass  # config_manager not built yet (very early startup failure)

        if verbose_errors:
            import traceback

            tb = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
            message += f"\n\n[dim]{tb.strip()}[/dim]"
        else:
            message += (
                "\n\n[dim]set verbose.errors true to see the full traceback.[/dim]"
            )

        self.write_log(render_box(title, message, style="red"))

    def _remove_echo(self, echo_id: str) -> None:
        """Removes the lightweight, unboxed "you typed this" line
        mounted right when the user hit Enter (see the chat branch of
        `_route_input`), just before the merged turn box (input +
        reply, or input + error) replaces it. Only possible because
        the scrollback is now one widget per entry instead of a
        write-once `RichLog` - see `SelectableStatic`/`write_log`."""
        try:
            self.query_one(f"#{echo_id}", SelectableStatic).remove()
        except NoMatches:
            pass

    def _route_input(self, event: ChatTextArea.Submitted) -> None:
        text = event.value
        input_widget = self.query_one("#main-input", ChatTextArea)
        input_widget.value = ""

        stripped = text.strip()
        if not stripped:
            return

        if self.stage not in _FREE_TEXT_STAGES:
            words_probe = stripped.split()
            head_probe = words_probe[0].lower() if words_probe else ""

            if head_probe in RESET_HEADS and len(words_probe) <= 2:
                self.write_log(f"[dim]›[/dim] {self._echo_text(text)}")
                self._start_reset_confirm(
                    words_probe[1].lower() if len(words_probe) == 2 else None
                )
                return

            if head_probe in RELOAD_HEADS and len(words_probe) <= 2:
                self.write_log(f"[dim]›[/dim] {self._echo_text(text)}")
                self._reload_config(
                    words_probe[1].lower() if len(words_probe) == 2 else None
                )
                return

            global_cmd = GLOBAL_ALIASES.get(stripped.lower())
            if global_cmd == "help":
                self.write_log(f"[dim]›[/dim] {self._echo_text(text)}")
                self.write_log("")
                self.write_log(render_help(self))
                return
            if global_cmd == "copy":
                self.write_log(f"[dim]›[/dim] {self._echo_text(text)}")
                self._copy_last_reply()
                return
            if global_cmd == "expand":
                self.write_log(f"[dim]›[/dim] {self._echo_text(text)}")
                self._expand_last_input()
                return
            if global_cmd == "clear":
                self._clear_scrollback()
                return
            if global_cmd == "packs":
                self.write_log(f"[dim]›[/dim] {self._echo_text(text)}")
                self.write_log(
                    settings_commands.render_packs_overview(self.config_manager)
                )
                return
            if global_cmd and self.stage in {"chat", "settings", "home"}:
                self._navigate(global_cmd)
                return

            # Compound shorthand, usable from chat/home directly (settings
            # already gets `_translate_compound_shorthand` applied to its
            # own grammar below) - "-rm -p <name>"/"add -md ..." reach
            # add/remove pack/model without opening `settings` first.
            if self.stage in {"chat", "home"}:
                words = stripped.split()
                first = words[0].lower()

                # Item 2: ANY settings-stage command now runs straight
                # from chat/home via "-cfg <rest>" - not just "-cfg
                # model [pack]" as before. `model`/`-ws`/`workspace`/
                # `reset`/`reload` stay special-cased since they're
                # app-level session state (selected_pack /
                # workspace_root / messages / config_manager), not
                # part of settings_commands' own grammar; everything
                # else is forwarded to settings_commands.handle()
                # verbatim, printed right where you are, with
                # self.stage untouched throughout.
                if first in ("-cfg", "--config") and len(words) >= 2:
                    self.write_log(f"[dim]›[/dim] {self._echo_text(text)}")
                    rest = words[1:]
                    head = rest[0].lower()
                    if head == "model":
                        self._handle_inline_model_switch(rest[1:])
                        return
                    if head in ("-ws", "--workspace", "workspace"):
                        self._handle_inline_workspace_switch(rest[1:])
                        return
                    if head in RESET_HEADS and len(rest) <= 2:
                        self._start_reset_confirm(
                            rest[1].lower() if len(rest) == 2 else None
                        )
                        return
                    if head in RELOAD_HEADS and len(rest) <= 2:
                        self._reload_config(rest[1].lower() if len(rest) == 2 else None)
                        return
                    translated = _translate_compound_shorthand(rest)
                    if translated is not None:
                        self._dispatch_settings_cmd(translated)
                        return
                    resolved_head = settings_commands.ALIASES.get(head, head)
                    if resolved_head == "suggest" and len(rest) == 1:
                        self._pack_return_stage = self.stage
                        self.start_suggest_wizard()
                        return
                    resolved_text = " ".join([resolved_head, *rest[1:]])
                    self._dispatch_settings_cmd(resolved_text)
                    return

                translated = _translate_compound_shorthand(words)
                if translated is not None:
                    self.write_log(f"[dim]›[/dim] {self._echo_text(text)}")
                    self._dispatch_settings_cmd(translated)
                    return

        if self.stage == "workspace":
            self._handle_workspace_input(stripped)
            return

        if self.stage == "workspace_edit":
            self._handle_workspace_edit_input(stripped)
            return

        if self.stage == "awaiting_approval":
            self._handle_approval_input(stripped)
            return

        if self.stage == "awaiting_reset_confirm":
            self._handle_reset_confirm_input(stripped)
            return

        if self.stage == "awaiting_remove_pack_confirm":
            self._handle_remove_pack_confirm_input(stripped)
            return

        if self.stage == "awaiting_activate_confirm":
            self._handle_activate_confirm_input(stripped)
            return

        if self.stage == "wiz_model_custom":
            self.write_log(f"[dim]›[/dim] {self._echo_text(text)}")
            self._wiz["model"] = stripped
            self._wiz_show_pack_step()
            return

        if self.stage == "wiz_pack_new":
            self.write_log(f"[dim]›[/dim] {self._echo_text(text)}")
            self._wiz["pack"] = stripped
            self._wiz["pack_is_new"] = True
            self._wiz_show_target_step()
            return

        normalized = NAV_ALIASES.get(stripped.lower(), stripped.lower())
        if normalized in NAV_COMMANDS and self.stage in {"chat", "settings", "home"}:
            self._navigate(normalized)
            return

        if self.stage == "settings":
            self.write_log(f"[dim]›[/dim] {self._echo_text(text)}")
            words = stripped.split()
            translated = _translate_compound_shorthand(words)
            if translated is not None:
                self._dispatch_settings_cmd(translated)
                return
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
            self._dispatch_settings_cmd(resolved_text)
            return

        if self.stage == "home":
            self.write_log(f"[dim]›[/dim] {self._echo_text(text)}")
            self.write_log(
                "[dim]type 'chat' to start chatting, or 'settings' for configuration[/dim]"
            )
            return

        # stage == "chat": a real message for the agent
        self._turn_counter += 1
        echo_id = f"turn-echo-{self._turn_counter}"
        self.write_log(f"[bold cyan]›[/bold cyan] {self._echo_text(text)}", id=echo_id)
        self.run_agent_turn(text, echo_id)

    def _navigate(self, command: str) -> None:
        if command == "chat":
            self.stage = "chat"
            self._write_nav_divider("chat")
        elif command == "settings":
            self.stage = "settings"
            self._write_nav_divider("settings")
            self.write_log(render_settings(self))
        elif command == "home":
            self.stage = "home"
            self._write_nav_divider("home")
            self.write_log(render_home(self))
        elif command == "model":
            self._pack_return_stage = self.stage
            self.show_model_picker()
        elif command == "workspace":
            self._workspace_return_stage = self.stage
            self.stage = "workspace_edit"
            self._write_nav_divider("workspace")
            self.write_log(f"[bold]Current workspace:[/bold] {self.workspace_root}")
            self.write_log(
                "[bold]New workspace folder:[/bold] (type a path, or 'cancel')"
            )
        elif command in ("exit", "quit"):
            self.exit()
        self._refresh_mode_line()

    # ------------------------------------------------------------------
    # workspace stage (replaces WorkspaceSelectorScreen)
    # ------------------------------------------------------------------

    def _handle_workspace_input(self, text: str) -> None:
        path = Path(text).expanduser().resolve()
        self.write_log(f"[dim]›[/dim] {self._echo_text(text)}")

        if not path.exists() or not path.is_dir():
            self.write_log(f"[red]not a valid directory: {path}[/red]")
            return

        self.workspace_root = path
        init_workspace_logging(path)
        self.messages.bind_store(ConversationStore(path))
        self.write_log(f"[green]✓[/green] workspace set: [#236f9b]{path}[/#236f9b]")
        self._pack_return_stage = "chat"
        self._refresh_mode_line()
        self.show_model_picker()

    def _handle_workspace_edit_input(self, text: str) -> None:
        """Backs the `workspace`/`-ws` command: lets the workspace root
        be changed later from chat/settings/home, reusing the same
        validation as the first-run `_handle_workspace_input` above but
        returning to wherever `workspace` was invoked from instead of
        always cascading into the model picker (that cascade only makes
        sense for the very first setup, not a later edit)."""
        self.write_log(f"[dim]›[/dim] {self._echo_text(text)}")

        if text.strip().lower() == "cancel":
            self.write_log("[dim]— workspace unchanged —[/dim]")
            self.stage = getattr(self, "_workspace_return_stage", "chat")
            self._refresh_mode_line()
            return

        path = Path(text).expanduser().resolve()
        if not path.exists() or not path.is_dir():
            self.write_log(f"[red]not a valid directory: {path}[/red]")
            return

        self.workspace_root = path
        init_workspace_logging(path)
        self.messages.bind_store(ConversationStore(path))
        self.write_log(f"[green]✓[/green] workspace set: {path}")
        self.stage = getattr(self, "_workspace_return_stage", "chat")
        self._refresh_mode_line()

    # ------------------------------------------------------------------
    # model pack picker (replaces ModelPickerScreen)
    # ------------------------------------------------------------------

    def _mount_options(
        self, options: list[Option], prompt: str, *, list_id: str
    ) -> None:
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
        self.query_one("#scrollback", VerticalScroll).remove_children()
        self._print_banner()
        self._refresh_mode_line()

    def _dispatch_settings_cmd(self, cmd_text: str) -> None:
        """Single funnel for every settings-grammar command, regardless
        of which of the several call sites reached it (settings stage,
        '-cfg <rest>' inline from chat/home, compound shorthand). Lets
        specific commands be intercepted for an app-level y/n confirm
        stage before ever reaching `settings_commands.handle()` - today
        that's just 'remove pack <name>' (see `_start_remove_pack_confirm`,
        same pattern as `_start_reset_confirm`); the old '... confirm'
        retyped-command flow inside `handle()` itself still works too,
        for anyone scripting commands directly."""
        parts = cmd_text.strip().split()
        if (
            len(parts) == 3
            and parts[0].lower() == "remove"
            and parts[1].lower() == "pack"
        ):
            self._start_remove_pack_confirm(parts[2])
            return
        self.write_log(settings_commands.handle(self.config_manager, cmd_text))
        self._after_settings_command(cmd_text)

    def _start_remove_pack_confirm(self, name: str) -> None:
        """Backs 'remove pack <name>' - same y/n confirm pattern as
        `_start_reset_confirm` (same box style, same 'y'/anything-else
        semantics), replacing the old flow that required retyping the
        whole command with 'confirm' appended. Permanently deletes the
        pack (and every model inside it) once confirmed; a backup is
        always taken first."""
        if name not in self.config_manager.packs.list_packs():
            self.write_log(
                render_box(
                    "Unknown pack",
                    f"'{name}' isn't a known pack.",
                    style="#C4374F",
                )
            )
            return
        self.write_log(
            render_box(
                "Confirm delete",
                f"This permanently deletes pack '[bold]{name}[/bold]' and every "
                "model inside it. A backup is taken automatically before the "
                "delete.\n\n"
                "Type [bold]y[/bold] to confirm, anything else to cancel.",
                style="#C4374F",
            )
        )
        self._remove_pack_return_stage = self.stage
        self._pending_remove_pack = name
        self.stage = "awaiting_remove_pack_confirm"

    def _handle_remove_pack_confirm_input(self, text: str) -> None:
        self.write_log(f"[dim]›[/dim] {self._echo_text(text)}")
        name = self._pending_remove_pack
        if text.strip().lower() in ("y", "yes"):
            try:
                self.config_manager.backup()
                self.config_manager.packs.remove_pack(name)
                self.config_manager.save()
                self._pack_choices = load_pack_choices(self.config_manager)
                self.write_log(
                    f"[green]✓[/green] removed pack '{name}'.\n"
                    "[dim]A backup was taken first - run 'restore' if this was a mistake.[/dim]"
                )
            except ConfigError as exc:
                self.write_log(f"[red]{exc}[/red]")
        else:
            self.write_log("[dim]Delete cancelled.[/dim]")
        self.stage = self._remove_pack_return_stage
        self._refresh_mode_line()

    def _start_reset_confirm(self, scope: str | None) -> None:
        """Backs 'reset'/-rst/--reset, optionally 'reset <scope>'
        (e.g. 'reset temp'). Only clears the in-memory `messages` list -
        the model's conversation context for this session - NOT the
        persisted history in conversation.db (that stays intact;
        resuming/inspecting old turns is untouched). `scope` defaults
        to the only target that exists today ("temp"/context); an
        explicit-but-unknown scope shows what's available instead of
        silently doing the default, so a typo doesn't quietly wipe
        context the user didn't mean to touch. Needs confirmation since
        it's one-way for the live context: the next agent turn starts
        as if the session had just begun."""
        resolved = DEFAULT_RESET_SCOPE if scope is None else RESET_SCOPES.get(scope)
        if resolved is None:
            self.write_log(
                render_box(
                    "Unknown reset target",
                    f"'{scope}' isn't something I can reset. Available: "
                    f"{', '.join(sorted(set(RESET_SCOPES.values())))}.",
                    style="#C4374F",
                )
            )
            return

        n = len(self.messages)
        self._write_nav_divider("reset")
        self.write_log(
            render_box(
                "Reset temp memory?",
                f"This clears the [bold]{n}[/bold] message(s) currently in "
                "context for the model. Nothing is deleted from the saved "
                "conversation history on disk.\n\n"
                "Type [bold]y[/bold] to confirm, anything else to cancel.",
                style="#C4374F",
            )
        )
        self._reset_return_stage = self.stage
        self._pending_reset_scope = resolved
        self.stage = "awaiting_reset_confirm"

    def _handle_reset_confirm_input(self, text: str) -> None:
        self.write_log(f"[dim]›[/dim] {self._echo_text(text)}")
        if text.strip().lower() in ("y", "yes"):
            # Only "temp" (LLM context) exists today - this dispatch
            # stays a dict-shaped switch so a future scope (e.g.
            # "session") is one branch, not a rewrite of the confirm flow.
            if self._pending_reset_scope == "temp":
                n = len(self.messages)
                self.messages.clear()
                self.write_log(
                    f"[green]Temp memory cleared[/green] ({n} message(s) dropped from context)."
                )
        else:
            self.write_log("[dim]Reset cancelled.[/dim]")
        self.stage = self._reset_return_stage
        self._refresh_mode_line()

    def _reload_config(self, scope: str | None) -> None:
        """Backs 'reload'/-rl/--reload, optionally 'reload <scope>'
        (e.g. 'reload cfg'/'reload config'/'reload settings' - all
        synonyms for the same, only, target today: global_config.yaml).
        Re-reads it from disk into the SAME shared ConfigManager
        instance the dispatcher/settings screen already use (see the
        on_mount comment on why there's only ever one instance) - no
        rebuild of the dispatcher or pack picker needed, they read
        through it live. On a bad file, `reload()` raises before
        touching the in-memory cache (see ConfigManager.load), so the
        previously working config keeps running; only the error is
        shown."""
        resolved = DEFAULT_RELOAD_SCOPE if scope is None else RELOAD_SCOPES.get(scope)
        if resolved is None:
            self.write_log(
                render_box(
                    "Unknown reload target",
                    f"'{scope}' isn't something I can reload. Available: "
                    f"{', '.join(sorted(set(RELOAD_SCOPES.values())))}.",
                    style="#C4374F",
                )
            )
            return

        try:
            self.config_manager.reload()
        except ConfigError as exc:
            self._report_error("Config reload failed", exc)
            self.write_log("[dim]Kept the previously loaded configuration.[/dim]")
            return
        self._pack_choices = load_pack_choices(self.config_manager)
        self.write_log("[green]global_config.yaml reloaded.[/green]")
        self._refresh_mode_line()

    def _copy_last_reply(self) -> None:
        """Backs the 'copy'/-c command. Scrollback text is now natively
        mouse-selectable (see `SelectableStatic`), but this stays as a
        one-shot fallback: it pushes the last agent reply straight to
        the system clipboard over OSC 52 via Textual's own
        `App.copy_to_clipboard`, which works even over SSH where
        drag-select copies the remote pane, not the local clipboard."""
        if not self._last_agent_reply:
            self.write_log(
                "[yellow]nothing to copy yet - no agent reply in this session.[/yellow]"
            )
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

    def _handle_inline_model_switch(self, args: list[str]) -> None:
        """Backs `-cfg model [pack]` typed straight into chat/home. With
        a name it's a direct one-shot switch (no picker, no leaving the
        stage you're in); without one it falls back to the normal
        `show_model_picker` picker, same as the bare `model` command."""
        if not args:
            self._pack_return_stage = self.stage
            self.show_model_picker()
            return
        name = args[0]
        if name in self.config_manager.config.providers:
            self.selected_pack = name
            self.write_log(f"[green]✓[/green] active pack: [#4ad851]{name}[/#4ad851]")
            self._refresh_mode_line()
        else:
            self.write_log(
                f"[red]no such pack: '{name}'.[/red] Try [bold]-p[/bold] to list packs."
            )

    def _handle_inline_workspace_switch(self, args: list[str]) -> None:
        """Backs `-cfg -ws <path>`/`-cfg workspace <path>` typed straight
        into chat/home (item 2). With a path it's a direct one-shot
        switch - no 'type a path or cancel' interactive stage, no
        leaving chat/home - same shape as `_handle_inline_model_switch`
        just above. Without one, falls back to the normal interactive
        `workspace_edit` stage, same as the bare `workspace`/`-ws`
        command."""
        if not args:
            self._workspace_return_stage = self.stage
            self.stage = "workspace_edit"
            self._write_nav_divider("workspace")
            self.write_log(f"[bold]Current workspace:[/bold] {self.workspace_root}")
            self.write_log(
                "[bold]New workspace folder:[/bold] (type a path, or 'cancel')"
            )
            self._refresh_mode_line()
            return
        path = Path(" ".join(args)).expanduser().resolve()
        if not path.exists() or not path.is_dir():
            self.write_log(f"[red]not a valid directory: {path}[/red]")
            return
        self.workspace_root = path
        self.messages.bind_store(ConversationStore(path))
        self.write_log(f"[green]✓[/green] workspace set: {path}")
        self._refresh_mode_line()

    def _after_settings_command(self, cmd_text: str) -> None:
        """Runs after every `settings_commands.handle()` call (from
        the settings stage itself, the compound-shorthand path, or the
        generalized `-cfg` forwarding - item 2) to keep app-level
        session state (`self.selected_pack`) in sync with pack-level
        edits `handle()`/`PacksManager` just made to global_config.yaml,
        since neither of those has any notion of "the pack this
        session currently has active":

        - (item 3) the active pack just got deleted -> clear it and
          reopen the pack picker immediately, so the user always has
          one selected rather than silently pointing at a pack that no
          longer exists.
        - (item 4) a brand new pack was just added -> make it active
          right away, same as picking "+ Add new pack" from the
          picker already does - anyone adding a pack is doing it to
          use it.
        - the active pack was just renamed -> follow the rename so
          `self.selected_pack` still refers to something real.

        `cmd_text` must be the already-alias-resolved/translated
        string actually passed to `handle()` (e.g. "remove pack foo
        confirm"), not the raw shorthand the user typed - the word
        positions checked below assume the canonical grammar."""
        words = cmd_text.strip().split()
        if len(words) < 3 or words[1].lower() != "pack":
            return
        head = words[0].lower()
        providers = self.config_manager.config.providers

        if head == "add":
            name = words[2]
            if name in providers:
                self.selected_pack = name
                self.write_log(
                    f"[green]✓[/green] active pack:[#4ad851]{name}[/#4ad851]"
                )
                self._refresh_mode_line()
            return

        if head == "remove":
            name = words[2]
            if name not in providers and self.selected_pack == name:
                self.selected_pack = None
                self.write_log(
                    "[yellow]active pack was removed - pick a new one:[/yellow]"
                )
                self._pack_return_stage = self.stage
                self.show_model_picker()
            return

        if head == "rename" and len(words) >= 4:
            old_name, new_name = words[2], words[3]
            if (
                self.selected_pack == old_name
                and new_name in providers
                and old_name not in providers
            ):
                self.selected_pack = new_name
                self._refresh_mode_line()
            return

    def show_model_picker(self) -> None:
        """Mounts an inline, arrow-key navigable OptionList in place of
        the Input - not a separate screen. Removed again the moment a
        choice is made (see `on_option_list_option_selected`).

        Always includes "+ Add new pack" (chains into the existing
        provider->model->pack `suggest` wizard, see `start_suggest_wizard`)
        and "Cancel" (backs out untouched) - previously an empty pack
        list was a dead end that told you to go type commands in
        `settings` instead; now both paths are reachable from right
        here, and picking nothing is always an option too."""
        self._pack_choices = load_pack_choices(self.config_manager)

        self.stage = "model_pick"
        options = [
            Option(choice.label, id=choice.name) for choice in self._pack_choices
        ]
        options.append(Option("+ Add new pack", id="__add_pack__"))
        options.append(Option("Cancel", id="__cancel__"))
        self._mount_options(options, "Select a model pack", list_id="pack-options")

    async def on_option_list_option_selected(
        self, event: OptionList.OptionSelected
    ) -> None:
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
            chosen_id = event.option.id
            await event.option_list.remove()

            if chosen_id == "__cancel__":
                self._restore_input()
                self.write_log("[dim]— cancelled, pack unchanged —[/dim]")
                self.stage = self._pack_return_stage
                self._refresh_mode_line()
                return

            if chosen_id == "__add_pack__":
                # `_restore_input()` deliberately not called here - the
                # suggest wizard mounts its own OptionList immediately
                # (`start_suggest_wizard`), same as every other wizard
                # step, so the input stays hidden until a free-text step
                # (custom model id / new pack name) actually needs it.
                self.start_suggest_wizard(
                    activate=True, return_stage=self._pack_return_stage
                )
                return

            self._restore_input()
            self.selected_pack = chosen_id
            self.write_log(
                f"[green]✓[/green] active pack: [#4ad851]{chosen_id}[/#4ad851]"
            )
            self.stage = self._pack_return_stage
            self._refresh_mode_line()
            self.write_log(
                "\n[#7F849C]────────────────Chat───────────────────[/#7F849C]\n"
            )
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

    def start_suggest_wizard(
        self, *, activate: bool = False, return_stage: str = "settings"
    ) -> None:
        """`activate`/`return_stage` let this be reused by the model
        picker's "+ Add new pack" option (see `_route_option_selected`):
        the settings-stage `suggest` command still gets the original
        behavior (activate=False, land back in settings), while a pack
        created from the picker becomes the active pack right away and
        returns to wherever `model` was invoked from - since picking
        "add new pack" there means "I want to use this pack now", not
        "file this away for later"."""
        self._wiz = {"activate": activate, "return_stage": return_stage}
        self.stage = "wiz_provider"
        self._refresh_mode_line()
        options = [
            Option(
                f"{e.provider:<14} {e.label} — e.g. {', '.join(e.example_models)}",
                id=e.provider,
            )
            for e in PROVIDER_CATALOG
        ]
        self._mount_options(options, "Pick a provider", list_id="wiz-options")

    def _wiz_show_model_step(self) -> None:
        self.stage = "wiz_model"
        self._refresh_mode_line()
        entry = next(e for e in PROVIDER_CATALOG if e.provider == self._wiz["provider"])
        options = [Option(model, id=model) for model in entry.example_models]
        options.append(Option("type a different model id…", id="__custom__"))
        self._mount_options(
            options, f"Pick a model ({entry.label})", list_id="wiz-options"
        )

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
        self._mount_options(
            options,
            f"Pool or fallback for '{self._wiz['pack']}'?",
            list_id="wiz-options",
        )

    def _wiz_commit(self) -> None:
        provider = self._wiz["provider"]
        model = self._wiz["model"]
        pack = self._wiz["pack"]
        target = self._wiz["target"]
        pack_is_new = self._wiz.get("pack_is_new", False)
        activate = self._wiz.get("activate", False)
        return_stage = self._wiz.get("return_stage", "settings")

        try:
            if pack_is_new:
                self.config_manager.packs.add_pack(pack)
            self.config_manager.packs.add_model(pack, provider, model, target=target)
            self.config_manager.save()
            self.write_log(
                f"[green]✓[/green] added {provider}/{model} to '{pack}' ({target})."
            )
        except ConfigError as exc:
            self.write_log(f"[red]{exc}[/red]")
            self._wiz = {}
            self.stage = return_stage
            self._restore_input()
            self._refresh_mode_line()
            return

        self._wiz = {}

        if activate:
            # Used to activate `pack` immediately here. Now asks first -
            # same y/n confirm pattern as `_start_reset_confirm` /
            # `_start_remove_pack_confirm`, green instead of red since
            # this is an additive/reversible action, not a delete.
            self.write_log(
                render_box(
                    "Activate this pack?",
                    f"Set '[bold]{pack}[/bold]' as the active pack now?\n\n"
                    "Type [bold]y[/bold] to confirm, anything else to keep "
                    "the current active pack.",
                    style="#A6E3A1",
                )
            )
            self._pending_activate_pack = pack
            self._activate_return_stage = return_stage
            self.stage = "awaiting_activate_confirm"
            self._restore_input()
            self._refresh_mode_line()
            return

        self.stage = return_stage
        self._restore_input()
        self._refresh_mode_line()

    def _handle_activate_confirm_input(self, text: str) -> None:
        self.write_log(f"[dim]›[/dim] {self._echo_text(text)}")
        pack = self._pending_activate_pack
        if text.strip().lower() in ("y", "yes"):
            self.selected_pack = pack
            self.write_log(f"[green]✓[/green] active pack: [#4ad851]{pack}[/#4ad851]")
        else:
            self.write_log("[dim]Kept the current active pack.[/dim]")
        self.stage = self._activate_return_stage
        self._restore_input()
        self._refresh_mode_line()

    # ------------------------------------------------------------------
    # chat stage (replaces ChatScreen)
    # ------------------------------------------------------------------

    @work(exclusive=True)
    async def run_agent_turn(self, user_text: str, echo_id: str) -> None:
        """Previously a raised exception here (provider error, hitting
        agent.max_iterations, a rate-limit that survived the
        dispatcher's own fallback/retry, a tool crash) would propagate
        out of this `@work` coroutine as a failed Worker, which Textual
        surfaces as an unhandled app-level exception - i.e. the whole
        app exits instead of just this one turn failing. Caught here
        now so a bad turn just prints an error and the user can keep
        chatting or fix their pack/config and retry, exactly the same
        as any other command failure.

        `echo_id` is the lightweight, unboxed "you typed this" line
        `_route_input` mounted the instant Enter was pressed (so typing
        never feels like it went nowhere while the agent is still
        "thinking"). Once the turn resolves - success or failure - that
        line is removed and replaced by a single bordered box holding
        both the original message and the outcome, so a finished turn
        reads as one unit instead of two separate scrollback entries."""
        self.set_status("[dim]● thinking…[/dim]")
        try:
            assert self.workspace_root is not None, "Workspace root must be set"
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
            self._remove_echo(echo_id)
            self._report_error("Agent turn failed", exc, user_text=user_text)
            return
        self.set_status("")
        self._last_agent_reply = reply
        self._remove_echo(echo_id)
        body = f"[bold cyan]›[/bold cyan] {self._echo_text(user_text)}\n\n{reply}"
        self.write_log(render_box("Agent", body, style="#b98cff"))

    # ------------------------------------------------------------------
    # tool approval (replaces ApprovalModal)
    # ------------------------------------------------------------------

    async def approve_via_ui(
        self, tool_name: str, args: dict[str, Any], workspace_root: Path
    ) -> bool:
        """The `approve_fn` injected into `run_inner_loop`. Same role as
        the old modal-backed version: block (from the agent loop's point
        of view) on an `asyncio.Event` so the Textual event loop stays
        responsive - the difference is the preview and y/n prompt are
        written straight into the log/input instead of a popup screen."""
        call = ToolCallInfo(
            tool_name=tool_name, args=args, workspace_root=workspace_root
        )

        self.write_log("")
        self.write_log(
            render_box("Tool approval", render_approval_preview(call), style="yellow")
        )

        previous_stage = self.stage
        self.stage = "awaiting_approval"
        self._approval_event = asyncio.Event()
        self._approval_result = False

        await self._approval_event.wait()

        self.stage = previous_stage
        return self._approval_result

    def _handle_approval_input(self, text: str) -> None:
        approved = text.strip().lower() in {"y", "yes"}
        self.write_log(f"[dim]›[/dim] {self._echo_text(text)}")
        self._approval_result = approved
        if self._approval_event is not None:
            self._approval_event.set()


def run() -> None:
    # No-op unless LANGSMITH_TRACING=true in .env - see
    # observability/tracing.py. Must happen before any dispatcher/model
    # call, so every model call made during this session is traced.
    from tesseractcli.observability import configure_tracing

    configure_tracing()
    TesseractApp().run()
