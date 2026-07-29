"""
tesseractcli/ui/views/help_view.py

`render_help()` backs the new global `help` command (see
`GLOBAL_ALIASES` in `ui/app.py`) - unlike the old help, which only
existed inside `settings_commands.py` and only fired while
`stage == "settings"`, this one is reachable from (almost) any stage:
chat, settings, home, mid-workspace-setup, mid pack-picker, mid the
'suggest' wizard. The only stages it's deliberately NOT intercepted in
are the three that are capturing literal free-text on purpose
(approval y/n, custom model id, new pack name) - see
`_FREE_TEXT_STAGES` in `ui/app.py`.

While in the settings stage, the settings-specific command grammar
(`settings_commands.HELP_TEXT`) is appended below the global commands,
so "help" is a strict superset of what the old settings-only help
showed rather than a replacement for it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from tesseractcli.ui.views.box import render_box

if TYPE_CHECKING:
    from tesseractcli.ui.app import TesseractApp

# Column widths for the command table below. Previously each row's
# spacing between "command", "(aliases)", and the description was
# hand-typed - since the command/alias text is a different length on
# almost every line, the description column landed at a different
# screen position row to row (it only ever lined up by accident on a
# handful of rows). Built from a plain (command, aliases, description,
# continuation_lines) table instead, so every row is padded to the
# same two column widths and the description column is always
# vertically aligned, continuation lines included.
_CMD_COL = 30
_ALIAS_COL = 22


def _row(command: str, aliases: str, description: str, *continuation: str) -> str:
    """One command row, left-padded to `_CMD_COL`/`_ALIAS_COL`, plus
    any continuation lines (extra description-only lines) indented to
    line up under the description column. `command`/`aliases` may
    contain a Rich-markup-escaped literal bracket (``\\[pack]``) for a
    placeholder like ``[pack]`` - the leading backslash is stripped by
    Rich at render time and so must not count towards the padding
    width, or that row\'s description column drifts by one space.
    """
    cmd_pad = max(_CMD_COL - (len(command) - command.count("\\")), 0)
    alias_pad = max(_ALIAS_COL - (len(aliases) - aliases.count("\\")), 0)
    first = f"  {command}{' ' * cmd_pad}{aliases}{' ' * alias_pad}{description}\n"
    indent = " " * (2 + _CMD_COL + _ALIAS_COL)
    rest = "".join(f"{indent}{line}\n" for line in continuation)
    return first + rest


GLOBAL_HELP_TEXT = (
    "[bold #b98cff]Navigation[/bold #b98cff]\n"
    + _row("chat", "", "go to the chat")
    + _row("settings", "(-cfg, --config)", "open settings")
    + _row("home", "", "go to the home screen")
    + _row("workspace", "(-ws, --workspace)", "change the workspace folder")
    + "\n"
    "[bold #4dd8ff]Model[/bold #4dd8ff]\n"
    + _row(
        "model", "", "pick a different model pack",
        "(also: '+ Add new pack' - asks y/n to activate it once added;",
        "'Cancel' in the list)",
    )
    + _row(
        "-cfg model \\[pack]", "", "switch pack without leaving chat",
        "(no pack name -> opens the picker instead)",
    )
    + _row(
        "-cfg -ws \\[path]", "", "switch workspace without leaving chat",
        "(no path -> opens the interactive prompt instead)",
    )
    + _row(
        "-cfg <settings command>", "", "run it inline, e.g. -cfg -a -p x y z,",
        "-cfg -rn pack old new, -cfg -rm -p mypack",
        "(-rm -p asks y/n before deleting the pack)",
    )
    + "\n"
    "[bold #4ddb9e]Utility[/bold #4ddb9e]\n"
    + _row("copy", "(-c)", "copy the last agent reply")
    + _row("expand", "(-e, --expand)", "show the full text of the last truncated input")
    + _row("clear", "(-cl, --clear, cls)", "clear the terminal")
    + _row(
        "reset \\[temp]", "(-rst, --reset)", "clear temp memory (LLM context), asks for y/n;",
        "conversation.db is untouched (also: -cfg reset \\[temp])",
    )
    + _row(
        "reload \\[cfg]", "(-rl, --reload)", "re-read global_config.yaml from disk",
        "(aliases: cfg/config/settings; also: -cfg reload \\[cfg])",
    )
    + _row("help", "(-h, --help, ?)", "show this help")
    + _row("exit / quit", "(-q, --quit)", "quit TesseractCLI")
    + "\n"
    # "[dim]Shortcut convention: a single leading '-' is a short flag (-h, -cfg,\n"
    # "-q, -c, -cl) and a leading '--' spells the same thing out in full (--help,\n"
    # "--config, --quit, --clear) - the same short/long option shape as any\n"
    # "getopt/argparse-style CLI, so nothing new to learn.[/dim]"
)


def render_help(app: "TesseractApp") -> Any:
    body = GLOBAL_HELP_TEXT
    if app.stage == "settings":
        from tesseractcli.ui.views.settings_commands import HELP_TEXT as SETTINGS_HELP_TEXT

        body += "\n\n[bold]Settings commands[/bold] (this list only while in settings)\n" + SETTINGS_HELP_TEXT
    return render_box("Help", body, style="#e8a33d")
