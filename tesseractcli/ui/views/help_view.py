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

GLOBAL_HELP_TEXT = (
    "  chat                                        go to the chat\n"
    "  settings         (-cfg, --config)            open settings\n"
    "  home                                         go to the home screen\n"
    "  model                                        pick a different model pack\n"
    "  copy             (-c)                        copy the last agent reply\n"
    "  help             (-h, --help, ?)              show this help\n"
    "  exit / quit      (-q, --quit)                quit TesseractCLI\n\n"
    "[dim]Shortcut convention: a single leading '-' is a short flag (-h, -cfg,\n"
    "-q, -c) and a leading '--' spells the same thing out in full (--help,\n"
    "--config, --quit) - the same short/long option shape as any\n"
    "getopt/argparse-style CLI, so nothing new to learn.[/dim]"
)


def render_help(app: "TesseractApp") -> Any:
    body = GLOBAL_HELP_TEXT
    if app.stage == "settings":
        from tesseractcli.ui.views.settings_commands import HELP_TEXT as SETTINGS_HELP_TEXT

        body += "\n\n[bold]Settings commands[/bold] (this list only while in settings)\n" + SETTINGS_HELP_TEXT
    return render_box("Help", body, style="#e8a33d")
