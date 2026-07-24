"""
tesseractcli/ui/views/home_view.py

New (there was no home/hub screen before). Backs the `home` command:
a short status + command list, written as a plain block into the
`RichLog`, same pattern as `settings_view.py`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from tesseractcli.config.settings import get_settings

if TYPE_CHECKING:
    from tesseractcli.ui.app import TesseractApp


def render_home(app: "TesseractApp") -> str:
    version = get_settings().APP_VERSION
    workspace = app.workspace_root or "[dim](not set)[/dim]"
    pack = app.selected_pack or "[dim](not set)[/dim]"

    return (
        f"[bold]TesseractCLI[/bold] {version}\n"
        f"  Workspace   {workspace}\n"
        f"  Active pack {pack}\n\n"
        "[bold]Commands[/bold]\n"
        "  chat       start / return to chatting with the agent\n"
        "  settings   view configuration and provider status\n"
        "  model      change the active model pack\n"
        "  home       show this screen\n"
    )
