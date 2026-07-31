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


def render_home(app: TesseractApp) -> str:
    version = get_settings().APP_VERSION
    workspace = app.workspace_root or "[dim](not set)[/dim]"
    pack = app.selected_pack or "[dim](not set)[/dim]"

    return (
        f"[bold][#4dd8ff]TesseractCLI[/#4dd8ff][/bold]   [#ccff3f]{version} [/#ccff3f]\n"
        f"  Workspace    [#236f9b] {workspace}[/#236f9b]\n"
        f"  Active pack   [#4ad851]{pack}[/#4ad851]\n\n"
        "[bold]Commands[/bold]\n"
        "  [yellow]chat[/yellow]       start / return to chatting with the agent\n"
        "  [yellow]settings[/yellow]   view configuration and provider status\n"
        "  [yellow]model[/yellow]      change the active model pack\n"
        "  [yellow]home[/yellow]       show this screen\n"
    )
