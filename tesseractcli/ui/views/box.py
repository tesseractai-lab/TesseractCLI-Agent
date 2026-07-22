"""
tesseractcli/ui/views/box.py

Shared bordered-panel helper. Every "branch" of the settings UI (pack
overview, add/remove pack, add/remove model, provider suggestions,
errors) goes through `render_box` so it renders as a consistent,
full-width bordered panel in the RichLog instead of unindented text
that just scrolls past. `expand=True` makes the border span the
terminal width rather than hugging the content, so it stays legible
as the terminal is resized.
"""

from __future__ import annotations

from typing import Any

from rich.panel import Panel


def render_box(title: str, body: Any, *, style: str = "#4dd8ff") -> Panel:
    """Wrap `body` (a string or any Rich renderable) in a bordered panel.

    Args:
        title: Short title shown in the top-left of the border.
        body: Panel content - plain/markup string or a Rich renderable.
        style: Border color.

    Returns:
        A `rich.panel.Panel` ready to hand to `RichLog.write()`.
    """
    return Panel(
        body,
        title=f"[bold]{title}[/bold]",
        title_align="left",
        border_style=style,
        padding=(1, 2),
        expand=True,
    )
