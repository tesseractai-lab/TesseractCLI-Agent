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
from rich.text import Text


def render_box(title: str, body: Any, *, style: str = "#4dd8ff") -> Panel:
    """Wrap `body` (a string or any Rich renderable) in a bordered panel.

    Args:
        title: Short title shown in the top-left of the border.
        body: Panel content - plain/markup string or a Rich renderable.
        style: Border color.

    Returns:
        A `rich.panel.Panel` ready to hand to `RichLog.write()`.
    """
    # A plain markup string is turned into a `Text` with
    # `overflow="fold"` explicitly, rather than left to Rich's default
    # (which only breaks on whitespace). Without this, a single long
    # unbroken token - a URL, a hash, a line of code with no spaces -
    # can render wider than the panel's content area and spill out past
    # the border instead of wrapping. Any renderable that isn't a plain
    # string (already-built Rich objects, e.g. the banner) is passed
    # through untouched.
    if isinstance(body, str):
        body = Text.from_markup(body, overflow="fold")
    return Panel(
        body,
        title=f"[bold]{title}[/bold]",
        title_align="left",
        border_style=style,
        padding=(1, 2),
        expand=True,
    )
