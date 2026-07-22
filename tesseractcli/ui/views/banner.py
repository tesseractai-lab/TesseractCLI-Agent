"""
tesseractcli/ui/views/banner.py

Replaces the old `WelcomeScreen`. That version was a full-screen Textual
`Screen` (Center/Middle layout, its own timer, "press any key to continue")
that got pushed onto the screen stack and later popped and discarded.

In the REPL-style app there is no screen stack, so the banner is just
Rich renderables written once into the persistent `RichLog` at startup -
it becomes the first lines of scrollback history, exactly like Claude
Code prints its own banner and then drops straight into the prompt loop
underneath it, rather than clearing to a different screen.
"""

from __future__ import annotations

from rich.align import Align
from rich.panel import Panel

from tesseractcli.ui.logo import FALLBACK_TITLE, GRADIENT_LOGO, LOGO_MIN_WIDTH, TAGLINE


def build_banner_panel(width: int) -> Panel:
    """Rich Panel for the startup logo, sized against terminal `width`
    the same way `WelcomeScreen._render_logo` did: fall back to the
    plain bold title if the terminal is too narrow for the full ASCII
    art. `RichLog.write()` accepts Rich renderables directly, so the
    caller writes this straight into the log.

    `expand=True` makes the panel's border span the full terminal
    width (matching `#scrollback`'s `width: 1fr` in the app CSS)
    instead of shrinking to the logo's own fixed width, with the logo
    itself centered inside via `Align.center` - this is what makes the
    banner sit "in the middle of the screen / full width" rather than
    left-hugging a narrow box. Note this is a one-shot render at
    startup (RichLog only appends, it can't redraw a past line), so
    resizing the terminal *after* the banner is printed won't reflow
    it retroactively - every other view (`settings`, `home`, and
    anything under `ui.views.box`) is rendered fresh on each visit and
    so reflows normally.
    """
    logo = GRADIENT_LOGO if width >= LOGO_MIN_WIDTH else FALLBACK_TITLE
    return Panel(
        Align.center(logo),
        border_style="#4dd8ff",
        padding=(1, 2),
        expand=True,
    )


def build_tagline() -> str:
    return TAGLINE
