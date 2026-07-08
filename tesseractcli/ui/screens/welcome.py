"""The first screen shown when `tesseract` is launched."""

from __future__ import annotations

from textual import events
from textual.app import ComposeResult
from textual.containers import Center, Middle, Vertical
from textual.screen import Screen
from textual.widgets import Static
from rich.panel import Panel
from rich.align import Align

from tesseractcli.ui.logo import (
    FALLBACK_TITLE,
    GRADIENT_LOGO,
    LOGO_MIN_WIDTH,
    TAGLINE,
)


class WelcomeScreen(Screen):
    """Splash screen: logo + a hint to continue.

    Press any key to move on. The caller (the App) is responsible for
    pushing the next screen (workspace selector / model picker) in
    response to the `ContinuePressed` message.
    """

    DEFAULT_CSS = """
    WelcomeScreen {
        align: center middle;
        background: $surface;
    }

    #logo {
        text-align: center;
        width: auto;
    }

    #hint {
        text-align: center;
        margin-top: 2;
        color: $text-muted;
    }
    """

    class ContinuePressed(events.Event):
        """Posted when the user presses any key to leave the welcome screen."""

    def compose(self) -> ComposeResult:
        with Center():
            with Middle():
                with Vertical():
                    yield Static(self._render_logo(), id="logo")
                    yield Static(TAGLINE, id="tagline")
                    yield Static(
                        "[dim]press any key to continue[/dim]", id="hint"
                    )

    def _render_logo(self):
        logo = (
            GRADIENT_LOGO
            if self.app.size.width >= LOGO_MIN_WIDTH
            else FALLBACK_TITLE
        )

        return Panel(
            Align.center(logo),
            border_style="#4dd8ff",
            padding=(1, 2),
            expand=False,
    )

    def on_key(self, event: events.Key) -> None:
        # Any key advances past the splash screen.
        self.post_message(self.ContinuePressed())
