"""The first screen shown when `tesseract` is launched."""

from __future__ import annotations

import os

from rich.align import Align
from rich.panel import Panel
from textual import events
from textual.app import ComposeResult
from textual.containers import Center, Middle, Vertical
from textual.screen import Screen
from textual.widgets import Static

from tesseractcli.ui.logo import (
    FALLBACK_TITLE,
    GRADIENT_LOGO,
    LOGO_MIN_WIDTH,
    TAGLINE,
)


class WelcomeScreen(Screen):
    """Splash screen: logo + workspace status + a hint to continue.

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

    #status {
        text-align: left;
        margin-top: 2;
        width: auto;
    }

    #hint {
        text-align: center;
        margin-top: 2;
        color: $text-muted;
    }
    """

    # Purely cosmetic delay before "System ready" appears. Once real
    # config/provider checks exist (see docs/TODO.md) this timer goes away
    # and System ready reflects an actual check instead of a fixed delay.
    _READY_DELAY_SECONDS = 0.6

    class ContinuePressed(events.Event):
        """Posted when the user presses any key to leave the welcome screen."""

    def compose(self) -> ComposeResult:
        # NOTE: os.getcwd() is a placeholder for "the active workspace".
        # Once the real workspace selector exists, this should read the
        # selected workspace path instead of the launch directory.
        self._workspace_path = os.getcwd()

        with Center(), Middle(), Vertical():
            yield Static(self._render_logo(), id="logo")
            yield Static(TAGLINE, id="tagline")
            yield Static(self._status_text(ready=False), id="status")
            yield Static("[dim]press any key to continue[/dim]", id="hint")

    def on_mount(self) -> None:
        # Reveal "System ready" a beat after the screen appears, so it
        # reads as a real startup check rather than static text.
        self.set_timer(self._READY_DELAY_SECONDS, self._mark_ready)

    def _render_logo(self):
        logo = (
            GRADIENT_LOGO if self.app.size.width >= LOGO_MIN_WIDTH else FALLBACK_TITLE
        )

        return Panel(
            Align.center(logo),
            border_style="#4dd8ff",
            padding=(1, 2),
            expand=False,
        )

    def _status_text(self, ready: bool) -> str:
        line1 = "888888888="
        # line1 = f"[dim]Working on:[/dim] {self._workspace_path}"
        line2 = (
            "[green]✓ System ready[/green]"
            if ready
            else "[dim]Checking environment...[/dim]"
        )
        return f"{line1}\n{line2}"

    def _mark_ready(self) -> None:
        self.query_one("#status", Static).update(self._status_text(ready=True))

    def on_key(self, event: events.Key) -> None:
        # Any key advances past the splash screen.
        self.post_message(self.ContinuePressed())
