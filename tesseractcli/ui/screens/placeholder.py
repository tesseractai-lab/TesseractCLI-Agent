"""Temporary stand-in for the workspace selector / model picker screens
(see TODO.md, Step 6). Replace this once those are built.
"""

from __future__ import annotations

from textual.app import ComposeResult
from textual.containers import Center, Middle
from textual.screen import Screen
from textual.widgets import Static


class PlaceholderScreen(Screen):
    def compose(self) -> ComposeResult:
        with Center():
            with Middle():
                yield Static(
                    "[bold]Workspace / model picker goes here.[/bold]\n\n"
                    "[dim]press q to quit[/dim]"
                )

    def on_key(self, event) -> None:
        if event.key == "q":
            self.app.exit()
