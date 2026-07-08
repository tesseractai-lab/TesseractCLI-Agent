"""The Textual App entry point for TesseractCLI."""

from __future__ import annotations

from textual.app import App

from tesseractcli.ui.screens.placeholder import PlaceholderScreen
from tesseractcli.ui.screens.welcome import WelcomeScreen


class TesseractApp(App):
    TITLE = "TesseractCLI"

    def on_mount(self) -> None:
        self.push_screen(WelcomeScreen())

    def on_welcome_screen_continue_pressed(
        self, message: WelcomeScreen.ContinuePressed
    ) -> None:
        self.pop_screen()
        self.push_screen(PlaceholderScreen())


def run() -> None:
    TesseractApp().run()
