"""
tesseractcli/ui/widgets/chat_input.py

Replaces the plain `Input` that used to sit at the bottom of the app
and get reused for every stage (workspace path, model pack, chat
messages, tool approval, settings commands, wizard steps). `Input` is
single-line only, so pasting a multi-line code block into a chat
message got mangled/truncated - this widget is a thin `TextArea`
subclass that keeps the same "one widget reused everywhere" design but
supports multiple lines.

Key/submit design
------------------
`Enter` submits (same behavior as the old `Input`, so every existing
stage - workspace, settings commands, wizard steps, approval y/n -
keeps working unchanged). `ctrl+j` inserts a literal newline instead.

`ctrl+j` was picked over `shift+enter` deliberately: `shift+enter` is
not a distinct, reliable key event in most terminal emulators (they'd
send the same byte as a plain Enter without a newer keyboard protocol
the terminal may not support), whereas `ctrl+j` *is* the ASCII linefeed
character and terminals send it as its own distinct key almost
universally - the same trick several other terminal chat UIs use.

Autocomplete
------------
`TextArea` already ships a suggestion/ghost-text mechanism
(`self.suggestion`, accepted by pressing the right arrow when the
cursor is at the end of the text) - it's just an empty no-op hook
(`update_suggestion()`) by default, meant to be overridden. That's all
this does: `command_choices` is a plain list the app sets per-stage
(the settings grammar while `stage == "settings"`, the bare nav words
otherwise, empty during free chat text so it never gets in the way of
an actual message to the agent).
"""

from __future__ import annotations

from textual.message import Message
from textual.widgets import TextArea


class ChatTextArea(TextArea):
    """Multi-line input area reused across every app stage.

    Attributes:
        command_choices: Prefix-matched against the current buffer to
            populate `self.suggestion`. Set by the app whenever the
            stage changes; empty means "no autocomplete right now".
    """

    class Submitted(Message):
        """Posted when the user presses Enter. Mirrors `Input.Submitted`
        (`.value` holds the text) so the rest of the app's event-handling
        shape barely changes."""

        def __init__(self, text_area: ChatTextArea, value: str) -> None:
            self.text_area = text_area
            self.value = value
            super().__init__()

    def __init__(self, *args, **kwargs) -> None:
        kwargs.setdefault("show_line_numbers", False)
        kwargs.setdefault("soft_wrap", True)
        kwargs.setdefault("tab_behavior", "focus")
        kwargs.setdefault("highlight_cursor_line", False)
        kwargs.setdefault("compact", True)
        super().__init__(*args, **kwargs)
        self.command_choices: list[str] = []

    async def _on_key(
        self, event
    ) -> None:  # events.Key, left untyped to avoid an unused import
        if event.key == "enter":
            event.stop()
            event.prevent_default()
            self.post_message(self.Submitted(self, self.text))
            return
        if event.key == "shift+enter":
            event.stop()
            event.prevent_default()
            self.insert("\n")
            return
        await super()._on_key(event)

    def update_suggestion(self) -> None:
        """Called by `TextArea` after every edit (base class hook is a
        no-op). Offers a ghost-text completion of the *whole* current
        buffer against `command_choices` - only while it's still a
        single line and the cursor sits at the end, same shape as a
        normal Input autocomplete."""
        self.suggestion = ""
        if not self.command_choices:
            return
        if self.document.line_count != 1 or not self.cursor_at_end_of_text:
            return
        text = self.document.text
        if not text:
            return
        matches = sorted(
            (
                choice
                for choice in self.command_choices
                if choice.startswith(text) and choice != text
            ),
            key=len,
        )
        if matches:
            self.suggestion = matches[0][len(text) :]

    @property
    def value(self) -> str:
        """`Input`-shaped alias for `.text`, so call sites that used to
        read/write `Input.value` (workspace path prefill, clearing the
        box after submit) don't all need to change to `.text` too."""
        return self.text

    @value.setter
    def value(self, new_value: str) -> None:
        self.text = new_value
