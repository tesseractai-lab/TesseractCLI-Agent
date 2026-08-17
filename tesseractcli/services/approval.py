"""
tesseractcli/services/approval.py

Extracted straight out of the old `TesseractApp.approve_via_ui` /
`_handle_approval_input` pair. The asyncio.Event handshake between the
agent loop (running in a background task, calling `tools/approval.py`'s
`approve_tool_call`) and the human typing y/n into the TUI belongs to
neither the View nor the SessionController on its own - it's a small
piece of concurrency plumbing, so it gets its own service.

Usage from the Controller:

    result = await approval_service.request(tool_name, tool_args, workspace_root)

`request()` suspends until `resolve()` is called from the y/n input
handler (`ApprovalInput` Command). Only one request can be in flight at
a time - same invariant the original single `self._approval_event` /
`self._approval_result` pair on `TesseractApp` had.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any


class ApprovalService:
    def __init__(self) -> None:
        self._event: asyncio.Event | None = None
        self._result: bool = False
        self.pending_tool: str | None = None
        self.pending_args: dict[str, Any] | None = None

    @property
    def awaiting(self) -> bool:
        return self._event is not None

    async def request(
        self, tool_name: str, tool_args: dict[str, Any], workspace_root: Path
    ) -> bool:
        """Called from the agent-loop side (via the tool-approval
        callback). Raises RuntimeError if a request is already pending -
        the UI only ever has one `awaiting_approval` stage active at
        once, so overlapping requests would indicate a real bug
        upstream rather than something to silently queue."""
        if self._event is not None:
            raise RuntimeError("ApprovalService.request() called while a request is pending")
        self.pending_tool = tool_name
        self.pending_args = tool_args
        self._event = asyncio.Event()
        self._result = False
        await self._event.wait()
        result = self._result
        self._event = None
        self.pending_tool = None
        self.pending_args = None
        return result

    def resolve(self, approved: bool) -> None:
        """Called from the View/Controller once the user answers y/n.
        No-op (not an error) if nothing is pending - mirrors the old
        code's tolerance of a stray Enter press after the stage already
        moved on."""
        self._result = approved
        if self._event is not None:
            self._event.set()
