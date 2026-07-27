"""
tesseractcli/ui/views/approval_view.py

Replaces the old `ApprovalModal`. `ToolCallInfo` and the preview-markup
logic (per-tool-name branches for write_file/edit_file/run_command) are
unchanged from that file - still Rich markup, still independent of
`tools/approval.py`'s ANSI-code version, for the same reason as before
(ANSI codes don't render right inside a Textual widget).

What's gone is `ApprovalModal` itself (a `ModalScreen` with its own
bordered box + Approve/Deny `Button`s that covered the whole terminal).
The preview text this module builds is now written as a normal block
into the persistent `RichLog`, and the y/n answer is typed into the
same `Input` used for everything else - see `TesseractApp.approve_via_ui`
and `_handle_approval_input` in `ui/app.py`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class ToolCallInfo:
    """Minimal view of a pending tool call, for display purposes only."""

    tool_name: str
    args: dict[str, Any] = field(default_factory=dict)
    workspace_root: Path | None = None


def render_approval_preview(call: ToolCallInfo) -> str:
    """Builds a short Rich-markup preview per known tool name, falling
    back to a raw name+args dump for anything without a dedicated
    branch yet (mirrors the branching that used to live in
    `ApprovalModal._preview_markup` / `tools/approval.py`).

    Returns just the preview body + y/n prompt - no title line of its
    own. `TesseractApp.approve_via_ui` (ui/app.py) wraps the returned
    string in `render_box("Tool approval", ..., style="yellow")`, so
    the panel's own title takes over what used to be a plain
    "Tool requesting approval" bold text line here."""
    name = call.tool_name
    args = call.args

    if name == "write_file":
        path = args.get("path", "?")
        content = args.get("content", "")
        lines = content.splitlines()
        preview_lines = "\n".join(lines[:12])
        more = f"\n[dim]... +{len(lines) - 12} more lines[/dim]" if len(lines) > 12 else ""
        body = f"[bold]write_file[/bold] → [cyan]{path}[/cyan]\n\n{preview_lines}{more}"
    elif name == "edit_file":
        path = args.get("path", "?")
        old_str = args.get("old_str", "")
        new_str = args.get("new_str", "")
        body = (
            f"[bold]edit_file[/bold] → [cyan]{path}[/cyan]\n\n"
            f"[red]- {old_str}[/red]\n"
            f"[green]+ {new_str}[/green]"
        )
    elif name == "run_command":
        command = args.get("command", [])
        rendered = " ".join(str(part) for part in command) if isinstance(command, list) else str(command)
        body = f"[bold]run_command[/bold]\n\n[yellow]$ {rendered}[/yellow]"
    else:
        body = f"[bold]{name}[/bold]\n{args}"

    return f"{body}\n\n[dim]approve this? (y/n)[/dim]"
