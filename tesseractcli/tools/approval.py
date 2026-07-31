"""
tesseractcli/tools/approval.py

Tool-call approval flow: renders a preview for side-effecting tools
and asks the user for confirmation before execution.

Filtering (whether a tool needs approval at all) happens in
agent/loop.py via registry.needs_approval() BEFORE this module is
even called — approve_tool_call() here assumes it's only invoked when
a prompt is actually needed. It is never registered as a tool itself
(see registry.py) and is imported/called directly by the loop.
"""

from __future__ import annotations

import difflib
import shlex
from pathlib import Path

from tesseractcli.models.exceptions import ApprovalError

RESET = "\033[0m"
BOLD = "\033[1m"
GREEN = "\033[32m"
RED = "\033[31m"
CYAN = "\033[36m"
YELLOW = "\033[33m"

_NEW_FILE_PREVIEW_LIMIT = 40  # lines


def _colorize_diff_line(line: str) -> str:
    if line.startswith("+") and not line.startswith("+++"):
        return f"{GREEN}{line}{RESET}"
    if line.startswith("-") and not line.startswith("---"):
        return f"{RED}{line}{RESET}"
    if line.startswith("@@"):
        return f"{CYAN}{line}{RESET}"
    return line


def render_write_diff(full_path: Path, new_content: str, context_lines: int = 3) -> str:
    """Return a colored, diff-style preview for a write_file call.
    `full_path` must already be resolved against workspace_root."""
    if not full_path.exists():
        preview_lines = new_content.splitlines()
        header = f"{BOLD}{GREEN}[new file]{RESET} {full_path}"
        if len(preview_lines) > _NEW_FILE_PREVIEW_LIMIT:
            shown = preview_lines[:_NEW_FILE_PREVIEW_LIMIT]
            hidden_count = len(preview_lines) - _NEW_FILE_PREVIEW_LIMIT
            body = "\n".join(shown) + f"\n{CYAN}... ({hidden_count} more lines){RESET}"
        else:
            body = "\n".join(preview_lines)
        return f"{header}\n{body}"

    old_content = full_path.read_text(encoding="utf-8")
    old_lines = old_content.splitlines(keepends=True)
    new_lines = new_content.splitlines(keepends=True)

    diff = list(
        difflib.unified_diff(
            old_lines,
            new_lines,
            fromfile=f"a/{full_path.name}",
            tofile=f"b/{full_path.name}",
            n=context_lines,
        )
    )

    if not diff:
        return f"{YELLOW}[no changes]{RESET} {full_path}"

    colored = [_colorize_diff_line(line.rstrip("\n")) for line in diff]
    return f"{BOLD}{full_path}{RESET}\n" + "\n".join(colored)


def render_edit_diff(
    path: str, old_str: str, new_str: str, context_lines: int = 3
) -> str:
    """Diff preview for edit_file. Unlike render_write_diff, old_str/new_str
    come directly from the tool call itself — no need to read anything
    from disk. Note: since old_str is wholesale replaced by new_str (not a
    line-by-line partial edit), the diff will show mostly-removed old_str
    lines followed by mostly-added new_str lines, not a fine-grained
    word-level change within a line."""
    old_lines = old_str.splitlines(keepends=True)
    new_lines = new_str.splitlines(keepends=True)

    diff = list(
        difflib.unified_diff(
            old_lines,
            new_lines,
            fromfile="old",
            tofile="new",
            n=context_lines,
        )
    )

    if not diff:
        return f"{YELLOW}[no changes]{RESET} {path}"

    colored = [_colorize_diff_line(line.rstrip("\n")) for line in diff]
    return f"{BOLD}{path}{RESET}\n" + "\n".join(colored)


def render_command_preview(command: list[str]) -> str:
    """Return a colored preview of the literal command string for run_command."""
    cmd_str = shlex.join(command)
    return f"{BOLD}{YELLOW}$ {cmd_str}{RESET}"


def approve_tool_call(tool_name: str, tool_args: dict, workspace_root: Path) -> bool:
    """
    Render a preview for `tool_name` and block on user confirmation.

    Called only when registry.needs_approval(tool_name) is True — the
    loop already filtered out tools that don't need this (read_file,
    list_directory), so no needs_approval check happens here.
    """
    path = tool_args.get("path")
    if path is None:
        raise ApprovalError ("`approve_tool_call` Path is None")

    if tool_name == "write_file":
        content = tool_args.get("content", "")
        full_path = Path(workspace_root) / path
        preview = render_write_diff(full_path, content)

    elif tool_name == "edit_file":
        old_str = tool_args.get("old_str", "")
        new_str = tool_args.get("new_str", "")
        preview = render_edit_diff(path, old_str, new_str)
    elif tool_name == "run_command":
        command = tool_args.get("command", [])
        preview = render_command_preview(command)
    else:
        # Fallback for any tool without a dedicated renderer yet.
        preview = f"{BOLD}{tool_name}{RESET}\n{tool_args}"

    print(preview)
    answer = input(f"{BOLD}approve? (y/n): {RESET}").strip().lower()
    return answer == "y"
