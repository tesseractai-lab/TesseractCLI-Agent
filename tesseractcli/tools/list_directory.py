"""
tesseractcli/tools/list_directory.py
"""

import time
from pathlib import Path
from typing import cast

from tesseractcli.models.exceptions import PathEscapesWorkspaceError
from tesseractcli.models.tool_models import (
    ListDirectoryArgs,
    ListDirectoryMetadata,
    ToolResult,
)
from tesseractcli.tools.registry import ToolRegistry
from tesseractcli.tools.sandbox import resolve_in_workspace


def list_directory(
    workspace_root: Path,
    path: str = ".",
) -> ToolResult:

    started = time.monotonic()
    metadata: ListDirectoryMetadata = {"path": path}

    try:
        full_path = resolve_in_workspace(workspace_root, path)

        if not full_path.exists():
            return ToolResult(
                tool_name="list_directory",
                success=False,
                output="",
                error=f"Directory '{path}' not found.",
            )

        if not full_path.is_dir():
            return ToolResult(
                tool_name="list_directory",
                success=False,
                output="",
                error=f"'{path}' is not a directory.",
            )

        entries = sorted(full_path.iterdir(), key=lambda p: p.name)

        if not entries:
            output = f"Directory '{path}' is empty."
        else:
            lines = [
                f"- {entry.name} ({'Directory' if entry.is_dir() else 'File'})"
                for entry in entries
            ]
            output = f"Contents of directory '{path}':\n" + "\n".join(lines)

        metadata["item_count"] = len(entries)
        metadata["duration_ms"] = round((time.monotonic() - started) * 1000, 2)

        return ToolResult(
            tool_name="list_directory", success=True, output=output, metadata=cast(dict, metadata)
        )

    except PathEscapesWorkspaceError as e:
        return ToolResult(
            tool_name="list_directory", success=False, output="", error=str(e)
        )
    except PermissionError:
        return ToolResult(
            tool_name="list_directory",
            success=False,
            output="",
            error=f"Permission denied to access '{path}'.",
        )


def register(registry: ToolRegistry) -> None:
    registry.add(
        name="list_directory",
        schema=ListDirectoryArgs,
        fn=list_directory,
        needs_approval=False,
        core=False,
    )
