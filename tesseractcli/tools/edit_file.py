"""
tesseractcli\tools\edit_file.py
"""

import time
from pathlib import Path


from tesseractcli.models import ToolResult, EditFileMetadata
from tesseractcli.models.exceptions import PathEscapesWorkspaceError, FileNotFoundInWorkspace
from tesseractcli.tools.sandbox import safe_open, resolve_in_workspace, atomic_write

def edit_file(
        workspace_root: Path,
        path: str,
        old_str: str,
        new_str: str,
) -> ToolResult:
    
    started = time.monotonic()
    metadata: EditFileMetadata = {"path": path}

    try:
        full_path = resolve_in_workspace(workspace_root, path)

        with safe_open(workspace_root, path, "r") as f:
            original = f.read()

        match_count = original.count(old_str)
        metadata["match_count"] = match_count

        if match_count == 0:
            return ToolResult(
                tool_name="edit_file", success=False, output="",
                error="old_str not found in file. Check exact whitespace/content.",
                metadata=metadata,
            )
        if match_count > 1:
            return ToolResult(
                tool_name="edit_file", success=False, output="",
                error=f"old_str is not unique ({match_count} matches). "
                      f"Provide more surrounding context to make it unique.",
                metadata=metadata,
            )

        updated = original.replace(old_str, new_str, 1)
        atomic_write(full_path, updated)

        metadata["chars_replaced"] = len(old_str)
        metadata["duration_ms"] = round((time.monotonic() - started) * 1000, 2)

        return ToolResult(tool_name="edit_file", success=True, output="", metadata=metadata)

    except PathEscapesWorkspaceError as e:
        return ToolResult(tool_name="edit_file", success=False, output="", error=str(e))
    except FileNotFoundInWorkspace as e:
        return ToolResult(tool_name="edit_file", success=False, output="", error=str(e))
    except PermissionError:
        return ToolResult(tool_name="edit_file", success=False, output="",
                           error=f"Permission denied editing '{path}'.")
