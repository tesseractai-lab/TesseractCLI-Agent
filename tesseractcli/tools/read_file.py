"""
tesseractcli/tools/read_file.py
"""

import time
from pathlib import Path

from tesseractcli.models.tool_models import ToolResult, ReadFileMetadata, ReadFileArgs
from tesseractcli.models.exceptions import FileNotFoundInWorkspace, PathEscapesWorkspaceError, SensitiveFileBlocked
from tesseractcli.tools.sandbox import safe_open
from tesseractcli.tools.registry import ToolRegistry
from tesseractcli.config.settings import get_settings as config



def read_file(
        workspace_root: Path,
        path: str,
        start_line: int | None = None,
        end_line: int | None = None,
) -> ToolResult:

    started = time.monotonic()
    metadata: ReadFileMetadata = {"path": path}

    try:
        with safe_open(workspace_root, path, "r") as f:
            lines = f.readlines()

        total_lines = len(lines)
        metadata["total_lines"] = total_lines
        metadata["encoding"] = "utf-8"

        if start_line is not None or end_line is not None:
            lo = max((start_line or 1) - 1, 0)
            hi = end_line if end_line is not None else total_lines
            selected = lines[lo:hi]
            metadata["truncated"] = hi < total_lines or lo > 0
            content = "".join(selected)
        elif total_lines > config().MAX_LINES_WITHOUT_RANGE:
            selected = lines[:config().MAX_LINES_WITHOUT_RANGE]
            metadata["truncated"] = True
            metadata["hint"] = (
                f"File has {total_lines} lines, showing first "
                f"{config().MAX_LINES_WITHOUT_RANGE}. Use start_line/end_line to see more."
            )
            content = "".join(selected)
        else:
            metadata["truncated"] = False
            content = "".join(lines)

        metadata["duration_ms"] = round((time.monotonic() - started) * 1000, 2)
        return ToolResult(tool_name="read_file", success=True, output=content, metadata=metadata)

    except PathEscapesWorkspaceError as e:
        return ToolResult(tool_name="read_file", success=False, output="", error=str(e))
    except FileNotFoundInWorkspace as e:
        return ToolResult(tool_name="read_file", success=False, output="", error=str(e))
    except SensitiveFileBlocked as e:
        return ToolResult(tool_name="read_file", success=False, output="", error=str(e))
    except PermissionError:
        return ToolResult(tool_name="read_file", success=False, output="",
                           error=f"Permission denied reading '{path}'.")
    except UnicodeDecodeError:
        return ToolResult(tool_name="read_file", success=False, output="",
                           error=f"'{path}' is not a valid UTF-8 text file (binary?).")

def register(registry: ToolRegistry) -> None:
    registry.add(name="read_file", schema=ReadFileArgs, fn=read_file,needs_approval=False)
