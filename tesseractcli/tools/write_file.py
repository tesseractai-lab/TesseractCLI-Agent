"""
tesseractcli/tools/write_file.py
"""

import time
from pathlib import Path

from tesseractcli.models.tool_models import ToolResult, WriteFileMetadata, WriteFileArgs
from tesseractcli.models.exceptions import (
    PathEscapesWorkspaceError,
    SensitiveFileBlocked,
)
from tesseractcli.tools.sandbox import safe_open, resolve_in_workspace, atomic_write
from tesseractcli.tools.registry import ToolRegistry


def write_file(
    workspace_root: Path,
    path: str,
    content: str,
    mode: str = "overwrite",
) -> ToolResult:

    started = time.monotonic()
    side_effects = []
    metadata: WriteFileMetadata = {"path": path, "mode": mode}

    try:
        if mode not in ("overwrite", "append"):
            return ToolResult(
                tool_name="write_file",
                success=False,
                output="",
                error=f"Invalid mode '{mode}', expected 'overwrite' or 'append'.",
            )

        full_path = resolve_in_workspace(workspace_root, path)

        already_existed = full_path.exists()
        if not full_path.parent.exists():
            full_path.parent.mkdir(parents=True, exist_ok=True)
            side_effects.append(
                {"type": "created_directory", "detail": str(full_path.parent)}
            )

        if mode == "append":
            with safe_open(workspace_root, path, "a") as f:
                f.write(content)
        else:
            if already_existed:
                side_effects.append(
                    {"type": "overwrote_existing_file", "detail": str(full_path)}
                )
            atomic_write(full_path, content, workspace_root=workspace_root)

        metadata["bytes_written"] = len(content.encode("utf-8"))
        if side_effects:
            metadata["side_effects"] = side_effects
        metadata["duration_ms"] = round((time.monotonic() - started) * 1000, 2)

        return ToolResult(
            tool_name="write_file", success=True, output="", metadata=metadata
        )

    except PathEscapesWorkspaceError as e:
        return ToolResult(
            tool_name="write_file", success=False, output="", error=str(e)
        )
    except SensitiveFileBlocked as e:
        return ToolResult(
            tool_name="write_file", success=False, output="", error=str(e)
        )
    except PermissionError:
        return ToolResult(
            tool_name="write_file",
            success=False,
            output="",
            error=f"Permission denied writing '{path}'.",
        )


def register(registry: ToolRegistry) -> None:
    registry.add(
        name="write_file",
        schema=WriteFileArgs,
        fn=write_file,
        needs_approval=True,
        core=False,
    )
