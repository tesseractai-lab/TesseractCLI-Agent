"""
tesseractcli/tools/sandbox/file_ops.py

Centralized, sandboxed file I/O.

safe_open() combines path validation with consistent error translation
for read/append/write operations. It does NOT create parent directories
or otherwise mutate the filesystem beyond what open() itself does —
any side effects (like directory creation) are the caller's
responsibility to perform and log explicitly. A silent auto-mkdir here
would make tools like write_file lie about their own side_effects
metadata, so that logic stays out of this file on purpose.

atomic_write() is a separate primitive for atomic overwrite via a temp
file + os.replace, since that doesn't fit the open()-context-manager
shape at all. It operates on an already-resolved, already-trusted Path
— the caller must have run resolve_in_workspace() first.
"""

import contextlib
import os
import tempfile
from pathlib import Path
from typing import IO, Iterator

from tesseractcli.models.exceptions import FileNotFoundInWorkspace
from tesseractcli.tools.sandbox.path_guard import resolve_in_workspace


@contextlib.contextmanager
def safe_open(
    workspace: Path,
    user_path: str | Path,
    mode: str = "r",
    encoding: str | None = "utf-8",
) -> Iterator[IO]:
    """
    Resolve `user_path` inside `workspace` and open it safely.

    Raises:
        PathEscapesWorkspaceError: propagated from resolve_in_workspace.
        FileNotFoundInWorkspace: if opening for read and the file doesn't exist.

    Note: does not create parent directories. If a caller needs to write
    into a possibly-missing directory and wants to report that as a
    side effect, it must create the directory itself before calling this.
    """
    full_path = resolve_in_workspace(workspace, user_path)

    is_read_mode = "r" in mode and "+" not in mode
    if is_read_mode and not full_path.is_file():
        raise FileNotFoundInWorkspace(f"'{user_path}' not found in workspace.")

    if "b" in mode:
        encoding = None

    f = open(full_path, mode, encoding=encoding)
    try:
        yield f
    finally:
        f.close()


def atomic_write(full_path: Path, content: str) -> None:
    """
    Atomically overwrite full_path with content via a temp file + os.replace.

    full_path must already be a resolved, validated Path (the caller
    already ran it through resolve_in_workspace). This function does
    not perform path validation itself since it never touches raw
    user input.
    """
    fd, tmp_path = tempfile.mkstemp(dir=full_path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(content)
        os.replace(tmp_path, full_path)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)