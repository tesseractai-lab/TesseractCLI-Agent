"""
tests/tools/test_sandbox.py
"""

from pathlib import Path

import pytest

from tesseractcli.models.exceptions import PathEscapesWorkspaceError
from tesseractcli.tools.sandbox import Sandbox


def test_resolve_valid_path(tmp_path: Path):
    """A normal file inside the workspace should be allowed."""

    workspace = tmp_path / "workspace"
    workspace.mkdir()

    sandbox = Sandbox(workspace)

    safe_path = sandbox.resolve_workspace_path("notes.txt")

    assert safe_path == workspace / "notes.txt"


def test_reject_path_traversal(tmp_path: Path):
    """Path traversal using '..' must be rejected."""

    workspace = tmp_path / "workspace"
    workspace.mkdir()

    sandbox = Sandbox(workspace)

    with pytest.raises(PathEscapesWorkspaceError):
        sandbox.resolve_workspace_path("../secret.txt")


def test_reject_absolute_path(tmp_path: Path):
    """Absolute paths are never allowed."""

    workspace = tmp_path / "workspace"
    workspace.mkdir()

    sandbox = Sandbox(workspace)

    absolute = Path("/etc/passwd")

    # Windows fallback
    if not absolute.is_absolute():
        absolute = tmp_path.resolve()

    with pytest.raises(PathEscapesWorkspaceError):
        sandbox.resolve_workspace_path(absolute)


def test_allow_nested_relative_path(tmp_path: Path):
    """Nested relative paths should be resolved inside the workspace."""

    workspace = tmp_path / "workspace"
    workspace.mkdir()

    sandbox = Sandbox(workspace)

    path = sandbox.resolve_workspace_path("docs/api/test.py")

    assert path == workspace / "docs" / "api" / "test.py"


def test_reject_symlink_escape(tmp_path: Path):
    """Symlinks escaping the workspace must be rejected."""

    workspace = tmp_path / "workspace"
    workspace.mkdir()

    outside = tmp_path / "outside"
    outside.mkdir()

    link = workspace / "link"

    try:
        link.symlink_to(outside, target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("Symlink creation is not permitted on this platform.")

    sandbox = Sandbox(workspace)

    with pytest.raises(PathEscapesWorkspaceError):
        sandbox.resolve_workspace_path("link/secret.txt")