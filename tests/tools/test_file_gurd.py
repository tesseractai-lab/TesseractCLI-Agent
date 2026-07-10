"""
tests/tools/test_sandbox.py
"""

from pathlib import Path

import pytest

from tesseractcli.models.exceptions import PathEscapesWorkspaceError
from tesseractcli.tools.sandbox import resolve_in_workspace


def test_resolve_valid_path(tmp_path: Path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    safe_path = resolve_in_workspace(workspace, "notes.txt")

    assert safe_path == workspace / "notes.txt"


def test_reject_path_traversal(tmp_path: Path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    with pytest.raises(PathEscapesWorkspaceError):
        resolve_in_workspace(workspace, "../secret.txt")


def test_reject_absolute_path(tmp_path: Path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    absolute = Path("/etc/passwd")

    # Windows fallback
    if not absolute.is_absolute():
        absolute = tmp_path.resolve()

    with pytest.raises(PathEscapesWorkspaceError):
        resolve_in_workspace(workspace, absolute)


def test_allow_nested_relative_path(tmp_path: Path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    path = resolve_in_workspace(workspace, "docs/api/test.py")

    assert path == workspace / "docs" / "api" / "test.py"


def test_reject_symlink_escape(tmp_path: Path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    outside = tmp_path / "outside"
    outside.mkdir()

    link = workspace / "link"

    try:
        link.symlink_to(outside, target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("Symlink creation is not permitted on this platform.")

    with pytest.raises(PathEscapesWorkspaceError):
        resolve_in_workspace(workspace, "link/secret.txt")