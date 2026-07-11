"""
tests/tools/test_list_directory.py
"""

from pathlib import Path

from tesseractcli.tools.list_directory import list_directory


def test_list_directory_empty_directory(workspace_root: Path) -> None:
    result = list_directory(workspace_root, path=".")

    assert result.success is True
    assert "empty" in result.output.lower()
    assert result.metadata["item_count"] == 0


def test_list_directory_lists_files_and_subdirectories(workspace_root: Path) -> None:
    (workspace_root / "file.txt").write_text("x")
    (workspace_root / "subdir").mkdir()

    result = list_directory(workspace_root, path=".")

    assert result.success is True
    assert "file.txt" in result.output
    assert "subdir" in result.output
    assert result.metadata["item_count"] == 2


def test_list_directory_nonexistent_path_returns_failed_result(workspace_root: Path) -> None:
    result = list_directory(workspace_root, path="does_not_exist")

    assert result.success is False
    assert result.error is not None


def test_list_directory_path_is_a_file_not_a_directory(workspace_root: Path) -> None:
    (workspace_root / "file.txt").write_text("x")

    result = list_directory(workspace_root, path="file.txt")

    assert result.success is False
    assert result.error is not None


def test_list_directory_rejects_path_escaping_workspace(workspace_root: Path) -> None:
    result = list_directory(workspace_root, path="../")

    assert result.success is False
    assert result.error is not None


def test_list_directory_default_path_is_current_workspace_root(workspace_root: Path) -> None:
    (workspace_root / "a.txt").write_text("x")

    # مفيش path متمررة خالص — لازم تستخدم "." كـ default
    result = list_directory(workspace_root)

    assert result.success is True
    assert "a.txt" in result.output
