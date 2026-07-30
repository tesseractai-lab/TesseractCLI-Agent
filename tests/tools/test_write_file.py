"""
tests/tools/test_write_file.py
"""

from pathlib import Path

from tesseractcli.tools.write_file import write_file


def test_write_file_creates_new_file_with_content(workspace_root: Path) -> None:
    result = write_file(workspace_root, path="notes.txt", content="hello")

    assert result.success is True
    assert (workspace_root / "notes.txt").read_text() == "hello"


def test_write_file_overwrite_mode_replaces_existing_content(
    workspace_root: Path,
) -> None:
    (workspace_root / "notes.txt").write_text("old content")

    result = write_file(
        workspace_root, path="notes.txt", content="new content", mode="overwrite"
    )

    assert result.success is True
    assert (workspace_root / "notes.txt").read_text() == "new content"


def test_write_file_append_mode_adds_to_existing_content(workspace_root: Path) -> None:
    (workspace_root / "notes.txt").write_text("line1\n")

    result = write_file(
        workspace_root, path="notes.txt", content="line2\n", mode="append"
    )

    assert result.success is True
    assert (workspace_root / "notes.txt").read_text() == "line1\nline2\n"


def test_write_file_creates_parent_directories_if_missing(workspace_root: Path) -> None:
    result = write_file(workspace_root, path="nested/dir/file.txt", content="x")

    assert result.success is True
    assert (workspace_root / "nested" / "dir" / "file.txt").exists()


def test_write_file_invalid_mode_returns_failed_result(workspace_root: Path) -> None:
    result = write_file(
        workspace_root, path="notes.txt", content="x", mode="invalid_mode"
    )

    assert result.success is False
    assert result.error is not None
    assert "invalid_mode" in result.error.lower() or "mode" in result.error.lower()


def test_write_file_rejects_path_escaping_workspace(workspace_root: Path) -> None:
    result = write_file(workspace_root, path="../outside.txt", content="x")

    assert result.success is False
    assert result.error is not None
    # الملف مفروض متتكتبش خالص برة الـ workspace
    assert not (workspace_root.parent / "outside.txt").exists()


def test_write_file_metadata_contains_bytes_written(workspace_root: Path) -> None:
    result = write_file(workspace_root, path="notes.txt", content="hello")

    assert result.metadata is not None
    assert result.metadata["bytes_written"] == len("hello".encode("utf-8"))
