"""
tests/tools/test_edit_file.py
"""

from pathlib import Path

from tesseractcli.tools.edit_file import edit_file


def test_edit_file_replaces_unique_match(workspace_root: Path) -> None:
    (workspace_root / "config.py").write_text("DEBUG = False\nNAME = 'app'\n")

    result = edit_file(
        workspace_root,
        path="config.py",
        old_str="DEBUG = False",
        new_str="DEBUG = True",
    )

    assert result.success is True
    assert (workspace_root / "config.py").read_text() == "DEBUG = True\nNAME = 'app'\n"


def test_edit_file_old_str_not_found_returns_failed_result(workspace_root: Path) -> None:
    (workspace_root / "config.py").write_text("DEBUG = False\n")

    result = edit_file(
        workspace_root,
        path="config.py",
        old_str="THIS_STRING_DOES_NOT_EXIST",
        new_str="anything",
    )

    assert result.success is False
    assert result.error is not None
    # الملف الأصلي لازم يفضل زي ما هو من غير أي تعديل
    assert (workspace_root / "config.py").read_text() == "DEBUG = False\n"


def test_edit_file_non_unique_old_str_returns_failed_result(workspace_root: Path) -> None:
    (workspace_root / "config.py").write_text("x = 1\nx = 1\n")

    result = edit_file(
        workspace_root,
        path="config.py",
        old_str="x = 1",
        new_str="x = 2",
    )

    assert result.success is False
    assert result.error is not None
    assert result.metadata is not None
    assert result.metadata["match_count"] == 2
    # الملف الأصلي لازم يفضل زي ما هو، مفيش تعديل جزئي
    assert (workspace_root / "config.py").read_text() == "x = 1\nx = 1\n"


def test_edit_file_missing_file_returns_failed_result(workspace_root: Path) -> None:
    result = edit_file(
        workspace_root,
        path="does_not_exist.py",
        old_str="x",
        new_str="y",
    )

    assert result.success is False
    assert result.error is not None


def test_edit_file_rejects_path_escaping_workspace(workspace_root: Path) -> None:
    result = edit_file(
        workspace_root,
        path="../outside.py",
        old_str="x",
        new_str="y",
    )

    assert result.success is False
    assert result.error is not None
