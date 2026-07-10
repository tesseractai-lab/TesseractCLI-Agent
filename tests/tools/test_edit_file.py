from pathlib import Path

from tesseractcli.tools.edit_file import edit_file


def test_edit_success(tmp_path: Path):
    file = tmp_path / "a.txt"
    file.write_text("hello world", encoding="utf-8")

    result = edit_file(
        tmp_path,
        "a.txt",
        "world",
        "python",
    )

    assert result.success

    assert file.read_text() == "hello python"

    assert result.metadata["match_count"] == 1
    assert result.metadata["chars_replaced"] == len("world")


def test_edit_missing_file(tmp_path: Path):
    result = edit_file(
        tmp_path,
        "missing.txt",
        "a",
        "b",
    )

    assert result.success is False
    assert "not found" in result.error.lower()


def test_old_string_not_found(tmp_path: Path):
    file = tmp_path / "a.txt"
    file.write_text("hello", encoding="utf-8")

    result = edit_file(
        tmp_path,
        "a.txt",
        "python",
        "world",
    )

    assert result.success is False
    assert "not found" in result.error.lower()


def test_old_string_not_unique(tmp_path: Path):
    file = tmp_path / "a.txt"
    file.write_text(
        "hello hello hello",
        encoding="utf-8",
    )

    result = edit_file(
        tmp_path,
        "a.txt",
        "hello",
        "python",
    )

    assert result.success is False
    assert "not unique" in result.error.lower()


def test_edit_outside_workspace(tmp_path: Path):
    result = edit_file(
        tmp_path,
        "../secret.txt",
        "a",
        "b",
    )

    assert result.success is False