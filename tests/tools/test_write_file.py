from pathlib import Path

from tesseractcli.tools.write_file import write_file


def test_write_new_file(tmp_path: Path):
    result = write_file(tmp_path, "notes.txt", "hello")

    assert result.success

    assert (tmp_path / "notes.txt").read_text() == "hello"

    assert result.metadata["bytes_written"] == len("hello".encode())


def test_append_file(tmp_path: Path):
    file = tmp_path / "notes.txt"
    file.write_text("Hello", encoding="utf-8")

    result = write_file(
        tmp_path,
        "notes.txt",
        " World",
        mode="append",
    )

    assert result.success
    assert file.read_text() == "Hello World"


def test_invalid_mode(tmp_path: Path):
    result = write_file(
        tmp_path,
        "a.txt",
        "text",
        mode="invalid",
    )

    assert result.success is False
    assert "Invalid mode" in result.error


def test_create_directory(tmp_path: Path):
    result = write_file(
        tmp_path,
        "docs/file.txt",
        "hello",
    )

    assert result.success

    assert (tmp_path / "docs").exists()

    assert result.metadata["side_effects"] == [
        {
            "type": "created_directory",
            "detail": str(tmp_path / "docs"),
        }
    ]


def test_overwrite_existing_file(tmp_path: Path):
    file = tmp_path / "file.txt"
    file.write_text("old", encoding="utf-8")

    result = write_file(
        tmp_path,
        "file.txt",
        "new",
    )

    assert result.success

    assert file.read_text() == "new"

    assert result.metadata["side_effects"] == [
        {
            "type": "overwrote_existing_file",
            "detail": str(file),
        }
    ]