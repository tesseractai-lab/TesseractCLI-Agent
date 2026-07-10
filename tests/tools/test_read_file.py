from pathlib import Path

from tesseractcli.tools.read_file import read_file


def test_read_existing_file(tmp_path: Path):
    file = tmp_path / "hello.txt"
    file.write_text("Hello\nWorld\n", encoding="utf-8")

    result = read_file(tmp_path, "hello.txt")

    assert result.success is True
    assert result.output == "Hello\nWorld\n"
    assert result.metadata["path"] == "hello.txt"
    assert result.metadata["total_lines"] == 2
    assert result.metadata["truncated"] is False


def test_read_line_range(tmp_path: Path):
    file = tmp_path / "data.txt"
    file.write_text("1\n2\n3\n4\n5\n", encoding="utf-8")

    result = read_file(
        tmp_path,
        "data.txt",
        start_line=2,
        end_line=4,
    )

    assert result.success
    assert result.output == "2\n3\n4\n"


def test_read_missing_file(tmp_path: Path):
    result = read_file(tmp_path, "missing.txt")

    assert result.success is False
    assert "not found" in result.error.lower()


def test_read_outside_workspace(tmp_path: Path):
    result = read_file(tmp_path, "../secret.txt")

    assert result.success is False