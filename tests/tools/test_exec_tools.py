"""
tests/tools/test_exec_tools.py

ملحوظة مهمة: التستات دي لازم تشتغل على أي OS (Windows/Linux/Mac)، فبنتجنب
عمدًا أي binary مش مضمون وجوده كـ standalone executable على كل الأنظمة
(زي echo و cat اللي على Windows مش موجودين كـ .exe مستقل، هما shell
builtins بس). البديل: نستخدم sys.executable (نفس الـ Python اللي التست
شغال بيه دلوقتي) مع -c لعمل نفس السلوك بشكل موحّد على أي نظام.
"""

import sys
from pathlib import Path

from tesseractcli.tools.exec_tool import run_command
from tesseractcli.config.settings import get_settings as config

MAX_OUTPUT_CHARS = config().MAX_OUTPUT_CHARS

PY = sys.executable  # بدل "python3" الثابتة اللي مش موجودة على كل الأنظمة


def test_run_command_success_returns_stdout(workspace_root: Path) -> None:
    result = run_command(workspace_root, command=[PY, "-c", "print('hello')"])

    assert result.success is True
    assert "hello" in result.output
    assert result.metadata["exit_code"] == 0


def test_run_command_nonzero_exit_code_is_failure(workspace_root: Path) -> None:
    result = run_command(workspace_root, command=[PY, "-c", "import sys; sys.exit(1)"])

    assert result.success is False
    assert result.metadata["exit_code"] == 1


def test_run_command_blocked_by_policy_never_executes(workspace_root: Path) -> None:
    result = run_command(workspace_root, command=["bash", "-c", "echo should_not_run"])

    assert result.success is False
    assert result.metadata["blocked_by_policy"] is True
    assert "should_not_run" not in result.output


def test_run_command_times_out(workspace_root: Path) -> None:
    result = run_command(
        workspace_root,
        command=[PY, "-c", "import time; time.sleep(5)"],
        timeout=1,
    )

    assert result.success is False
    assert result.metadata["timed_out"] is True


def test_run_command_nonexistent_binary_returns_clear_error(
    workspace_root: Path,
) -> None:
    result = run_command(workspace_root, command=["this_binary_does_not_exist_xyz"])

    assert result.success is False
    assert "not found" in (result.error or "").lower()


def test_run_command_truncates_large_output(workspace_root: Path) -> None:
    huge_length = MAX_OUTPUT_CHARS + 500
    code = f"print('a' * {huge_length})"

    result = run_command(workspace_root, command=[PY, "-c", code])

    assert result.metadata["stdout_truncated"] is True
    assert len(result.output) <= MAX_OUTPUT_CHARS + len(
        "\n... [truncated, 500 more characters]"
    )


def test_run_command_executes_inside_workspace_root(workspace_root: Path) -> None:
    (workspace_root / "marker.txt").write_text("found_me")

    # بدل cat (مش موجود على Windows)، بنستخدم Python نفسه لقراءة الملف —
    # ده كمان بيتأكد إن الأمر بيتنفذ فعليًا جوه workspace_root (cwd الصح)
    code = "print(open('marker.txt').read())"
    result = run_command(workspace_root, command=[PY, "-c", code])

    assert result.success is True
    assert "found_me" in result.output
