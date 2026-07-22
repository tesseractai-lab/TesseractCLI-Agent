"""
exec_tool.py

run_command — executes a command with shell=False (command as list[str],
never a raw string) to eliminate shell-injection risk by construction.
See command_policy.py for the pre-execution safety check this integrates
with; see the approval-gate design (Phase 4 of the roadmap) for where
`requires_approval` results actually get surfaced to a human.
"""

import subprocess
import time
from pathlib import Path

from tesseractcli.models import ToolResult
from tesseractcli.models.tool_models import RunCommandMetadata, RunCommandArgs
from tesseractcli.tools.registry import ToolRegistry
from tesseractcli.tools.sandbox.command_policy import check_command
from tesseractcli.config.settings import get_settings as config



def _truncate(text: str, limit: int = config().MAX_OUTPUT_CHARS) -> tuple[str, bool]:
    if len(text) <= limit:
        return text, False
    return text[:limit] + f"\n... [truncated, {len(text) - limit} more characters]", True


def run_command(
    workspace_root: Path,
    command: list[str],
    timeout: int = config().DEFAULT_TIMEOUT_SECONDS,
) -> ToolResult:
    metadata: RunCommandMetadata = {"command": command}

    policy_result = check_command(command)
    if policy_result.blocked:
        metadata["blocked_by_policy"] = True
        metadata["blocked_reason"] = policy_result.reason or ""
        return ToolResult(
            tool_name="run_command", success=False, output="",
            error=f"Command blocked by security policy: {policy_result.reason}",
            metadata=metadata,
        )

    # NOTE: policy_result.requires_approval is intentionally NOT enforced
    # here. This function's job is execution, not authorization — the
    # approval gate (Phase 4) is the layer responsible for calling
    # check_command() before ever reaching run_command, and deciding
    # whether to prompt a human. run_command trusts that if it was called,
    # that decision has already been made upstream.

    started = time.monotonic()
    try:
        result = subprocess.run(
            command,
            shell=False,
            cwd=workspace_root,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        duration_ms = round((time.monotonic() - started) * 1000, 2)

        stdout, stdout_truncated = _truncate(result.stdout)
        stderr, stderr_truncated = _truncate(result.stderr)

        metadata["exit_code"] = result.returncode
        metadata["timed_out"] = False
        metadata["stdout_truncated"] = stdout_truncated
        metadata["stderr_truncated"] = stderr_truncated
        metadata["duration_ms"] = duration_ms

        return ToolResult(
            tool_name="run_command",
            success=(result.returncode == 0),
            output=stdout,
            error=stderr or None,
            metadata=metadata,
        )

    except subprocess.TimeoutExpired:
        duration_ms = round((time.monotonic() - started) * 1000, 2)
        metadata["timed_out"] = True
        metadata["duration_ms"] = duration_ms
        metadata["side_effects"] = [
            {"type": "process_killed", "detail": f"SIGKILL after {timeout}s timeout"}
        ]
        return ToolResult(
            tool_name="run_command", success=False, output="",
            error=f"Command timed out after {timeout}s.",
            metadata=metadata,
        )

    except FileNotFoundError:
        return ToolResult(
            tool_name="run_command", success=False, output="",
            error=f"Command not found: '{command[0]}'.",
            metadata=metadata,
        )

    except PermissionError:
        return ToolResult(
            tool_name="run_command", success=False, output="",
            error=f"Permission denied executing '{command[0]}'.",
            metadata=metadata,
        )


def register(registry: ToolRegistry) -> None:
    registry.add(name="run_command", schema=RunCommandArgs, fn=run_command,needs_approval=True)
