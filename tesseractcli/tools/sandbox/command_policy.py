"""
tesseractcli/tools/sandbox/command_policy.py

Responsibility: answer an open-ended, heuristic question — "is this command
safe to run?" — as opposed to path_guard.py's deterministic geometric check.

Three layers, deliberately combined because none is sufficient alone:

  1. Shell interpreter block. Invoking a shell interpreter directly
     (e.g. `bash -c "..."`) hands it a full script to interpret, which
     defeats the shell=False protection at the subprocess boundary —
     the interpreter itself becomes the shell. Blocked outright,
     regardless of arguments.

  2. Denylist of known-dangerous patterns. Useful, but NOT exhaustive —
     e.g. `python3 -c "import shutil; shutil.rmtree('/')"` achieves the same
     result as `rm -rf /` without matching any rm-based pattern. Regex-based
     checks are inherently incomplete against a determined adversary.

  3. Default-deny allowlist for auto-approval. Instead of trying to
     enumerate every dangerous command (open-ended, impossible to complete),
     we enumerate the SMALL set of read-only-ish commands considered safe
     to run without human confirmation. Everything else requires the
     approval gate (Phase 4 of the roadmap) — this function only decides
     whether a human confirmation is required, it does not obtain that
     confirmation itself.

Explicit non-goal: this module is NOT a complete security boundary by
itself. True isolation ultimately needs OS/process-level measures
(restricted user, resource limits, container/sandbox isolation) — this is
one layer of defense-in-depth, not the whole defense. It also does not
inspect paths embedded inside arguments (e.g. `cat /etc/passwd` is not
caught) — known gap, tracked in TODO.md, acceptable for MVP.
"""

import re

# Shell interpreters that must never be invoked directly, since doing so
# reintroduces shell-script interpretation even under shell=False.
SHELL_INTERPRETERS: set[str] = {
    "bash",
    "sh",
    "zsh",
    "dash",
    "ksh",
    "csh",
    "fish",
    "cmd",
    "cmd.exe",
    "powershell",
    "pwsh",
}

DANGEROUS_PATTERNS: list[str] = [
    r"rm\s+-rf\s+/(?:\s|$)",
    r"\bsudo\b",
    r":\(\)\{.*:\|:&.*\};:",  # fork bomb
    r"dd\s+if=.*of=/dev/(sd|nvme|hd)",
    r"chmod\s+-R\s+777\s+/",
    r"curl[^|]*\|\s*(sh|bash)\b",
    r"wget[^|]*\|\s*(sh|bash)\b",
    r">\s*/dev/(sd|nvme|hd)",
    r"\bmkfs\.",
]

# Command prefixes considered safe enough to auto-approve without a human
# confirmation step. Deliberately small and conservative — default-deny,
# not default-allow.
SAFE_READONLY_PREFIXES: set[str] = {
    "ls",
    "cat",
    "grep",
    "find",
    "pwd",
    "echo",
    "git status",
    "git diff",
    "git log",
    "pytest",
    "python -m pytest",
}


class CommandPolicyResult:
    """Outcome of a command_policy check. Distinct from ToolResult on
    purpose — this is a pre-execution decision, not an execution result."""

    def __init__(
        self, blocked: bool, requires_approval: bool, reason: str | None = None
    ):
        self.blocked = blocked
        self.requires_approval = requires_approval
        self.reason = reason


def check_command(command: list[str]) -> CommandPolicyResult:
    if not command:
        return CommandPolicyResult(
            blocked=True,
            requires_approval=False,
            reason="Empty command.",
        )

    # Layer 1: block direct shell interpreter invocation, regardless of
    # arguments — this must run before any pattern matching below.
    executable = command[0].rsplit("/", 1)[-1]
    if executable in SHELL_INTERPRETERS:
        return CommandPolicyResult(
            blocked=True,
            requires_approval=False,
            reason=f"Direct shell interpreter invocation ('{executable}') is not allowed.",
        )

    joined = " ".join(command)

    # Layer 2: known-dangerous patterns.
    for pattern in DANGEROUS_PATTERNS:
        if re.search(pattern, joined):
            return CommandPolicyResult(
                blocked=True,
                requires_approval=False,  # blocked outright, approval is moot
                reason=f"Command matches a blocked pattern ({pattern}).",
            )

    # Layer 3: default-deny allowlist for auto-approval.
    is_safe_readonly = any(
        joined.startswith(prefix) for prefix in SAFE_READONLY_PREFIXES
    )

    return CommandPolicyResult(
        blocked=False,
        requires_approval=not is_safe_readonly,
        reason=None
        if is_safe_readonly
        else "Command is not in the read-only allowlist.",
    )
