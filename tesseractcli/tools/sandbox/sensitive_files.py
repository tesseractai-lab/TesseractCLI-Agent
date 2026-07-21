# tesseractcli/tools/sandbox/sensitive_files.py
"""
Content-class denylist for sensitive files (secrets, credentials, private keys).

This is DELIBERATELY separate from path_guard.py:
  - path_guard.py answers: "is this path inside the sandbox boundary?" (geometric)
  - sensitive_files.py answers: "is this path a class of file we never touch,
    even if it's fully inside the sandbox?" (content-class denylist)

Two different questions, two different lifecycles — path_guard's boundary
rarely changes, but this denylist will grow constantly as we discover new
secret-file patterns. Keeping them separate means editing this file never
risks touching the boundary-check logic.
"""

from __future__ import annotations

from dataclasses import dataclass
from fnmatch import fnmatch
from pathlib import Path
from typing import Literal

Action = Literal["read", "write"]

# Patterns are matched against the file's *name* AND its path relative to
# workspace root, both case-sensitively (secrets are usually case-sensitive
# anyway, and case-insensitive matching would just widen the surface for
# false negatives on Linux/macOS).
#
# NOTE: "list" is intentionally NOT covered here. Knowing a secret file
# EXISTS (its name showing up in list_directory output) is not itself a
# leak — the agent needs to know .env is there to tell the user "hey, set
# your API key in .env". What's forbidden is reading or mutating its
# CONTENT.
DENYLIST: dict[Action, list[str]] = {
    "read": [
        ".env",
        ".env.*",
        "*.pem",
        "*.key",
        "id_rsa*",
        "id_ed25519*",
        "*.pfx",
        "*.p12",
        "**/secrets/**",
        "**/credentials/**",
        ".git/config",
        ".aws/credentials",
        ".ssh/*",
        ".docker/config.json",
    ],
    "write": [
        ".env",
        ".env.*",
        "*.pem",
        "*.key",
        "id_rsa*",
        "id_ed25519*",
        "*.pfx",
        "*.p12",
        ".git/config",
        ".aws/credentials",
        ".ssh/*",
        ".docker/config.json",
        # NOTE: "**/secrets/**" and "**/credentials/**" deliberately NOT
        # blocked for write here — a directory-level rule blocking write
        # would also block *creating new* non-secret files inside a
        # secrets/ folder, which is a stricter call than we've made yet.
        # Revisit if that turns out to be too permissive in practice.
    ],
}

# Explicit allowlist that overrides the denylist — e.g. ".env.example"
# matches ".env.*" above but must never be blocked, since it's a template
# with no real secret values. Checked BEFORE the denylist.
ALLOWLIST_OVERRIDES: list[str] = [
    ".env.example",
    ".env.sample",
    ".env.template",
]


@dataclass(frozen=True)
class SensitiveFileCheck:
    allowed: bool
    matched_pattern: str | None = None


def _relative_and_name(path: Path, workspace_root: Path) -> tuple[str, str]:
    """
    Resolve symlinks first — a symlink named notes.txt pointing at .env
    must not bypass the denylist. resolve() follows symlinks and gives us
    the real target path to match against.
    """
    real = path.resolve()
    try:
        rel = real.relative_to(workspace_root.resolve())
    except ValueError:
        # Outside workspace entirely — path_guard should already have
        # rejected this before we ever get here, but fall back safely.
        rel = real
    return str(rel), real.name


def check(path: Path, action: Action, workspace_root: Path) -> SensitiveFileCheck:
    """
    Returns SensitiveFileCheck(allowed=False, matched_pattern=...) if `path`
    is denied for `action`, else allowed=True.

    Call this AFTER path_guard.check() — no point classifying sensitivity
    for a path that's already outside the sandbox boundary.
    """
    rel_str, name = _relative_and_name(path, workspace_root)

    # Allowlist override wins first, regardless of denylist matches.
    for pattern in ALLOWLIST_OVERRIDES:
        if fnmatch(name, pattern) or fnmatch(rel_str, pattern):
            return SensitiveFileCheck(allowed=True)

    for pattern in DENYLIST.get(action, []):
        if fnmatch(name, pattern) or fnmatch(rel_str, pattern):
            return SensitiveFileCheck(allowed=False, matched_pattern=pattern)

    return SensitiveFileCheck(allowed=True)
