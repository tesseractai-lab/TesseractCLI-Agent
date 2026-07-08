"""
tesseractcli/tools/sandbox.py

Utilities for safely resolving paths inside the user's workspace.

This module ensures that file operations cannot escape the configured
workspace directory through path traversal, absolute paths, or symlinks.
"""

from __future__ import annotations
from pathlib import Path

from tesseractcli.config.exceptions import SandboxViolationError


class Sandbox:
    """Resolve and validate paths inside a configured workspace."""

    def __init__(self, workspace: Path):
        self.workspace = workspace.resolve()

    def resolve_workspace_path(self, user_path: str | Path) -> Path:
        """
        Resolve a user-supplied path and ensure it remains inside
        the configured workspace.

        Args:
            user_path: Relative path supplied by the user.

        Returns:
            A resolved absolute Path inside the workspace.

        Raises:
            SandboxViolationError:
                If the path escapes the workspace or an absolute path
                is supplied.
        """

        # Normalize input.
        user_path = Path(user_path)

        # Absolute paths are never allowed.
        if user_path.is_absolute():
            raise SandboxViolationError(
                "Absolute paths are not allowed."
            )

        # Resolve against the workspace.
        candidate = (self.workspace / user_path).resolve(strict=False)

        # Ensure the resolved path is still inside the workspace.
        if not candidate.is_relative_to(self.workspace):
            raise SandboxViolationError("Path escapes the workspace.")

        
        return candidate