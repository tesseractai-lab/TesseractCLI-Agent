"""
tesseractcli/models/exceptions.py

"""

class SandboxError(Exception):
    """Base class for all sandbox-related errors."""


class PathEscapesWorkspaceError(SandboxError):
    """Raised when a resolved path falls outside the workspace root
    (via absolute path, traversal, or symlink)."""


class FileNotFoundInWorkspace(SandboxError):
    """Raised when a file does not exist inside the workspace."""


class CommandNotAllowedError(SandboxError):
    """Raised when a command is blocked by policy. (future use)"""


class ResourceLimitExceededError(SandboxError):
    """Raised when a process exceeds CPU/memory/time limits. (future use)"""