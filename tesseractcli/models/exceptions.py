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

class ConfigError(Exception):
    """Base class for all configuration-related errors.

    Every other exception raised by the ``config`` package inherits
    from this class.
    """


class ConfigFileNotFoundError(ConfigError):
    """Raised when a required configuration file cannot be found on disk.

    This covers both the active ``global_config.yaml`` and the
    distributed ``default_config.yaml`` template.
    """


class InvalidConfigError(ConfigError):
    """Raised when configuration content fails schema validation.

    This is raised whenever raw YAML content (or an in-memory mapping)
    cannot be parsed into a valid :class:`~config.models.GlobalConfig`.
    """


class ConfigKeyError(ConfigError):
    """Raised when a dot-notation configuration path cannot be resolved.

    Examples of invalid paths include referencing a section that does
    not exist (``providers.missing.max_tokens``) or attempting to
    descend into a scalar value as though it were a mapping.
    """


class ConfigPackError(ConfigError):
    """Raised for errors related to provider pack management.

    Examples include adding a pack that already exists, or removing,
    renaming, or reading a pack that does not exist.
    """

class ConfigModelError(ConfigError):
    """Raised for errors related to model entries inside a provider pack.

    Examples include removing a model that is not present in a pack's
    pool, or referencing an unknown pack while managing its models.
    """
