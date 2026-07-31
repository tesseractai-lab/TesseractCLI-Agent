"""Configuration manager for the application.

:class:`ConfigManager` is the single object responsible for every
configuration operation in the application. No other module should
read or write ``global_config.yaml`` directly; everything goes through
this class (or its ``packs`` sub-manager) so that validation, caching,
and error handling stay in one place.

This module stays focused on *lifecycle* (create, load, save, reload,
validate, reset, backup) and *dot-notation access* (``get``/``set``).
Everything about provider packs and the models inside them lives in
:class:`~config.packs.PacksManager`, exposed as ``manager.packs``.
Dot-path traversal lives in :class:`~config.path_access.PathAccessor`.

Typical usage::

    from config.manager import ConfigManager

    manager = ConfigManager(config_dir="~/.myapp")
    manager.load()
    max_tokens = manager.get("providers.main.max_tokens")
    manager.set("agent.temperature", 0.2)
    manager.packs.add_pack("vision")
    manager.packs.add_model("vision", "openai", "gpt-4o")
    manager.save()
"""

from __future__ import annotations

import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Final

import yaml
from pydantic import ValidationError

from tesseractcli.config.global_config.packs_manager import PacksManager
from tesseractcli.config.global_config.paths_manager import PathAccessor
from tesseractcli.models.config_models.global_config_models import GlobalConfig
from tesseractcli.models.exceptions import (
    ConfigError,
    ConfigFileNotFoundError,
    InvalidConfigError,
)

__all__ = ["ConfigManager"]

# Public constants describing the on-disk layout. Centralized here so
# no filename or directory name is ever hardcoded more than once.
GLOBAL_CONFIG_FILENAME: Final[str] = "global_config.yaml"
DEFAULT_CONFIG_FILENAME: Final[str] = "default_config.yaml"
BACKUP_DIRNAME: Final[str] = "backups"
BACKUP_TIMESTAMP_FORMAT: Final[str] = "%Y%m%d_%H%M%S_%f"

# Permission bits applied to the active configuration file, its backups,
# and their containing directories. The configuration may hold
# credentials in the future (API keys inside a provider pack), so it is
# treated as sensitive from day one rather than hardened retroactively.
# Only enforced on POSIX platforms - see _secure_file/_secure_dir.
_FILE_MODE: Final[int] = 0o600
_DIR_MODE: Final[int] = 0o700

# Sentinel used to distinguish "no default given" from "default is None"
# in ConfigManager.get().
_MISSING: Final[object] = object()

CONFIG_DIR = Path(__file__).parents[3] / "app_config"


class ConfigManager:
    """Single source of truth for reading, writing, and mutating configuration.

    The manager owns exactly one in-memory :class:`~config.models.GlobalConfig`
    instance at a time. It is loaded lazily on first access and cached;
    the YAML file is never re-read on every operation, only on
    :meth:`load` (first call), :meth:`reload`, and :meth:`reset`.

    Provider-pack and model management is delegated entirely to the
    ``packs`` attribute; this class does not implement any of that
    logic itself, only the two hooks (``_dump_config`` /
    ``_commit_config``) that let ``packs`` read and persist changes.

    Attributes:
        config_path: Absolute path to the active ``global_config.yaml``.
        default_path: Absolute path to the packaged ``default_config.yaml``
            template used to seed a missing configuration file.
        packs: The :class:`~config.packs.PacksManager` bound to this
            manager, e.g. ``manager.packs.add_pack("vision")``.
    """

    def __init__(
        self,
        config_dir: str | Path = CONFIG_DIR,
        *,
        config_filename: str = GLOBAL_CONFIG_FILENAME,
        default_config_path: str | Path | None = None,
    ) -> None:
        """Initialize the manager without touching the filesystem.

        Args:
            config_dir: Directory in which the active configuration
                file lives (or will be created).
            config_filename: Name of the active configuration file
                within ``config_dir``. Defaults to
                ``"global_config.yaml"``.
            default_config_path: Explicit path to the default
                configuration template. When omitted, the template
                distributed alongside this package
                (``config/default_config.yaml``) is used.
        """
        self._config_dir: Path = Path(config_dir).expanduser()
        self._config_path: Path = self._config_dir / config_filename
        self._default_path: Path = (
            Path(default_config_path).expanduser()
            if default_config_path is not None
            else Path(__file__).resolve().parent / DEFAULT_CONFIG_FILENAME
        )
        self._config: GlobalConfig | None = None
        self.packs: PacksManager = PacksManager(self)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def config_path(self) -> Path:
        """Path to the active ``global_config.yaml`` file."""
        return self._config_path

    @property
    def default_path(self) -> Path:
        """Path to the ``default_config.yaml`` template."""
        return self._default_path

    @property
    def config(self) -> GlobalConfig:
        """Currently cached configuration, loading it first if needed.

        Returns:
            The cached :class:`~config.models.GlobalConfig` instance.
        """
        self._ensure_loaded()
        assert self._config is not None
        return self._config

    # ------------------------------------------------------------------
    # Lifecycle: creation, loading, saving, resetting, backing up
    # ------------------------------------------------------------------

    def create_if_missing(self) -> bool:
        """Create ``global_config.yaml`` from the default template if absent.

        Returns:
            True if the file was created by this call, False if it
            already existed.

        Raises:
            ConfigFileNotFoundError: If the file is missing and the
                default template cannot be found either.
        """
        if self._config_path.exists():
            return False
        if not self._default_path.exists():
            raise ConfigFileNotFoundError(
                f"Cannot create '{self._config_path}': default template "
                f"'{self._default_path}' does not exist."
            )
        self._config_dir.mkdir(parents=True, exist_ok=True)
        self._secure_dir(self._config_dir)
        shutil.copy2(self._default_path, self._config_path)
        self._secure_file(self._config_path)
        return True

    def load(self, *, force: bool = False) -> GlobalConfig:
        """Load configuration from disk into the in-memory cache.

        If a cached configuration already exists and ``force`` is
        False, the cache is returned unchanged and the file is not
        touched.

        Args:
            force: If True, bypass the cache and re-read the file.

        Returns:
            The loaded (and validated) :class:`~config.models.GlobalConfig`.

        Raises:
            ConfigFileNotFoundError: If the file is missing and cannot
                be created from the default template.
            InvalidConfigError: If the file content is not valid YAML
                or does not match the configuration schema.
        """
        if self._config is not None and not force:
            return self._config

        self.create_if_missing()
        raw = self._read_yaml(self._config_path)
        self._config = self._validate_mapping(raw or {})
        return self._config

    def reload(self) -> GlobalConfig:
        """Discard the in-memory cache and re-read the file from disk.

        Returns:
            The freshly loaded :class:`~config.models.GlobalConfig`.
        """
        return self.load(force=True)

    def save(self) -> None:
        """Persist the in-memory configuration to ``global_config.yaml``.

        Raises:
            InvalidConfigError: If the file cannot be written.
        """
        self._ensure_loaded()
        self._config_dir.mkdir(parents=True, exist_ok=True)
        self._secure_dir(self._config_dir)
        payload = self._config.model_dump(mode="json")  # type: ignore[union-attr]
        try:
            with self._config_path.open("w", encoding="utf-8") as handle:
                yaml.safe_dump(payload, handle, sort_keys=False, allow_unicode=True)
        except OSError as exc:
            raise InvalidConfigError(
                f"Failed to write configuration to '{self._config_path}': {exc}"
            ) from exc
        self._secure_file(self._config_path)

    def validate(self, data: dict[str, Any] | None = None) -> GlobalConfig:
        """Validate configuration content without mutating manager state.

        Args:
            data: Raw mapping to validate. When omitted, the file on
                disk is read and validated (falling back to the cached
                in-memory configuration if the file is missing and a
                configuration was already loaded).

        Returns:
            The validated :class:`~config.models.GlobalConfig`. This is
            a standalone instance; it does not replace the cache.

        Raises:
            InvalidConfigError: If the content does not match the schema.
            ConfigFileNotFoundError: If ``data`` is omitted, the file is
                missing, and nothing has been loaded yet.
        """
        if data is not None:
            return self._validate_mapping(data)

        if self._config_path.exists():
            raw = self._read_yaml(self._config_path)
            return self._validate_mapping(raw or {})

        self._ensure_loaded()
        assert self._config is not None
        return self._config

    def reset(self, *, keep_backup: bool = True) -> GlobalConfig:
        """Reset the active configuration back to the default template.

        Args:
            keep_backup: If True (default) and an active configuration
                file exists, it is backed up before being overwritten.

        Returns:
            The reloaded, default :class:`~config.models.GlobalConfig`.

        Raises:
            ConfigFileNotFoundError: If the default template does not
                exist.
        """
        if not self._default_path.exists():
            raise ConfigFileNotFoundError(
                f"Cannot reset configuration: default template "
                f"'{self._default_path}' does not exist."
            )
        if keep_backup and self._config_path.exists():
            self.backup()

        self._config_dir.mkdir(parents=True, exist_ok=True)
        self._secure_dir(self._config_dir)
        shutil.copy2(self._default_path, self._config_path)
        self._secure_file(self._config_path)
        return self.reload()

    def backup(self) -> Path:
        """Create a timestamped copy of the active configuration file.

        Returns:
            Path to the newly created backup file, stored under a
            ``backups`` directory next to the active configuration.

        Raises:
            ConfigFileNotFoundError: If there is no active configuration
                file to back up.
        """
        if not self._config_path.exists():
            raise ConfigFileNotFoundError(
                f"Cannot back up: '{self._config_path}' does not exist."
            )
        backup_dir = self._config_dir / BACKUP_DIRNAME
        backup_dir.mkdir(parents=True, exist_ok=True)
        self._secure_dir(backup_dir)

        timestamp = datetime.now().strftime(BACKUP_TIMESTAMP_FORMAT)
        stem = f"{self._config_path.stem}_{timestamp}"
        suffix = self._config_path.suffix
        backup_path = backup_dir / f"{stem}{suffix}"

        # Guard against overwriting a prior backup on the rare chance two
        # calls land on the same microsecond-precision timestamp.
        collision_index = 1
        while backup_path.exists():
            backup_path = backup_dir / f"{stem}_{collision_index}{suffix}"
            collision_index += 1

        shutil.copy2(self._config_path, backup_path)
        self._secure_file(backup_path)
        return backup_path

    # ------------------------------------------------------------------
    # Dot-notation access
    # ------------------------------------------------------------------

    def get(self, path: str, default: Any = _MISSING) -> Any:
        """Read a configuration value using dot-notation.

        Args:
            path: Dot-separated path, e.g. ``"agent.temperature"`` or
                ``"providers.main.max_tokens"``.
            default: Value to return if ``path`` cannot be resolved.
                When omitted, an unresolved path raises instead.

        Returns:
            The value at ``path``.

        Raises:
            ConfigKeyError: If ``path`` cannot be resolved and no
                ``default`` was given.
        """
        data = self._dump_config()
        parts = PathAccessor.split(path)
        try:
            return PathAccessor.get(data, parts, path)
        except ConfigError:
            if default is not _MISSING:
                return default
            raise

    def set(self, path: str, value: Any) -> None:
        """Write a configuration value using dot-notation.

        The full configuration is re-validated after the write, so an
        invalid value (e.g. a negative ``max_tokens``) is rejected and
        the in-memory configuration is left unchanged.

        Args:
            path: Dot-separated path to an *existing* key, e.g.
                ``"paths.data_dir"`` or ``"providers.main.temperature"``.
            value: New value to assign at ``path``.

        Raises:
            ConfigKeyError: If ``path`` does not resolve to an existing key.
            InvalidConfigError: If the resulting configuration is invalid.
        """
        data = self._dump_config()
        parts = PathAccessor.split(path)
        PathAccessor.set(data, parts, value, path)
        self._commit_config(data)

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Export the current configuration as a plain dictionary.

        Returns:
            A JSON-compatible mapping (paths and nested models are
            converted to primitive types).
        """
        self._ensure_loaded()
        return self._config.model_dump(mode="json")  # type: ignore[union-attr]

    def to_yaml(self) -> str:
        """Export the current configuration as a YAML-formatted string.

        Returns:
            The configuration serialized with ``yaml.safe_dump``.
        """
        return yaml.safe_dump(self.to_dict(), sort_keys=False, allow_unicode=True)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _ensure_loaded(self) -> None:
        """Load configuration into the cache if it has not been loaded yet."""
        if self._config is None:
            self.load()

    def _dump_config(self) -> dict[str, Any]:
        """Return the current configuration as a mutable plain ``dict``.

        This is the read half of the internal hook used by
        :class:`~config.packs.PacksManager` (and by ``get``/``set``
        above) to work with the configuration without depending on
        Pydantic model internals directly.

        Returns:
            A fresh ``dict`` (mutating it never affects the cache
            until it is passed to :meth:`_commit_config`).
        """
        self._ensure_loaded()
        return self._config.model_dump(mode="python")  # type: ignore[union-attr]

    def _commit_config(self, data: dict[str, Any]) -> None:
        """Validate a mutated mapping and swap it into the cache.

        This is the write half of the internal hook used by
        :class:`~config.packs.PacksManager`. Validation happens here,
        in one place, so no caller can install an invalid configuration
        into the cache.

        Args:
            data: A mapping derived from :meth:`_dump_config` and then
                mutated by the caller.

        Raises:
            InvalidConfigError: If ``data`` does not match the schema.
        """
        self._config = self._validate_mapping(data)

    def _validate_mapping(self, data: dict[str, Any]) -> GlobalConfig:
        """Validate a raw mapping against the GlobalConfig schema.

        Args:
            data: Raw mapping, typically produced by ``yaml.safe_load``
                or ``model_dump``.

        Returns:
            A validated :class:`~config.models.GlobalConfig` instance.

        Raises:
            InvalidConfigError: If ``data`` does not match the schema.
        """
        try:
            return GlobalConfig.model_validate(data)
        except ValidationError as exc:
            raise InvalidConfigError(
                f"Configuration failed schema validation: {exc}"
            ) from exc

    def _read_yaml(self, path: Path) -> dict[str, Any] | None:
        """Read and parse a YAML file.

        Args:
            path: Path to the YAML file to read.

        Returns:
            The parsed content, or None if the file is empty.

        Raises:
            ConfigFileNotFoundError: If the file cannot be read.
            InvalidConfigError: If the file is not valid YAML.
        """
        try:
            with path.open("r", encoding="utf-8") as handle:
                return yaml.safe_load(handle)
        except OSError as exc:
            raise ConfigFileNotFoundError(f"Cannot read '{path}': {exc}") from exc
        except yaml.YAMLError as exc:
            raise InvalidConfigError(
                f"Failed to parse YAML in '{path}': {exc}"
            ) from exc

    @staticmethod
    def _secure_file(path: Path) -> None:
        """Restrict a file to owner read/write only (``0o600``), on POSIX.

        A no-op on non-POSIX platforms (e.g. Windows), where
        ``Path.chmod`` does not offer the same permission model.

        Args:
            path: File to restrict.

        Raises:
            ConfigError: If the permission change fails on a POSIX system.
        """
        if os.name != "posix":
            return
        try:
            path.chmod(_FILE_MODE)
        except OSError as exc:
            raise ConfigError(
                f"Failed to set secure permissions on '{path}': {exc}"
            ) from exc

    @staticmethod
    def _secure_dir(path: Path) -> None:
        """Restrict a directory to owner access only (``0o700``), on POSIX.

        A no-op on non-POSIX platforms (e.g. Windows).

        Args:
            path: Directory to restrict.

        Raises:
            ConfigError: If the permission change fails on a POSIX system.
        """
        if os.name != "posix":
            return
        try:
            path.chmod(_DIR_MODE)
        except OSError as exc:
            raise ConfigError(
                f"Failed to set secure permissions on '{path}': {exc}"
            ) from exc
