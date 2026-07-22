"""Dot-notation path resolution over plain configuration mappings.

:class:`PathAccessor` is used internally by
:class:`~config.manager.ConfigManager` to implement its ``get()`` and
``set()`` methods. It operates on plain ``dict`` structures (typically
produced by ``GlobalConfig.model_dump()``) rather than on Pydantic
models directly, so a single code path handles both regular fields
(e.g. ``agent.temperature``) and the open-ended ``providers`` mapping
(e.g. ``providers.main.max_tokens``).

The class holds no state of its own - every method is a
``staticmethod``. It exists as a class (rather than a module of free
functions) for structural symmetry with :class:`~config.packs.PacksManager`:
every unit of configuration-mutation logic in this package lives in
its own dedicated class, and ``ConfigManager`` is the only thing that
touches the filesystem or the cache.
"""

from __future__ import annotations

from typing import Any, Final

from tesseractcli.models.exceptions import ConfigKeyError

PATH_SEPARATOR: Final[str] = "."

__all__ = ["PathAccessor"]


class PathAccessor:
    """Stateless dot-notation resolver for plain configuration mappings.

    Every method is a ``staticmethod``; instances carry no state and
    are never required. ``ConfigManager`` calls these directly as
    ``PathAccessor.split(...)`` / ``PathAccessor.get(...)`` /
    ``PathAccessor.set(...)``.
    """

    @staticmethod
    def split(path: str) -> list[str]:
        """Split a dot-notation path into its component keys.

        Args:
            path: Dot-separated configuration path, e.g.
                ``"agent.temperature"``.

        Returns:
            The list of individual key components.

        Raises:
            ConfigKeyError: If ``path`` is empty or blank.
        """
        if not path or not path.strip():
            raise ConfigKeyError("Configuration path must be a non-empty string.")
        return path.split(PATH_SEPARATOR)

    @staticmethod
    def get(data: Any, parts: list[str], full_path: str) -> Any:
        """Traverse ``data`` following ``parts`` and return the final value.

        Args:
            data: Mapping to traverse.
            parts: Sequence of keys produced by :meth:`split`.
            full_path: Original dotted path, used for error messages.

        Returns:
            The value found at the end of the path.

        Raises:
            ConfigKeyError: If any component of the path cannot be resolved.
        """
        current = data
        for part in parts:
            if not isinstance(current, dict) or part not in current:
                raise ConfigKeyError(f"Unknown configuration key: '{full_path}'.")
            current = current[part]
        return current

    @staticmethod
    def set(data: dict[str, Any], parts: list[str], value: Any, full_path: str) -> None:
        """Traverse ``data`` following all but the last of ``parts`` and assign.

        Only existing keys may be assigned; this method never creates
        new keys, since new structural elements (packs, models) must go
        through :class:`~config.packs.PacksManager` instead.

        Args:
            data: Mapping to mutate in place.
            parts: Sequence of keys produced by :meth:`split`.
            value: Value to assign at the final key.
            full_path: Original dotted path, used for error messages.

        Raises:
            ConfigKeyError: If any component of the path (including the
                final key) cannot be resolved.
        """
        current: Any = data
        for part in parts[:-1]:
            if not isinstance(current, dict) or part not in current:
                raise ConfigKeyError(f"Unknown configuration key: '{full_path}'.")
            current = current[part]

        last = parts[-1]
        if not isinstance(current, dict) or last not in current:
            raise ConfigKeyError(f"Unknown configuration key: '{full_path}'.")
        current[last] = value
