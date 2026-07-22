"""Provider pack and model mutation operations.

:class:`PacksManager` owns every operation that changes the shape of
the ``providers`` section of the configuration: adding, removing, or
renaming a pack, and adding, removing, or clearing models inside a
pack's ``pool``/``fallback`` lists.

It never touches the filesystem or the schema directly. Instead, each
method goes through its parent :class:`~config.manager.ConfigManager`
via two narrow hooks:

- ``manager._dump_config()`` - read the current configuration as a
  plain, mutable ``dict``.
- ``manager._commit_config(data)`` - validate a mutated ``dict``
  against the schema and, if valid, swap it into the manager's cache.

This keeps exactly one object - :class:`ConfigManager` - responsible
for the cache, the schema, and the filesystem, while every unit of
pack/model-specific business logic lives here, independent of I/O.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

from tesseractcli.models.exceptions import ConfigModelError, ConfigPackError
from tesseractcli.models.config_models.provider_models import ModelPack

if TYPE_CHECKING:
    from tesseractcli.config.global_config.manager import ConfigManager

PoolTarget = Literal["pool", "fallback"]

__all__ = ["PacksManager", "PoolTarget"]


class PacksManager:
    """Provider pack and model management, scoped to one ConfigManager.

    An instance is created once, internally, by
    ``ConfigManager.__init__`` and exposed as the ``packs`` attribute
    (e.g. ``manager.packs.add_pack("vision")``). There is no reason to
    construct one directly.
    """

    def __init__(self, manager: "ConfigManager") -> None:
        """Bind this object to the ConfigManager it will read from and write to.

        Args:
            manager: The owning ConfigManager instance.
        """
        self._manager = manager

    # ------------------------------------------------------------------
    # Pack management
    # ------------------------------------------------------------------

    def add_pack(self, name: str, pack: ModelPack | None = None) -> None:
        """Add a new, empty (or given) provider pack.

        Args:
            name: Name of the new pack, e.g. ``"vision"``.
            pack: Optional pre-built :class:`~config.models.ModelPack`
                to insert. When omitted, an empty pack with default
                settings is created.

        Raises:
            ConfigPackError: If ``name`` is empty or a pack with that
                name already exists.
            InvalidConfigError: If the resulting configuration is invalid.
        """
        if not name or not name.strip():
            raise ConfigPackError("Provider pack name must be a non-empty string.")

        data = self._manager._dump_config()
        if name in data["providers"]:
            raise ConfigPackError(f"Provider pack '{name}' already exists.")

        new_pack = pack if pack is not None else ModelPack()
        data["providers"][name] = new_pack.model_dump(mode="python")
        self._manager._commit_config(data)

    def remove_pack(self, name: str) -> None:
        """Remove an existing provider pack.

        Args:
            name: Name of the pack to remove.

        Raises:
            ConfigPackError: If no pack named ``name`` exists.
        """
        data = self._manager._dump_config()
        self._require_pack(data, name)
        del data["providers"][name]
        self._manager._commit_config(data)

    def rename_pack(self, old_name: str, new_name: str) -> None:
        """Rename an existing provider pack.

        Args:
            old_name: Current name of the pack.
            new_name: New name for the pack.

        Raises:
            ConfigPackError: If ``old_name`` does not exist or
                ``new_name`` already exists.
        """
        data = self._manager._dump_config()
        self._require_pack(data, old_name)
        if not new_name or not new_name.strip():
            raise ConfigPackError("New provider pack name must be a non-empty string.")
        if new_name in data["providers"]:
            raise ConfigPackError(f"Provider pack '{new_name}' already exists.")

        data["providers"][new_name] = data["providers"].pop(old_name)
        self._manager._commit_config(data)

    def get_pack(self, name: str) -> ModelPack:
        """Return a single provider pack by name.

        Args:
            name: Name of the pack to retrieve.

        Returns:
            The :class:`~config.models.ModelPack` registered under ``name``.

        Raises:
            ConfigPackError: If no pack named ``name`` exists.
        """
        data = self._manager._dump_config()
        self._require_pack(data, name)
        return self._manager.config.providers[name]

    def list_packs(self) -> list[str]:
        """List the names of every configured provider pack.

        Returns:
            Pack names in insertion order.
        """
        return list(self._manager.config.providers.keys())

    # ------------------------------------------------------------------
    # Model management inside a pack
    # ------------------------------------------------------------------

    def add_model(
        self,
        pack: str,
        provider: str,
        model: str,
        *,
        target: PoolTarget = "pool",
    ) -> None:
        """Add a model entry to a provider pack.

        Args:
            pack: Name of the pack to modify.
            provider: Provider name for the new entry (e.g. ``"openai"``).
            model: Model identifier for the new entry (e.g. ``"gpt-4o"``).
            target: Which list to append to, ``"pool"`` (default) or
                ``"fallback"``.

        Raises:
            ConfigPackError: If ``pack`` does not exist.
            ConfigModelError: If ``target`` is invalid or the entry is
                already present in the target list.
        """
        data = self._manager._dump_config()
        self._require_pack(data, pack)
        self._require_target(target)

        entries: list[dict[str, str]] = data["providers"][pack][target]
        new_entry = {"provider": provider, "model": model}
        if new_entry in entries:
            raise ConfigModelError(
                f"Model '{provider}/{model}' is already present in "
                f"pack '{pack}' ({target})."
            )
        entries.append(new_entry)
        self._manager._commit_config(data)

    def remove_model(
        self,
        pack: str,
        provider: str,
        model: str,
        *,
        target: PoolTarget = "pool",
    ) -> None:
        """Remove a model entry from a provider pack.

        Args:
            pack: Name of the pack to modify.
            provider: Provider name of the entry to remove.
            model: Model identifier of the entry to remove.
            target: Which list to remove from, ``"pool"`` (default) or
                ``"fallback"``.

        Raises:
            ConfigPackError: If ``pack`` does not exist.
            ConfigModelError: If ``target`` is invalid or no matching
                entry is found.
        """
        data = self._manager._dump_config()
        self._require_pack(data, pack)
        self._require_target(target)

        entries: list[dict[str, str]] = data["providers"][pack][target]
        filtered = [
            entry
            for entry in entries
            if not (entry["provider"] == provider and entry["model"] == model)
        ]
        if len(filtered) == len(entries):
            raise ConfigModelError(
                f"Model '{provider}/{model}' was not found in pack '{pack}' ({target})."
            )
        data["providers"][pack][target] = filtered
        self._manager._commit_config(data)

    def clear_pool(self, pack: str) -> None:
        """Remove every model from a pack's primary pool.

        Args:
            pack: Name of the pack to clear.

        Raises:
            ConfigPackError: If ``pack`` does not exist.
        """
        self._clear_list(pack, "pool")

    def clear_fallback(self, pack: str) -> None:
        """Remove every model from a pack's fallback list.

        Args:
            pack: Name of the pack to clear.

        Raises:
            ConfigPackError: If ``pack`` does not exist.
        """
        self._clear_list(pack, "fallback")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _clear_list(self, pack: str, target: PoolTarget) -> None:
        """Empty the ``pool`` or ``fallback`` list of a given pack."""
        data = self._manager._dump_config()
        self._require_pack(data, pack)
        data["providers"][pack][target] = []
        self._manager._commit_config(data)

    @staticmethod
    def _require_pack(data: dict[str, Any], name: str) -> None:
        """Raise ConfigPackError if ``name`` is not a known provider pack."""
        if name not in data["providers"]:
            raise ConfigPackError(f"Provider pack '{name}' does not exist.")

    @staticmethod
    def _require_target(target: str) -> None:
        """Raise ConfigModelError if ``target`` is not 'pool' or 'fallback'."""
        if target not in ("pool", "fallback"):
            raise ConfigModelError(
                f"Invalid target '{target}': expected 'pool' or 'fallback'."
            )
