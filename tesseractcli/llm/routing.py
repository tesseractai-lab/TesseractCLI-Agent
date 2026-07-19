"""
tesseractcli/llm/routing.py
Domain logic for "how do we pick a model for a given pack" - owned
entirely by the LLM/agent layer. Deliberately independent from
Settings: Settings is flat, env-backed config; routing here just
resolves a pack name against the structured config already owned by
`config.ConfigManager` (providers/packs/models loaded from
global_config.yaml). The TUI settings screen and the dispatcher both
*consume* RoutingResolver; neither owns the pack data itself -
ConfigManager does.

This replaces the old RoutingStep/RoutingConfig/RoutingTable classes:
those duplicated a schema (provider + model, primary + fallbacks) that
now lives in config.models (ModelConfig, ModelPack). Resolving a task
now means resolving a *pack name* straight to a ModelPack, instead of
looking up a hand-maintained in-memory task table.
"""
from __future__ import annotations

from functools import lru_cache

from tesseractcli.models.exceptions import ConfigPackError
from tesseractcli.models.config_models.provider_models import ModelConfig, ModelPack
from tesseractcli.config.global_config.manager import ConfigManager

# Used when the requested pack doesn't exist (e.g. an unconfigured
# task name, or a typo) - "main" is guaranteed to exist because it
# ships in default_config.yaml.
DEFAULT_PACK_NAME = "main"


class RoutingResolver:
    """Resolves a pack name to its ModelPack (models + provider info).

    Construct directly (e.g. in tests, with an isolated ConfigManager),
    or use get_routing_resolver() for the process-wide cached instance.
    """

    def __init__(
        self, manager: ConfigManager, *, default_pack: str = DEFAULT_PACK_NAME
    ) -> None:
        self._manager = manager
        self._default_pack = default_pack

    def resolve(self, pack_name: str | None) -> ModelPack:
        """Return the ModelPack for `pack_name`.

        Falls back to the default pack ("main") when `pack_name` is
        None or refers to a pack that doesn't exist in the config.
        """
        name = pack_name or self._default_pack
        try:
            return self._manager.packs.get_pack(name)
        except ConfigPackError:
            return self._manager.packs.get_pack(self._default_pack)

    def resolve_primary(self, pack_name: str | None) -> ModelConfig:
        """Return just the first (primary) provider/model in the pack's pool.

        Convenience for callers that only need one provider/model pair
        rather than the full pool + fallback list (e.g. a quick manual
        `--model` style lookup). Raises ConfigPackError if the resolved
        pack's pool is empty.
        """
        pack = self.resolve(pack_name)
        if not pack.pool:
            raise ConfigPackError(
                f"Pack '{pack_name or self._default_pack}' has an empty pool."
            )
        return pack.pool[0]


@lru_cache(maxsize=1)
def get_routing_resolver() -> RoutingResolver:
    """Process-wide cached resolver, backed by the on-disk global config."""
    manager = ConfigManager(config_dir="~/.tesseractcli")
    manager.load()
    return RoutingResolver(manager)
