import pytest

from tesseractcli.models.exceptions import  ConfigPackError
from tesseractcli.config.global_config.manager import ConfigManager
from tesseractcli.llm.routing import DEFAULT_PACK_NAME, RoutingResolver


@pytest.fixture
def manager(tmp_path):
    """A real ConfigManager backed by a throwaway directory, seeded from
    the packaged default_config.yaml (packs: main, secondary)."""
    m = ConfigManager(config_dir=tmp_path)
    m.load()
    return m


class TestRoutingResolverResolve:
    def test_resolve_returns_main_pack_by_default(self, manager):
        resolver = RoutingResolver(manager)

        pack = resolver.resolve(None)

        assert pack.pool[0].provider == "anthropic"
        assert pack.pool[0].model == "claude-sonnet-4-6"

    def test_resolve_returns_named_pack_when_present(self, manager):
        resolver = RoutingResolver(manager)

        pack = resolver.resolve("secondary")

        assert pack.pool[0].provider == "openai"
        assert pack.pool[0].model == "gpt-4o-mini"

    def test_resolve_falls_back_to_main_for_unknown_pack(self, manager):
        resolver = RoutingResolver(manager)

        main_pack = resolver.resolve(DEFAULT_PACK_NAME)
        fallback_pack = resolver.resolve("some_pack_nobody_configured")

        assert fallback_pack == main_pack

    def test_resolve_respects_custom_default_pack(self, manager):
        resolver = RoutingResolver(manager, default_pack="secondary")

        pack = resolver.resolve("some_pack_nobody_configured")

        assert pack.pool[0].provider == "openai"
        assert pack.pool[0].model == "gpt-4o-mini"

    def test_resolve_sees_packs_added_after_construction(self, manager):
        resolver = RoutingResolver(manager)
        manager.packs.add_pack("vision")
        manager.packs.add_model("vision", "openai", "gpt-4o")

        pack = resolver.resolve("vision")

        assert pack.pool[0].provider == "openai"
        assert pack.pool[0].model == "gpt-4o"


class TestRoutingResolverResolvePrimary:
    def test_resolve_primary_returns_first_pool_entry(self, manager):
        resolver = RoutingResolver(manager)

        primary = resolver.resolve_primary("secondary")

        assert primary.provider == "openai"
        assert primary.model == "gpt-4o-mini"

    def test_resolve_primary_falls_back_to_main_for_unknown_pack(self, manager):
        resolver = RoutingResolver(manager)

        primary = resolver.resolve_primary("some_pack_nobody_configured")

        assert primary.provider == "anthropic"
        assert primary.model == "claude-sonnet-4-6"

    def test_resolve_primary_raises_on_empty_pool(self, manager):
        manager.packs.add_pack("empty_pack")
        resolver = RoutingResolver(manager)

        with pytest.raises(ConfigPackError):
            resolver.resolve_primary("empty_pack")
