from tesseractcli.llm.routing import RoutingStep, RoutingConfig, RoutingTable


def test_routing_config_defaults():
    config = RoutingConfig(primary=RoutingStep(provider="groq", model="llama3"))
    assert config.fallbacks == []
    assert config.temperature == 0.3
    assert config.max_tokens == 4096


def test_routing_config_with_fallback_chain():
    config = RoutingConfig(
        primary=RoutingStep(provider="groq", model="llama3"),
        fallbacks=[
            RoutingStep(provider="cerebras", model="llama3.1-8b"),
            RoutingStep(provider="openai", model="gpt-4o-mini"),
        ],
    )
    assert len(config.fallbacks) == 2
    assert config.fallbacks[0].provider == "cerebras"
    assert config.fallbacks[1].model == "gpt-4o-mini"


def test_routing_table_resolves_task_specific_config():
    main = RoutingConfig(primary=RoutingStep(provider="groq", model="llama3"))
    task_config = RoutingConfig(
        primary=RoutingStep(provider="anthropic", model="claude-sonnet-5")
    )
    table = RoutingTable(main_model=main, default_routing={"code_generation": task_config})

    assert table.resolve("code_generation") is task_config


def test_routing_table_falls_back_to_main_model_for_unknown_task():
    main = RoutingConfig(primary=RoutingStep(provider="groq", model="llama3"))
    table = RoutingTable(main_model=main)

    assert table.resolve("nonexistent_task") is main
    assert table.resolve(None) is main
