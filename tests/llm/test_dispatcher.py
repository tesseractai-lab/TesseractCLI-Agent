from unittest.mock import AsyncMock, MagicMock

import pytest
from langchain_core.messages import HumanMessage

from tesseractcli.models.config_models.provider_models import ModelConfig, ModelPack
from tesseractcli.llm.dispatcher import LLMDispatcher


def _pack(
    provider: str,
    model: str,
    *,
    fallback: list[tuple[str, str]] | None = None,
    temperature: float = 0.3,
    max_tokens: int = 4096,
) -> ModelPack:
    """Build a single-entry-pool ModelPack, optionally with a fallback
    chain - the shape LLMDispatcher expects from RoutingResolver.resolve()."""
    return ModelPack(
        pool=[ModelConfig(provider=provider, model=model)],
        fallback=[ModelConfig(provider=p, model=m) for p, m in (fallback or [])],
        temperature=temperature,
        max_tokens=max_tokens,
    )


class _FakeResolver:
    """Stand-in for RoutingResolver: resolves a named pack, or falls
    back to `main_pack` for anything unrecognized - same contract as
    the real RoutingResolver.resolve()."""

    def __init__(self, main_pack: ModelPack, packs: dict[str, ModelPack] | None = None):
        self._main = main_pack
        self._packs = packs or {}

    def resolve(self, pack_name: str | None) -> ModelPack:
        if pack_name and pack_name in self._packs:
            return self._packs[pack_name]
        return self._main


def _fake_chat_model(content: str = "ok", side_effect=None) -> MagicMock:
    model = MagicMock()
    if side_effect is not None:
        model.ainvoke = AsyncMock(side_effect=side_effect)
    else:
        result = MagicMock()
        result.content = content
        model.ainvoke = AsyncMock(return_value=result)
    return model


class TestRoutingResolution:
    def test_get_llm_uses_main_pack_when_no_pack_given(self, mocker):
        main_pack = _pack("groq", "llama3")
        dispatcher = LLMDispatcher(resolver=_FakeResolver(main_pack))

        fake_provider = MagicMock()
        fake_provider.get_model_safe.return_value = "the-model"
        mocker.patch.object(dispatcher, "_get_provider", return_value=fake_provider)

        result = dispatcher.get_llm()

        fake_provider.get_model_safe.assert_called_once_with(
            "llama3",
            temperature=0.3,
            max_tokens=4096,
        )
        assert result == "the-model"

    def test_get_llm_uses_pack_specific_routing_when_present(self, mocker):
        main_pack = _pack("groq", "llama3")
        task_pack = _pack("anthropic", "claude-sonnet-5", temperature=0.1)

        dispatcher = LLMDispatcher(
            resolver=_FakeResolver(main_pack, packs={"code_generation": task_pack})
        )

        fake_provider = MagicMock()
        fake_provider.get_model_safe.return_value = MagicMock()
        mocker.patch.object(dispatcher, "_get_provider", return_value=fake_provider)

        dispatcher.get_llm(pack_name="code_generation")

        fake_provider.get_model_safe.assert_called_once_with(
            "claude-sonnet-5",
            temperature=0.1,
            max_tokens=4096,
        )

    def test_unknown_pack_name_falls_back_to_main_pack(self, mocker):
        main_pack = _pack("groq", "llama3")
        dispatcher = LLMDispatcher(resolver=_FakeResolver(main_pack))

        fake_provider = MagicMock()
        fake_provider.get_model_safe.return_value = MagicMock()
        mocker.patch.object(dispatcher, "_get_provider", return_value=fake_provider)

        dispatcher.get_llm(pack_name="some_pack_nobody_configured")

        fake_provider.get_model_safe.assert_called_once_with(
            "llama3",
            temperature=0.3,
            max_tokens=4096,
        )

    def test_unregistered_provider_name_raises_value_error(self):
        pack = _pack("not_a_real_provider", "x")
        dispatcher = LLMDispatcher(resolver=_FakeResolver(pack))

        with pytest.raises(ValueError, match="not_a_real_provider"):
            dispatcher.get_llm()

    def test_get_llm_raises_on_empty_pool(self):
        empty_pack = ModelPack(pool=[], fallback=[])
        dispatcher = LLMDispatcher(resolver=_FakeResolver(empty_pack))

        with pytest.raises(ValueError, match="empty pool"):
            dispatcher.get_llm()


@pytest.mark.asyncio
class TestFallbackChain:
    async def test_primary_success_returns_content_directly(self, mocker):
        main_pack = _pack("groq", "llama3")
        dispatcher = LLMDispatcher(
            resolver=_FakeResolver(main_pack), max_context_messages=50
        )

        fake_provider = MagicMock()
        fake_provider.get_model_safe.return_value = _fake_chat_model(content="hi")
        mocker.patch.object(dispatcher, "_get_provider", return_value=fake_provider)

        result = await dispatcher.ainvoke_with_fallback([HumanMessage(content="hey")])

        assert result.content == "hi"

    async def test_falls_back_when_primary_provider_unavailable(self, mocker):
        main_pack = _pack("groq", "llama3", fallback=[("cerebras", "llama3.1-8b")])
        dispatcher = LLMDispatcher(
            resolver=_FakeResolver(main_pack), max_context_messages=50
        )

        groq_provider = MagicMock()
        groq_provider.get_model_safe.return_value = None

        cerebras_provider = MagicMock()
        cerebras_provider.get_model_safe.return_value = _fake_chat_model(
            content="from cerebras"
        )

        def fake_get_provider(name):
            return {
                "groq": groq_provider,
                "cerebras": cerebras_provider,
            }[name]

        mocker.patch.object(dispatcher, "_get_provider", side_effect=fake_get_provider)

        result = await dispatcher.ainvoke_with_fallback([HumanMessage(content="hey")])

        assert result.content == "from cerebras"

    async def test_falls_back_on_non_rate_limit_exception(self, mocker):
        main_pack = _pack("groq", "llama3", fallback=[("cerebras", "llama3.1-8b")])
        dispatcher = LLMDispatcher(
            resolver=_FakeResolver(main_pack), max_context_messages=50
        )

        groq_provider = MagicMock()
        groq_provider.get_model_safe.return_value = _fake_chat_model(
            side_effect=ConnectionError("network blip")
        )

        cerebras_provider = MagicMock()
        cerebras_provider.get_model_safe.return_value = _fake_chat_model(
            content="from cerebras"
        )

        def fake_get_provider(name):
            return {
                "groq": groq_provider,
                "cerebras": cerebras_provider,
            }[name]

        mocker.patch.object(dispatcher, "_get_provider", side_effect=fake_get_provider)

        result = await dispatcher.ainvoke_with_fallback([HumanMessage(content="hey")])

        assert result.content == "from cerebras"
        assert groq_provider.get_model_safe.return_value.ainvoke.call_count == 1

    async def test_truncates_and_retries_same_step_on_rate_limit_error(self, mocker):
        main_pack = _pack("groq", "llama3")
        dispatcher = LLMDispatcher(
            resolver=_FakeResolver(main_pack), max_context_messages=50
        )

        model = MagicMock()

        success_result = MagicMock()
        success_result.content = "truncated success"

        model.ainvoke = AsyncMock(
            side_effect=[
                Exception("429 rate limit exceeded"),
                success_result,
            ]
        )

        fake_provider = MagicMock()
        fake_provider.get_model_safe.return_value = model
        mocker.patch.object(dispatcher, "_get_provider", return_value=fake_provider)

        long_message = HumanMessage(content="x" * 5000)

        result = await dispatcher.ainvoke_with_fallback([long_message])

        assert result.content == "truncated success"
        assert model.ainvoke.call_count == 2

        second_call_messages = model.ainvoke.call_args_list[1].args[0]
        assert len(second_call_messages[-1].content) < 5000

    async def test_all_steps_exhausted_raises_runtime_error(self, mocker):
        main_pack = _pack("groq", "llama3", fallback=[("cerebras", "llama3.1-8b")])
        dispatcher = LLMDispatcher(
            resolver=_FakeResolver(main_pack), max_context_messages=50
        )

        groq_provider = MagicMock()
        groq_provider.get_model_safe.return_value = None

        cerebras_provider = MagicMock()
        cerebras_provider.get_model_safe.return_value = None

        def fake_get_provider(name):
            return {
                "groq": groq_provider,
                "cerebras": cerebras_provider,
            }[name]

        mocker.patch.object(dispatcher, "_get_provider", side_effect=fake_get_provider)

        with pytest.raises(RuntimeError, match="exhausted"):
            await dispatcher.ainvoke_with_fallback([HumanMessage(content="hey")])


class TestMaxContextMessages:
    def test_explicit_override_is_used_without_touching_resolver(self):
        dispatcher = LLMDispatcher(
            resolver=_FakeResolver(_pack("groq", "llama3")),
            max_context_messages=7,
        )

        assert dispatcher.max_context_messages == 7

    def test_reads_live_from_resolver_manager_when_not_overridden(self):
        resolver = MagicMock()
        resolver.manager.config.agent.max_context_messages = 40
        dispatcher = LLMDispatcher(resolver=resolver)

        assert dispatcher.max_context_messages == 40

        resolver.manager.config.agent.max_context_messages = 5
        assert dispatcher.max_context_messages == 5


class TestTruncateMessages:
    def test_short_message_untouched(self):
        messages = [HumanMessage(content="short")]
        result = LLMDispatcher._truncate_messages(messages)
        assert result[0].content == "short"

    def test_long_message_truncated(self):
        messages = [HumanMessage(content="x" * 5000)]
        result = LLMDispatcher._truncate_messages(messages, max_chars=100)
        assert len(result[0].content) < 5000
        assert "truncated" in result[0].content

    def test_only_last_message_truncated(self):
        messages = [
            HumanMessage(content="x" * 5000),
            HumanMessage(content="y" * 5000),
        ]
        result = LLMDispatcher._truncate_messages(messages, max_chars=100)

        assert result[0].content == "x" * 5000
        assert len(result[1].content) < 5000

    def test_empty_messages_returns_empty(self):
        assert LLMDispatcher._truncate_messages([]) == []
