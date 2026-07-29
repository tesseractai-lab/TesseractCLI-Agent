"""
tesseractcli/llm/dispatcher.py
Built on the ideas mined from TesseractResearch's old dispatcher, layered
on top of the new BaseLLMProvider (which already solves the "rebuilding
the model on every loop iteration" problem via its internal cache).
"""

from __future__ import annotations

import random

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage, ToolMessage

from tesseractcli.models.config_models.provider_models import ModelConfig, ModelPack
from tesseractcli.config.logger import logger
from tesseractcli.llm.providers import (
    AnthropicProvider,
    BaseLLMProvider,
    CerebrasProvider,
    CohereProvider,
    GitHubModelsProvider,
    GroqProvider,
    HuggingFaceProvider,
    LocalGGUFProvider,
    MistralProvider,
    OpenAIProvider,
    OpenRouterProvider,
    TogetherProvider,
    GoogleProvider,
)
from tesseractcli.llm.routing import RoutingResolver, get_routing_resolver

_RATE_LIMIT_SIZE_MARKERS = (
    "rate limit",
    "rate_limit",
    "429",
    "too many tokens",
    "context_length_exceeded",
    "maximum context length",
    "request too large",
    "payload too large",
)


class LLMDispatcher:
    """Resolves a pack name to a ModelPack, and walks pool -> fallback,
    truncating and retrying a step once if the failure looks like a
    rate-limit/size problem rather than giving up on that step
    outright."""

    _PROVIDER_REGISTRY: dict[str, type[BaseLLMProvider]] = {
        "cerebras": CerebrasProvider,
        "groq": GroqProvider,
        "mistral": MistralProvider,
        "huggingface": HuggingFaceProvider,
        "anthropic": AnthropicProvider,
        "together": TogetherProvider,
        "openai": OpenAIProvider,
        "cohere": CohereProvider,
        "openrouter": OpenRouterProvider,
        "github_models": GitHubModelsProvider,
        "local_gguf": LocalGGUFProvider,
        "google": GoogleProvider,
    }

    def __init__(
        self,
        resolver: RoutingResolver | None = None,
        max_context_messages: int | None = None,
    ) -> None:
        self.resolver = resolver or get_routing_resolver()
        self._providers: dict[str, BaseLLMProvider] = {}
        # None (the default) means "read `agent.max_context_messages`
        # live off the resolver's ConfigManager every call" - so
        # `set agent.max_context_messages <n>` in the TUI settings
        # screen takes effect immediately, no restart needed. Pass an
        # explicit int to pin it instead (e.g. in tests).
        self._max_context_messages_override = max_context_messages

    @property
    def max_context_messages(self) -> int:
        if self._max_context_messages_override is not None:
            return self._max_context_messages_override
        return self.resolver.manager.config.agent.max_context_messages

    @property
    def max_iterations(self) -> int:
        """Live off the same shared ConfigManager as max_context_messages -
        `agent.max_iterations` (default 10, see config_models/agent_models.py).
        Reading it here (rather than agent/loop.py constructing its own
        ConfigManager) is what makes `settings --set agent.max_iterations n`
        (or the TUI settings screen) take effect on the very next turn,
        with no restart - the same "never cache it separately" principle
        already applied to max_context_messages above."""
        return self.resolver.manager.get("agent.max_iterations", default=10)

    @property
    def lazy_tool_loading(self) -> bool:
        """Live off the same shared ConfigManager - `agent.lazy_tool_loading`
        (default False, see config_models/agent_models.py). agent/loop.py
        reads this once per turn to decide whether `core=False` tools
        start deferred (True) or are just sent every request like
        before (False) - so flipping it via `set agent.lazy_tool_loading
        true` (or a config reload) changes behavior on the very next
        turn, no restart needed, same as max_iterations above."""
        return bool(self.resolver.manager.get("agent.lazy_tool_loading", default=False))

    def _get_provider(self, name: str) -> BaseLLMProvider:
        if name not in self._PROVIDER_REGISTRY:
            raise ValueError(
                f"Unknown provider '{name}'. Known providers: "
                f"{sorted(self._PROVIDER_REGISTRY)}"
            )
        if name not in self._providers:
            self._providers[name] = self._PROVIDER_REGISTRY[name]()
        return self._providers[name]

    def _pack_for(self, pack_name: str | None) -> ModelPack:
        return self.resolver.resolve(pack_name)

    def get_llm(
        self, pack_name: str | None = None, pinned: tuple[str, str] | None = None
    ) -> BaseChatModel:
        """Primary-only getter, no fallback walk - for callers that want
        a plain model handle (e.g. for `.bind_tools()`) rather than the
        retry/fallback-aware `ainvoke_with_fallback`.

        `pinned`, if given, is a (provider, model) pair that must already
        exist somewhere in the pack's pool/fallback - use that exact
        entry instead of the pack's first pool entry."""
        pack = self._pack_for(pack_name)
        primary = (
            self.resolver.resolve_step(pack_name, *pinned)
            if pinned is not None
            else self._primary_of(pack, pack_name)
        )
        provider = self._get_provider(primary.provider)
        logger.debug(
            "get_llm → pack={} provider={} model={}",
            pack_name,
            primary.provider,
            primary.model,
        )
        return provider.get_model_safe(
            primary.model,
            temperature=pack.temperature,
            max_tokens=pack.max_tokens,
        )

    def get_llm_with_tools(
        self,
        tools: list[dict],
        pack_name: str | None = None,
        pinned: tuple[str, str] | None = None,
    ) -> BaseChatModel:
        """Like get_llm(), but returns a tool-bound model - bind_tools()
        happens before with_retry() inside the provider, since RunnableRetry
        doesn't forward bind_tools(). NOTE: this has no fallback/rate-limit
        handling of its own - use ainvoke_with_fallback(..., tools=...) for
        the full resilience path. This stays around for callers that only
        need a raw bound model handle (e.g. streaming) without invoking it
        through the dispatcher.

        `pinned`, if given, is a (provider, model) pair that must already
        exist somewhere in the pack's pool/fallback - use that exact
        entry instead of the pack's first pool entry."""
        pack = self._pack_for(pack_name)
        primary = (
            self.resolver.resolve_step(pack_name, *pinned)
            if pinned is not None
            else self._primary_of(pack, pack_name)
        )
        provider = self._get_provider(primary.provider)

        logger.debug(
            "get_llm_with_tools → pack={} provider={} model={} tool_count={} tools={}",
            pack_name,
            primary.provider,
            primary.model,
            len(tools),
            [t.get("name") for t in tools],
        )

        return provider.get_model_with_tools_safe(
            primary.model,
            tools,
            temperature=pack.temperature,
            max_tokens=pack.max_tokens,
        )

    async def ainvoke_with_fallback(
        self,
        messages: list[BaseMessage],
        tools: list[dict] | None = None,
        pack_name: str | None = None,
        pinned: tuple[str, str] | None = None,
    ) -> BaseMessage:
        """Try the pack's pool in a random order every call - NOT always
        starting from pool[0] - so repeated calls spread load/usage
        across every configured model rather than hammering one entry
        and only ever reaching the others when it happens to fail.
        Fallback is a second group, touched ONLY once every single pool
        entry has been tried and failed this call - at that point it's
        treated exactly like its own pool (same random order), not as a
        fixed last-resort sequence.

        Within a single step: `.with_retry()` (already baked into every
        provider's get_model/get_model_with_tools) absorbs transient
        failures. If the step still fails and it looks rate-limit/size-
        shaped, truncate the last message and retry that SAME step once
        before moving to a different step in the same group. Raises
        RuntimeError only if pool and fallback are both fully exhausted.

        Pass `tools` to get a tool-bound model at every step (pool AND
        fallback) - this is the single entry point for tool-calling +
        fallback + rate-limit resilience combined. Returns the full
        BaseMessage (not just .content), since tool calls live on
        `.tool_calls`, not in `.content`.

        `pinned`, if given, is a (provider, model) pair the caller
        explicitly wants used for this call, bypassing pool/fallback
        selection entirely - resolved via
        `RoutingResolver.resolve_step`. It still gets the same rate-
        limit-triggered truncate-and-retry-once treatment as any other
        step, it just never moves on to a different model if that retry
        also fails. Raises ConfigModelError up front if `pinned` doesn't
        match any pool/fallback entry in the resolved pack.
        """
        pack = self._pack_for(pack_name)
        if pinned is not None:
            groups = [[self.resolver.resolve_step(pack_name, *pinned)]]
        else:
            # Two groups, each shuffled independently every call - this is
            # what makes model selection an actual round-robin/random pick
            # across the pool (not always starting from pool[0] in fixed
            # order): fallback is only ever touched once every single
            # pool entry has been tried and failed this call, at which
            # point it's treated exactly like its own pool (same random
            # approach), not as a fixed-order last resort.
            groups = [list(pack.pool), list(pack.fallback)]
        last_error: Exception | None = None
        messages = self._window_messages(messages, self.max_context_messages)

        for group in groups:
            if not group:
                continue
            order = list(group)
            random.shuffle(order)

            for step in order:
                provider = self._get_provider(step.provider)

                if tools:
                    model = provider.get_model_with_tools_safe(
                        step.model,
                        tools,
                        temperature=pack.temperature,
                        max_tokens=pack.max_tokens,
                    )
                else:
                    model = provider.get_model_safe(
                        step.model,
                        temperature=pack.temperature,
                        max_tokens=pack.max_tokens,
                    )

                if model is None:
                    logger.warning(
                        "skipping {}/{} - provider unavailable (missing key/config)",
                        step.provider,
                        step.model,
                    )
                    continue

                try:
                    result = await model.ainvoke(messages)
                    return self._stamp_attribution(result, step, pack_name, pack)
                except Exception as e:  # noqa: BLE001 - intentional: any failure -> try next
                    if not self._looks_like_rate_limit_error(e):
                        last_error = e
                        logger.error("{}/{} failed: {}", step.provider, step.model, e)
                        continue

                    logger.warning(
                        "{}/{} hit a rate-limit/size-shaped error, truncating "
                        "and retrying this step once",
                        step.provider,
                        step.model,
                    )
                    truncated = self._truncate_messages(messages)
                    try:
                        result = await model.ainvoke(truncated)
                        return self._stamp_attribution(result, step, pack_name, pack)
                    except Exception as e2:  # noqa: BLE001
                        last_error = e2
                        logger.error(
                            "{}/{} failed again after truncation: {}",
                            step.provider,
                            step.model,
                            e2,
                        )
                        continue

        raise RuntimeError(
            f"All routing steps exhausted for pack '{pack_name}'"
        ) from last_error

    @staticmethod
    def _stamp_attribution(
        message: BaseMessage, step: ModelConfig, pack_name: str | None, pack: ModelPack
    ) -> BaseMessage:
        """Records which provider/model/pack/hyperparameters actually
        produced this reply under `response_metadata` - raw provider
        `response_metadata` shapes aren't consistent across all 11
        providers, so this is the one normalized source
        `memory/store.py` reads from to populate the `model_meta` table
        (model_name, pack_name, temperature, max_tokens). Token counts
        (input/output) are NOT stamped here - they already live on
        `message.usage_metadata` as a standard LangChain attribute when
        a provider reports them, so `memory/store.py` reads that
        directly instead of duplicating it into `response_metadata`."""
        message.response_metadata = {
            **(message.response_metadata or {}),
            "tesseract_provider": step.provider,
            "tesseract_model": step.model,
            "tesseract_pack_name": pack_name,
            "tesseract_temperature": pack.temperature,
            "tesseract_max_tokens": pack.max_tokens,
        }
        return message

    @staticmethod
    def _primary_of(pack: ModelPack, pack_name: str | None) -> ModelConfig:
        """First model in the pack's pool - used by the single-model
        getters (get_llm / get_llm_with_tools), which don't walk fallback."""
        if not pack.pool:
            raise ValueError(f"Pack '{pack_name}' has an empty pool.")
        return pack.pool[0]

    @staticmethod
    def _looks_like_rate_limit_error(error: Exception) -> bool:
        text = str(error).lower()
        return any(marker in text for marker in _RATE_LIMIT_SIZE_MARKERS)

    @staticmethod
    def _window_messages(
        messages: list[BaseMessage], max_messages: int
    ) -> list[BaseMessage]:
        """Keeps only the most recent `max_messages` messages before
        sending to the model - a stopgap for unbounded context/token
        growth until real summarization/compaction exists. Nothing is
        deleted from the caller's list or from `conversation.db`; this
        only shrinks what actually gets sent on THIS call.

        Never starts the window on a ToolMessage: cutting between an
        AIMessage's tool_calls and its ToolMessage response would send
        an orphaned tool result with no matching call, which every
        provider rejects as an invalid request.
        """
        if len(messages) <= max_messages:
            return messages

        start = len(messages) - max_messages
        while start < len(messages) and isinstance(messages[start], ToolMessage):
            start += 1
        return messages[start:]

    @staticmethod
    def _truncate_messages(
        messages: list[BaseMessage], max_chars: int = 4000
    ) -> list[BaseMessage]:
        """Truncates only the LAST message's content (usually the newest
        user turn / tool result - the most likely oversized part),
        leaving earlier conversation history untouched."""
        if not messages:
            return messages

        truncated = list(messages)
        last = truncated[-1]
        content = last.content if isinstance(last.content, str) else str(last.content)

        if len(content) <= max_chars:
            return truncated

        new_content = content[:max_chars] + "\n...[truncated by dispatcher]"
        truncated[-1] = last.model_copy(update={"content": new_content})
        return truncated
