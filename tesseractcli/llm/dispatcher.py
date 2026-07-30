"""
tesseractcli/llm/dispatcher.py

`LLMDispatcher` is the single entry point `agent/loop.py` (and anything
else that needs a model call) goes through. It does NOT own pack/model
data itself - that's `RoutingResolver` (see llm/routing.py), which reads
straight from the shared `ConfigManager`. This module only knows how to:

1. Turn a resolved `ModelPack` into an ordered list of candidate
   `ModelConfig` entries to try (pool, shuffled per call so repeated
   calls spread load rather than always hammering pool[0] first, then
   fallback in its declared, fixed order) - or, if the caller pinned an
   exact (provider, model) pair, just that one entry.
2. Get a ready-to-call model `Runnable` for a candidate via the
   corresponding provider's `get_model_safe` / `get_model_with_tools_safe`
   (these already cache the underlying client and wrap it with
   `.with_retry()` - see llm/providers/base.py).
3. Invoke it, and on a rate-limit/size error, truncate the last
   message's content and retry the SAME candidate once before moving on
   to the next one in the list.
4. Stamp which provider/model/pack/temperature/max_tokens actually
   answered onto the returned `AIMessage.response_metadata`, so
   `memory/store.py`'s `_model_meta_row` can record it.

`agent.max_iterations` / `agent.lazy_tool_loading` are exposed as
properties that read straight off `resolver.manager.config` every
access (never cached), so a `set agent.max_iterations ...` from the
settings screen takes effect on the very next turn - same "never cache
it" principle `RoutingResolver.resolve()` already applies to packs.
"""

from __future__ import annotations

import random
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.runnables import Runnable

from tesseractcli.config.logger import logger
from tesseractcli.llm.providers import (
    AnthropicProvider,
    CerebrasProvider,
    CohereProvider,
    GitHubModelsProvider,
    GoogleProvider,
    GroqProvider,
    HuggingFaceProvider,
    LocalGGUFProvider,
    MistralProvider,
    OpenAIProvider,
    OpenRouterProvider,
    TogetherProvider,
)
from tesseractcli.llm.providers.base import BaseLLMProvider
from tesseractcli.llm.routing import RoutingResolver, get_routing_resolver
from tesseractcli.models.config_models.provider_models import ModelConfig, ModelPack

# Substring markers used to detect a rate-limit or oversized-request
# error from a provider's raised exception text. There's no single
# shared exception type across every LangChain provider integration
# used here, so this stays a heuristic string match rather than a
# `except SomeRateLimitError` catch.
_RATE_LIMIT_OR_SIZE_MARKERS: tuple[str, ...] = (
    "rate limit",
    "rate_limit",
    "429",
    "too many tokens",
    "context_length_exceeded",
    "maximum context length",
    "request too large",
    "payload too large",
)

# Cap applied to the last message's content on a truncate-and-retry.
# Only the last message is touched - earlier history is left as-is.
DEFAULT_TRUNCATE_MAX_CHARS = 4000

# Pack name in global_config.yaml pack entries -> provider class.
# "google" isn't in config/provider_catalog.py's suggestion list (it's
# not a default-shipped provider), but GoogleProvider exists and is
# routed the same OpenAI-compatible way as the others, so it's still
# registered here for anyone who adds it to a pack manually.
_PROVIDER_REGISTRY: dict[str, type[BaseLLMProvider]] = {
    "anthropic": AnthropicProvider,
    "cerebras": CerebrasProvider,
    "cohere": CohereProvider,
    "github_models": GitHubModelsProvider,
    "google": GoogleProvider,
    "groq": GroqProvider,
    "huggingface": HuggingFaceProvider,
    "local_gguf": LocalGGUFProvider,
    "mistral": MistralProvider,
    "openai": OpenAIProvider,
    "openrouter": OpenRouterProvider,
    "together": TogetherProvider,
}


def _is_rate_limit_or_size_error(exc: Exception) -> bool:
    text = str(exc).lower()
    return any(marker in text for marker in _RATE_LIMIT_OR_SIZE_MARKERS)


def _truncate_last_message(
    messages: list[BaseMessage], max_chars: int
) -> list[BaseMessage]:
    """Returns a new list with only the last message's content
    truncated (when it's a plain string over `max_chars`), so a retry
    after a rate-limit/size error has a real chance of fitting -
    earlier history is left untouched."""
    if not messages:
        return messages
    patched = list(messages)
    last = patched[-1]
    content = last.content
    if isinstance(content, str) and len(content) > max_chars:
        patched[-1] = last.model_copy(update={"content": content[:max_chars]})
    return patched


class LLMDispatcher:
    """Resolves a routing pack name to a pool+fallback chain of models
    and drives a single model call across it, with rate-limit/size-aware
    truncate-and-retry.

    Construct with the SAME `RoutingResolver` (and therefore the same
    `ConfigManager`) the rest of the app uses - e.g.
    `LLMDispatcher(RoutingResolver(self.config_manager))` in
    `ui/app.py`, NOT `LLMDispatcher()` with no args, which would fall
    back to `get_routing_resolver()`'s own separate, second
    `ConfigManager` instance and silently diverge from the shared one.
    """

    def __init__(
        self,
        resolver: RoutingResolver | None = None,
        *,
        truncate_max_chars: int = DEFAULT_TRUNCATE_MAX_CHARS,
    ) -> None:
        self._resolver = resolver or get_routing_resolver()
        self._truncate_max_chars = truncate_max_chars
        # One provider instance per name, reused across calls so each
        # provider's own model cache (get_model_safe) actually caches
        # instead of rebuilding a LangChain client every call.
        self._providers: dict[str, BaseLLMProvider] = {}

    # ------------------------------------------------------------------ #
    # Live config
    # ------------------------------------------------------------------ #

    @property
    def resolver(self) -> RoutingResolver:
        return self._resolver

    @property
    def max_iterations(self) -> int:
        """Live from `agent.max_iterations` (see agent/loop.py)."""
        return self._resolver.manager.config.agent.max_iterations

    @property
    def lazy_tool_loading(self) -> bool:
        """Live from `agent.lazy_tool_loading` (see agent/loop.py)."""
        return self._resolver.manager.config.agent.lazy_tool_loading

    # ------------------------------------------------------------------ #
    # Provider access
    # ------------------------------------------------------------------ #

    def _provider_for(self, name: str) -> BaseLLMProvider:
        if name not in self._providers:
            try:
                provider_cls = _PROVIDER_REGISTRY[name]
            except KeyError:
                raise ValueError(f"Unknown provider '{name}'.") from None
            self._providers[name] = provider_cls()
        return self._providers[name]

    # ------------------------------------------------------------------ #
    # Candidate resolution
    # ------------------------------------------------------------------ #

    def _candidates(
        self, pack_name: str | None, pinned: tuple[str, str] | None
    ) -> list[ModelConfig]:
        """The ordered list of (provider, model) entries to try.

        `pinned`, if given, short-circuits to exactly that one entry
        (still resolved against the pack via `resolve_step`, so an
        unknown pin fails loudly with `ConfigModelError` instead of
        silently constructing an ad-hoc entry). Otherwise: the pack's
        pool, shuffled (spreads load across the whole pool instead of
        always trying pool[0] first), followed by the fallback list in
        its declared, fixed order - fallbacks are a deliberate priority
        order and are never shuffled.
        """
        if pinned is not None:
            provider, model = pinned
            return [self._resolver.resolve_step(pack_name, provider, model)]

        pack = self._resolver.resolve(pack_name)
        pool = list(pack.pool)
        random.shuffle(pool)
        return pool + list(pack.fallback)

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def get_llm(self, pack_name: str | None = None, **kwargs) -> Runnable | None:
        """Primary-only lookup (the pack's first pool entry) - for
        callers that just want a bindable model (e.g. to call
        `.bind_tools()` themselves) rather than the full fallback-aware
        `ainvoke_with_fallback` path."""
        pack = self._resolver.resolve(pack_name)
        primary = self._resolver.resolve_primary(pack_name)
        provider = self._provider_for(primary.provider)
        return provider.get_model_safe(
            primary.model, max_tokens=pack.max_tokens, temperature=pack.temperature, **kwargs
        )

    def get_llm_with_tools(
        self, tools: list[dict], pack_name: str | None = None, **kwargs
    ) -> Runnable | None:
        """Same as `get_llm`, but with `tools` bound - primary entry only."""
        pack = self._resolver.resolve(pack_name)
        primary = self._resolver.resolve_primary(pack_name)
        provider = self._provider_for(primary.provider)
        return provider.get_model_with_tools_safe(
            primary.model,
            tools,
            max_tokens=pack.max_tokens,
            temperature=pack.temperature,
            **kwargs,
        )

    async def ainvoke_with_fallback(
        self,
        messages: list[BaseMessage],
        *,
        tools: list[dict] | None = None,
        pack_name: str | None = None,
        pinned: tuple[str, str] | None = None,
    ) -> AIMessage:
        """Walks the resolved candidate list (pinned entry, or pool
        then fallback) in order, invoking each through its provider.
        On a rate-limit/size error, truncates the last message and
        retries the SAME candidate once before moving to the next one.
        Raises the last error seen if every candidate fails outright
        (or every provider failed to even construct a client, e.g. a
        missing API key - already logged inside get_model*_safe).
        """
        pack = self._resolver.resolve(pack_name)
        candidates = self._candidates(pack_name, pinned)
        if not candidates:
            raise RuntimeError(
                f"Pack '{pack_name or self._resolver.manager.config}' has no models configured."
            )

        last_error: Exception | None = None
        for candidate in candidates:
            provider = self._provider_for(candidate.provider)
            model = (
                provider.get_model_with_tools_safe(
                    candidate.model,
                    tools,
                    max_tokens=pack.max_tokens,
                    temperature=pack.temperature,
                )
                if tools
                else provider.get_model_safe(
                    candidate.model,
                    max_tokens=pack.max_tokens,
                    temperature=pack.temperature,
                )
            )
            if model is None:
                # Provider failed to construct the client at all (e.g.
                # missing API key) - already logged inside get_model*_safe.
                continue

            current_messages = messages
            for attempt in range(2):  # first try, then one truncate-and-retry
                try:
                    result = await model.ainvoke(current_messages)
                    return self._stamp_metadata(result, candidate, pack, pack_name)
                except Exception as exc:  # noqa: BLE001 - fallback chain by design
                    last_error = exc
                    if attempt == 0 and _is_rate_limit_or_size_error(exc):
                        logger.warning(
                            "rate-limit/size error on {}/{}, truncating last "
                            "message and retrying once",
                            candidate.provider,
                            candidate.model,
                        )
                        current_messages = _truncate_last_message(
                            current_messages, self._truncate_max_chars
                        )
                        continue
                    logger.error(
                        "model call failed on {}/{}: {}",
                        candidate.provider,
                        candidate.model,
                        exc,
                    )
                    break  # move to the next candidate

        assert last_error is not None
        raise last_error

    @staticmethod
    def _stamp_metadata(
        result: Any,
        candidate: ModelConfig,
        pack: ModelPack,
        pack_name: str | None,
    ) -> AIMessage:
        """Stamps which provider/model/pack/temperature/max_tokens
        actually answered onto the AIMessage's `response_metadata`, so
        `memory/store.py`'s `_model_meta_row` can record it
        (`tesseract_provider`, `tesseract_model`, `tesseract_pack_name`,
        `tesseract_temperature`, `tesseract_max_tokens`)."""
        if not isinstance(result, AIMessage):
            result = AIMessage(content=str(getattr(result, "content", result)))
        meta = dict(result.response_metadata or {})
        meta.setdefault("tesseract_provider", candidate.provider)
        meta.setdefault("tesseract_model", candidate.model)
        meta.setdefault("tesseract_pack_name", pack_name)
        meta.setdefault("tesseract_temperature", pack.temperature)
        meta.setdefault("tesseract_max_tokens", pack.max_tokens)
        return result.model_copy(update={"response_metadata": meta})
