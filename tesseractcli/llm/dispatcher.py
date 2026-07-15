"""
tesseractcli/llm/dispatcher.py
Built on the ideas mined from TesseractResearch's old dispatcher, layered
on top of the new BaseLLMProvider (which already solves the "rebuilding
the model on every loop iteration" problem via its internal cache).
"""
from __future__ import annotations

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage

from tesseractcli.config.logger import logger
from tesseractcli.llm.providers import(AnthropicProvider, BaseLLMProvider, CerebrasProvider,
                                    CohereProvider, GitHubModelsProvider, GroqProvider, HuggingFaceProvider,
                                    LocalGGUFProvider, MistralProvider, OpenAIProvider, OpenRouterProvider,
                                    TogetherProvider)
from tesseractcli.llm.routing import RoutingConfig, RoutingTable, get_routing_table

# Error-message substrings that indicate "the request was too big for
# this provider/model", as opposed to some other kind of failure. This
# is a heuristic (providers don't share a common exception type), and
# is deliberately kept in one place so it's easy to extend once we see
# real provider error strings in the wild.
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
    """Resolves a task name to a routing chain, and walks primary ->
    fallbacks, truncating and retrying a step once if the failure looks
    like a rate-limit/size problem rather than giving up on that step
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
    }

    def __init__(self, routing_table: RoutingTable | None = None) -> None:
        self.routing_table = routing_table or get_routing_table()
        self._providers: dict[str, BaseLLMProvider] = {}

    def _get_provider(self, name: str) -> BaseLLMProvider:
        if name not in self._PROVIDER_REGISTRY:
            raise ValueError(
                f"Unknown provider '{name}'. Known providers: "
                f"{sorted(self._PROVIDER_REGISTRY)}"
            )
        if name not in self._providers:
            self._providers[name] = self._PROVIDER_REGISTRY[name]()
        return self._providers[name]

    def _routing_for(self, task_name: str | None) -> RoutingConfig:
        return self.routing_table.resolve(task_name)

    def get_llm(self, task_name: str | None = None) -> BaseChatModel:
        """Primary-only getter, no fallback walk - for callers that want
        a plain model handle (e.g. for `.bind_tools()`) rather than the
        retry/fallback-aware `ainvoke_with_fallback`."""
        routing = self._routing_for(task_name)
        provider = self._get_provider(routing.primary.provider)
        logger.debug(
        "get_llm_with_tools → task={} provider={} model={} ",
        task_name, routing.primary.provider, routing.primary.model,
        # len(tools), [t.get("name") for t in tools],
    )
        return provider.get_model(
            routing.primary.model,
            temperature=routing.temperature,
            max_tokens=routing.max_tokens,
        )

    def get_llm_with_tools(self, tools: list[dict], task_name: str | None = None) -> BaseChatModel:
        """Like get_llm(), but returns a tool-bound model - bind_tools()
        happens before with_retry() inside the provider, since RunnableRetry
        doesn't forward bind_tools()."""
        routing = self._routing_for(task_name)
        provider = self._get_provider(routing.primary.provider)

        logger.debug(
            "get_llm_with_tools → task={} provider={} model={} tool_count={} tools={}",
            task_name,
            routing.primary.provider,
            routing.primary.model,
            len(tools),
            [t.get("name") for t in tools],
        )
        # Full schemas at TRACE-ish detail — descriptions are exactly
        # what the model reasons over when deciding whether to call a
        # tool, so log them too (not just the names).
        # for t in tools:
        #     logger.debug(
        #         "  tool_def name={} description={!r}",
        #         t.get("name"),
        #         (t.get("description") or "")[:200],
        #     )

        return provider.get_model_with_tools(
            routing.primary.model,
            tools,
            temperature=routing.temperature,
            max_tokens=routing.max_tokens,
        )

    async def ainvoke_with_fallback(
        self, messages: list[BaseMessage], task_name: str | None = None
    ) -> str:
        """Try primary, then each fallback in order. Within a single
        step: `.with_retry()` (already baked into every provider's
        `get_model`) absorbs transient failures. If the step still fails
        and it looks rate-limit/size-shaped, truncate the last message
        and retry that SAME step once before moving to the next
        fallback. Raises RuntimeError only if every step is exhausted.
        """
        routing = self._routing_for(task_name)
        steps = [routing.primary, *routing.fallbacks]
        last_error: Exception | None = None

        for step in steps:
            provider = self._get_provider(step.provider)
            model = provider.get_model_safe(
                step.model,
                temperature=routing.temperature,
                max_tokens=routing.max_tokens,
            )
            if model is None:
                logger.warning(
                    "skipping %s/%s - provider unavailable (missing key/config)",
                    step.provider,
                    step.model,
                )
                continue

            try:
                result = await model.ainvoke(messages)
                return result.content
            except Exception as e:  # noqa: BLE001 - intentional: any failure -> try next
                if not self._looks_like_rate_limit_error(e):
                    last_error = e
                    logger.error(
                        "%s/%s failed: %s", step.provider, step.model, e
                    )
                    continue

                logger.warning(
                    "%s/%s hit a rate-limit/size-shaped error, truncating "
                    "and retrying this step once",
                    step.provider,
                    step.model,
                )
                truncated = self._truncate_messages(messages)
                try:
                    result = await model.ainvoke(truncated)
                    return result.content
                except Exception as e2:  # noqa: BLE001
                    last_error = e2
                    logger.error(
                        "%s/%s failed again after truncation: %s",
                        step.provider,
                        step.model,
                        e2,
                    )
                    continue

        raise RuntimeError(
            f"All routing steps exhausted for task '{task_name}'"
        ) from last_error

    @staticmethod
    def _looks_like_rate_limit_error(error: Exception) -> bool:
        text = str(error).lower()
        return any(marker in text for marker in _RATE_LIMIT_SIZE_MARKERS)

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
