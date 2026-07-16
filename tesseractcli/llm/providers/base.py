"""
tesseractcli/llm/providers/base.py
Abstract base class for all LLM providers (LangChain-based).
"""
from __future__ import annotations

from abc import ABC, abstractmethod

from langchain_core.language_models import BaseChatModel

from tesseractcli.config.logger import logger
from tesseractcli.config.settings import get_settings


class BaseLLMProvider(ABC):
    """Every provider (Cerebras, Groq, Mistral, HF, Anthropic, Together,
    OpenAI, Local GGUF, ...) inherits from here."""

    DEFAULT_MAX_RETRIES: int = 3
    DEFAULT_RATE_LIMIT_RPS: float | None = None

    def __init__(self) -> None:
        self.config = get_settings()
        self._cache: dict[str, BaseChatModel] = {}
        self._raw_cache: dict[str, BaseChatModel] = {}   # unwrapped models, keyed same as _cache

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def get_model_safe(self, model_name: str, **kwargs) -> BaseChatModel | None:
        """Same as get_model, but swallows errors so a dispatcher can
        fall back to the next provider in the chain instead of crashing."""
        try:
            model = self._get_model(model_name, **kwargs)
            logger.debug("model loaded: {}", model_name)
            return model
        except Exception as e:  # noqa: BLE001 - intentional catch-all for fallback
            logger.error("failed to load '{}': {}", model_name, e)
            return None


    def get_model_with_tools_safe(
        self, model_name: str, tools: list[dict], **kwargs
    ) -> BaseChatModel | None:
        """Same as get_model_with_tools, but swallows errors so a dispatcher
        can fall back to the next provider in the chain instead of crashing."""
        try:
            model = self._get_model_with_tools(model_name, tools, **kwargs)
            logger.debug("model with tools loaded: {}", model_name)
            return model
        except Exception as e:  # noqa: BLE001 - intentional catch-all for fallback
            logger.error("failed to load '{}' with tools: {}", model_name, e)
            return None

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def _cache_key(model_name: str, **kwargs) -> str:
        """Single source of truth for cache-key shape, shared by _cache
        and _raw_cache so both stay in sync if the shape ever changes."""
        return f"{model_name}:{sorted(kwargs.items())}"

    def _get_raw_model(self, model_name: str, **kwargs) -> BaseChatModel:
        """Loads (and caches) the model with NO wrapping - no retry, no
        tools bound. The one shared source both get_model() and
        get_model_with_tools() build on top of, so the underlying model
        is only ever constructed once regardless of which wrapper is needed."""
        cache_key = self._cache_key(model_name, **kwargs)
        if cache_key not in self._raw_cache:
            self._raw_cache[cache_key] = self._load_model(model_name, **kwargs)
        return self._raw_cache[cache_key]

    def _get_model(self, model_name: str, **kwargs) -> BaseChatModel:
        """Public factory. Caches the instance and wraps it with retry."""
        cache_key = self._cache_key(model_name, **kwargs)
        if cache_key not in self._cache:
            raw_model = self._get_raw_model(model_name, **kwargs)
            self._cache[cache_key] = raw_model.with_retry(stop_after_attempt=self.DEFAULT_MAX_RETRIES)
        return self._cache[cache_key]

    def _get_model_with_tools(self, model_name: str, tools: list[dict], **kwargs) -> BaseChatModel:
        """Same underlying model as get_model(), but bind_tools() happens
        BEFORE with_retry() - RunnableRetry doesn't proxy bind_tools, so
        this order is not optional. Not cached across calls (bind_tools
        output isn't hashable the way raw models are), but the expensive
        part - _load_model - IS cached via _get_raw_model."""
        raw_model = self._get_raw_model(model_name, **kwargs)
        bound = raw_model.bind_tools(tools)
        return bound.with_retry(stop_after_attempt=self.DEFAULT_MAX_RETRIES)

    # ------------------------------------------------------------------ #
    # Provider contract
    # ------------------------------------------------------------------ #

    @abstractmethod
    def _load_model(self, model_name: str, **kwargs) -> BaseChatModel:
        """Each provider builds its raw ChatModel instance here."""
        ...
