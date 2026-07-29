"""
tesseractcli/llm/providers/openrouter_provider.py
OpenRouter is OpenAI-compatible - same pattern as Cerebras/Mistral/HF.
"""

from __future__ import annotations

from langchain_core.language_models import BaseChatModel
from langchain_core.rate_limiters import InMemoryRateLimiter
from langchain_openai import ChatOpenAI

from tesseractcli.llm.providers.base import BaseLLMProvider

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# https://openrouter.ai/api/v1/chat/completions


class OpenRouterProvider(BaseLLMProvider):
    DEFAULT_RATE_LIMIT_RPS: float | None = 1.0

    def _load_model(self, model_name: str, **kwargs) -> BaseChatModel:
        if not self.config.OPENROUTER_API_KEY:
            raise ValueError(
                "OPENROUTER_API_KEY is not set - add it to your .env file."
            )

        rate_limiter = None
        if self.DEFAULT_RATE_LIMIT_RPS:
            rate_limiter = InMemoryRateLimiter(
                requests_per_second=self.DEFAULT_RATE_LIMIT_RPS
            )

        return ChatOpenAI(
            model=model_name,
            api_key=self.config.OPENROUTER_API_KEY,
            base_url=OPENROUTER_BASE_URL,
            rate_limiter=rate_limiter,
            **kwargs,
        )
