"""
tesseractcli/llm/providers/openai_provider.py
Direct OpenAI (no custom base_url) - kept separate from the OpenAI-
compatible providers (Cerebras/Mistral/HF) even though it uses the same
ChatOpenAI class, so each has its own key/config/rate-limit lifecycle.
"""

from __future__ import annotations

from langchain_core.language_models import BaseChatModel
from langchain_core.rate_limiters import InMemoryRateLimiter
from langchain_openai import ChatOpenAI

from tesseractcli.llm.providers.base import BaseLLMProvider


class OpenAIProvider(BaseLLMProvider):
    DEFAULT_RATE_LIMIT_RPS: float | None = 1.0

    def _load_model(self, model_name: str, **kwargs) -> BaseChatModel:
        if not self.config.OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY is not set - add it to your .env file.")

        rate_limiter = None
        if self.DEFAULT_RATE_LIMIT_RPS:
            rate_limiter = InMemoryRateLimiter(
                requests_per_second=self.DEFAULT_RATE_LIMIT_RPS
            )

        return ChatOpenAI(
            model=model_name,
            api_key=self.config.OPENAI_API_KEY,
            rate_limiter=rate_limiter,
            **kwargs,
        )
