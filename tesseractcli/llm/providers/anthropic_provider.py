"""
tesseractcli/llm/providers/anthropic_provider.py
"""

from __future__ import annotations

from langchain_anthropic import ChatAnthropic
from langchain_core.language_models import BaseChatModel
from langchain_core.rate_limiters import InMemoryRateLimiter
from pydantic import SecretStr

from tesseractcli.llm.providers.base import BaseLLMProvider


class AnthropicProvider(BaseLLMProvider):
    DEFAULT_RATE_LIMIT_RPS: float | None = 1.0

    def _load_model(self, model_name: str, **kwargs) -> BaseChatModel:
        if not self.config.ANTHROPIC_API_KEY:
            raise ValueError("ANTHROPIC_API_KEY is not set - add it to your .env file.")

        rate_limiter = None
        if self.DEFAULT_RATE_LIMIT_RPS:
            rate_limiter = InMemoryRateLimiter(
                requests_per_second=self.DEFAULT_RATE_LIMIT_RPS
            )

        return ChatAnthropic(
            model_name=model_name,
            api_key=SecretStr(self.config.ANTHROPIC_API_KEY),
            rate_limiter=rate_limiter,
            **kwargs,
        )
