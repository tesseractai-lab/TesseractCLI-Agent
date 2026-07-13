"""
tesseractcli/llm/providers/together_provider.py
"""
from __future__ import annotations

from langchain_core.language_models import BaseChatModel
from langchain_core.rate_limiters import InMemoryRateLimiter
from langchain_together import ChatTogether

from tesseractcli.llm.providers.base import BaseLLMProvider


class TogetherProvider(BaseLLMProvider):
    DEFAULT_RATE_LIMIT_RPS: float | None = 1.5

    def _load_model(self, model_name: str, **kwargs) -> BaseChatModel:
        if not self.config.TOGETHER_API_KEY:
            raise ValueError("TOGETHER_API_KEY is not set - add it to your .env file.")

        rate_limiter = None
        if self.DEFAULT_RATE_LIMIT_RPS:
            rate_limiter = InMemoryRateLimiter(
                requests_per_second=self.DEFAULT_RATE_LIMIT_RPS
            )

        return ChatTogether(
            model=model_name,
            api_key=self.config.TOGETHER_API_KEY,
            rate_limiter=rate_limiter,
            **kwargs,
        )
