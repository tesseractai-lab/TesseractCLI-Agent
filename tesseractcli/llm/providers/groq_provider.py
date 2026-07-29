"""
tesseractcli/llm/providers/groq_provider.py
"""

from __future__ import annotations

from langchain_core.language_models import BaseChatModel
from langchain_core.rate_limiters import InMemoryRateLimiter
from langchain_groq import ChatGroq

from tesseractcli.llm.providers.base import BaseLLMProvider


class GroqProvider(BaseLLMProvider):
    # Groq's free tier is far tighter than Cerebras (~6k TPM on many
    # models) - starting point only, tune with real telemetry.
    DEFAULT_RATE_LIMIT_RPS: float | None = 0.5

    def _load_model(self, model_name: str, **kwargs) -> BaseChatModel:
        if not self.config.GROQ_API_KEY:
            raise ValueError("GROQ_API_KEY is not set - add it to your .env file.")

        rate_limiter = None
        if self.DEFAULT_RATE_LIMIT_RPS:
            rate_limiter = InMemoryRateLimiter(
                requests_per_second=self.DEFAULT_RATE_LIMIT_RPS
            )

        return ChatGroq(
            model=model_name,
            api_key=self.config.GROQ_API_KEY,
            rate_limiter=rate_limiter,
            **kwargs,
        )
