"""
tesseractcli/llm/providers/cohere_provider.py
Cohere has a free "trial" API key tier - separate rate limits from paid
keys, but the same LangChain integration either way.
"""
from __future__ import annotations

from langchain_cohere import ChatCohere
from langchain_core.language_models import BaseChatModel
from langchain_core.rate_limiters import InMemoryRateLimiter

from tesseractcli.llm.providers.base import BaseLLMProvider


class CohereProvider(BaseLLMProvider):
    # Cohere's free trial keys are capped fairly low (~20 calls/min on
    # many endpoints) - conservative starting point, tune later.
    DEFAULT_RATE_LIMIT_RPS: float | None = 0.3

    def _load_model(self, model_name: str, **kwargs) -> BaseChatModel:
        if not self.config.COHERE_API_KEY:
            raise ValueError("COHERE_API_KEY is not set - add it to your .env file.")

        rate_limiter = None
        if self.DEFAULT_RATE_LIMIT_RPS:
            rate_limiter = InMemoryRateLimiter(
                requests_per_second=self.DEFAULT_RATE_LIMIT_RPS
            )

        return ChatCohere(
            model=model_name,
            cohere_api_key=self.config.COHERE_API_KEY,
            rate_limiter=rate_limiter,
            **kwargs,
        )
