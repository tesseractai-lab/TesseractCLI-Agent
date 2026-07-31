"""
tesseractcli/llm/providers/google_provider.py
Routed through ChatOpenAI + custom base_url, same pattern as HuggingFace/Cerebras.
"""

from __future__ import annotations

from langchain_core.language_models import BaseChatModel
from langchain_core.rate_limiters import InMemoryRateLimiter
from langchain_openai import ChatOpenAI
from pydantic import SecretStr

from tesseractcli.llm.providers.base import BaseLLMProvider

GOOGLE_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"


class GoogleProvider(BaseLLMProvider):
    DEFAULT_RATE_LIMIT_RPS: float | None = (
        0.15  # ~10 RPM (Flash) to stay safe across models
    )

    def _load_model(self, model_name: str, **kwargs) -> BaseChatModel:
        if not self.config.GOOGLE_API_KEY:
            raise ValueError("GOOGLE_API_KEY is not set - add it to your .env file.")

        rate_limiter = None
        if self.DEFAULT_RATE_LIMIT_RPS:
            rate_limiter = InMemoryRateLimiter(
                requests_per_second=self.DEFAULT_RATE_LIMIT_RPS
            )

        return ChatOpenAI(
            model=model_name,
            api_key=SecretStr(self.config.GOOGLE_API_KEY),
            base_url=GOOGLE_BASE_URL,
            rate_limiter=rate_limiter,
            **kwargs,
        )
