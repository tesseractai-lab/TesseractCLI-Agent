"""
tesseractcli/llm/providers/cerebras_provider.py
Cerebras is OpenAI-compatible, so we route through ChatOpenAI with a
custom base_url instead of a dedicated SDK.
"""

from __future__ import annotations

from langchain_core.language_models import BaseChatModel
from langchain_core.rate_limiters import InMemoryRateLimiter
from langchain_openai import ChatOpenAI
from pydantic import SecretStr

from tesseractcli.llm.providers.base import BaseLLMProvider

CEREBRAS_BASE_URL = "https://api.cerebras.ai/v1"


class CerebrasProvider(BaseLLMProvider):
    # Cerebras' free/dev tier is roughly ~30k TPM - generous relative to
    # Groq. This is a starting point, not a measured value; tune once
    # real usage/telemetry exists.
    DEFAULT_RATE_LIMIT_RPS: float | None = 2.0

    def _load_model(self, model_name: str, **kwargs) -> BaseChatModel:
        if not self.config.CEREBRAS_API_KEY:
            raise ValueError("CEREBRAS_API_KEY is not set - add it to your .env file.")

        rate_limiter = None
        if self.DEFAULT_RATE_LIMIT_RPS:
            rate_limiter = InMemoryRateLimiter(
                requests_per_second=self.DEFAULT_RATE_LIMIT_RPS
            )

        return ChatOpenAI(
            model=model_name,
            api_key=SecretStr(self.config.CEREBRAS_API_KEY),
            base_url=CEREBRAS_BASE_URL,
            rate_limiter=rate_limiter,
            **kwargs,
        )
