"""
tesseractcli/llm/providers/hf_provider.py
Routed through ChatOpenAI + custom base_url, same pattern as Cerebras.
"""

from __future__ import annotations

from langchain_core.language_models import BaseChatModel
from langchain_core.rate_limiters import InMemoryRateLimiter
from langchain_openai import ChatOpenAI
from pydantic import SecretStr

from tesseractcli.llm.providers.base import BaseLLMProvider

HF_BASE_URL = "https://router.huggingface.co/v1"


class HuggingFaceProvider(BaseLLMProvider):
    DEFAULT_RATE_LIMIT_RPS: float | None = 2.0

    def _load_model(self, model_name: str, **kwargs) -> BaseChatModel:
        if not self.config.HUGGINGFACE_API_KEY:
            raise ValueError(
                "HUGGINGFACE_API_KEY is not set - add it to your .env file."
            )

        rate_limiter = None
        if self.DEFAULT_RATE_LIMIT_RPS:
            rate_limiter = InMemoryRateLimiter(
                requests_per_second=self.DEFAULT_RATE_LIMIT_RPS
            )

        return ChatOpenAI(
            model=model_name,
            api_key=SecretStr(self.config.HUGGINGFACE_API_KEY),
            base_url=HF_BASE_URL,
            rate_limiter=rate_limiter,
            **kwargs,
        )
