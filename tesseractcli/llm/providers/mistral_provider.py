"""
tesseractcli/llm/providers/mistral_provider.py
Routed through ChatOpenAI + custom base_url, same pattern as Cerebras,
per the agreed design (rather than the native langchain_mistralai
package) to keep the OpenAI-compatible providers consistent.
"""

from __future__ import annotations

from langchain_core.language_models import BaseChatModel
from langchain_core.rate_limiters import InMemoryRateLimiter
from pydantic import SecretStr
from langchain_openai import ChatOpenAI

from tesseractcli.llm.providers.base import BaseLLMProvider

MISTRAL_BASE_URL = "https://api.mistral.ai/v1"


class MistralProvider(BaseLLMProvider):
    DEFAULT_RATE_LIMIT_RPS: float | None = 1.0

    def _load_model(self, model_name: str, **kwargs) -> BaseChatModel:
        if not self.config.MISTRAL_API_KEY:
            raise ValueError("MISTRAL_API_KEY is not set - add it to your .env file.")

        rate_limiter = None
        if self.DEFAULT_RATE_LIMIT_RPS:
            rate_limiter = InMemoryRateLimiter(
                requests_per_second=self.DEFAULT_RATE_LIMIT_RPS
            )

        return ChatOpenAI(
            model=model_name,
            api_key=SecretStr(self.config.MISTRAL_API_KEY),
            base_url=MISTRAL_BASE_URL,
            rate_limiter=rate_limiter,
            **kwargs,
        )
