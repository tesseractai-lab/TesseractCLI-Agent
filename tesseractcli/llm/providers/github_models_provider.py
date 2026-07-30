"""
tesseractcli/llm/providers/github_models_provider.py
GitHub Models exposes an OpenAI-compatible inference endpoint, auth'd
with a GitHub personal access token (not the same as a repo-scoped
GITHUB_TOKEN some CI setups use, hence the distinct settings field name).
"""

from __future__ import annotations

from langchain_core.language_models import BaseChatModel
from langchain_core.rate_limiters import InMemoryRateLimiter
from pydantic import SecretStr
from langchain_openai import ChatOpenAI

from tesseractcli.llm.providers.base import BaseLLMProvider

GITHUB_MODELS_BASE_URL = "https://models.github.ai/inference"


class GitHubModelsProvider(BaseLLMProvider):
    # Free tier rate limits vary by model/org plan - conservative default.
    DEFAULT_RATE_LIMIT_RPS: float | None = 0.3

    def _load_model(self, model_name: str, **kwargs) -> BaseChatModel:
        if not self.config.GITHUB_MODELS_TOKEN:
            raise ValueError(
                "GITHUB_MODELS_TOKEN is not set - add it to your .env file."
            )

        rate_limiter = None
        if self.DEFAULT_RATE_LIMIT_RPS:
            rate_limiter = InMemoryRateLimiter(
                requests_per_second=self.DEFAULT_RATE_LIMIT_RPS
            )

        return ChatOpenAI(
            model=model_name,
            api_key=SecretStr(self.config.GITHUB_MODELS_TOKEN),
            base_url=GITHUB_MODELS_BASE_URL,
            rate_limiter=rate_limiter,
            **kwargs,
        )
