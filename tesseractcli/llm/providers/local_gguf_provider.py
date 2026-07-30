"""
tesseractcli/llm/providers/local_gguf_provider.py
No API key involved - this provider loads a local .gguf model file via
llama-cpp-python. `llama-cpp-python` is an optional dependency (it needs
a C++ build toolchain), so the import happens lazily inside
_load_model rather than at module level - importing this module should
never fail just because the optional package isn't installed on a
machine that only uses cloud providers.

`model_name` here is interpreted as a path to a .gguf file. If not
given, falls back to `Settings.LOCAL_GGUF_MODEL_PATH`.
"""

from __future__ import annotations

from pathlib import Path

from langchain_core.language_models import BaseChatModel

from tesseractcli.llm.providers.base import BaseLLMProvider


class LocalGGUFProvider(BaseLLMProvider):
    # No cloud rate limit applies to a local process.
    DEFAULT_RATE_LIMIT_RPS: float | None = None

    def _load_model(self, model_name: str, **kwargs) -> BaseChatModel:
        model_path = model_name or self.config.LOCAL_GGUF_MODEL_PATH
        if not model_path:
            raise ValueError(
                "No GGUF model path given - pass model_name or set "
                "LOCAL_GGUF_MODEL_PATH in your .env file."
            )
        if not Path(model_path).exists():
            raise FileNotFoundError(f"GGUF model file not found: {model_path}")

        try:
            from langchain_community.chat_models import ChatLlamaCpp  # type: ignore
        except ImportError as e:
            raise ImportError(
                "llama-cpp-python is required for the local GGUF provider. "
                "Install it with `uv add llama-cpp-python`."
            ) from e

        return ChatLlamaCpp(model_path=model_path, **kwargs)
