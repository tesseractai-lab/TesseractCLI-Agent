"""
tests/llm/providers/conftest.py

Shared fixtures for every provider test file.
"""

from __future__ import annotations

import pytest

from tesseractcli.config.settings import Settings

# Every env var a provider might read - cleared before each test so tests
# never leak into each other or pick up a real key from the host env.
ALL_PROVIDER_ENV_VARS = [
    "CEREBRAS_API_KEY",
    "GROQ_API_KEY",
    "MISTRAL_API_KEY",
    "HUGGINGFACE_API_KEY",
    "ANTHROPIC_API_KEY",
    "TOGETHER_API_KEY",
    "OPENAI_API_KEY",
    "GITHUB_MODELS_TOKEN",
    "COHERE_API_KEY",
    "OPENROUTER_API_KEY",
    "LOCAL_GGUF_MODEL_PATH",
]


@pytest.fixture(autouse=True)
def clean_provider_env(monkeypatch):
    """Wipe every provider-related env var before each test."""
    for var in ALL_PROVIDER_ENV_VARS:
        monkeypatch.delenv(var, raising=False)
    yield


@pytest.fixture
def make_settings(monkeypatch):
    """
    Factory that returns an isolated Settings instance.

    Usage:
        settings = make_settings()
        settings = make_settings(GROQ_API_KEY="xxx")
        settings = make_settings(OPENAI_API_KEY="yyy")
    """

    def _make(**env_vars: str) -> Settings:
        # Inject only the env vars this test needs.
        for key, value in env_vars.items():
            monkeypatch.setenv(key, value)

        # Build Settings without reading .env files.
        return Settings(
            _env_file=None,
            ENV_MODE="dev",
            APP_NAME="TesseractCLI",
            APP_VERSION="0.0.1",
            DEBUG=False,
            LOG_LEVEL="INFO",
            LOG_DIR="logs",
            LOG_ROTATION="10 MB",
            LOG_RETENTION="15 days",
            MAX_LINES_WITHOUT_RANGE=2000,
            DEFAULT_TIMEOUT_SECONDS=30,
            MAX_OUTPUT_CHARS=10000,
        )

    return _make
