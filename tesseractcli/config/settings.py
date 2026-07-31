"""
tesseractcli/config/settings.py
"""

import os
from enum import StrEnum
from functools import lru_cache
from pathlib import Path

from dotenv import dotenv_values
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


def _resolve_base_dir() -> Path:
    override = os.getenv("TESSERACT_BASE_DIR")
    if override:
        return Path(override)
    return Path(__file__).parents[2]


BASE_DIR = _resolve_base_dir()
MAIN_ENV = BASE_DIR / ".env"

_main_values = dotenv_values(MAIN_ENV)


class EnvFileMode(StrEnum):
    DEVELOPMENT = "dev"
    PRODUCTION = "prod"


def _resolve_env_mode() -> EnvFileMode:
    raw = (_main_values.get("ENV_MODE") or "dev").strip('"').lower()
    return (
        EnvFileMode.PRODUCTION
        if raw in ("prod", "production")
        else EnvFileMode.DEVELOPMENT
    )


ENV_MODE = _resolve_env_mode()
MODE_ENV = BASE_DIR / f".env.{ENV_MODE.value}"


class Settings(BaseSettings):
    ENV_MODE: EnvFileMode = Field(...)

    APP_NAME: str = Field(..., max_length=100)
    APP_VERSION: str = Field(...)

    DEBUG: bool = Field(...)

    # ===== Logging =====
    LOG_LEVEL: str = Field(...)
    # Superseded by the per-workspace layout under ~/.tesseract/logs
    # (see tesseractcli/logging/workspace.py, and LOGS_ROOT /
    # TESSERACT_LOGS_DIR in tesseractcli/config/logger.py). Kept here,
    # unused, only so an existing .env with LOG_DIR set still validates.
    LOG_DIR: str = Field(default="logs")
    LOG_ROTATION: str = Field(default="10 MB")
    LOG_RETENTION: str = Field(default="15 days")

    # ===== Tools =====
    MAX_LINES_WITHOUT_RANGE: int = Field(...)
    DEFAULT_TIMEOUT_SECONDS: int = Field(...)
    MAX_OUTPUT_CHARS: int = Field(...)

    # ===== Cloud provider API keys ====
    CEREBRAS_API_KEY: str | None = None
    GROQ_API_KEY: str | None = None
    MISTRAL_API_KEY: str | None = None
    HUGGINGFACE_API_KEY: str | None = None
    ANTHROPIC_API_KEY: str | None = None
    TOGETHER_API_KEY: str | None = None
    OPENAI_API_KEY: str | None = None
    COHERE_API_KEY: str | None = None
    OPENROUTER_API_KEY: str | None = None
    GITHUB_MODELS_TOKEN: str | None = None
    GOOGLE_API_KEY: str | None = None
    # --- Local GGUF provider ---
    LOCAL_GGUF_MODEL_PATH: str | None = None

    # ===== LangSmith tracing (optional) =====
    # Off by default - installing the `langsmith` package alone does
    # NOT start exporting traces; this flag must also be explicitly
    # set to true. See observability/tracing.py for how these get
    # bridged into the plain os.environ vars the langsmith SDK reads.
    LANGSMITH_TRACING: bool = False
    LANGSMITH_API_KEY: str | None = None
    LANGSMITH_PROJECT: str = "tesseractcli"
    LANGSMITH_ENDPOINT: str | None = None  # only needed for self-hosted LangSmith

    model_config = SettingsConfigDict(
        env_file=(MAIN_ENV, MODE_ENV),  # mode-specific override
        env_file_encoding="utf-8",
        extra="ignore",
    )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()  # type: ignore[call-arg]  # fields populated from .env at runtime
