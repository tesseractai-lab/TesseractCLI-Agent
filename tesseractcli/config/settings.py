"""
tesseractcli/config/settings.py
"""
import os
from enum import StrEnum
from dotenv import dotenv_values
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field
from pathlib import Path
from functools import lru_cache


def _resolve_base_dir() -> Path:
    """
    بيسمح بعمل override لـ BASE_DIR عن طريق env var - مهم جدًا للتستات،
    عشان importlib.reload() يقدر يعيد قراءتها من مكان مؤقت بدل المشروع الحقيقي.
    """
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
    raw = _main_values.get("ENV_MODE", "dev").strip('"').lower()
    return EnvFileMode.PRODUCTION if raw in ("prod", "production") else EnvFileMode.DEVELOPMENT


ENV_MODE = _resolve_env_mode()
MODE_ENV = BASE_DIR / f".env.{ENV_MODE.value}"


class Settings(BaseSettings):

    ENV_MODE: EnvFileMode = Field(...)

    APP_NAME: str = Field(..., max_length=100)
    APP_VERSION: str = Field(...)

    DEBUG: bool = Field(...)

    # ===== Logging =====
    LOG_LEVEL: str = Field(...)
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
    OPENAI_API_KEY: str | None = None
    COHERE_API_KEY: str | None = None
    OPENROUTER_API_KEY: str | None = None
    GITHUB_MODELS_TOKEN: str | None = None
    # --- Local GGUF provider ---
    LOCAL_GGUF_MODEL_PATH: str | None = None
    
    model_config = SettingsConfigDict(
    env_file=(MAIN_ENV, MODE_ENV),  # mode-specific override
    env_file_encoding="utf-8",
    extra="ignore",)
@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()