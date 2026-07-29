"""tesseractcli/config/logger.py

Workspace-agnostic logger bootstrap. Owns exactly two things:

  - the console sink (stderr) - dev only. The app runs as a Textual
    TUI in production, and a raw log line written to stderr while the
    TUI owns the terminal corrupts the screen, so no console sink is
    ever added outside of dev.
  - a small "bootstrap" file sink that catches anything logged before
    a workspace has been picked (startup, `tesseract settings`, an
    error during workspace selection itself).

This module has no knowledge of workspaces. Per-workspace, per-session
file logging is layered on top of this by
`tesseractcli.logging.workspace.init_workspace_logging()`, which adds
its own sink alongside (not instead of) the bootstrap one below.
"""
import os
import sys
from pathlib import Path

from loguru import logger

from .settings import get_settings, EnvFileMode

settings = get_settings()

logger.remove()

IS_DEV = settings.ENV_MODE == EnvFileMode.DEVELOPMENT

# ===== Console sink (dev only) =====
# Never added in production - see module docstring.
if IS_DEV:
    logger.add(
        sys.stderr,
        level=settings.LOG_LEVEL,
        colorize=True,
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
            "<level>{message}</level>"
        ),
        backtrace=True,
        diagnose=True,
    )


def _resolve_logs_root() -> Path:
    """`~/.tesseract/logs` by default. Overridable via `TESSERACT_LOGS_DIR`
    (same override convention as `TESSERACT_BASE_DIR` in settings.py)
    so tests never write into the real home directory."""
    override = os.getenv("TESSERACT_LOGS_DIR")
    if override:
        return Path(override)
    project_root = Path(__file__).resolve().parent.parent.parent
    return project_root / "app_config" / "logs"


# Root of ALL log output, workspace-specific or not - re-exported so
# `tesseractcli/logging/workspace.py` builds per-workspace paths under
# the exact same root rather than re-deriving it.
LOGS_ROOT = _resolve_logs_root()

# ===== Bootstrap file sink (dev + prod) =====
# Catches everything logged before a workspace is bound. Left running
# for the lifetime of the process - init_workspace_logging() only ever
# *adds* a sink, it never removes this one.
_bootstrap_dir = LOGS_ROOT / "_bootstrap"
_bootstrap_dir.mkdir(parents=True, exist_ok=True)

if IS_DEV:
    logger.add(
        _bootstrap_dir / "tesseract_dev_{time:YYYY-MM-DD}.log",
        level=settings.LOG_LEVEL,
        rotation=settings.LOG_ROTATION,
        retention=settings.LOG_RETENTION,
        format=(
            "{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | "
            "{name}:{function}:{line} - {message}"
        ),
        serialize=False,   # نص عادي، مش JSON - أسهل فى القراءة وقت التطوير
        backtrace=True,
        diagnose=True,     # آمن هنا لأنه ملف محلي وقت التطوير بس
        encoding="utf-8",  # مهم على ويندوز، الـ default مش UTF-8 دايمًا
    )
else:
    logger.add(
        _bootstrap_dir / "tesseract_prod_{time:YYYY-MM-DD}.log",
        level=settings.LOG_LEVEL,
        rotation=settings.LOG_ROTATION,
        retention=settings.LOG_RETENTION,
        compression="zip",
        serialize=True,     # JSON structured logging
        backtrace=False,
        diagnose=False,     # مهم أمنيًا: يمنع تسريب قيم متغيرات فى ملف prod
        encoding="utf-8",
    )

__all__ = ["logger", "LOGS_ROOT", "IS_DEV"]
