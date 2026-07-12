"""tesseractcli/config/logger.py"""
import sys
from loguru import logger

from .settings import get_settings, EnvFileMode, BASE_DIR

settings = get_settings()

logger.remove()

_is_dev = settings.ENV_MODE == EnvFileMode.DEVELOPMENT

# ===== Console sink (dev + prod) =====
logger.add(
    sys.stderr,
    level=settings.LOG_LEVEL,
    colorize=_is_dev,
    format=(
        "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
        "<level>{message}</level>"
    ),
    backtrace=_is_dev,
    diagnose=_is_dev,
)

# ===== File sink (dev + prod) =====
# لازم يبقى نسبي لـ BASE_DIR مش لـ cwd، وإلا ملفات اللوج هتتكتب فى مكان
# مختلف حسب مكان تشغيل الـ process (وده بالظبط اللي كان بيكسر التستات)
log_dir = BASE_DIR / settings.LOG_DIR
log_dir.mkdir(parents=True, exist_ok=True)

if _is_dev:
    logger.add(
        log_dir / "tesseract_dev_{time:YYYY-MM-DD}.log",
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
        log_dir / "tesseract_prod_{time:YYYY-MM-DD}.log",
        level=settings.LOG_LEVEL,
        rotation=settings.LOG_ROTATION,
        retention=settings.LOG_RETENTION,
        compression="zip",
        serialize=True,     # JSON structured logging
        backtrace=False,
        diagnose=False,     # مهم أمنيًا: يمنع تسريب قيم متغيرات فى ملف prod
        encoding="utf-8",
    )

__all__ = ["logger"]