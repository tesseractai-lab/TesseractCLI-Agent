"""
tests/conftest.py

Shared fixtures لكل الـ tool tests. الفكرة الأساسية: كل test لازم ياخد
workspace_root نظيف ومعزول (isolated) عشان تجربة معينة متأثرش في تجربة تانية.
tmp_path هو fixture جاهز من pytest بيدّي مجلد مؤقت فريد لكل test function.
"""

import importlib
import pytest
from pathlib import Path


@pytest.fixture
def workspace_root(tmp_path: Path) -> Path:
    """مجلد workspace فاضي ومعزول لكل test على حدة."""
    return tmp_path


@pytest.fixture
def configured_env(workspace_root, monkeypatch):
    """
    بيبني بيئة .env كاملة جوه workspace_root، ويرجع دالة بتاخد المود المطلوب
    ("dev" أو "prod") وتظبط عليه config + logger.

    ملاحظة مهمة: بنستخدم monkeypatch.setenv("TESSERACT_BASE_DIR", ...) مش
    monkeypatch.setattr(config, "BASE_DIR", ...) - لأن importlib.reload()
    بيعيد تنفيذ كود الموديول من الأول، فأي setattr هيتمسح، لكن env var
    فى os.environ بتفضل موجودة وبيتقرا منها تاني وقت الـ reload.
    """

    def _write_env_files(mode: str):
        (workspace_root / ".env").write_text(
            f'ENV_MODE = "{mode}"\n'
            'APP_NAME = "TessersctClI"\n'
            'APP_VERSION = "1.0.0"\n'
            "LOG_DIR=logs\n"
            'LOG_ROTATION = "10 MB"\n'
            'LOG_RETENTION = "15 days"\n'
            "MAX_LINES_WITHOUT_RANGE=2000\n"
            "DEFAULT_TIMEOUT_SECONDS=30\n"
            "MAX_OUTPUT_CHARS=10000\n"
        )
        (workspace_root / ".env.dev").write_text("DEBUG=True\nLOG_LEVEL=DEBUG\n")
        (workspace_root / ".env.prod").write_text("DEBUG=False\nLOG_LEVEL=INFO\n")

    def _setup(mode: str):
        _write_env_files(mode)

        import tesseractcli.config.settings as config
        import tesseractcli.config.logger as logger_module

        monkeypatch.setenv("TESSERACT_BASE_DIR", str(workspace_root))
        # logger.py resolves its own root independently of BASE_DIR/LOG_DIR
        # (see LOGS_ROOT / TESSERACT_LOGS_DIR in tesseractcli/config/logger.py) -
        # without this, tests would write real log files into the actual
        # project's app_config/logs/ directory instead of the isolated
        # workspace_root.
        monkeypatch.setenv("TESSERACT_LOGS_DIR", str(workspace_root / "logs"))

        # لازم نشيل الكاش القديم قبل الـ reload، وإلا get_settings() هترجع
        # instance قديمة اتبنت بقيم مختلفة من قبل
        config.get_settings.cache_clear()
        importlib.reload(config)
        importlib.reload(logger_module)

        return config, logger_module

    return _setup
