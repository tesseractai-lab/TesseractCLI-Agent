"""
tests/test_config/test_settings.py

اختبارات Settings: التأكد إن القيم بتتفرق صح بين dev/prod،
إن الـ validation شغالة، وإن lru_cache بيتصرف زي ما متوقع.
"""

import pytest
from pydantic import ValidationError


def test_dev_mode_resolves_correct_values(configured_env):
    config, _ = configured_env("dev")
    settings = config.get_settings()

    assert settings.ENV_MODE.value == "dev"
    assert settings.DEBUG is True
    assert settings.LOG_LEVEL == "DEBUG"


def test_prod_mode_resolves_correct_values(configured_env):
    config, _ = configured_env("prod")
    settings = config.get_settings()

    assert settings.ENV_MODE.value == "prod"
    assert settings.DEBUG is False
    assert settings.LOG_LEVEL == "INFO"


def test_missing_required_field_raises(configured_env, workspace_root):
    config, _ = configured_env("dev")

    # نمسح حقول required من المين .env وكمان من .env.dev عشان نتأكد
    # إن الفاليديشن شغالة فعلاً من كل المصادر مش بس من واحد فيهم
    (workspace_root / ".env").write_text(
        'APP_NAME="X"\n'
    )  # ENV_MODE/APP_VERSION/... ناقصين
    (workspace_root / ".env.dev").write_text("")  # LOG_LEVEL كمان بقى ناقص

    config.get_settings.cache_clear()
    with pytest.raises(ValidationError):
        config.Settings()


def test_get_settings_is_cached(configured_env):
    config, _ = configured_env("dev")
    s1 = config.get_settings()
    s2 = config.get_settings()
    assert s1 is s2  # نفس الـ instance، من غير إعادة بناء


def test_cache_clear_forces_new_instance(configured_env):
    config, _ = configured_env("dev")
    s1 = config.get_settings()
    config.get_settings.cache_clear()
    s2 = config.get_settings()
    assert s1 is not s2
