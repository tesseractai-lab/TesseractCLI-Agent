"""
tests/test_config/test_logger.py
"""

import json


def test_dev_log_file_is_plain_text(configured_env, workspace_root):
    config, logger_module = configured_env("dev")

    logger_module.logger.info("test dev message")

    log_files = list((workspace_root / "logs" / "_bootstrap").glob("tesseract_dev_*.log"))
    assert len(log_files) == 1

    content = log_files[0].read_text(encoding="utf-8")
    assert "test dev message" in content
    assert not content.strip().startswith("{")


def test_prod_log_file_is_json(configured_env, workspace_root):
    config, logger_module = configured_env("prod")

    logger_module.logger.info("test prod message")

    log_files = list((workspace_root / "logs" / "_bootstrap").glob("tesseract_prod_*.log"))
    assert len(log_files) == 1

    first_line = log_files[0].read_text(encoding="utf-8").strip().splitlines()[0]
    parsed = json.loads(first_line)
    assert parsed["record"]["message"] == "test prod message"


def test_prod_traceback_does_not_leak_local_variables(configured_env, workspace_root):
    """
    بديل سلوكي لفحص diagnose=False، بدل ما نلمس private attributes فى loguru
    (اللي بيتغيروا بين نسخ المكتبة). القيمة السرية دي متفروضش تظهر فى الملف
    لو diagnose فعلاً False.
    """
    config, logger_module = configured_env("prod")

    def _raise_with_secret():
        fake_secret_value = "sk-super-secret-token-123"  # noqa
        raise ValueError("boom")

    try:
        _raise_with_secret()
    except ValueError:
        logger_module.logger.exception("caught an error")

    log_files = list((workspace_root / "logs" / "_bootstrap").glob("tesseract_prod_*.log"))
    content = log_files[0].read_text(encoding="utf-8")
    assert "sk-super-secret-token-123" not in content
