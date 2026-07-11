"""
tests/conftest.py

Shared fixtures لكل الـ tool tests. الفكرة الأساسية: كل test لازم ياخد
workspace_root نظيف ومعزول (isolated) عشان تجربة معينة متأثرش في تجربة تانية.
tmp_path هو fixture جاهز من pytest بيدّي مجلد مؤقت فريد لكل test function.
"""

import pytest
from pathlib import Path


@pytest.fixture
def workspace_root(tmp_path: Path) -> Path:
    """مجلد workspace فاضي ومعزول لكل test على حدة."""
    return tmp_path
