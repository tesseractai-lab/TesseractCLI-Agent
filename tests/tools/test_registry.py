"""
tests/tools/test_registry.py

بيختبر الـ ToolRegistry نفسها (registry.py) بمعزل تام عن أي tool حقيقي —
عن طريق بناء fake tool وهمي بسيط جدًا، عشان نتأكد إن الـ registry بتعمل
شغلها الصح (add / dispatch / validation / error handling) من غير ما نعتمد
على تفاصيل tool حقيقي زي write_file أو edit_file.
"""

import pytest
from pathlib import Path
from pydantic import BaseModel

from tesseractcli.tools.registry import ToolRegistry
from tesseractcli.models.tool_models import ToolResult


class _FakeArgs(BaseModel):
    """Args model وهمي بس لغرض الاختبار، بيحاكي شكل أي tool حقيقي."""
    value: int


def _fake_tool(workspace_root: Path, value: int) -> ToolResult:
    """Tool وهمي: بيرجع الـ value اللي استلمها مضروبة في 2."""
    return ToolResult(tool_name="fake_tool", success=True, output=str(value * 2))


def test_add_then_dispatch_calls_correct_function(workspace_root: Path) -> None:
    registry = ToolRegistry()
    registry.add(name="fake_tool", schema=_FakeArgs, fn=_fake_tool)

    result = registry.dispatch("fake_tool", {"value": 5}, workspace_root)

    assert result.success is True
    assert result.output == "10"


def test_dispatch_unknown_tool_name_returns_failed_result_not_exception(
    workspace_root: Path,
) -> None:
    registry = ToolRegistry()
    registry.add(name="fake_tool", schema=_FakeArgs, fn=_fake_tool)

    # الاسم ده مش مسجل خالص — المفروض ترجع ToolResult فاشلة، مش ترمي exception
    result = registry.dispatch("does_not_exist", {"value": 5}, workspace_root)

    assert result.success is False
    assert result.error is not None
    assert "does_not_exist" in result.error


def test_dispatch_invalid_arguments_returns_failed_result(workspace_root: Path) -> None:
    registry = ToolRegistry()
    registry.add(name="fake_tool", schema=_FakeArgs, fn=_fake_tool)

    # value لازم تكون int، لو بعتنا حاجة مينفعش تتحول لـ int، الـ pydantic
    # المفروض ترفض، والـ dispatch يلفها كـ ToolResult فاشلة بدل ما تكسر البرنامج
    result = registry.dispatch("fake_tool", {"value": "not_a_number"}, workspace_root)

    assert result.success is False
    assert result.error is not None


def test_add_duplicate_tool_name_raises_error() -> None:
    registry = ToolRegistry()
    registry.add(name="fake_tool", schema=_FakeArgs, fn=_fake_tool)

    with pytest.raises(ValueError):
        registry.add(name="fake_tool", schema=_FakeArgs, fn=_fake_tool)


def test_schemas_returns_one_schema_per_registered_tool() -> None:
    registry = ToolRegistry()
    registry.add(name="fake_tool", schema=_FakeArgs, fn=_fake_tool)

    schemas = registry.schemas()

    assert len(schemas) == 1
    assert schemas[0] is _FakeArgs
