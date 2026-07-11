"""
tests/tools/test_registry_builder.py

الهدف الأساسي من الملف ده: يمسك بالظبط النوع اللي حصل في المحادثة —
اسم tool اتكتب غلط أو اتنسي من build_registry() ومحدش لاحظ إلا وقت
الـ runtime الفعلي. لو أي tool اتنسى أو اسمه اتكتب غلط، الـ test ده
لازم يفشل فورًا وقت الـ CI، مش وقت استخدام الـ agent.
"""

from tesseractcli.tools.registry_builder import build_registry

EXPECTED_TOOL_NAMES = {
    "write_file",
    "read_file",
    "edit_file",
    "list_directory",
    "run_command",
}


def test_build_registry_registers_every_expected_tool() -> None:
    registry = build_registry()

    # الوصول لـ _tools هنا مباشر (private attribute) لأن الهدف من التست
    # فحص داخلي (internal invariant)، مش استخدام public API عادي.
    registered_names = set(registry._tools.keys())  # type: ignore[attr-defined]

    assert registered_names == EXPECTED_TOOL_NAMES


def test_build_registry_returns_fresh_instance_each_call() -> None:
    """
    مهم نتأكد إن build_registry() بترجع registry جديد كل مرة، مش نفس
    الـ instance بترجع تاني (لو فيه global state متخبي، ده هيفشل).
    """
    registry_a = build_registry()
    registry_b = build_registry()

    assert registry_a is not registry_b
