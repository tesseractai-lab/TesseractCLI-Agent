"""
tesseractcli/tools/registry.py
"""

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from pydantic import BaseModel, ValidationError

from tesseractcli.models import ToolResult
from tesseractcli.observability import traced_tool_call


@dataclass
class ToolRegistry:
    # (schema, fn, needs_approval, core). `core=True` (the default) means
    # the tool's full JSON schema is sent to the model on every request.
    # `core=False` means only its name + one-line description are ever
    # shown up front (via brief_specs, used in the system prompt) - the
    # full schema is only added to what's sent to the model once the
    # model calls the `search_tools` meta-tool (see agent/loop.py). This
    # is what keeps per-request tool context flat as more tools get
    # added: mark anything not used on nearly every turn as core=False.
    _tools: dict[str, tuple[type[BaseModel], Callable, bool, bool]] = field(
        default_factory=dict
    )

    def add(
        self,
        name: str,
        schema: type[BaseModel],
        fn: Callable,
        needs_approval: bool = True,
        core: bool = True,
    ) -> None:
        if name in self._tools:
            raise ValueError(f"Tool '{name}' is already registered.")
        self._tools[name] = (schema, fn, needs_approval, core)

    def schemas(self) -> list[type[BaseModel]]:
        """All the plans for that were tracked for the LLM Provider."""
        return [schema for schema, _, _, _ in self._tools.values()]

    def tool_specs(
        self, names: set[str] | None = None
    ) -> list[tuple[str, type[BaseModel]]]:
        """Name + schema pairs, for building tool definitions that keep
        the LLM-facing tool name in sync with the registry key. Pass
        `names` to restrict to a subset (e.g. only the currently-active
        tools) instead of every registered tool."""
        items = (
            self._tools.items()
            if names is None
            else ((name, self._tools[name]) for name in names if name in self._tools)
        )
        return [(name, schema) for name, (schema, _, _, _) in items]

    def core_tool_names(self) -> list[str]:
        """Tools whose full schema is always sent to the model."""
        return [name for name, (_, _, _, core) in self._tools.items() if core]

    def deferred_tool_names(self) -> list[str]:
        """Tools only discoverable via `search_tools` (see agent/loop.py)."""
        return [name for name, (_, _, _, core) in self._tools.items() if not core]

    def brief_specs(self, names: list[str] | set[str] | None = None) -> dict[str, str]:
        """name -> first line of its schema's docstring. Cheap (no JSON
        schema dump) - used for the always-on tool summary in the system
        prompt and for `search_tools` results, neither of which need the
        full parameter schema, just enough for the model to know a tool
        exists and what it's for."""
        keys = (
            self._tools.keys()
            if names is None
            else [n for n in names if n in self._tools]
        )
        out: dict[str, str] = {}
        for name in keys:
            schema = self._tools[name][0]
            doc_lines = (schema.__doc__ or "").strip().splitlines()
            out[name] = doc_lines[0].strip() if doc_lines else ""
        return out

    def needs_approval(self, name: str) -> bool:
        """Used by the approval flow to decide whether to prompt the user.
        Unknown tool names fail safe (treated as needing approval)."""
        if name not in self._tools:
            return True
        _, _, needs, _ = self._tools[name]
        return needs

    def dispatch(self, name: str, raw_args: dict, workspace_root: Path) -> ToolResult:
        """ "It is called when the LLM returns a tool_use request."""
        if name not in self._tools:
            return ToolResult(
                tool_name=name,
                success=False,
                output="",
                error=f"Unknown tool '{name}'.",
            )

        schema, fn, _, _ = self._tools[name]

        try:
            validated = schema(**raw_args)
        except ValidationError as e:
            return ToolResult(
                tool_name=name,
                success=False,
                output="",
                error=f"Invalid arguments for '{name}': {e}",
            )

        # Plain Python call, not a LangChain Runnable, so it is NOT
        # auto-traced the way model.ainvoke() calls are - traced_tool_call
        # wraps it in a LangSmith run named after the real tool (e.g.
        # "edit_file"), and is a no-op when tracing is off.
        with traced_tool_call(name, raw_args):
            return fn(workspace_root=workspace_root, **validated.model_dump())
