"""
tesseractcli/observability/tracing.py

Optional LangSmith tracing, wired through the same "safe" degradation
pattern already used everywhere else in this codebase (see
`BaseLLMProvider.get_model_safe` in llm/providers/base.py): if
`langsmith` isn't installed, or tracing isn't enabled in Settings,
every helper here becomes a no-op. Nothing in agent/loop.py or
tools/registry.py has to branch on "is tracing on?" - they just call
these helpers unconditionally.

Three distinct things get traced, for three different reasons:

  1. LLM calls (dispatcher.py's `model.ainvoke(...)`) are traced
     AUTOMATICALLY once LANGCHAIN_TRACING_V2=true is set - every model
     here is a LangChain `BaseChatModel`, and LangChain's runnables
     report to a global callback manager on their own. No code change
     needed in dispatcher.py for this part.

  2. Tool execution (`ToolRegistry.dispatch`) is plain Python, NOT a
     LangChain Runnable - it is NOT auto-traced. `traced_tool_call()`
     wraps one dispatch call in an explicit LangSmith run, named after
     the actual tool being invoked (e.g. "edit_file") - a static
     `@traceable` can't do this, since the tool name is only known at
     dispatch time, not at function-definition time.

  3. The whole agent turn (`run_inner_loop`, one call per user message)
     is wrapped with `@traceable` so every LLM attempt (across pool +
     fallback) and every tool call for that turn show up as ONE trace
     tree in LangSmith, instead of disconnected top-level runs.
"""

from __future__ import annotations

import os
from collections.abc import Callable, Generator
from contextlib import contextmanager
from typing import Any, TypeVar

from tesseractcli.config.logger import logger
from tesseractcli.config.settings import get_settings

F = TypeVar("F", bound=Callable[..., Any])

try:
    from langsmith import trace as _ls_trace
    from langsmith import traceable as _ls_traceable

    _LANGSMITH_INSTALLED = True
except ImportError:  # langsmith is an optional dependency
    _LANGSMITH_INSTALLED = False


def _tracing_enabled() -> bool:
    """Single source of truth, checked by every helper below. Requires
    BOTH the package to be installed AND the setting to be explicitly
    on - installing langsmith should never silently start exporting
    traces for someone who didn't ask for it."""
    settings = get_settings()
    return _LANGSMITH_INSTALLED and bool(getattr(settings, "LANGSMITH_TRACING", False))


def configure_tracing() -> None:
    """Call once at startup - `ui/app.py::run()` and the CLI entry in
    `__main__.py` both do this before anything else touches the
    dispatcher. Bridges `Settings` (.env-driven, same pattern as every
    other config value in this project - see settings.py) into the
    plain `os.environ` vars the langsmith/langchain SDKs read
    internally on their own; they don't go through pydantic-settings.

    No-ops entirely if tracing isn't enabled, so it's always safe to
    call unconditionally."""
    if not _tracing_enabled():
        return

    settings = get_settings()
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    if settings.LANGSMITH_API_KEY:
        os.environ["LANGCHAIN_API_KEY"] = settings.LANGSMITH_API_KEY
    os.environ["LANGCHAIN_PROJECT"] = settings.LANGSMITH_PROJECT or "tesseractcli"
    if settings.LANGSMITH_ENDPOINT:
        os.environ["LANGCHAIN_ENDPOINT"] = settings.LANGSMITH_ENDPOINT

    logger.debug("LangSmith tracing enabled (project={})", settings.LANGSMITH_PROJECT)


def traceable(**kwargs: Any) -> Callable[[F], F]:
    """Drop-in for `langsmith.traceable`, always used WITH keyword args
    in this codebase (e.g. `@traceable(name="agent_turn", run_type="chain")`).
    Degrades to a plain identity decorator when tracing is off, so
    `agent/loop.py` never needs an `if tracing_enabled:` branch around
    its own definition."""
    if not _tracing_enabled():

        def _identity(fn: F) -> F:
            return fn

        return _identity
    return _ls_traceable(**kwargs)


@contextmanager
def traced_tool_call(name: str, inputs: dict) -> Generator[Any, None, None]:
    """Wraps one `ToolRegistry.dispatch` call in a LangSmith run named
    after the real tool being executed. Yields the LangSmith run object
    when tracing is on (unused by callers today, available for
    `run.end(outputs=...)`-style enrichment later), or `None` when
    tracing is off - `tools/registry.py` uses this as a plain
    `with traced_tool_call(...):` block either way."""
    if not _tracing_enabled():
        yield None
        return

    with _ls_trace(name=name, run_type="tool", inputs=inputs) as run:
        yield run
