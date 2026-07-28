"""
tesseractcli/agent/loop.py

The inner agent loop: runs ONE full turn for a single user message.
Sends messages + tool schemas to the model, prints and approves any
tool_use request, dispatches it through the registry, feeds the
ToolResult back as a ToolMessage (matched by tool_call_id), and
repeats until the model replies with no tool_calls (final answer) or
until `agent.max_iterations` (read live off the dispatcher's shared
ConfigManager, see llm/dispatcher.py) is hit.

This is NOT the outer session loop (waiting on repeated user input) -
that's a separate, later concern (owned by the TUI / CLI entrypoint).
This function is called once per user message; `messages` is passed in
and mutated in place so the caller keeps full history across calls.

Step 6 change (Textual integration): this is now `async def`.

- Model calls go through `dispatcher.ainvoke_with_fallback(...)` instead
  of a single bound model's `.invoke()`. This is a deliberate change
  from the Step 5 version (which cached one `bound_model` and called
  `.invoke()` per iteration): `ainvoke_with_fallback` re-resolves the
  pack's pool + fallback chain and does rate-limit/size-aware retry on
  every call, which is the "full resilience path" it's built for. The
  cost is re-doing pack resolution each iteration; the provider layer's
  own model cache (`get_model_safe`) means this doesn't rebuild
  LangChain client objects each time, so the overhead is small.

- Tool approval is now injected via `approve_fn` (async callable)
  instead of calling `approve_tool_call` directly. The default
  (`_default_approve_fn`) preserves the exact terminal behavior from
  Step 5 (print preview + blocking `input()`), just run in a worker
  thread via `asyncio.to_thread` so it never blocks the event loop -
  this matters even for terminal usage now that the function is async,
  and it's what lets the Textual UI swap in its own modal-based
  `approve_fn` with zero changes to this file.
"""
from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Awaitable, Callable, TypeAlias

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage

from tesseractcli.config.logger import logger
from tesseractcli.llm.dispatcher import LLMDispatcher
from tesseractcli.models.tool_models.tools_result import ToolResult
from tesseractcli.observability import traceable
from tesseractcli.tools.approval import approve_tool_call
from tesseractcli.tools.registry import ToolRegistry

# (tool_name, tool_args, workspace_root) -> approved?
ApproveFn : TypeAlias = Callable[[str, dict, Path], Awaitable[bool]]


def _build_tool_defs(registry: ToolRegistry) -> list[dict]:
    """LLM-facing tool schemas, built from (name, schema) pairs so the
    tool name the model sees matches the registry key exactly - NOT
    registry.schemas(), which drops the name and would leak the Python
    class name (e.g. "ReadFileArgs") instead of "read_file"."""
    tool_defs = []
    for name, schema in registry.tool_specs():
        tool_defs.append(
            {
                "name": name,
                "description": (schema.__doc__ or "").strip(),
                "parameters": schema.model_json_schema(),
            }
        )
    return tool_defs


def _normalize_for_cross_provider_replay(ai_message: AIMessage) -> AIMessage:
    """Some models return an AIMessage with BOTH `.content` (a narration
    like "let me check that file...") AND `.tool_calls` populated at
    the same time. Anthropic tolerates this when the message is replayed
    back as history, but OpenAI-compatible endpoints (Mistral, Cerebras
    - see llm/providers/mistral_provider.py, both routed through
    ChatOpenAI) reject it outright with a 400:
    'Assistant message must have either content or tool_calls, but not both.' """
    if ai_message.tool_calls and ai_message.content:
        return ai_message.model_copy(update={"content": ""})
    return ai_message


def _format_tool_result(result: ToolResult) -> str:
    """What the model sees back for a tool call - success or failure,
    both go back as a normal ToolMessage so the model can react to
    errors itself (retry, change approach) rather than crashing the loop."""
    if result.success:
        return result.output
    return f"ERROR: {result.error}"


async def _default_approve_fn(tool_name: str, tool_args: dict, workspace_root: Path) -> bool:
    """Terminal fallback approve_fn - identical behavior to Step 5
    (renders a preview, blocks on `input("approve? (y/n): ")`), just
    run off the event loop thread so `await approve_fn(...)` is always
    safe to call from async code, terminal or Textual alike."""
    return await asyncio.to_thread(approve_tool_call, tool_name, tool_args, workspace_root)


@traceable(name="agent_turn", run_type="chain")
async def run_inner_loop(
    user_input: str,
    messages: list[BaseMessage],
    registry: ToolRegistry,
    dispatcher: LLMDispatcher,
    workspace_root: Path,
    name_pack: str | None ,
    approve_fn: ApproveFn = _default_approve_fn,
    pinned_model: tuple[str, str] | None = None,
) -> str:
    """Runs one full agent turn for `user_input`. Returns the model's
    final text reply. Mutates `messages` in place (appends the human
    message, every AI message, and every tool result) so the caller can
    pass the same list back in on the next call for continuity.

    `name_pack` is a routing pack name (e.g. "main_pack", "second_pack",
    "third_pack" - see `llm/routing.py`'s RoutingResolver), NOT a raw
    provider/model pair. `approve_fn` defaults to the terminal prompt;
    pass a UI-backed one (e.g. Textual's modal) to override it.

    `pinned_model`, if given, is a (provider, model) pair the caller
    explicitly selected out of `name_pack`'s pool/fallback - every model
    call in this turn uses exactly that entry (still with the same
    rate-limit-triggered truncate-and-retry-once as any other step), and
    never silently falls back to a different model. Leave it as None to
    keep the original "walk pool then fallback" behavior.
    """
    max_iterations = dispatcher.max_iterations

    messages.append(HumanMessage(content=user_input))

    tool_defs = _build_tool_defs(registry)

    for _ in range(max_iterations):
        ai_message: AIMessage = await dispatcher.ainvoke_with_fallback(
            messages, tools=tool_defs, pack_name=name_pack, pinned=pinned_model
        )
        ai_message = _normalize_for_cross_provider_replay(ai_message)
        messages.append(ai_message)

        if not ai_message.tool_calls:
            return ai_message.content

        # tool_calls entry, and each needs its own ToolMessage matched
        # back by that call's own id.
        for call in ai_message.tool_calls:
            name = call["name"]
            args = call["args"]
            call_id = call["id"]

            # Read-only tools (needs_approval=False in the registry)
            # skip the prompt entirely - approve_fn is never even
            # called for them.
            if registry.needs_approval(name):
                approved = await approve_fn(name, args, workspace_root)
            else:
                approved = True

            if approved:
                result = registry.dispatch(name, args, workspace_root)
            else:
                result = ToolResult(
                    tool_name=name,
                    success=False,
                    output="",
                    error="User declined to approve this tool call.",
                )

            messages.append(
                ToolMessage(
                    content=_format_tool_result(result),
                    tool_call_id=call_id,
                    additional_kwargs={
                        "tool_name": name,
                        "tool_args": args,
                        "tool_success": result.success,
                    },
                )
            )

    logger.warning(
        "Inner loop hit agent.max_iterations (%d) without a final reply",
        max_iterations,
    )
    return f"[stopped after {max_iterations} iterations without a final answer - check the log]"
