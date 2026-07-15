"""
tesseractcli/agent/loop.py

The inner agent loop: runs ONE full turn for a single user message.
Sends messages + tool schemas to the model, prints and approves any
tool_use request, dispatches it through the registry, feeds the
ToolResult back as a ToolMessage (matched by tool_call_id), and
repeats until the model replies with no tool_calls (final answer) or
until INNER_LOOP_MAX_ITERATIONS is hit.

This is NOT the outer session loop (waiting on repeated user input) -
that's a separate, later concern (likely owned by the TUI). This
function is called once per user message; `messages` is passed in and
mutated in place so the caller keeps full history across calls.
"""
from __future__ import annotations

import json
from pathlib import Path

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage

from tesseractcli.config.logger import logger
from tesseractcli.config.settings import get_settings
from tesseractcli.llm.dispatcher import LLMDispatcher
from tesseractcli.models.tool_models.tools_result import ToolResult
from tesseractcli.tools.registry import ToolRegistry


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


def _format_tool_result(result: ToolResult) -> str:
    """What the model sees back for a tool call - success or failure,
    both go back as a normal ToolMessage so the model can react to
    errors itself (retry, change approach) rather than crashing the loop."""
    if result.success:
        return result.output
    return f"ERROR: {result.error}"


def _approve(tool_name: str, args: dict) -> bool:
    """Placeholder approval gate. Deliberately isolated in its own
    function (not inlined in the loop body) so swapping this for the
    real Textual approval UI later means changing this function only -
    the loop body doesn't need to know how approval is obtained."""
    print(f"\n[tool_use] {tool_name}({json.dumps(args, ensure_ascii=False)})")
    answer = input("approve? y/n: ").strip().lower()
    logger.debug(f"[tool]: {tool_name} ask approve to run ({json.dumps(args, ensure_ascii=False)}) and user say [{answer}]")
    return answer == "y"



def run_inner_loop(
    user_input: str,
    messages: list[BaseMessage],
    registry: ToolRegistry,
    dispatcher: LLMDispatcher,
    workspace_root: Path,
    task_name: str | None = None,
) -> str:
    """Runs one full agent turn for `user_input`. Returns the model's
    final text reply. Mutates `messages` in place (appends the human
    message, every AI message, and every tool result) so the caller can
    pass the same list back in on the next call for continuity."""
    settings = get_settings()
    max_iterations = settings.INNER_LOOP_MAX_ITERATIONS

    messages.append(HumanMessage(content=user_input))

    tool_defs = _build_tool_defs(registry)
    bound_model = dispatcher.get_llm_with_tools(tool_defs, task_name)

    for _ in range(max_iterations):
        ai_message: AIMessage = bound_model.invoke(messages)
        messages.append(ai_message)

        if not ai_message.tool_calls:
            return ai_message.content

        # Handle every tool call the model asked for in this turn, not
        # just the first - a single AIMessage can carry more than one
        # tool_calls entry, and each needs its own ToolMessage matched
        # back by that call's own id.
        for call in ai_message.tool_calls:
            name = call["name"]
            args = call["args"]
            call_id = call["id"]

# here customize approve logic
            if _approve(name, args):
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
                )
            )

    logger.warning(
        "Inner loop hit INNER_LOOP_MAX_ITERATIONS (%d) without a final reply",
        max_iterations,
    )
    return f"[stopped after {max_iterations} iterations without a final answer - check the log]"
