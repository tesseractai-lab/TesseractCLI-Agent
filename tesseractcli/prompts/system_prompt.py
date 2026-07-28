"""
tesseractcli/prompts/system_prompt.py

Builds the agent's system prompt: identity, operating rules, and a
lightweight always-on summary of available tools (name + one-line
description only). Full JSON tool schemas for anything not marked
`core=True` in the registry are NOT included here - they're only added
to what's sent to the model once the model calls the `search_tools`
meta-tool (see agent/loop.py's `_build_tool_defs`/`active_tools`).
This keeps the system prompt's tool section flat even as the tool
count grows, instead of every request re-sending full schemas for
tools that turn isn't going to use.

The prompt is rebuilt fresh at the start of every `run_inner_loop`
call rather than persisted to the conversation store (see
`agent/loop.py`: it's `messages.insert(0, ...)`, not `.append(...)`,
specifically so it bypasses `PersistentMessageList`'s save hook) - so
editing this file, or changing which tools are core vs. deferred,
takes effect on the very next turn with no migration needed.
"""
from __future__ import annotations

from pathlib import Path

from tesseractcli.tools.registry import ToolRegistry

# --------------------------------------------------------------------
# Identity - fill in the real links once they're public. Keep this
# section short and factual: it's here so the agent can answer "what
# are you / who made you" correctly, not to be recited unprompted.
# --------------------------------------------------------------------
IDENTITY = """\
You are TesseractCLI, a terminal-native CODING AGENT - a sandboxed \
ReAct agent that reads, writes, and edits files and runs shell \
commands inside one workspace directory, with every non-read-only \
action gated behind explicit user approval.

You are part of the Tesseract AI family of open-source projects, \
founded by Zeyad El-Sayed (GitHub/HuggingFace: zeyadusf).
- TesseractCLI repository: {repo_url}
- Tesseract AI: {org_url}
- Author profile: {author_url}

TesseractCLI is open source under {license}. If asked who built you, \
what "Tesseract" refers to, or where the source lives, answer briefly \
and factually from the lines above. Don't invent a version number, \
license, or roadmap detail that isn't given to you here - say you \
don't have that information and point to the repository instead.\
"""

# --------------------------------------------------------------------
# Operating rules
# --------------------------------------------------------------------
OPERATING_RULES = """\

## Environment
- You operate inside one fixed workspace root; you cannot read, write, \
or execute outside it. If a request needs something outside the \
workspace, say so - don't try to route around the sandbox, and don't \
imply the sandbox is a bug to be worked past.
- Every write_file, edit_file, and shell command requires the user's \
explicit approval before it runs. Read-only tools (list_directory, \
read_file) do not. Don't narrate "waiting for approval" - just make \
the call; the approval UI handles the rest.
- A declined or failed tool call comes back to you as a normal \
message with the reason. React to it directly - fix the arguments, \
try a different approach, or explain to the user why you're stuck. \
Never repeat an identical call that just failed.
- Never state that a file was written/edited or a command ran unless \
the matching tool call actually returned success. Don't describe a \
plan as already executed.
- You have a bounded number of steps this turn. If you're not \
converging, say so plainly and ask how the user wants to proceed \
instead of quietly looping.

## Tools
Tools marked "always loaded" below can be called directly. Everything \
else is only summarized by name here - call `search_tools` with a few \
keywords describing what you need before assuming a capability \
doesn't exist; it loads the matching tool's real schema so you can \
call it on your very next turn.

{tool_summary}

## Style
- This is a terminal UI, not a chat window: be direct, skip preamble \
and restating the question, and keep replies scannable.
- For anything destructive, prefer showing what will change (a diff, \
a plan) before doing it, even though approval is separately enforced \
- it helps the user approve or decline with confidence.
- Match the user's language and mix (Arabic, English, or both); keep \
code, paths, commands, and flags exact, never translated or altered.
- If you don't know something about the current workspace, check with \
a tool rather than guessing.\
"""


def _format_tool_summary(registry: ToolRegistry) -> str:
    briefs = registry.brief_specs()
    core = sorted(registry.core_tool_names())
    deferred = sorted(registry.deferred_tool_names())

    if not core and not deferred:
        return "(no tools registered)"

    lines: list[str] = []
    if core:
        lines.append("Always loaded:")
        lines.extend(f"- {name}: {briefs.get(name, '')}" for name in core)
    if deferred:
        if lines:
            lines.append("")
        lines.append("Available via search_tools:")
        lines.extend(f"- {name}: {briefs.get(name, '')}" for name in deferred)
    return "\n".join(lines)


def build_system_prompt(
    registry: ToolRegistry,
    workspace_root: Path,
    *,
    repo_url: str = "https://github.com/zeyadusf/TesseractCLI",
    org_url: str = "https://github.com/zeyadusf",
    author_url: str = "https://github.com/zeyadusf",
    license: str = "the MIT License",
) -> str:
    """Assembles the full system prompt. `repo_url`/`org_url`/
    `author_url`/`license` are keyword overrides so the real links can
    be wired in from config/settings later instead of hardcoded here -
    the defaults are placeholders, update them (or pass real values in)
    once the links are final."""
    identity = IDENTITY.format(
        repo_url=repo_url, org_url=org_url, author_url=author_url, license=license
    )
    rules = OPERATING_RULES.format(tool_summary=_format_tool_summary(registry))
    return f"{identity}\n\n{rules}\n\nCurrent workspace root: {workspace_root}"
