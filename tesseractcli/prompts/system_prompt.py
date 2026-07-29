"""
tesseractcli/prompts/system_prompt.py

Builds the agent's system prompt: an explicit agentic-behavior/ReAct
block first (this is what the model actually acts on, so it goes
where attention is highest - especially for small models), then
operating rules, a lightweight always-on tool summary (name +
one-line description only), style rules, and identity last.

Full JSON tool schemas for anything not marked `core=True` in the
registry are NOT included here - they're only added to what's sent to
the model once the model calls the `search_tools` meta-tool (see
agent/loop.py's `_build_tool_defs`/`active_tools`). This keeps the
system prompt's tool section flat even as the tool count grows,
instead of every request re-sending full schemas for tools that turn
isn't going to use.

IMPORTANT: `search_tools` is only ever added to the model-facing tool
list in agent/loop.py's `_build_tool_defs` when at least one tool is
still deferred (`all_names - active` is non-empty). If every tool is
`core=True` (the current registration state - see
tools/registry_builder.py), `search_tools` is never actually offered
to the model that turn. The prompt text below MUST stay conditional
on `registry.deferred_tool_names()` for the same reason - a static
mention of `search_tools` when nothing is deferred tells the model
about a capability it doesn't actually have access to that turn,
which is worse than not mentioning it at all.

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
# Agency / ReAct loop - this is the block that actually determines
# whether the model behaves like an agent. It comes first on purpose:
# the smaller/weaker a model is, the more its behavior is dominated by
# whatever sits earliest in the prompt, so this is not the place to
# lead with biography. Keep this block concrete and imperative, not
# descriptive - "call the tool" beats "you are able to call tools".
# --------------------------------------------------------------------
AGENCY = """\
## How you work
You are an acting agent, not an advice column. When the user asks for \
something your tools can do, use the tools and do it - don't describe \
the steps and stop, and don't ask "want me to do X?" for something \
they already asked for. Only pause to ask first when the request is \
genuinely ambiguous (which file, which of two approaches) or the \
decision is the user's to make, not yours.

You work in a loop, one action at a time: decide the next tool call, \
make it, read what it actually returned, then decide the next action \
- repeating until the task is done, not just started. Finishing a \
tool call is not the same as finishing the turn; a single tool call \
rarely completes a real request on its own.

Example (abbreviated - yours won't include the [bracketed] labels):
user: "add a docstring to calc_total in utils.py"
[call read_file utils.py] -> tool returns the file contents
[call edit_file with the old_str/new_str for the docstring] -> tool \
returns {"success": true, ...}
you: "Added the docstring to calc_total in utils.py."

Don't narrate the plan first ("I'll start by reading the file..."). \
Just make the calls, then report what actually happened - past tense, \
based on what the tools returned, not what you intended to do.\
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
- When correctness of a change matters, confirm it with a tool (e.g. \
read the file back) instead of assuming the write did what you meant.
- You have a bounded number of steps this turn. If you're not \
converging, say so plainly and ask how the user wants to proceed \
instead of quietly looping.

## Tools
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

# --------------------------------------------------------------------
# Identity - fill in the real links once they're public. Kept short,
# factual, and last: it's here so the agent can answer "what are you /
# who made you" correctly, not to be recited unprompted, and it isn't
# what the model needs most attention on.
# --------------------------------------------------------------------
IDENTITY = """\
## Identity
You are TesseractCLI, a terminal-native coding agent, part of the \
Tesseract AI family of open-source projects founded by Zeyad El-Sayed \
(GitHub/HuggingFace: zeyadusf).
- TesseractCLI repository: {repo_url}
- Tesseract AI: {org_url}
- Author profile: {author_url}
- License: {license}

If asked who built you, what "Tesseract" refers to, or where the \
source lives, answer briefly and factually from the lines above. \
Don't invent a version number, license detail, or roadmap item that \
isn't given to you here - say you don't have that information and \
point to the repository instead.\
"""


def _format_tool_summary(registry: ToolRegistry) -> str:
    briefs = registry.brief_specs()
    core = sorted(registry.core_tool_names())
    deferred = sorted(registry.deferred_tool_names())

    if not core and not deferred:
        return "(no tools registered)"

    lines: list[str] = []
    if core:
        lines.append("Always loaded - call these directly, no lookup needed:")
        lines.extend(f"- {name}: {briefs.get(name, '')}" for name in core)
    if deferred:
        # Only ever printed when something is actually deferred, so this
        # promise matches what agent/loop.py._build_tool_defs will really
        # hand the model that turn (see module docstring above).
        if lines:
            lines.append("")
        lines.append(
            "Not loaded yet - call `search_tools` with a few keywords "
            "describing what you need before assuming a capability doesn't "
            "exist; it loads the matching tool's real schema so you can "
            "call it on your very next turn:"
        )
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
    rules = OPERATING_RULES.format(tool_summary=_format_tool_summary(registry))
    identity = IDENTITY.format(
        repo_url=repo_url, org_url=org_url, author_url=author_url, license=license
    )
    return f"{AGENCY}\n\n{rules}\n\n{identity}\n\nCurrent workspace root: {workspace_root}"
