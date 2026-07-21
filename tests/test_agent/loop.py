"""
Manual test for agent/loop.py — run this directly (not pytest) and
watch stdout. You'll be prompted to approve each tool call.

"""
from pathlib import Path

from tesseractcli.agent.loop import run_inner_loop
from tesseractcli.llm.dispatcher import LLMDispatcher
from tesseractcli.tools.registry_builder import build_registry

WORKSPACE_ROOT = Path.cwd()

registry = build_registry()
dispatcher = LLMDispatcher()
messages = []  # shared across every stage -> model keeps context between them

# Ordered so each stage's tool needs something the previous stage set up
# (write before read/edit, edit before the file is listed/inspected again).
# Expected tool per stage is a note for YOU to check against the
# "[tool_use] <name>(...)" line the loop already prints before asking
# for approval - it's not enforced in code, the model still decides.
stages = [
    ("write_file", "Create a new file called scratch.txt in dir tests/test_agent  with the text 'hello from the agent' and new lines write create your text"),
    ("read_file", "Read the contents of scratch.txt dir tests/test_agent and show me exactly what's in it"),
    ("edit_file", "In scratch.txt in dir tests/test_agent, replace the text 'hello from the agent' with 'edited by the agent'"),
    ("list_directory", "List the files in the current directory (don't run a shell command, use the directory listing tool)"),
    ("exec_tool", "Run the shell command `cat tests/test_agent/scratch.txt` and show me the output"),
]

for expected_tool, prompt in stages:
    print(f"\n{'=' * 60}\nUSER (expecting {expected_tool}): {prompt}\n{'=' * 60}")
    reply = run_inner_loop(
        user_input=prompt,
        messages=messages,
        registry=registry,
        dispatcher=dispatcher,
        workspace_root=WORKSPACE_ROOT,
    )
    print(f"\nAGENT: {reply}")

