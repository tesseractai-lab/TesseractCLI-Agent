from pathlib import Path
from unittest.mock import patch

import pytest

from tesseractcli.agent.loop import run_inner_loop
from tesseractcli.llm.dispatcher import LLMDispatcher
from tesseractcli.tools.registry_builder import build_registry

WORKSPACE_ROOT = Path.cwd()

STAGES = [
    (
        "write_file",
        "Create a new file called scratch.txt in tests/test_agent. "
        "Write a short multi-line note of 6-10 lines that you create yourself. "
        "Each line should contain different text.",
    ),
    (
        "read_file",
        "Read the contents of tests/test_agent/scratch.txt and show me the exact contents.",
    ),
    (
        "edit_file",
        "Edit tests/test_agent/scratch.txt by modifying only a small part of one line in the middle of the file. "
        "Do not replace the whole line or the first line. Change only a short phrase within that line and leave everything else unchanged.",
    ),
    (
        "list_directory",
        "List the files in the current directory (don't run a shell command, use the directory listing tool)",
    ),
    (
        "exec_tool",
        "Run the shell command `cat tests/test_agent/scratch.txt` and show me the output",
    ),
]


@pytest.mark.manual
def test_loop_manual():
    registry = build_registry()
    dispatcher = LLMDispatcher()
    messages = []

    # Automatically answer "y" to every approval prompt.
    with patch("builtins.input", return_value="y"):
        for expected_tool, prompt in STAGES:
            print(f"\n{'=' * 60}")
            print(f"USER (expecting {expected_tool}): {prompt}")
            print(f"{'=' * 60}")

            reply = run_inner_loop(
                task_name="test_loop_manual",
                user_input=prompt,
                messages=messages,
                registry=registry,
                dispatcher=dispatcher,
                workspace_root=WORKSPACE_ROOT,
            )

            print(f"\nAGENT: {reply}")
