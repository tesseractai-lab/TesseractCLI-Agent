# TesseractCLI-Agent

A terminal-native, multi-provider coding agent — inspired by Claude Code — built **from scratch** (hand-built agent loop and tools layer, no LangGraph/LangChain) as a deep dive into how agentic coding tools work internally.

Invoked from anywhere via:

```bash
tesseract
```

> **Status:** early-stage / MVP in progress. See [`docs/TODO.md`](docs/TODO.md) for the current build checklist.

---

## What it does (target MVP)

- A keyboard-navigable terminal UI (built with [Textual](https://textual.textualize.io/)) — workspace selector, model/provider picker, and a chat screen
- An agent loop that lets the LLM itself decide which tool to call (native tool-calling, not a hardcoded router)
- File and shell tools (`read_file`, `write_file`, `run_command`) sandboxed to the active workspace
- Every mutating action (file writes, shell commands) requires explicit user approval before it runs

## Architecture

Three strictly separated layers:

```
ui/      -> screens & widgets, never talks to the LLM directly
agent/   -> the ReAct loop, LLM providers, conversation state
tools/   -> file/shell execution, sandboxed, gated by approval
```

See [`docs/PROJECT_VISION.md`](docs/PROJECT_VISION.md) for the full design rationale.

## Installation (development)

```bash
git clone <this-repo>
cd TesseractCLI-Agent
pip install -e .
tesseract
```

`pip install -e .` installs the package in editable mode, so code changes take effect immediately without reinstalling.

## Configuration

On first run, Tesseract needs at least one LLM provider API key. For now this is configured by hand at `~/.tesseract/global_config.yaml` (a setup wizard is planned — see the roadmap).

## Project structure

```
tesseractcli/
├── __main__.py         # entry point (`tesseract` console command)
├── config/              # loading/validating global + workspace config
├── ui/                  # Textual screens, widgets, styles
├── agent/               # ReAct loop, providers, dispatcher, prompts
├── tools/                # sandboxed file/shell execution + registry
└── memory/               # session history storage
```

## Roadmap

1. **Phase 1 (MVP):** TUI + agent loop + 3 tools + approval gate + path sandboxing
2. **Phase 2:** Hierarchical memory (summarization tiers) + real Settings screen
3. **Phase 3:** Reasoning-effort/retrieval-depth levels + Docker-based command sandboxing + PyPI packaging
4. **Phase 4 (experimental):** Planner-led multi-model parallel code generation

## Author

Built by [zeyadusf](https://github.com/zeyadusf) as part of the Tesseract project family (alongside TesseractRAG and TesseractResearch).
