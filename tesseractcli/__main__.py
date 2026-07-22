"""Entry point module.

`pyproject.toml` maps the `tesseract` console command to `main()` here:

    [project.scripts]
    tesseract = "tesseractcli.__main__:main"

Two ways to run it now:

    tesseract                      launch the interactive TUI (unchanged
                                    default: workspace -> pack picker ->
                                    chat; type 'exit' or 'quit' anywhere
                                    in the TUI to close it)

    tesseract settings [options]   manage global_config.yaml directly
                                    from the shell, no TUI - for
                                    scripting/CI or a quick edit without
                                    booting the whole app. Examples:

        tesseract -h
        tesseract settings --list
        tesseract settings --suggest
        tesseract settings --add pool --pack vision
        tesseract settings --add model --pack vision --provider openai --model gpt-4o-mini
        tesseract settings --add model --pack vision --provider openai --model gpt-4o-mini --fallback
        tesseract settings --remove model --pack vision --provider openai --model gpt-4o-mini
        tesseract settings --remove pool --pack vision
        tesseract settings --set paths.logs_dir /tmp/tesseract-logs
        tesseract settings --get agent.temperature

This subcommand reuses the exact same `ConfigManager`/`PacksManager`
that the in-app `settings` stage does (via `settings_commands.py`), so
a pack added from the shell shows up in the TUI's pack picker and vice
versa - there's only ever one `global_config.yaml`.
"""

from __future__ import annotations

import argparse
import sys

from tesseractcli.config.global_config.manager import ConfigManager


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tesseract",
        description="TesseractCLI - a terminal-native coding agent.",
    )
    sub = parser.add_subparsers(dest="command")

    settings = sub.add_parser(
        "settings",
        help="manage global_config.yaml (packs, models, paths, agent config) without launching the TUI",
    )
    settings.add_argument("--list", action="store_true", help="list all packs and the models inside them")
    settings.add_argument("--suggest", action="store_true", help="show provider/model suggestions")
    settings.add_argument("--add", choices=["pool", "model"], help="add a new pool (pack) or a model inside one")
    settings.add_argument("--remove", choices=["pool", "model"], help="remove a pool (pack) or a model inside one")
    settings.add_argument("--pack", help="pack name (required with --add/--remove)")
    settings.add_argument("--provider", help="provider name (required with --add/--remove model)")
    settings.add_argument("--model", help="model identifier (required with --add/--remove model)")
    settings.add_argument("--fallback", action="store_true", help="target the fallback list instead of the primary pool")
    settings.add_argument("--set", nargs=2, metavar=("PATH", "VALUE"), help="set a dot-notation config value, e.g. paths.logs_dir /tmp/logs")
    settings.add_argument("--get", metavar="PATH", help="read a dot-notation config value, e.g. agent.temperature")

    return parser


def _require(args: argparse.Namespace, *names: str) -> None:
    missing = [f"--{n}" for n in names if not getattr(args, n)]
    if missing:
        sys.exit(f"missing required option(s): {', '.join(missing)}")


def _run_settings(args: argparse.Namespace) -> None:
    # Imported lazily so `tesseract -h` stays fast and doesn't need
    # rich/pydantic import overhead just to print usage.
    from rich.console import Console

    from tesseractcli.ui.views.settings_commands import render_packs_overview, suggestions_text

    console = Console()
    manager = ConfigManager()
    manager.load()

    if args.suggest:
        console.print(suggestions_text())
        return

    if args.list or not any([args.add, args.remove, args.set, args.get]):
        console.print(render_packs_overview(manager))
        return

    if args.add == "pool":
        _require(args, "pack")
        manager.packs.add_pack(args.pack)
        manager.save()
        console.print(f"[green]created pack '{args.pack}'[/green]")
    elif args.add == "model":
        _require(args, "pack", "provider", "model")
        manager.packs.add_model(args.pack, args.provider, args.model, target="fallback" if args.fallback else "pool")
        manager.save()
        console.print(f"[green]added {args.provider}/{args.model} to '{args.pack}'[/green]")
    elif args.remove == "pool":
        _require(args, "pack")
        manager.packs.remove_pack(args.pack)
        manager.save()
        console.print(f"[green]removed pack '{args.pack}'[/green]")
    elif args.remove == "model":
        _require(args, "pack", "provider", "model")
        manager.packs.remove_model(args.pack, args.provider, args.model, target="fallback" if args.fallback else "pool")
        manager.save()
        console.print(f"[green]removed {args.provider}/{args.model} from '{args.pack}'[/green]")
    elif args.set:
        path, value = args.set
        manager.set(path, value)
        manager.save()
        console.print(f"[green]{path} = {value}[/green]")
    elif args.get:
        console.print(f"{args.get} = {manager.get(args.get)}")


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    if args.command == "settings":
        _run_settings(args)
        return

    # No subcommand: launch the interactive TUI (unchanged default
    # behavior) - starting on the workspace prompt, same as before.
    # Imported here, not at module level, so `-h`/`settings` don't pull
    # in textual/langchain just to print usage or edit the config.
    from tesseractcli.ui.app import run

    run()


if __name__ == "__main__":
    # Also allows `python -m tesseractcli` as an alternative to the
    # installed `tesseract` command, useful while developing.
    main()
