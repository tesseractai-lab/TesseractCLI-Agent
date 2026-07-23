"""
tesseractcli/ui/views/settings_commands.py

Parses and executes the command grammar available while
`TesseractApp.stage == "settings"`. Every branch (list packs, show one
pack, add/remove a pack, add/remove a model, edit a path, show
suggestions) goes through `handle()` here and returns a bordered Rich
panel (via `ui.views.box.render_box`) to write into the log -
`app.py` never touches `ConfigManager` directly for any of this.

Grammar (bare words, case-insensitive first token):

    help                                     show this command list
    packs                                    list all packs + their models
    pack <name>                              show one pack in full detail
    suggest                                  provider/model suggestions
    add pack <name>                          create a new empty pack
    remove pack <name>                       delete a pack
    add model <pack> <provider> <model> [fallback]
    remove model <pack> <provider> <model> [fallback]
    set <dot.path> <value>                   edit any scalar config value
    get <dot.path>                           read any scalar config value

`set`/`get` use the same dot-notation as `ConfigManager.get`/`.set`
(e.g. `paths.logs_dir`, `agent.temperature`,
`providers.main_pack.temperature`) so every field in global_config.yaml
is reachable, not just packs/models.
"""

from __future__ import annotations

import shutil
from typing import TYPE_CHECKING, Any

from tesseractcli.config.provider_catalog import suggestions_text
from tesseractcli.models.exceptions import ConfigError
from tesseractcli.ui.views.box import render_box

if TYPE_CHECKING:
    from tesseractcli.config.global_config.manager import ConfigManager
    from tesseractcli.models.config_models.provider_models import ModelPack

# Short aliases for the commands people actually type by hand a lot.
# `app.py` resolves the first word through this dict before dispatching,
# so `s` / `h` / `?` etc. work everywhere `suggest` / `help` would.
# NOTE: this was previously referenced from app.py (`settings_commands.ALIASES`)
# without being defined here at all - any settings-stage input triggered an
# AttributeError that crashed the whole app instead of showing an error.
# Full command templates for ghost-text autocomplete on `#main-input`
# while `stage == "settings"` (see `ui/widgets/chat_input.py`). Entries
# that take an argument end with a trailing space so accepting the
# suggestion (right arrow) leaves the cursor ready to type the name,
# rather than needing a space typed manually afterwards.
COMMAND_CHOICES: list[str] = [
    "help",
    "packs",
    "pack ",
    "suggest",
    "add pack ",
    "add model ",
    "remove pack ",
    "remove model ",
    "set ",
    "get ",
    "backup",
    "backups",
    "restore",
    "restore latest",
    "validate",
    "chat",
    "exit",
]

# NOTE: "back" used to be listed here (and in HELP_TEXT below) as a
# synonym for "chat", but it was never actually wired to anything -
# NAV_COMMANDS in app.py (the thing that makes bare words like "chat"
# navigate anywhere) never included "back", so typing it just fell
# through to "unrecognized command". Removed rather than fixed: "chat"
# already does the job, and keeping both invited confusion with
# "backup"/"backups"/"restore" right below it. -cnf's mapping to
# "settings" was similarly dead here (this dict is only consulted
# *while already in* the settings stage, and "settings" was never a
# recognized command in `handle()` either) - it's now a real, working
# global shortcut instead (see `GLOBAL_ALIASES` in `ui/app.py`).
ALIASES: dict[str, str] = {
    "-s": "suggest",
    "-h": "help",
    "?": "help",
    "-p": "packs",
    "ls-p": "packs",
    "-a": "add",
    "-rm": "remove",
    "-r": "remove",
}

HELP_TEXT = (
    "  packs            (-p, ls-p)             list packs and their models\n"
    "  pack <name>                                show one pack in detail\n"
    "  suggest          (-s)                     provider/model suggestions\n"
    "  add pack <name>                             create a new empty pack\n"
    "  remove pack <name>                          delete a pack\n"
    "  add model <pack> <provider> <model> [fallback]\n"
    "  remove model <pack> <provider> <model> [fallback]\n"
    "  set <dot.path> <value>                      edit a config value\n"
    "  get <dot.path>                              read a config value\n"
    "  backup                                      snapshot the config now\n"
    "  backups                                     list saved backups\n"
    "  restore [latest|<file>]                     roll back to a backup\n"
    "  validate                                    check config against schema\n"
    "  chat                                        return to chat\n"
    "  help             (-h, --help, ?)              show this list\n"
    "  exit             (-q, --quit)                quit TesseractCLI\n\n"
    "[dim]remove pack/model always asks for '... confirm' before deleting,\n"
    "and takes a backup first - nothing is a one-way door.[/dim]"
)


def _format_pack(name: str, pack: "ModelPack", *, color: str) -> str:
    """Each pack gets its own header color (cycled from `_PACK_PALETTE`
    by position, see `pack_color()`), and inside a pack, `pool` and
    `fallback` are their own color families (green / amber) so you can
    tell "this is a primary model" from "this is a fallback" at a
    glance instead of reading the label - the model lines under each
    are a lighter tint of that same family, same header/body pairing
    used in `settings_view.py`."""
    lines = [f"[bold {color}]{name}[/bold {color}]  (max_tokens={pack.max_tokens}, temperature={pack.temperature})"]

    pool_header, pool_body = _POOL_COLORS
    lines.append(f"  [bold {pool_header}]pool:[/bold {pool_header}]")
    if pack.pool:
        lines.extend(f"    [{pool_body}]- {m.provider}/{m.model}[/{pool_body}]" for m in pack.pool)
    else:
        lines.append("    [dim](empty)[/dim]")

    fb_header, fb_body = _FALLBACK_COLORS
    lines.append(f"  [bold {fb_header}]fallback:[/bold {fb_header}]")
    if pack.fallback:
        lines.extend(f"    [{fb_body}]- {m.provider}/{m.model}[/{fb_body}]" for m in pack.fallback)
    else:
        lines.append("    [dim](empty)[/dim]")
    return "\n".join(lines)


# One color per pack, cycled by position so any number of packs stays
# readable rather than reusing `settings_view.py`'s section palette
# (which is fixed to 4 named sections and wouldn't scale to N packs).
# `pool`/`fallback` are deliberately NOT in this list - they're a
# separate, fixed color family (see `_format_pack`) shared by every
# pack, so "this is a pool entry" reads the same way in every pack
# rather than shifting color depending on which pack it's in.
_PACK_PALETTE = ["#4dd8ff", "#b98cff", "#ff6b9d", "#6bcaff", "#e8a33d", "#4ddb9e"]
_POOL_COLORS = ("#4ddb9e", "#a8f2d4")      # pool (primary): green / light green
_FALLBACK_COLORS = ("#e8a33d", "#f5cf94")  # fallback: amber / light amber


def pack_color(index: int) -> str:
    """Exposed (not `_`-prefixed) so `settings_view.py`'s pack summary
    list can use the exact same color per pack as the detailed
    `packs`/`pack <name>` panels below - one pack, one color,
    everywhere it's shown."""
    return _PACK_PALETTE[index % len(_PACK_PALETTE)]


def render_packs_overview(manager: "ConfigManager") -> Any:
    """Bordered panel listing every pack and the models inside it -
    this is the piece that was missing before: picking a pack used to
    be blind, with no way to see what's actually in it."""
    cfg = manager.config
    if not cfg.providers:
        body = "[dim](no packs configured yet - try 'add pack <name>')[/dim]"
    else:
        body = "\n\n".join(
            _format_pack(name, pack, color=pack_color(i))
            for i, (name, pack) in enumerate(cfg.providers.items())
        )
    return render_box("Packs", body)


def handle(manager: "ConfigManager", raw: str) -> Any:
    """Execute one settings-stage command and return a Rich renderable.

    Never raises: any `ConfigError` (bad pack/model/path, invalid
    value, etc.) is caught and rendered as an error panel instead of
    crashing the app.
    """
    parts = raw.strip().split()
    if not parts:
        return render_box("Settings", "[dim]type 'help' for commands[/dim]")

    cmd = parts[0].lower()

    try:
        if cmd == "help":
            return render_box("Settings help", HELP_TEXT)

        if cmd == "packs":
            return render_packs_overview(manager)

        if cmd == "pack" and len(parts) >= 2:
            name = parts[1]
            pack = manager.packs.get_pack(name)
            index = list(manager.config.providers.keys()).index(name) if name in manager.config.providers else 0
            return render_box(f"Pack: {name}", _format_pack(name, pack, color=pack_color(index)))

        if cmd == "suggest":
            return render_box("Provider / model suggestions", suggestions_text())

        if cmd == "add" and len(parts) >= 3 and parts[1].lower() == "pack":
            name = parts[2]
            manager.packs.add_pack(name)
            manager.save()
            return render_box(
                "Pack added",
                f"[green]✓[/green] created empty pack '{name}'.\n"
                f"Next: [bold]add model {name} <provider> <model>[/bold] (see 'suggest' for ideas).",
            )

        if cmd == "remove" and len(parts) >= 3 and parts[1].lower() == "pack":
            name = parts[2]
            confirmed = len(parts) >= 4 and parts[3].lower() == "confirm"
            if not confirmed:
                return render_box(
                    "Confirm delete",
                    f"This permanently deletes pack '{name}' and every model inside it.\n"
                    f"To confirm, run: [bold]remove pack {name} confirm[/bold]",
                    style="yellow",
                )
            manager.backup()
            manager.packs.remove_pack(name)
            manager.save()
            return render_box(
                "Pack removed",
                f"[green]✓[/green] removed pack '{name}'.\n"
                "[dim]A backup was taken first - run 'restore' if this was a mistake.[/dim]",
            )

        if cmd == "add" and len(parts) >= 5 and parts[1].lower() == "model":
            pack, provider, model = parts[2], parts[3], parts[4]
            target = "fallback" if len(parts) >= 6 and parts[5].lower() == "fallback" else "pool"
            manager.packs.add_model(pack, provider, model, target=target)
            manager.save()
            return render_box("Model added", f"[green]✓[/green] added {provider}/{model} to '{pack}' ({target}).")

        if cmd == "remove" and len(parts) >= 5 and parts[1].lower() == "model":
            pack, provider, model = parts[2], parts[3], parts[4]
            confirmed = parts[-1].lower() == "confirm"
            tail = parts[5:-1] if confirmed else parts[5:]
            target = "fallback" if tail and tail[0].lower() == "fallback" else "pool"
            if not confirmed:
                suffix = " fallback" if target == "fallback" else ""
                return render_box(
                    "Confirm delete",
                    f"This removes {provider}/{model} from '{pack}' ({target}).\n"
                    f"To confirm, run: [bold]remove model {pack} {provider} {model}{suffix} confirm[/bold]",
                    style="yellow",
                )
            manager.backup()
            manager.packs.remove_model(pack, provider, model, target=target)
            manager.save()
            return render_box(
                "Model removed",
                f"[green]✓[/green] removed {provider}/{model} from '{pack}' ({target}).\n"
                "[dim]A backup was taken first - run 'restore' if this was a mistake.[/dim]",
            )

        if cmd == "set" and len(parts) >= 3:
            path = parts[1]
            value = " ".join(parts[2:])
            manager.set(path, _coerce(value))
            manager.save()
            return render_box("Updated", f"[green]✓[/green] {path} = {value}")

        if cmd == "get" and len(parts) >= 2:
            path = parts[1]
            value = manager.get(path)
            return render_box("Value", f"{path} = {value}")

        if cmd == "backup":
            path = manager.backup()
            return render_box("Backup created", f"[green]✓[/green] {path}")

        if cmd == "backups":
            return render_box("Backups", _list_backups(manager))

        if cmd == "restore":
            target = parts[1] if len(parts) >= 2 else "latest"
            return _restore(manager, target)

        if cmd == "validate":
            manager.validate()
            return render_box("Valid", "[green]✓[/green] global_config.yaml matches the schema.")

        return render_box("Unknown command", f"[red]unrecognized: '{raw}'[/red]\n\n{HELP_TEXT}")

    except ConfigError as exc:
        return render_box("Error", f"[red]{exc}[/red]", style="red")


def _backup_dir(manager: "ConfigManager") -> "Any":
    return manager.config_path.parent / "backups"


def _list_backups(manager: "ConfigManager") -> str:
    """Lists what `manager.backup()` has already written to disk
    (`<config_dir>/backups/`), newest first. `ConfigManager` has no
    listing method of its own - it only knows how to create one backup
    at a time - so this reads the directory directly the same way any
    other file-listing command would."""
    backup_dir = _backup_dir(manager)
    if not backup_dir.exists():
        return "[dim](no backups yet - 'backup' creates one, and pack/model removal takes one automatically)[/dim]"
    files = sorted(backup_dir.glob("*.yaml"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not files:
        return "[dim](no backups yet)[/dim]"
    lines = [f"  {f.name}" for f in files]
    lines.append("\n[dim]restore latest[/dim]  or  [dim]restore <filename>[/dim]")
    return "[bold]Available backups (newest first)[/bold]\n" + "\n".join(lines)


def _restore(manager: "ConfigManager", target: str) -> Any:
    """Copies a previously-created backup back over the active
    `global_config.yaml` and reloads it. Takes a fresh backup of the
    *current* (about-to-be-overwritten) file first, so restoring is
    itself undoable rather than a one-way door."""
    backup_dir = _backup_dir(manager)
    files = sorted(backup_dir.glob("*.yaml"), key=lambda p: p.stat().st_mtime, reverse=True) if backup_dir.exists() else []
    if not files:
        return render_box("Restore", "[red]no backups found.[/red]")

    if target.lower() == "latest":
        chosen = files[0]
    else:
        matches = [f for f in files if f.name == target]
        if not matches:
            return render_box(
                "Restore",
                f"[red]no backup named '{target}'.[/red]\n\n{_list_backups(manager)}",
            )
        chosen = matches[0]

    manager.backup()  # snapshot the current file before overwriting it
    shutil.copy2(chosen, manager.config_path)
    manager.reload()
    return render_box(
        "Restored",
        f"[green]✓[/green] restored config from {chosen.name}.\n"
        "[dim]The config that was active before this restore was itself backed up first.[/dim]",
    )


def _coerce(value: str) -> Any:
    """Best-effort string->bool/int/float coercion for `set` values;
    falls back to the raw string (e.g. for paths or provider names)."""
    low = value.lower()
    if low in ("true", "false"):
        return low == "true"
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value
