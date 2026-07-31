"""
tesseractcli/ui/views/model_picker.py

`PackChoice`/`load_pack_choices` back the inline `OptionList` mounted
by `TesseractApp.show_model_picker` (see `ui/app.py`).

Previously `load_pack_choices()` was a STUB returning three hardcoded
pack names (main_pack/second_pack/third_pack) because there was no
live "list configured packs" call available. That's fixed now:
`ConfigManager` is loaded and `.config.providers` is real, so this
builds the option list straight from `global_config.yaml` - whatever
packs actually exist, including ones added/removed at runtime through
the settings command grammar (`settings_commands.py`).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tesseractcli.config.global_config.manager import ConfigManager


@dataclass
class PackChoice:
    name: str
    description: str

    @property
    def label(self) -> str:
        return f"{self.name} — {self.description}"


def load_pack_choices(manager: ConfigManager) -> list[PackChoice]:
    """Build the picker's option list from the live global_config.yaml.

    The description previews the primary model (plus a count of any
    others) so a pack is no longer a blind name - you can see what's
    actually in it before selecting it.
    """
    cfg = manager.config
    choices: list[PackChoice] = []
    for name, pack in cfg.providers.items():
        if pack.pool:
            primary = f"{pack.pool[0].provider}/{pack.pool[0].model}"
            extra = f" +{len(pack.pool) - 1} more" if len(pack.pool) > 1 else ""
            description = f"{primary}{extra}"
        else:
            description = "empty pool"
        choices.append(PackChoice(name, description))
    return choices
