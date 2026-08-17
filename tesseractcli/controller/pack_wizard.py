"""
tesseractcli/controller/pack_wizard.py

The provider -> model -> pack -> pool/fallback "suggest" wizard as an
isolated, typed state machine. Replaces `TesseractApp._wiz: dict[str,
Any]` and the five `_wiz_*` methods that used to read/write it directly
off `self`. Nothing here touches a widget - every step method returns
plain data (the next `WizardState`, or a list of choices to present),
and `PackWizard` itself is stateless between calls: the `SessionController`
holds the current `WizardState` and passes it back in on every step.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Literal

from tesseractcli.config.provider_catalog import PROVIDER_CATALOG
from tesseractcli.models.exceptions import ConfigError

WizardStep = Literal[
    "provider", "model", "model_custom", "pack", "pack_new", "target"
]


@dataclass(frozen=True)
class WizardState:
    step: WizardStep
    provider: str | None = None
    model: str | None = None
    pack: str | None = None
    pack_is_new: bool = False
    target: Literal["pool", "fallback"] | None = None
    activate: bool = False
    return_stage: str = "settings"


@dataclass(frozen=True)
class WizardChoice:
    """One option to present to the user for the current step - maps
    1:1 onto a Textual `Option(label, id=value)` at the View layer,
    without this module knowing that."""

    label: str
    value: str


@dataclass(frozen=True)
class WizardPrompt:
    """What the Controller hands back to the View after a step advance:
    either a fresh set of choices to show, or a request for free text."""

    state: WizardState
    prompt: str
    choices: list[WizardChoice] | None = None  # None => needs free text
    free_text_hint: str | None = None


@dataclass(frozen=True)
class WizardResult:
    """Outcome of `PackWizard.commit()` - either the pack/model was
    saved (and `activate` says whether an activate-confirm should
    follow), or it failed with a `ConfigError` message to show."""

    ok: bool
    provider: str = ""
    model: str = ""
    pack: str = ""
    target: str = ""
    activate: bool = False
    return_stage: str = "settings"
    error: str | None = None


class PackWizard:
    """Pure orchestration: talks to `ConfigManager.packs` only inside
    `commit()`, everything else is in-memory state transitions."""

    def __init__(self, config_manager) -> None:  # type: ignore[no-untyped-def]
        self._config_manager = config_manager

    def start(self, *, activate: bool = False, return_stage: str = "settings") -> WizardPrompt:
        state = WizardState(step="provider", activate=activate, return_stage=return_stage)
        choices = [
            WizardChoice(
                label=f"{e.provider:<14} {e.label} — e.g. {', '.join(e.example_models)}",
                value=e.provider,
            )
            for e in PROVIDER_CATALOG
        ]
        return WizardPrompt(state=state, prompt="Pick a provider", choices=choices)

    def choose_provider(self, state: WizardState, provider: str) -> WizardPrompt:
        state = replace(state, step="model", provider=provider)
        entry = next(e for e in PROVIDER_CATALOG if e.provider == provider)
        choices = [WizardChoice(label=m, value=m) for m in entry.example_models]
        choices.append(WizardChoice(label="type a different model id…", value="__custom__"))
        return WizardPrompt(
            state=state, prompt=f"Pick a model ({entry.label})", choices=choices
        )

    def choose_model(self, state: WizardState, model_id: str) -> WizardPrompt:
        if model_id == "__custom__":
            state = replace(state, step="model_custom")
            return WizardPrompt(
                state=state,
                prompt="Type the model id",
                choices=None,
                free_text_hint="e.g. gpt-4o-mini",
            )
        return self._advance_to_pack_step(replace(state, model=model_id))

    def submit_custom_model(self, state: WizardState, model_id: str) -> WizardPrompt:
        return self._advance_to_pack_step(replace(state, model=model_id))

    def _advance_to_pack_step(self, state: WizardState) -> WizardPrompt:
        state = replace(state, step="pack")
        pack_names = self._config_manager.packs.list_packs()
        choices = [WizardChoice(label=name, value=name) for name in pack_names]
        choices.append(WizardChoice(label="+ create a new pack…", value="__new__"))
        return WizardPrompt(state=state, prompt="Add to which pack?", choices=choices)

    def choose_pack(self, state: WizardState, pack_id: str) -> WizardPrompt:
        if pack_id == "__new__":
            state = replace(state, step="pack_new")
            return WizardPrompt(
                state=state, prompt="Type the new pack's name", choices=None
            )
        return self._advance_to_target_step(
            replace(state, pack=pack_id, pack_is_new=False)
        )

    def submit_new_pack_name(self, state: WizardState, name: str) -> WizardPrompt:
        return self._advance_to_target_step(replace(state, pack=name, pack_is_new=True))

    def _advance_to_target_step(self, state: WizardState) -> WizardPrompt:
        state = replace(state, step="target")
        choices = [
            WizardChoice(label="pool (primary)", value="pool"),
            WizardChoice(label="fallback", value="fallback"),
        ]
        return WizardPrompt(
            state=state,
            prompt=f"Pool or fallback for '{state.pack}'?",
            choices=choices,
        )

    def choose_target(self, state: WizardState, target: str) -> WizardState:
        """Last step before commit - returns the final state, the
        Controller calls `commit()` with it."""
        return replace(state, target=target)  # type: ignore[arg-type]

    def commit(self, state: WizardState) -> WizardResult:
        assert state.provider and state.model and state.pack and state.target
        try:
            if state.pack_is_new:
                self._config_manager.packs.add_pack(state.pack)
            self._config_manager.packs.add_model(
                state.pack, state.provider, state.model, target=state.target
            )
            self._config_manager.save()
        except ConfigError as exc:
            return WizardResult(ok=False, return_stage=state.return_stage, error=str(exc))
        return WizardResult(
            ok=True,
            provider=state.provider,
            model=state.model,
            pack=state.pack,
            target=state.target,
            activate=state.activate,
            return_stage=state.return_stage,
        )
