"""
Unit tests for `controller/pack_wizard.py`. `PackWizard` only touches
`config_manager.packs`/`config_manager.save()` inside `commit()` -
everything before that is pure in-memory state transitions, so a tiny
fake config manager is enough to exercise the whole flow without any
real project dependency.
"""

from __future__ import annotations

import pytest

from tesseractcli.controller.pack_wizard import PackWizard, WizardState
from tesseractcli.models.exceptions import ConfigError


class FakePacksManager:
    def __init__(self, existing_packs: list[str] | None = None) -> None:
        self.packs: dict[str, list[tuple[str, str, str]]] = {
            name: [] for name in (existing_packs or [])
        }
        self.add_pack_calls: list[str] = []
        self.add_model_calls: list[tuple[str, str, str, str]] = []
        self.fail_add_model_with: Exception | None = None

    def list_packs(self) -> list[str]:
        return list(self.packs)

    def add_pack(self, name: str) -> None:
        self.add_pack_calls.append(name)
        self.packs.setdefault(name, [])

    def add_model(self, pack: str, provider: str, model: str, *, target: str) -> None:
        if self.fail_add_model_with is not None:
            raise self.fail_add_model_with
        self.add_model_calls.append((pack, provider, model, target))
        self.packs.setdefault(pack, []).append((provider, model, target))


class FakeConfigManager:
    def __init__(self, existing_packs: list[str] | None = None) -> None:
        self.packs = FakePacksManager(existing_packs)
        self.save_calls = 0

    def save(self) -> None:
        self.save_calls += 1


@pytest.fixture
def config_manager() -> FakeConfigManager:
    return FakeConfigManager(existing_packs=["main_pack", "second_pack"])


@pytest.fixture
def wizard(config_manager: FakeConfigManager) -> PackWizard:
    return PackWizard(config_manager)


# ---------------------------------------------------------------------
# start()
# ---------------------------------------------------------------------


def test_start_lists_every_provider_in_the_catalog(wizard: PackWizard):
    prompt = wizard.start()
    assert prompt.state.step == "provider"
    assert {c.value for c in prompt.choices} == {
        "groq",
        "cerebras",
        "mistral",
        "huggingface",
        "anthropic",
        "together",
        "openai",
        "cohere",
        "openrouter",
        "github_models",
        "local_gguf",
    }


def test_start_carries_activate_and_return_stage_through(wizard: PackWizard):
    prompt = wizard.start(activate=True, return_stage="chat")
    assert prompt.state.activate is True
    assert prompt.state.return_stage == "chat"


def test_wizard_state_is_immutable_between_steps(wizard: PackWizard):
    prompt1 = wizard.start()
    prompt2 = wizard.choose_provider(prompt1.state, "groq")
    # the original state object handed to choose_provider must be
    # untouched - a real bug class this design specifically prevents
    # (the old `self._wiz` dict could be mutated by any method).
    assert prompt1.state.step == "provider"
    assert prompt1.state.provider is None
    assert prompt2.state.step == "model"
    assert prompt2.state.provider == "groq"


# ---------------------------------------------------------------------
# provider -> model
# ---------------------------------------------------------------------


def test_choose_provider_offers_its_example_models_plus_custom(wizard: PackWizard):
    prompt = wizard.start()
    prompt = wizard.choose_provider(prompt.state, "groq")
    values = {c.value for c in prompt.choices}
    assert "llama-3.3-70b-versatile" in values
    assert "__custom__" in values


def test_choose_provider_with_unknown_provider_raises():
    # PROVIDER_CATALOG lookup uses next(...) with no default - an
    # unknown provider id is a programming error (the View can only
    # ever pass an id it got from our own choices), so this should
    # blow up loudly rather than silently produce a broken state.
    w = PackWizard(FakeConfigManager())
    prompt = w.start()
    with pytest.raises(StopIteration):
        w.choose_provider(prompt.state, "does-not-exist")


# ---------------------------------------------------------------------
# model -> pack (including the custom-model detour)
# ---------------------------------------------------------------------


def test_choose_model_from_catalog_advances_straight_to_pack_step(wizard: PackWizard):
    prompt = wizard.start()
    prompt = wizard.choose_provider(prompt.state, "groq")
    prompt = wizard.choose_model(prompt.state, "kimi-k2")
    assert prompt.state.step == "pack"
    assert prompt.state.model == "kimi-k2"


def test_choose_model_custom_detours_to_free_text_step(wizard: PackWizard):
    prompt = wizard.start()
    prompt = wizard.choose_provider(prompt.state, "groq")
    prompt = wizard.choose_model(prompt.state, "__custom__")
    assert prompt.state.step == "model_custom"
    assert prompt.choices is None  # signals "needs free text" to the caller
    assert prompt.state.model is None  # not set yet


def test_submit_custom_model_advances_to_pack_step(wizard: PackWizard):
    prompt = wizard.start()
    prompt = wizard.choose_provider(prompt.state, "groq")
    prompt = wizard.choose_model(prompt.state, "__custom__")
    prompt = wizard.submit_custom_model(prompt.state, "gpt-4o-mini")
    assert prompt.state.step == "pack"
    assert prompt.state.model == "gpt-4o-mini"


# ---------------------------------------------------------------------
# pack -> target (including the new-pack detour)
# ---------------------------------------------------------------------


def test_pack_step_lists_existing_packs_plus_new(wizard: PackWizard):
    prompt = wizard.start()
    prompt = wizard.choose_provider(prompt.state, "groq")
    prompt = wizard.choose_model(prompt.state, "kimi-k2")
    values = {c.value for c in prompt.choices}
    assert values == {"main_pack", "second_pack", "__new__"}


def test_choose_existing_pack_advances_to_target(wizard: PackWizard):
    prompt = wizard.start()
    prompt = wizard.choose_provider(prompt.state, "groq")
    prompt = wizard.choose_model(prompt.state, "kimi-k2")
    prompt = wizard.choose_pack(prompt.state, "main_pack")
    assert prompt.state.step == "target"
    assert prompt.state.pack == "main_pack"
    assert prompt.state.pack_is_new is False


def test_choose_new_pack_detours_to_free_text(wizard: PackWizard):
    prompt = wizard.start()
    prompt = wizard.choose_provider(prompt.state, "groq")
    prompt = wizard.choose_model(prompt.state, "kimi-k2")
    prompt = wizard.choose_pack(prompt.state, "__new__")
    assert prompt.state.step == "pack_new"
    assert prompt.choices is None


def test_submit_new_pack_name_advances_to_target_and_flags_pack_is_new(
    wizard: PackWizard,
):
    prompt = wizard.start()
    prompt = wizard.choose_provider(prompt.state, "groq")
    prompt = wizard.choose_model(prompt.state, "kimi-k2")
    prompt = wizard.choose_pack(prompt.state, "__new__")
    prompt = wizard.submit_new_pack_name(prompt.state, "third_pack")
    assert prompt.state.step == "target"
    assert prompt.state.pack == "third_pack"
    assert prompt.state.pack_is_new is True


# ---------------------------------------------------------------------
# target -> commit
# ---------------------------------------------------------------------


def test_choose_target_returns_final_state_without_committing(
    wizard: PackWizard, config_manager: FakeConfigManager
):
    prompt = wizard.start()
    prompt = wizard.choose_provider(prompt.state, "groq")
    prompt = wizard.choose_model(prompt.state, "kimi-k2")
    prompt = wizard.choose_pack(prompt.state, "main_pack")
    final_state = wizard.choose_target(prompt.state, "pool")
    assert final_state.target == "pool"
    assert config_manager.packs.add_model_calls == []  # nothing written yet


def test_commit_existing_pack_writes_model_only(
    wizard: PackWizard, config_manager: FakeConfigManager
):
    state = WizardState(
        step="target",
        provider="groq",
        model="kimi-k2",
        pack="main_pack",
        pack_is_new=False,
        target="pool",
    )
    result = wizard.commit(state)
    assert result.ok is True
    assert config_manager.packs.add_pack_calls == []
    assert config_manager.packs.add_model_calls == [
        ("main_pack", "groq", "kimi-k2", "pool")
    ]
    assert config_manager.save_calls == 1


def test_commit_new_pack_creates_pack_before_adding_model(
    wizard: PackWizard, config_manager: FakeConfigManager
):
    state = WizardState(
        step="target",
        provider="groq",
        model="kimi-k2",
        pack="third_pack",
        pack_is_new=True,
        target="fallback",
    )
    result = wizard.commit(state)
    assert result.ok is True
    assert config_manager.packs.add_pack_calls == ["third_pack"]
    assert config_manager.packs.add_model_calls == [
        ("third_pack", "groq", "kimi-k2", "fallback")
    ]


def test_commit_carries_activate_and_return_stage_into_the_result(
    wizard: PackWizard, config_manager: FakeConfigManager
):
    state = WizardState(
        step="target",
        provider="groq",
        model="kimi-k2",
        pack="main_pack",
        pack_is_new=False,
        target="pool",
        activate=True,
        return_stage="chat",
    )
    result = wizard.commit(state)
    assert result.activate is True
    assert result.return_stage == "chat"


def test_commit_failure_does_not_save_and_returns_error(
    wizard: PackWizard, config_manager: FakeConfigManager
):
    config_manager.packs.fail_add_model_with = ConfigError(
        "model already exists in this pack"
    )
    state = WizardState(
        step="target",
        provider="groq",
        model="kimi-k2",
        pack="main_pack",
        pack_is_new=False,
        target="pool",
        return_stage="settings",
    )
    result = wizard.commit(state)
    assert result.ok is False
    assert result.error == "model already exists in this pack"
    assert result.return_stage == "settings"
    assert config_manager.save_calls == 0


def test_commit_requires_provider_model_pack_target(wizard: PackWizard):
    incomplete = WizardState(step="target")
    with pytest.raises(AssertionError):
        wizard.commit(incomplete)
