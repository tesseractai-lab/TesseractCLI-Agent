"""
Unit tests for `controller/session.py`.

`SessionController` is constructed with fakes for everything it depends
on (`config_manager`, `dispatcher`, `tool_registry`) - none of them touch
a real file or a real LLM, and nothing here imports Textual.

Every test drives the controller purely through `Command` objects
(the same ones `commands.parser.parse` produces) and asserts on the
`list[RenderInstruction]` it returns plus the resulting `stage`/state.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tesseractcli.commands.types import (
    ApprovalInput,
    ChatMessage,
    ConfirmInput,
    InlineModelSwitch,
    InlineWorkspaceSwitch,
    MetaCommand,
    NavCommand,
    ReloadCommand,
    ResetCommand,
    SettingsCommand,
    StartWizard,
    UnknownInput,
    WizardFreeText,
    WorkspacePathInput,
)
from tesseractcli.controller.session import SessionController
from tesseractcli.models.config_models.provider_models import (
    ModelConfig,
    ModelPack,
)
from tesseractcli.models.exceptions import ConfigError

# ---------------------------------------------------------------------
# fakes
# ---------------------------------------------------------------------


class FakeVerbose:
    def __init__(self, errors: bool = False) -> None:
        self.errors = errors


class FakeConfig:
    def __init__(
        self,
        providers: dict[str, ModelPack] | None = None,
        verbose_errors: bool = False,
    ) -> None:
        self.providers = providers or {
            "main_pack": ModelPack(),
        }
        self.verbose = FakeVerbose(errors=verbose_errors)


class FakePacksManager:
    def __init__(self, config: FakeConfig) -> None:
        self._config = config

    def list_packs(self) -> list[str]:
        return list(self._config.providers)

    def get_pack(self, name: str) -> ModelPack:
        if name not in self._config.providers:
            raise ConfigError(f"no such pack: {name}")

        return self._config.providers[name]

    def add_pack(self, name: str) -> None:
        if name in self._config.providers:
            raise ConfigError(f"pack already exists: {name}")

        self._config.providers[name] = ModelPack()

    def add_model(
        self,
        pack: str,
        provider: str,
        model: str,
        *,
        target: str,
    ) -> None:
        pack_obj = self._config.providers.setdefault(
            pack,
            ModelPack(),
        )

        getattr(pack_obj, target).append(
            ModelConfig(
                provider=provider,
                model=model,
            )
        )

    def remove_pack(self, name: str) -> None:
        if name not in self._config.providers:
            raise ConfigError(f"no such pack: {name}")

        del self._config.providers[name]

    def rename_pack(self, old_name: str, new_name: str) -> None:
        if old_name not in self._config.providers:
            raise ConfigError(f"no such pack: {old_name}")

        if new_name in self._config.providers:
            raise ConfigError(f"pack already exists: {new_name}")

        self._config.providers[new_name] = self._config.providers.pop(old_name)


class FakeConfigManager:
    def __init__(
        self,
        providers: dict[str, ModelPack] | None = None,
        verbose_errors: bool = False,
    ) -> None:
        self.config = FakeConfig(
            providers=providers,
            verbose_errors=verbose_errors,
        )

        self.packs = FakePacksManager(self.config)

        self.save_calls = 0
        self.backup_calls = 0
        self.reload_calls = 0
        self.reset_calls = 0

        self.fail_reload_with: Exception | None = None

    def save(self) -> None:
        self.save_calls += 1

    def backup(self) -> None:
        self.backup_calls += 1

    def reload(self) -> None:
        self.reload_calls += 1

        if self.fail_reload_with is not None:
            raise self.fail_reload_with

    def reset(self, *, keep_backup: bool = True) -> None:
        self.reset_calls += 1

        self.config = FakeConfig()
        self.packs = FakePacksManager(self.config)


class FakeDispatcher:
    pass


class FakeToolRegistry:
    pass


# ---------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------


@pytest.fixture
def config_manager() -> FakeConfigManager:
    return FakeConfigManager(
        providers={
            "main_pack": ModelPack(),
            "second_pack": ModelPack(),
        }
    )


@pytest.fixture
def controller(
    config_manager: FakeConfigManager,
) -> SessionController:
    ctl = SessionController(
        config_manager=config_manager,  # type: ignore[arg-type]
        dispatcher=FakeDispatcher(),  # type: ignore[arg-type]
        tool_registry=FakeToolRegistry(),  # type: ignore[arg-type]
    )

    ctl.stage = "chat"
    ctl.workspace_root = Path("/tmp/workspace")

    return ctl


def kinds(instrs) -> list[str]:
    return [i.kind for i in instrs]


# ---------------------------------------------------------------------
# navigation
# ---------------------------------------------------------------------


def test_nav_chat_settings_home(controller: SessionController):
    out = controller.handle(NavCommand(target="settings"))

    assert controller.stage == "settings"
    assert kinds(out) == ["divider", "render_settings"]

    out = controller.handle(NavCommand(target="home"))

    assert controller.stage == "home"
    assert kinds(out) == ["divider", "render_home"]

    out = controller.handle(NavCommand(target="chat"))

    assert controller.stage == "chat"
    assert kinds(out) == ["divider"]


def test_nav_exit_emits_exit_instruction(
    controller: SessionController,
):
    out = controller.handle(NavCommand(target="exit"))

    assert kinds(out) == ["exit"]


def test_nav_model_opens_picker_and_remembers_return_stage(
    controller: SessionController,
):
    controller.stage = "settings"

    out = controller.handle(NavCommand(target="model"))

    assert controller.stage == "model_pick"
    assert controller._pack_return_stage == "settings"

    assert out[0].kind == "mount_options"

    values = {option["value"] for option in out[0].content["options"]}

    assert {
        "main_pack",
        "second_pack",
        "__add_pack__",
        "__cancel__",
    } <= values


def test_nav_workspace_enters_edit_stage(
    controller: SessionController,
):
    out = controller.handle(NavCommand(target="workspace"))

    assert controller.stage == "workspace_edit"
    assert kinds(out) == [
        "divider",
        "log",
        "log",
    ]


# ---------------------------------------------------------------------
# settings dispatch
# ---------------------------------------------------------------------


def test_settings_command_echoes_original_text_not_resolved(
    controller: SessionController,
):
    out = controller.handle(
        SettingsCommand(
            raw_text="packs",
            original_text="p",
        )
    )

    assert out[0].kind == "echo"
    assert out[0].content["text"] == "p"


def test_settings_add_pack_activates_it(
    controller: SessionController,
):
    controller.config_manager.config.providers["brand_new"] = ModelPack()

    out = controller.handle(
        SettingsCommand(
            raw_text="add pack brand_new",
            original_text="add pack brand_new",
        )
    )

    assert controller.selected_pack == "brand_new"
    assert any(i.kind == "mode_line" for i in out)


def test_settings_remove_pack_goes_through_confirm_flow_not_direct(
    controller: SessionController,
):
    out = controller.handle(
        SettingsCommand(
            raw_text="remove pack main_pack",
            original_text="rm pack main_pack",
        )
    )

    assert controller.stage == "awaiting_remove_pack_confirm"
    assert out[-1].kind == "box"
    assert "main_pack" in out[-1].content["body"]


def test_settings_remove_unknown_pack_shows_error_box(
    controller: SessionController,
):
    out = controller.handle(
        SettingsCommand(
            raw_text="remove pack ghost",
            original_text="rm pack ghost",
        )
    )

    assert out[-1].kind == "box"
    assert "ghost" in out[-1].content["body"]
    assert controller.stage == "chat"


def test_settings_rename_pack_updates_selected_pack_if_it_was_active(
    controller: SessionController,
):
    controller.selected_pack = "main_pack"

    out = controller.handle(
        SettingsCommand(
            raw_text="rename pack main_pack renamed_pack",
            original_text="rn pack main_pack renamed_pack",
        )
    )

    assert controller.selected_pack == "renamed_pack"

    assert "renamed_pack" in controller.config_manager.config.providers

    assert "main_pack" not in controller.config_manager.config.providers

    assert any(i.kind == "mode_line" for i in out)


# ---------------------------------------------------------------------
# remove-pack confirm flow, end to end
# ---------------------------------------------------------------------


def test_remove_pack_confirm_yes_deletes_and_backs_up(
    controller: SessionController,
):
    controller.handle(
        SettingsCommand(
            raw_text="remove pack main_pack",
            original_text="rm pack main_pack",
        )
    )

    assert controller.stage == "awaiting_remove_pack_confirm"

    out = controller.handle(
        ConfirmInput(
            confirmed=True,
            text="y",
            stage="awaiting_remove_pack_confirm",
        )
    )

    assert "main_pack" not in controller.config_manager.config.providers

    assert controller.config_manager.backup_calls == 1
    assert controller.config_manager.save_calls == 1
    assert controller.stage == "chat"

    assert any(i.kind == "mode_line" for i in out)


def test_remove_pack_confirm_no_leaves_pack_intact(
    controller: SessionController,
):
    controller.handle(
        SettingsCommand(
            raw_text="remove pack main_pack",
            original_text="rm pack main_pack",
        )
    )

    out = controller.handle(
        ConfirmInput(
            confirmed=False,
            text="n",
            stage="awaiting_remove_pack_confirm",
        )
    )

    assert "main_pack" in controller.config_manager.config.providers

    assert controller.config_manager.backup_calls == 0
    assert "cancelled" in out[1].content.lower()


def test_settings_remove_pack_active_reroutes_to_picker(
    controller: SessionController,
):
    controller.selected_pack = "main_pack"

    controller.handle(
        SettingsCommand(
            raw_text="remove pack main_pack",
            original_text="rm pack main_pack",
        )
    )

    controller.handle(
        ConfirmInput(
            confirmed=True,
            text="y",
            stage="awaiting_remove_pack_confirm",
        )
    )

    assert "main_pack" not in controller.config_manager.config.providers


# ---------------------------------------------------------------------
# inline model / workspace switch
# ---------------------------------------------------------------------


def test_inline_model_switch_with_known_pack(
    controller: SessionController,
):
    out = controller.handle(
        InlineModelSwitch(
            args=("second_pack",),
            text="-cfg model second_pack",
        )
    )

    assert controller.selected_pack == "second_pack"
    assert out[0].kind == "echo"
    assert out[0].content["text"] == "-cfg model second_pack"


def test_inline_model_switch_with_unknown_pack_shows_error(
    controller: SessionController,
):
    out = controller.handle(
        InlineModelSwitch(
            args=("ghost",),
            text="-cfg model ghost",
        )
    )

    assert controller.selected_pack is None
    assert "no such pack" in out[-1].content


def test_inline_model_switch_no_args_opens_picker(
    controller: SessionController,
):
    out = controller.handle(
        InlineModelSwitch(
            args=(),
            text="-cfg model",
        )
    )

    assert controller.stage == "model_pick"
    assert out[-1].kind == "mount_options"


def test_inline_workspace_switch_valid_path(
    controller: SessionController,
    tmp_path,
):
    out = controller.handle(
        InlineWorkspaceSwitch(
            args=(str(tmp_path),),
            text=f"-cfg -ws {tmp_path}",
        )
    )

    assert controller.workspace_root == tmp_path.resolve()

    assert any("workspace set" in i.content for i in out if i.kind == "log")


def test_inline_workspace_switch_invalid_path(
    controller: SessionController,
):
    out = controller.handle(
        InlineWorkspaceSwitch(
            args=("/no/such/dir",),
            text="-cfg -ws /no/such/dir",
        )
    )

    assert "not a valid directory" in out[-1].content


# ---------------------------------------------------------------------
# reset flow
# ---------------------------------------------------------------------


def test_reset_confirm_yes_clears_temp_messages(
    controller: SessionController,
):
    controller.messages.append("m1")
    controller.messages.append("m2")

    out = controller.handle(
        ResetCommand(
            scope=None,
            text="reset",
        )
    )

    assert controller.stage == "awaiting_reset_confirm"

    assert kinds(out) == [
        "echo",
        "divider",
        "box",
    ]

    out = controller.handle(
        ConfirmInput(
            confirmed=True,
            text="y",
            stage="awaiting_reset_confirm",
        )
    )

    assert len(controller.messages) == 0
    assert controller.stage == "chat"
    assert "cleared" in out[1].content.lower()


def test_reset_confirm_no_keeps_messages(
    controller: SessionController,
):
    controller.messages.append("m1")

    controller.handle(
        ResetCommand(
            scope=None,
            text="reset",
        )
    )

    controller.handle(
        ConfirmInput(
            confirmed=False,
            text="n",
            stage="awaiting_reset_confirm",
        )
    )

    assert len(controller.messages) == 1


def test_reset_unknown_scope_shows_error_and_does_not_change_stage(
    controller: SessionController,
):
    out = controller.handle(
        ResetCommand(
            scope="bogus",
            text="reset bogus",
        )
    )

    assert controller.stage == "chat"
    assert out[-1].kind == "box"

    assert "Unknown reset target" in out[-1].content["title"]


# ---------------------------------------------------------------------
# reload flow
# ---------------------------------------------------------------------


def test_reload_success(
    controller: SessionController,
):
    out = controller.handle(
        ReloadCommand(
            scope=None,
            text="reload",
        )
    )

    assert controller.config_manager.reload_calls == 1

    assert any("reloaded" in i.content for i in out if i.kind == "log")


def test_reload_failure_shows_error_and_keeps_old_config(
    controller: SessionController,
):
    controller.config_manager.fail_reload_with = ConfigError("bad yaml")

    out = controller.handle(
        ReloadCommand(
            scope=None,
            text="reload",
        )
    )

    assert out[0].kind == "echo"
    assert out[1].kind == "box"
    assert "bad yaml" in out[1].content["body"]

    assert any(
        "Kept the previously loaded" in i.content for i in out if i.kind == "log"
    )


def test_reload_unknown_scope(
    controller: SessionController,
):
    out = controller.handle(
        ReloadCommand(
            scope="bogus",
            text="reload bogus",
        )
    )

    assert controller.config_manager.reload_calls == 0

    assert "Unknown reload target" in out[-1].content["title"]


# ---------------------------------------------------------------------
# meta commands
# ---------------------------------------------------------------------


def test_meta_help(
    controller: SessionController,
):
    out = controller.handle(MetaCommand(kind="help"))

    assert kinds(out) == [
        "log",
        "render_help",
    ]


def test_meta_copy_with_no_reply_yet(
    controller: SessionController,
):
    out = controller.handle(MetaCommand(kind="copy"))

    assert "nothing to copy" in out[0].content.lower()


def test_meta_copy_with_reply(
    controller: SessionController,
):
    controller._last_agent_reply = "here's your answer"

    out = controller.handle(MetaCommand(kind="copy"))

    assert out[0].kind == "copy_to_clipboard"
    assert out[0].content == "here's your answer"


def test_meta_expand_with_nothing_truncated(
    controller: SessionController,
):
    out = controller.handle(MetaCommand(kind="expand"))

    assert "Nothing truncated" in out[0].content


def test_meta_expand_after_truncated_echo(
    controller: SessionController,
):
    long_text = "line\n" * 20

    controller.echo_text(long_text)

    out = controller.handle(MetaCommand(kind="expand"))

    assert out[0].kind == "box"
    assert out[0].content["body"] == long_text


def test_meta_clear(
    controller: SessionController,
):
    out = controller.handle(MetaCommand(kind="clear"))

    assert kinds(out) == [
        "clear",
        "mode_line",
    ]


def test_meta_packs(
    controller: SessionController,
):
    from io import StringIO

    from rich.console import Console

    out = controller.handle(MetaCommand(kind="packs"))

    assert out[0].kind == "log"

    buffer = StringIO()
    console = Console(
        file=buffer,
        force_terminal=False,
        color_system=None,
    )

    console.print(out[0].content)

    rendered = buffer.getvalue()

    assert "main_pack" in rendered


# ---------------------------------------------------------------------
# unknown input
# ---------------------------------------------------------------------


def test_unknown_input_in_home_suggests_next_step(
    controller: SessionController,
):
    out = controller.handle(
        UnknownInput(
            text="huh",
            stage="home",
        )
    )

    assert len(out) == 2
    assert "chat" in out[1].content


def test_unknown_input_elsewhere_just_echoes(
    controller: SessionController,
):
    out = controller.handle(
        UnknownInput(
            text="huh",
            stage="chat",
        )
    )

    assert kinds(out) == ["echo"]


# ---------------------------------------------------------------------
# workspace path stage
# ---------------------------------------------------------------------


def test_workspace_first_run_cascades_into_model_picker(
    controller: SessionController,
    tmp_path,
):
    controller.stage = "workspace"

    out = controller.handle(WorkspacePathInput(text=str(tmp_path)))

    assert controller.workspace_root == tmp_path.resolve()

    assert controller.stage == "model_pick"
    assert out[-1].kind == "mount_options"


def test_workspace_edit_cancel_reverts_stage(
    controller: SessionController,
):
    controller.stage = "workspace_edit"
    controller._workspace_return_stage = "settings"

    out = controller.handle(
        WorkspacePathInput(
            text="cancel",
            is_cancel=True,
        )
    )

    assert controller.stage == "settings"

    assert any("unchanged" in i.content for i in out if i.kind == "log")


def test_workspace_edit_invalid_path_stays_in_stage(
    controller: SessionController,
):
    controller.stage = "workspace_edit"

    out = controller.handle(WorkspacePathInput(text="/definitely/not/a/dir"))

    assert controller.stage == "workspace_edit"

    assert "not a valid directory" in out[-1].content


# ---------------------------------------------------------------------
# model / pack picker via handle_option
# ---------------------------------------------------------------------


def test_pack_pick_cancel_restores_input_and_return_stage(
    controller: SessionController,
):
    controller.stage = "settings"

    controller.handle(NavCommand(target="model"))

    out = controller.handle_option(
        "pack-options",
        "__cancel__",
    )

    assert controller.stage == "settings"

    assert kinds(out) == [
        "restore_input",
        "log",
        "mode_line",
    ]


def test_pack_pick_selecting_a_pack_activates_it(
    controller: SessionController,
):
    controller._pack_return_stage = "chat"
    controller.stage = "model_pick"

    out = controller.handle_option(
        "pack-options",
        "second_pack",
    )

    assert controller.selected_pack == "second_pack"

    assert controller.stage == "chat"

    assert any(i.kind == "restore_input" for i in out)


def test_pack_pick_add_pack_launches_wizard(
    controller: SessionController,
):
    controller._pack_return_stage = "settings"
    controller.stage = "model_pick"

    out = controller.handle_option(
        "pack-options",
        "__add_pack__",
    )

    assert controller.stage == "wiz_provider"
    assert out[-1].kind == "mount_options"

    assert out[-1].content["list_id"] == "wiz-options"


def test_option_selected_ignored_when_stage_and_list_id_dont_match(
    controller: SessionController,
):
    controller.stage = "chat"

    out = controller.handle_option(
        "pack-options",
        "second_pack",
    )

    assert out == []


# ---------------------------------------------------------------------
# wizard end-to-end through the controller
# ---------------------------------------------------------------------


def test_wizard_full_flow_existing_pack_no_activate(
    controller: SessionController,
):
    controller.stage = "settings"

    out = controller.handle(StartWizard())

    assert controller.stage == "wiz_provider"

    provider_value = out[-1].content["options"][0]["value"]

    out = controller.handle_option(
        "wiz-options",
        provider_value,
    )

    assert controller.stage == "wiz_model"

    model_value = out[-1].content["options"][0]["value"]

    out = controller.handle_option(
        "wiz-options",
        model_value,
    )

    assert controller.stage == "wiz_pack"

    out = controller.handle_option(
        "wiz-options",
        "main_pack",
    )

    assert controller.stage == "wiz_target"

    out = controller.handle_option(
        "wiz-options",
        "pool",
    )

    assert controller.stage == "settings"

    assert any("added" in i.content.lower() for i in out if i.kind == "log")

    assert controller.config_manager.save_calls == 1

    # The current config schema stores models inside ModelPack.pool.
    pack = controller.config_manager.config.providers["main_pack"]

    assert isinstance(pack, ModelPack)
    assert len(pack.pool) == 1


def test_wizard_custom_model_and_new_pack_paths(
    controller: SessionController,
):
    controller.stage = "settings"

    controller.handle(StartWizard())

    assert controller._wizard_state is not None

    controller.handle_option(
        "wiz-options",
        "groq",
    )

    out = controller.handle_option(
        "wiz-options",
        "__custom__",
    )

    assert controller.stage == "wiz_model_custom"

    out = controller.handle(
        WizardFreeText(
            text="my-custom-model",
            field="model",
        )
    )

    assert controller.stage == "wiz_pack"
    assert out[0].kind == "echo"

    out = controller.handle_option(
        "wiz-options",
        "__new__",
    )

    assert controller.stage == "wiz_pack_new"

    out = controller.handle(
        WizardFreeText(
            text="brand_new_pack",
            field="pack_new",
        )
    )

    assert controller.stage == "wiz_target"

    out = controller.handle_option(
        "wiz-options",
        "fallback",
    )

    assert controller.stage == "settings"

    assert "brand_new_pack" in controller.config_manager.config.providers

    pack = controller.config_manager.config.providers["brand_new_pack"]

    assert isinstance(pack, ModelPack)
    assert len(pack.fallback) == 1

    model = pack.fallback[0]

    assert model.provider == "groq"
    assert model.model == "my-custom-model"


def test_wizard_activate_confirm_yes_sets_active_pack(
    controller: SessionController,
):
    controller._pack_return_stage = "chat"
    controller.stage = "model_pick"

    out = controller.handle_option(
        "pack-options",
        "__add_pack__",
    )

    assert controller.stage == "wiz_provider"

    controller.handle_option(
        "wiz-options",
        "groq",
    )

    out = controller.handle_option(
        "wiz-options",
        "kimi-k2",
    )

    controller.handle_option(
        "wiz-options",
        "main_pack",
    )

    out = controller.handle_option(
        "wiz-options",
        "pool",
    )

    assert controller.stage == "awaiting_activate_confirm"

    out = controller.handle(
        ConfirmInput(
            confirmed=True,
            text="y",
            stage="awaiting_activate_confirm",
        )
    )

    assert controller.selected_pack == "main_pack"

    assert controller.stage == "chat"

    assert any(i.kind == "restore_input" for i in out)


def test_wizard_commit_failure_shows_error_and_returns(
    controller: SessionController,
):
    controller.stage = "settings"

    controller.handle(StartWizard())

    controller.handle_option(
        "wiz-options",
        "groq",
    )

    controller.handle_option(
        "wiz-options",
        "kimi-k2",
    )

    controller.handle_option(
        "wiz-options",
        "main_pack",
    )

    def boom(*args, **kwargs):
        raise ConfigError("already exists")

    controller.config_manager.packs.add_model = boom  # type: ignore[method-assign]

    out = controller.handle_option(
        "wiz-options",
        "pool",
    )

    assert "already exists" in out[0].content

    assert controller.stage == "settings"


# ---------------------------------------------------------------------
# chat turn
# ---------------------------------------------------------------------


@pytest.mark.asyncio
async def test_chat_turn_happy_path(
    controller: SessionController,
    monkeypatch,
):
    async def fake_run_inner_loop(**kwargs):
        assert kwargs["user_input"] == "hello"

        assert kwargs["workspace_root"] == controller.workspace_root

        return "hi there!"

    monkeypatch.setattr(
        "tesseractcli.controller.session.run_inner_loop",
        fake_run_inner_loop,
    )

    instrs = [i async for i in controller.handle_chat_turn(ChatMessage(text="hello"))]

    kinds_seen = kinds(instrs)

    assert kinds_seen[0] == "echo"
    assert "status" in kinds_seen
    assert kinds_seen[-1] == "box"

    assert instrs[-1].content["body"].endswith("hi there!")

    assert controller._last_agent_reply == "hi there!"


@pytest.mark.asyncio
async def test_chat_turn_error_path(
    controller: SessionController,
    monkeypatch,
):
    async def fake_run_inner_loop(**kwargs):
        raise RuntimeError("provider exploded")

    monkeypatch.setattr(
        "tesseractcli.controller.session.run_inner_loop",
        fake_run_inner_loop,
    )

    instrs = [i async for i in controller.handle_chat_turn(ChatMessage(text="hello"))]

    assert instrs[-1].kind == "box"

    assert "provider exploded" in instrs[-1].content["body"]

    assert "RuntimeError" in instrs[-1].content["body"]


@pytest.mark.asyncio
async def test_chat_turn_error_shows_traceback_when_verbose(
    controller: SessionController,
    monkeypatch,
):
    controller.config_manager.config.verbose.errors = True

    async def fake_run_inner_loop(**kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(
        "tesseractcli.controller.session.run_inner_loop",
        fake_run_inner_loop,
    )

    instrs = [i async for i in controller.handle_chat_turn(ChatMessage(text="hi"))]

    assert (
        "Traceback" in instrs[-1].content["body"]
        or "boom" in instrs[-1].content["body"]
    )


@pytest.mark.asyncio
async def test_chat_turn_asserts_workspace_root_required():
    ctl = SessionController(
        config_manager=FakeConfigManager(),  # type: ignore[arg-type]
        dispatcher=FakeDispatcher(),  # type: ignore[arg-type]
        tool_registry=FakeToolRegistry(),  # type: ignore[arg-type]
    )

    ctl.stage = "chat"
    ctl.workspace_root = None

    instrs = [i async for i in ctl.handle_chat_turn(ChatMessage(text="hi"))]

    assert instrs[-1].kind == "box"

    assert "AssertionError" in instrs[-1].content["body"]


# ---------------------------------------------------------------------
# approval bridging
# ---------------------------------------------------------------------


@pytest.mark.asyncio
async def test_approval_bridge_resolves_pending_request(
    controller: SessionController,
):
    import asyncio

    results = []

    async def approve_and_wait():
        approved = await controller._approve_via_service(
            "write_file",
            {"path": "x.py"},
            controller.workspace_root,
        )

        results.append(approved)

    task = asyncio.ensure_future(approve_and_wait())

    await asyncio.sleep(0)

    assert controller.stage == "awaiting_approval"

    preview = controller.pending_approval_preview()

    assert preview is not None
    assert "write_file" in preview

    controller.handle(
        ApprovalInput(
            approved=True,
            text="y",
        )
    )

    await task

    assert results == [True]

    assert controller.stage == "chat"


def test_report_error_public_wrapper(
    controller: SessionController,
):
    out = controller.report_error(
        "Internal error",
        ValueError("bad"),
    )

    assert out[0].kind == "box"
    assert "bad" in out[0].content["body"]


# ---------------------------------------------------------------------
# handle() rejects ChatMessage
# ---------------------------------------------------------------------


def test_handle_rejects_chat_message(
    controller: SessionController,
):
    with pytest.raises(TypeError):
        controller.handle(ChatMessage(text="hi"))
