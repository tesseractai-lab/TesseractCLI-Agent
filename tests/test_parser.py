"""
Unit tests for `commands/parser.py`. Every test here is plain strings
in / Command out - no Textual, no App, no I/O. This is the whole point
of pulling the parser out of `TesseractApp`: these tests could never
have existed against the old code without spinning up a full Textual
App and driving fake keystrokes through it.
"""

from __future__ import annotations

from tesseractcli.commands.parser import parse
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

# ---------------------------------------------------------------------
# chat stage: default case is a real message
# ---------------------------------------------------------------------


def test_plain_text_in_chat_is_a_chat_message():
    assert parse("how do I reverse a linked list?", "chat") == ChatMessage(
        text="how do I reverse a linked list?"
    )


def test_chat_message_is_stripped():
    assert parse("  hello there  ", "chat") == ChatMessage(text="hello there")


# ---------------------------------------------------------------------
# bare nav words
# ---------------------------------------------------------------------


def test_bare_nav_words_from_chat():
    for word, target in [
        ("settings", "settings"),
        ("chat", "chat"),
        ("home", "home"),
        ("model", "model"),
        ("exit", "exit"),
        ("quit", "quit"),
    ]:
        assert parse(word, "chat") == NavCommand(target=target)


def test_nav_words_case_insensitive():
    assert parse("SETTINGS", "chat") == NavCommand(target="settings")
    assert parse("ExIt", "home") == NavCommand(target="exit")


def test_q_alias_resolves_to_exit():
    assert parse("q", "chat") == NavCommand(target="exit")


def test_nav_words_not_recognized_outside_chat_settings_home():
    # documented tradeoff: free-text stages don't get nav interception
    assert parse("chat", "wiz_model_custom") == WizardFreeText(
        text="chat", field="model"
    )


# ---------------------------------------------------------------------
# GLOBAL_ALIASES: meta commands and long-form nav
# ---------------------------------------------------------------------


def test_help_aliases():
    for token in ("-h", "--help", "?"):
        assert parse(token, "chat") == MetaCommand(kind="help")


def test_meta_aliases_from_settings_stage_too():
    assert parse("-h", "settings") == MetaCommand(kind="help")
    assert parse("clear", "settings") == MetaCommand(kind="clear")


def test_long_form_nav_aliases():
    assert parse("--chat", "settings") == NavCommand(target="chat")
    assert parse("--home", "chat") == NavCommand(target="home")


def test_quit_shortcuts():
    assert parse("-q", "chat") == NavCommand(target="exit")
    assert parse("--quit", "chat") == NavCommand(target="exit")


def test_copy_expand_clear_packs():
    assert parse("-c", "chat").kind == "copy"
    assert parse("-e", "chat").kind == "expand"
    assert parse("cls", "chat").kind == "clear"
    assert parse("-p", "chat").kind == "packs"


# ---------------------------------------------------------------------
# reset / reload with optional scope
# ---------------------------------------------------------------------


def test_reset_bare():
    cmd = parse("reset", "chat")
    assert isinstance(cmd, ResetCommand)
    assert cmd.scope is None
    assert cmd.text == "reset"


def test_reset_with_scope():
    cmd = parse("reset temp", "chat")
    assert cmd == ResetCommand(scope="temp", text="reset temp")


def test_reset_shorthand_head():
    cmd = parse("-rst context", "settings")
    assert cmd == ResetCommand(scope="context", text="-rst context")


def test_reload_bare_and_scoped():
    assert parse("reload", "chat") == ReloadCommand(scope=None, text="reload")
    assert parse("reload cfg", "chat") == ReloadCommand(scope="cfg", text="reload cfg")


def test_reset_head_with_too_many_words_falls_through():
    # "reset a b c" doesn't match the <=2-word reset grammar, so it
    # should NOT be treated as a reset command in the chat/home branch;
    # it falls through to compound-shorthand / settings-command handling.
    cmd = parse("reset a b c", "chat")
    assert not isinstance(cmd, ResetCommand)


# ---------------------------------------------------------------------
# compound shorthand ("-rm -p X" -> "remove pack X")
# ---------------------------------------------------------------------


def test_compound_shorthand_remove_pack_from_settings():
    cmd = parse("-rm -p mypack", "settings")
    assert cmd == SettingsCommand(
        raw_text="remove pack mypack", original_text="-rm -p mypack"
    )


def test_compound_shorthand_add_model_from_settings():
    cmd = parse("add -md mypack groq llama-3.3", "settings")
    assert cmd == SettingsCommand(
        raw_text="add model mypack groq llama-3.3",
        original_text="add -md mypack groq llama-3.3",
    )


def test_compound_shorthand_from_chat_via_dash_prefix():
    # bare shorthand (no "-cfg" prefix) is also recognized directly
    # from chat/home, per the original grammar.
    cmd = parse("-rm -p mypack", "chat")
    assert cmd == SettingsCommand(
        raw_text="remove pack mypack", original_text="-rm -p mypack"
    )


def test_canonical_spelling_already_matches_shorthand_translator():
    cmd = parse("remove pack mypack", "settings")
    assert cmd == SettingsCommand(
        raw_text="remove pack mypack", original_text="remove pack mypack"
    )


def test_non_add_remove_head_is_not_shorthand():
    # "rename" isn't "add"/"remove", so this must NOT be caught by the
    # shorthand translator and instead falls through to normal settings
    # alias resolution.
    cmd = parse("rename oldpack newpack", "settings")
    assert cmd == SettingsCommand(
        raw_text="rename oldpack newpack", original_text="rename oldpack newpack"
    )


# ---------------------------------------------------------------------
# settings stage: alias resolution, "suggest" -> StartWizard
# ---------------------------------------------------------------------


def test_settings_alias_resolution():
    cmd = parse("-rm pack foo", "settings")
    assert cmd == SettingsCommand(
        raw_text="remove pack foo", original_text="-rm pack foo"
    )


def test_settings_bare_suggest_starts_wizard():
    assert parse("suggest", "settings") == StartWizard()
    assert parse("-s", "settings") == StartWizard()


def test_settings_suggest_with_extra_words_is_not_wizard():
    # "suggest" only triggers StartWizard with exactly one word - with
    # trailing args it's forwarded as an ordinary settings command.
    cmd = parse("suggest foo", "settings")
    assert isinstance(cmd, SettingsCommand)
    assert not isinstance(cmd, StartWizard)


def test_settings_empty_input_is_unknown():
    cmd = parse("   ", "settings")
    assert isinstance(cmd, UnknownInput)
    assert cmd.stage == "settings"


# ---------------------------------------------------------------------
# "-cfg ..." inline forwarding from chat/home
# ---------------------------------------------------------------------


def test_cfg_model_with_arg_is_inline_switch():
    cmd = parse("-cfg model mypack", "chat")
    assert cmd == InlineModelSwitch(args=("mypack",), text="-cfg model mypack")


def test_cfg_model_no_arg_is_inline_switch_empty():
    cmd = parse("-cfg model", "home")
    assert cmd == InlineModelSwitch(args=(), text="-cfg model")


def test_cfg_workspace_switch():
    cmd = parse("-cfg -ws /tmp/proj", "chat")
    assert cmd == InlineWorkspaceSwitch(args=("/tmp/proj",), text="-cfg -ws /tmp/proj")


def test_cfg_reset_forwarding():
    cmd = parse("-cfg reset temp", "chat")
    assert cmd == ResetCommand(scope="temp", text="-cfg reset temp")


def test_cfg_reload_forwarding():
    cmd = parse("-cfg reload", "home")
    assert cmd == ReloadCommand(scope=None, text="-cfg reload")


def test_cfg_suggest_forwarding_starts_wizard():
    assert parse("-cfg suggest", "chat") == StartWizard()


def test_cfg_generic_settings_forwarding():
    cmd = parse("-cfg -rm pack foo", "chat")
    assert cmd == SettingsCommand(
        raw_text="remove pack foo", original_text="-cfg -rm pack foo"
    )


def test_cfg_only_recognized_from_chat_home_not_settings():
    # from "settings" stage, "-cfg ..." isn't special-cased - it falls
    # through to ordinary settings-stage alias resolution instead.
    cmd = parse("-cfg model mypack", "settings")
    assert not isinstance(cmd, InlineModelSwitch)


# ---------------------------------------------------------------------
# free-text stages: parser must NOT apply grammar here
# ---------------------------------------------------------------------


def test_awaiting_approval_never_intercepted_by_grammar():
    cmd = parse("-h", "awaiting_approval")
    assert cmd == ApprovalInput(approved=False, text="-h")


def test_approval_yes_variants():
    assert parse("y", "awaiting_approval") == ApprovalInput(approved=True, text="y")
    assert parse("Yes", "awaiting_approval") == ApprovalInput(approved=True, text="Yes")
    assert parse("n", "awaiting_approval") == ApprovalInput(approved=False, text="n")
    assert parse("nah", "awaiting_approval") == ApprovalInput(
        approved=False, text="nah"
    )


def test_wiz_model_custom_captures_literal_text():
    # "-h" would normally be the help shortcut - here it must be taken
    # as a literal (if unlikely) model id instead.
    cmd = parse("-h", "wiz_model_custom")
    assert cmd == WizardFreeText(text="-h", field="model")


def test_wiz_pack_new_captures_literal_text():
    cmd = parse("my-new-pack", "wiz_pack_new")
    assert cmd == WizardFreeText(text="my-new-pack", field="pack_new")


def test_confirm_stages_resolve_to_confirm_input():
    for stage in (
        "awaiting_reset_confirm",
        "awaiting_remove_pack_confirm",
        "awaiting_activate_confirm",
    ):
        cmd = parse("y", stage)
        assert cmd == ConfirmInput(confirmed=True, text="y", stage=stage)
        cmd = parse("nope", stage)
        assert cmd == ConfirmInput(confirmed=False, text="nope", stage=stage)


# ---------------------------------------------------------------------
# workspace / workspace_edit stages
# ---------------------------------------------------------------------


def test_workspace_stage_captures_path():
    cmd = parse("/home/user/project", "workspace")
    assert cmd == WorkspacePathInput(text="/home/user/project", is_cancel=False)


def test_workspace_edit_cancel_word():
    cmd = parse("cancel", "workspace_edit")
    assert cmd == WorkspacePathInput(text="cancel", is_cancel=True)


def test_workspace_edit_cancel_is_case_insensitive_for_the_flag_but_not_the_text():
    cmd = parse("CANCEL", "workspace_edit")
    assert cmd.is_cancel is True
    assert cmd.text == "CANCEL"  # text itself is preserved verbatim


def test_workspace_stage_never_has_a_cancel_option():
    # documented distinction: the first-run 'workspace' stage has no
    # cancel path, only 'workspace_edit' does.
    cmd = parse("cancel", "workspace")
    assert cmd == WorkspacePathInput(text="cancel", is_cancel=False)


# ---------------------------------------------------------------------
# home stage: unmatched text
# ---------------------------------------------------------------------


def test_home_stage_unknown_text():
    cmd = parse("blah blah", "home")
    assert cmd == UnknownInput(text="blah blah", stage="home")


def test_parser_never_raises_on_garbage():
    for text in ("", "   ", "!@#$%^&*()", "-cfg", "-cfg -ws", "add"):
        for stage in ("chat", "settings", "home", "workspace"):
            parse(text, stage)  # must not raise
