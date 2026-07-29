"""
tests/test_tools/test_sensitive_files.py

Coverage for tools/sandbox/sensitive_files.py:
  - denylist matching for read vs write (they differ — write denylist
    doesn't cover **/secrets/**, **/credentials/**)
  - allowlist overrides beating the denylist (.env.example etc.)
  - name-based AND relative-path-based pattern matching
  - symlink resolution (a symlink with an innocent name pointing at a
    denylisted target must still be blocked)
  - unknown / unmatched files are allowed
"""

import pytest

from tesseractcli.tools.sandbox.sensitive_files import check


@pytest.fixture
def workspace(tmp_path):
    return tmp_path


# ---------------------------------------------------------------------
# Basic denylist matches — read
# ---------------------------------------------------------------------


def test_env_file_blocked_for_read(workspace):
    target = workspace / ".env"
    target.write_text("SECRET=123")
    result = check(target, "read", workspace)
    assert result.allowed is False
    assert result.matched_pattern == ".env"


def test_env_variant_blocked_for_read(workspace):
    target = workspace / ".env.production"
    target.write_text("SECRET=123")
    result = check(target, "read", workspace)
    assert result.allowed is False
    assert result.matched_pattern == ".env.*"


def test_pem_file_blocked_for_read(workspace):
    target = workspace / "server.pem"
    target.write_text("-----BEGIN CERTIFICATE-----")
    result = check(target, "read", workspace)
    assert result.allowed is False


def test_ssh_key_blocked_for_read(workspace):
    target = workspace / "id_rsa"
    target.write_text("fake key")
    result = check(target, "read", workspace)
    assert result.allowed is False


def test_nested_secrets_dir_blocked_for_read(workspace):
    secrets_dir = workspace / "config" / "secrets"
    secrets_dir.mkdir(parents=True)
    target = secrets_dir / "anything.txt"
    target.write_text("whatever")
    result = check(target, "read", workspace)
    assert result.allowed is False
    assert result.matched_pattern == "**/secrets/**"


# ---------------------------------------------------------------------
# Basic denylist matches — write
# ---------------------------------------------------------------------


def test_env_file_blocked_for_write(workspace):
    target = workspace / ".env"
    result = check(target, "write", workspace)
    assert result.allowed is False
    assert result.matched_pattern == ".env"


def test_nested_secrets_dir_NOT_blocked_for_write(workspace):
    """Deliberately not covered: a directory-level write block would also
    block creating new, non-secret files inside a secrets/ folder."""
    secrets_dir = workspace / "config" / "secrets"
    secrets_dir.mkdir(parents=True)
    target = secrets_dir / "new_file.txt"
    result = check(target, "write", workspace)
    assert result.allowed is True


def test_git_config_blocked_for_write(workspace):
    git_dir = workspace / ".git"
    git_dir.mkdir()
    target = git_dir / "config"
    result = check(target, "write", workspace)
    assert result.allowed is False


# ---------------------------------------------------------------------
# Allowlist overrides
# ---------------------------------------------------------------------


def test_env_example_allowed_despite_matching_env_star(workspace):
    target = workspace / ".env.example"
    target.write_text("KEY=your_key_here")
    result = check(target, "read", workspace)
    assert result.allowed is True


def test_env_sample_allowed(workspace):
    target = workspace / ".env.sample"
    result = check(target, "write", workspace)
    assert result.allowed is True


def test_env_template_allowed(workspace):
    target = workspace / ".env.template"
    result = check(target, "read", workspace)
    assert result.allowed is True


# ---------------------------------------------------------------------
# Symlink resolution — the important security case
# ---------------------------------------------------------------------


def test_symlink_to_env_blocked_even_with_innocent_name(workspace):
    """A file named notes.txt that's actually a symlink to .env must
    still be blocked — resolve() must run before pattern matching."""
    real_env = workspace / ".env"
    real_env.write_text("SECRET=123")

    fake_notes = workspace / "notes.txt"
    fake_notes.symlink_to(real_env)

    result = check(fake_notes, "read", workspace)
    assert result.allowed is False
    assert result.matched_pattern == ".env"


def test_symlink_to_safe_file_allowed(workspace):
    real_file = workspace / "real_data.txt"
    real_file.write_text("nothing sensitive")

    link = workspace / "shortcut.txt"
    link.symlink_to(real_file)

    result = check(link, "read", workspace)
    assert result.allowed is True


# ---------------------------------------------------------------------
# Unmatched files
# ---------------------------------------------------------------------


def test_normal_python_file_allowed(workspace):
    target = workspace / "main.py"
    target.write_text("print('hello')")
    result = check(target, "read", workspace)
    assert result.allowed is True
    assert result.matched_pattern is None


def test_normal_python_file_allowed_for_write(workspace):
    target = workspace / "main.py"
    result = check(target, "write", workspace)
    assert result.allowed is True


# ---------------------------------------------------------------------
# Path outside workspace (fallback branch in _relative_and_name)
# ---------------------------------------------------------------------


def test_path_outside_workspace_falls_back_safely(workspace, tmp_path_factory):
    """path_guard should reject this before sensitive_files ever sees
    it, but the function must not crash if called directly on an
    out-of-workspace path — it should fall back to matching on the
    absolute path instead of raising."""
    outside_root = tmp_path_factory.mktemp("outside")
    outside_file = outside_root / ".env"
    outside_file.write_text("SECRET=123")

    result = check(outside_file, "read", workspace)
    # Even outside the workspace, the .env name pattern still matches
    # on the file's own name, so this should still be blocked.
    assert result.allowed is False
