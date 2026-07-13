import sys
import types

import pytest
from langchain_core.language_models import BaseChatModel

from tesseractcli.llm.providers.local_gguf_provider import LocalGGUFProvider


@pytest.fixture
def fake_gguf_file(tmp_path):
    model_file = tmp_path / "model.Q4_K_M.gguf"
    model_file.write_bytes(b"not a real gguf, just needs to exist")
    return str(model_file)


@pytest.fixture
def stub_llama_cpp_module(mocker):
    """langchain_community.chat_models.ChatLlamaCpp is imported lazily
    inside _load_model, so we inject a stub module into sys.modules
    rather than needing llama-cpp-python actually installed/compiled."""
    fake_chat_model = mocker.MagicMock(spec=BaseChatModel)
    fake_chat_model.with_retry.return_value = mocker.MagicMock()

    fake_chat_llama_cpp = mocker.MagicMock(return_value=fake_chat_model)
    stub_module = types.ModuleType("langchain_community.chat_models")
    stub_module.ChatLlamaCpp = fake_chat_llama_cpp

    original = sys.modules.get("langchain_community.chat_models")
    sys.modules["langchain_community.chat_models"] = stub_module
    yield fake_chat_llama_cpp
    if original is not None:
        sys.modules["langchain_community.chat_models"] = original
    else:
        del sys.modules["langchain_community.chat_models"]


class TestLocalGGUFProvider:
    def test_missing_path_raises_value_error(self, make_settings):
        provider = LocalGGUFProvider()
        provider.config = make_settings()  # no LOCAL_GGUF_MODEL_PATH
        with pytest.raises(ValueError, match="LOCAL_GGUF_MODEL_PATH"):
            provider.get_model("")

    def test_nonexistent_file_raises_file_not_found(self, make_settings):
        provider = LocalGGUFProvider()
        provider.config = make_settings()
        with pytest.raises(FileNotFoundError):
            provider.get_model("/tmp/definitely-not-a-real-model.gguf")

    def test_falls_back_to_settings_path(
        self, make_settings, fake_gguf_file, stub_llama_cpp_module
    ):
        provider = LocalGGUFProvider()
        provider.config = make_settings(LOCAL_GGUF_MODEL_PATH=fake_gguf_file)

        result = provider.get_model("")  # empty -> falls back to settings

        assert hasattr(result, "invoke")
        stub_llama_cpp_module.assert_called_once_with(model_path=fake_gguf_file)

    def test_explicit_path_overrides_settings(
        self, make_settings, fake_gguf_file, stub_llama_cpp_module
    ):
        provider = LocalGGUFProvider()
        provider.config = make_settings()  # no default path set

        provider.get_model(fake_gguf_file)

        stub_llama_cpp_module.assert_called_once_with(model_path=fake_gguf_file)

    def test_missing_llama_cpp_package_raises_import_error(
        self, make_settings, fake_gguf_file, monkeypatch
    ):
        # Simulate llama-cpp-python not being installed at all.
        monkeypatch.setitem(sys.modules, "langchain_community.chat_models", None)
        provider = LocalGGUFProvider()
        provider.config = make_settings()

        with pytest.raises(ImportError, match="llama-cpp-python"):
            provider.get_model(fake_gguf_file)
