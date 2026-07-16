"""
tests/llm/providers/_contract.py
Shared contract every cloud/API-key provider must satisfy. Concrete test
files subclass ProviderContractMixin, set four class attributes, and get
the whole contract for free - no copy-pasted test bodies per provider.

NOT collected directly by pytest (no test_ functions at module level,
and the mixin itself is abstract/incomplete without the attributes a
subclass provides).
"""
from __future__ import annotations

import pytest
from langchain_core.language_models import BaseChatModel

from tesseractcli.llm.providers.base import BaseLLMProvider


class ProviderContractMixin:
    """Subclass this in a concrete test_<provider>_provider.py file and set:

    provider_class: the BaseLLMProvider subclass under test
    env_var_name:   the Settings field name holding its API key
    model_name:     a representative model string to request
    patch_target:   dotted path to the underlying LangChain chat class,
                     as imported inside the provider module, e.g.
                     "tesseractcli.llm.providers.groq_provider.ChatGroq"
    """

    provider_class: type[BaseLLMProvider]
    env_var_name: str
    model_name: str
    patch_target: str

    def _make_provider(self, make_settings, **env_overrides):
        provider = self.provider_class()
        provider.config = make_settings(**env_overrides)
        return provider

    def test_missing_api_key_raises_value_error(self, make_settings):
        provider = self._make_provider(make_settings)  # no key set
        with pytest.raises(ValueError, match=self.env_var_name):
            provider._get_model(self.model_name)

    def test_get_model_returns_invokable_runnable(self, make_settings, mocker):
        fake_chat_model = mocker.MagicMock(spec=BaseChatModel)
        # with_retry() is what _get_model() actually returns to the caller,
        # so the mock needs to survive that wrapping too.
        fake_chat_model.with_retry.return_value = mocker.MagicMock(
            invoke=mocker.MagicMock(), stream=mocker.MagicMock()
        )
        mocker.patch(self.patch_target, return_value=fake_chat_model)

        provider = self._make_provider(make_settings, **{self.env_var_name: "test-key"})
        result = provider._get_model(self.model_name)

        assert hasattr(result, "invoke")
        assert hasattr(result, "stream")

    def test_get_model_caches_same_instance(self, make_settings, mocker):
        fake_chat_model = mocker.MagicMock(spec=BaseChatModel)
        fake_chat_model.with_retry.return_value = mocker.MagicMock()
        mocker.patch(self.patch_target, return_value=fake_chat_model)

        provider = self._make_provider(make_settings, **{self.env_var_name: "test-key"})
        first = provider._get_model(self.model_name)
        second = provider._get_model(self.model_name)

        assert first is second

    def test_get_model_safe_returns_none_when_key_missing(self, make_settings):
        provider = self._make_provider(make_settings)  # no key set
        assert provider.get_model_safe(self.model_name) is None

    def test_get_model_safe_returns_model_when_key_present(self, make_settings, mocker):
        fake_chat_model = mocker.MagicMock(spec=BaseChatModel)
        fake_chat_model.with_retry.return_value = mocker.MagicMock()
        mocker.patch(self.patch_target, return_value=fake_chat_model)

        provider = self._make_provider(make_settings, **{self.env_var_name: "test-key"})
        assert provider.get_model_safe(self.model_name) is not None
