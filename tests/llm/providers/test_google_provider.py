"""
tests/llm/providers/test_google_provider.py
"""
from tesseractcli.llm.providers.google_provider import GoogleProvider
from tests.llm.providers._contract import ProviderContractMixin


class TestGoogleProvider(ProviderContractMixin):
    provider_class = GoogleProvider
    env_var_name = "GOOGLE_API_KEY"
    model_name = "gemini-2.5-flash"
    patch_target = "tesseractcli.llm.providers.google_provider.ChatOpenAI"

    def test_uses_google_openai_compat_base_url(self, make_settings, mocker):
        patched = mocker.patch(self.patch_target)
        provider = self._make_provider(make_settings, GOOGLE_API_KEY="test-key")
        provider._get_model(self.model_name)

        _, kwargs = patched.call_args
        assert kwargs["base_url"] == "https://generativelanguage.googleapis.com/v1beta/openai/"
