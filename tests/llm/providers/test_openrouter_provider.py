from tesseractcli.llm.providers.openrouter_provider import OpenRouterProvider
from tests.llm.providers._contract import ProviderContractMixin


class TestOpenRouterProvider(ProviderContractMixin):
    provider_class = OpenRouterProvider
    env_var_name = "OPENROUTER_API_KEY"
    model_name = "meta-llama/llama-3.3-70b-instruct:free"
    patch_target = "tesseractcli.llm.providers.openrouter_provider.ChatOpenAI"

    def test_uses_openrouter_base_url(self, make_settings, mocker):
        patched = mocker.patch(self.patch_target)
        provider = self._make_provider(make_settings, OPENROUTER_API_KEY="test-key")
        provider._get_model(self.model_name)

        _, kwargs = patched.call_args
        assert kwargs["base_url"] == "https://openrouter.ai/api/v1"
