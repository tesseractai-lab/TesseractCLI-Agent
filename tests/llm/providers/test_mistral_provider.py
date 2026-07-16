from tesseractcli.llm.providers.mistral_provider import MistralProvider
from tests.llm.providers._contract import ProviderContractMixin


class TestMistralProvider(ProviderContractMixin):
    provider_class = MistralProvider
    env_var_name = "MISTRAL_API_KEY"
    model_name = "mistral-large-latest"
    patch_target = "tesseractcli.llm.providers.mistral_provider.ChatOpenAI"

    def test_uses_mistral_base_url(self, make_settings, mocker):
        patched = mocker.patch(self.patch_target)
        provider = self._make_provider(make_settings, MISTRAL_API_KEY="test-key")
        provider._get_model(self.model_name)

        _, kwargs = patched.call_args
        assert kwargs["base_url"] == "https://api.mistral.ai/v1"
