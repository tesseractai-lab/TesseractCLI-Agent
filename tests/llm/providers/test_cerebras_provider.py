from tesseractcli.llm.providers.cerebras_provider import CerebrasProvider
from tests.llm.providers._contract import ProviderContractMixin


class TestCerebrasProvider(ProviderContractMixin):
    provider_class = CerebrasProvider
    env_var_name = "CEREBRAS_API_KEY"
    model_name = "llama3.1-8b"
    patch_target = "tesseractcli.llm.providers.cerebras_provider.ChatOpenAI"

    def test_uses_cerebras_base_url(self, make_settings, mocker):
        patched = mocker.patch(self.patch_target)
        provider = self._make_provider(make_settings, CEREBRAS_API_KEY="test-key")
        provider._get_model(self.model_name)

        _, kwargs = patched.call_args
        assert kwargs["base_url"] == "https://api.cerebras.ai/v1"
