from tesseractcli.llm.providers.openai_provider import OpenAIProvider
from tests.llm.providers._contract import ProviderContractMixin


class TestOpenAIProvider(ProviderContractMixin):
    provider_class = OpenAIProvider
    env_var_name = "OPENAI_API_KEY"
    model_name = "gpt-4o-mini"
    patch_target = "tesseractcli.llm.providers.openai_provider.ChatOpenAI"

    def test_does_not_set_custom_base_url(self, make_settings, mocker):
        patched = mocker.patch(self.patch_target)
        provider = self._make_provider(make_settings, OPENAI_API_KEY="test-key")
        provider.get_model(self.model_name)

        _, kwargs = patched.call_args
        assert "base_url" not in kwargs
