from tesseractcli.llm.providers.github_models_provider import GitHubModelsProvider
from tests.llm.providers._contract import ProviderContractMixin


class TestGitHubModelsProvider(ProviderContractMixin):
    provider_class = GitHubModelsProvider
    env_var_name = "GITHUB_MODELS_TOKEN"
    model_name = "openai/gpt-4.1"
    patch_target = "tesseractcli.llm.providers.github_models_provider.ChatOpenAI"

    def test_uses_github_models_base_url(self, make_settings, mocker):
        patched = mocker.patch(self.patch_target)
        provider = self._make_provider(make_settings, GITHUB_MODELS_TOKEN="test-key")
        provider._get_model(self.model_name)

        _, kwargs = patched.call_args
        assert kwargs["base_url"] == "https://models.github.ai/inference"
