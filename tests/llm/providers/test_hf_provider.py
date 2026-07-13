from tesseractcli.llm.providers.hf_provider import HuggingFaceProvider
from tests.llm.providers._contract import ProviderContractMixin


class TestHuggingFaceProvider(ProviderContractMixin):
    provider_class = HuggingFaceProvider
    env_var_name = "HUGGINGFACE_API_KEY"
    model_name = "meta-llama/Llama-3.1-8B-Instruct"
    patch_target = "tesseractcli.llm.providers.hf_provider.ChatOpenAI"

    def test_uses_hf_router_base_url(self, make_settings, mocker):
        patched = mocker.patch(self.patch_target)
        provider = self._make_provider(make_settings, HUGGINGFACE_API_KEY="test-key")
        provider.get_model(self.model_name)

        _, kwargs = patched.call_args
        assert kwargs["base_url"] == "https://router.huggingface.co/v1"
