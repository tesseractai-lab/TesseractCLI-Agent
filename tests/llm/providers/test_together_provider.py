from tesseractcli.llm.providers.together_provider import TogetherProvider
from tests.llm.providers._contract import ProviderContractMixin


class TestTogetherProvider(ProviderContractMixin):
    provider_class = TogetherProvider
    env_var_name = "TOGETHER_API_KEY"
    model_name = "meta-llama/Llama-3.3-70B-Instruct-Turbo"
    patch_target = "tesseractcli.llm.providers.together_provider.ChatTogether"
