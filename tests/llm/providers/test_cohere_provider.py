from tesseractcli.llm.providers.cohere_provider import CohereProvider
from tests.llm.providers._contract import ProviderContractMixin


class TestCohereProvider(ProviderContractMixin):
    provider_class = CohereProvider
    env_var_name = "COHERE_API_KEY"
    model_name = "command-r-plus"
    patch_target = "tesseractcli.llm.providers.cohere_provider.ChatCohere"
