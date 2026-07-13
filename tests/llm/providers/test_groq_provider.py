from tesseractcli.llm.providers.groq_provider import GroqProvider
from tests.llm.providers._contract import ProviderContractMixin


class TestGroqProvider(ProviderContractMixin):
    provider_class = GroqProvider
    env_var_name = "GROQ_API_KEY"
    model_name = "llama-3.1-70b-versatile"
    patch_target = "tesseractcli.llm.providers.groq_provider.ChatGroq"
