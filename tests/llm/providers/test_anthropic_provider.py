from tesseractcli.llm.providers.anthropic_provider import AnthropicProvider
from tests.llm.providers._contract import ProviderContractMixin


class TestAnthropicProvider(ProviderContractMixin):
    provider_class = AnthropicProvider
    env_var_name = "ANTHROPIC_API_KEY"
    model_name = "claude-sonnet-5"
    patch_target = "tesseractcli.llm.providers.anthropic_provider.ChatAnthropic"
