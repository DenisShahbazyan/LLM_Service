from llm.providers._base import BaseProvider, ModelConfig
from llm.providers.anthropic import AnthropicProvider
from llm.providers.cerebras import CerebrasProvider
from llm.providers.deepseek import DeepSeekProvider
from llm.providers.gigachat import GigaChatProvider
from llm.providers.google import GoogleProvider
from llm.providers.openai import OpenAIProvider
from llm.providers.openrouter import OpenRouterProvider
from llm.providers.xai import XAIProvider


class ProviderFactory:
    _provider_classes = [
        OpenAIProvider,
        GigaChatProvider,
        AnthropicProvider,
        GoogleProvider,
        XAIProvider,
        DeepSeekProvider,
        CerebrasProvider,
        OpenRouterProvider,
    ]

    def __init__(self, usd_rate: float, model_name: str) -> None:
        self.usd_rate = usd_rate
        self._provider: BaseProvider = self._init_provider(model_name)

    def _init_provider(self, model_name: str) -> BaseProvider:
        """Инициализирует провайдер для модели

        Args:
            model_name (str): Название модели

        Returns:
            BaseProvider: Инициализированный провайдер
        """
        for provider_class in self._provider_classes:
            provider: BaseProvider = provider_class(self.usd_rate)
            if provider.has_model(model_name):
                self._provider = provider
                return provider

        raise ValueError(f'Model {model_name} not found in any provider')

    def get_model_config(self, model_name: str) -> ModelConfig:
        """Получает конфигурацию модели

        Args:
            model_name (str): Название модели

        Returns:
            ModelConfig: Конфигурация модели
        """
        return self._provider.get_model_config(model_name)
