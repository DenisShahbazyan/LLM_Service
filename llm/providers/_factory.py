import importlib
import inspect

from llm.providers._base import BaseProvider, ModelConfig


class ProviderFactory:
    def __init__(self, usd_rate: float, model_name: str) -> None:
        self.usd_rate = usd_rate
        self._provider_classes = self._get_available_providers()
        self._provider: BaseProvider = self._init_provider(model_name)

    def _get_available_providers(self) -> list[type[BaseProvider]]:
        """Получает список доступных классов провайдеров из __init__.py

        Returns:
            list[type[BaseProvider]]: Список классов провайдеров
        """
        provider_classes = []

        try:
            providers_module = importlib.import_module('llm.providers')

            exported_names = getattr(providers_module, '__all__', [])

            for name in exported_names:
                obj = getattr(providers_module, name, None)
                if (
                    obj
                    and inspect.isclass(obj)
                    and issubclass(obj, BaseProvider)
                    and obj != BaseProvider
                ):
                    provider_classes.append(obj)

        except ImportError:
            pass

        return provider_classes

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
                return provider

        raise ValueError(f'Model {model_name} not found in any available provider.')

    def get_model_config(self, model_name: str) -> ModelConfig:
        """Получает конфигурацию модели

        Args:
            model_name (str): Название модели

        Returns:
            ModelConfig: Конфигурация модели
        """
        return self._provider.get_model_config(model_name)
