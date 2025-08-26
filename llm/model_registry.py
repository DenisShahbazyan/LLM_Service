import warnings
from typing import Any

from langchain.schema import BaseMessage

from llm.direction import TokenDirection
from llm.providers._factory import ProviderFactory
from llm.types import LLMClientInstance


class ModelRegistry:
    def __init__(self, usd_rate: float, config: dict[str, Any]) -> None:
        self.client: LLMClientInstance | None = None
        self.config = config
        self.usd_rate = usd_rate
        self._factory = ProviderFactory(usd_rate, config.get('model'))

    def get_model_config(self, model_name: str):
        """Получает конфигурацию модели"""
        return self._factory.get_model_config(model_name)

    async def get_tokens(self, model_name: str, messages: list[BaseMessage]) -> int:
        """Получает нужную функцию счетчика токенов и вызывает ее

        Args:
            model_name (str): Название модели
            messages (list[BaseMessage]): Сообщения

        Returns:
            int: Количество токенов
        """
        config = self.get_model_config(model_name)
        return await config.token_counter(messages, model_name, self.client)

    async def get_moderation(
        self,
        model_name: str,
        messages: list[BaseMessage],
    ) -> None:
        """Получает нужную функцию модерации и вызывает ее

        Args:
            model_name (str): Название модели
            messages (list[BaseMessage]): Сообщения
        """
        config = self.get_model_config(model_name)
        if config.moderation is None:
            warnings.warn(f'No moderation for model {model_name}')
            return None
        return await config.moderation(messages, self.client)

    async def get_test_connections(self, model_name: str) -> bool | None:
        """Получает нужную функцию теста соединения и вызывает ее

        Args:
            model_name (str): Название модели
            messages (list[BaseMessage]): Сообщения

        Returns:
            bool | None: True, если соединение работает
        """
        config = self.get_model_config(model_name)
        if config.test_connection is None:
            return None
        return await config.test_connection(self.client)

    def init_client(self) -> LLMClientInstance:
        """Инициализирует клиента LLM

        Returns:
            LLMClientInstance: Клиент LLM
        """
        model_name = self.config.get('model')
        model_config = self.get_model_config(model_name)
        self.client = model_config.client_class(**self.config)
        return self.client

    def get_price(self, model_name: str, direction: TokenDirection) -> float:
        """Получает нужную цену

        Args:
            model_name (str): Название модели
            direction (TokenDirection): Направление

        Returns:
            float: Цена
        """
        config = self.get_model_config(model_name)
        return config.pricing[direction]
