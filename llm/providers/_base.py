from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable

from langchain.schema import BaseMessage

from llm.direction import TokenDirection
from llm.types import LLMClientClass, LLMClientInstance


@dataclass
class ModelConfig:
    """Конфигурация для конкретной модели"""

    client_class: LLMClientClass
    token_counter: Callable[[list[BaseMessage], str, LLMClientInstance], int]
    pricing: dict[TokenDirection, float]
    moderation: Callable[[list[BaseMessage], LLMClientInstance], None] | None = None
    test_connection: Callable[[LLMClientInstance], bool | None] | None = None


class BaseProvider(ABC):
    """Базовый класс для провайдеров LLM"""

    def __init__(self, usd_rate: float) -> None:
        self.usd_rate = usd_rate

    @property
    @abstractmethod
    def name(self) -> str:
        """Название провайдера

        Returns:
            str: Название провайдера
        """

    @abstractmethod
    def get_models(self) -> dict[str, ModelConfig]:
        """Возвращает словарь моделей провайдера

        Returns:
            dict[str, ModelConfig]: Словарь моделей провайдера
        """

    def has_model(self, model_name: str) -> bool:
        """Проверяет, поддерживается ли модель провайдером

        Args:
            model_name (str): Название модели

        Returns:
            bool: True, если модель поддерживается, False в противном случае
        """
        return model_name in self.get_models()

    def get_model_config(self, model_name: str) -> ModelConfig:
        """Получает конфигурацию модели

        Args:
            model_name (str): Название модели

        Returns:
            ModelConfig: Конфигурация модели
        """
        models = self.get_models()
        if model_name not in models:
            raise ValueError(f'Model {model_name} not found in {self.name}')
        return models[model_name]
