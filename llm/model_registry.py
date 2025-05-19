from dataclasses import dataclass
from typing import Any, Callable, Type

import tiktoken
from langchain.schema import BaseMessage
from langchain_anthropic import ChatAnthropic
from langchain_gigachat import GigaChat
from langchain_openai import ChatOpenAI

from llm.direction import TokenDirection

LLMClientInstance = ChatOpenAI | GigaChat | ChatAnthropic
LLMClientClass = Type[ChatOpenAI] | Type[GigaChat] | Type[ChatAnthropic]


@dataclass
class ModelConfig:
    """Конфигурация для конкретной модели"""

    client_class: LLMClientClass
    token_counter: Callable
    pricing: dict[TokenDirection, float]


class ModelRegistry:
    """Реестр моделей"""

    def __init__(self, usd_rate: float) -> None:
        self.usd_rate = usd_rate
        self.client: LLMClientInstance = None
        self._models = self._init_models()

    def _init_models(self) -> dict[str, ModelConfig]:
        return {
            'gpt-4o-mini': ModelConfig(
                client_class=ChatOpenAI,
                token_counter=self._count_tokens_openai,
                pricing={
                    TokenDirection.ENCODE: 0.15 / 1e6,
                    TokenDirection.DECODE: 0.6 / 1e6,
                },
            ),
            'gpt-4o': ModelConfig(
                client_class=ChatOpenAI,
                token_counter=self._count_tokens_openai,
                pricing={
                    TokenDirection.ENCODE: 2.5 / 1e6,
                    TokenDirection.DECODE: 10.0 / 1e6,
                },
            ),
            'GigaChat-Pro': ModelConfig(
                client_class=GigaChat,
                token_counter=self._create_gigachat_counter(),
                pricing={
                    # 10_500 рублей / 7_000_000 токенов / курс доллара
                    TokenDirection.ENCODE: 10_500 / 7_000_000 / self.usd_rate,
                    TokenDirection.DECODE: 10_500 / 7_000_000 / self.usd_rate,
                },
            ),
            'GigaChat-2-Max': ModelConfig(
                client_class=GigaChat,
                token_counter=self._create_gigachat_counter(),
                pricing={
                    # 15_600 рублей / 8_000_000 токенов / курс доллара
                    TokenDirection.ENCODE: 15_600 / 8_000_000 / self.usd_rate,
                    TokenDirection.DECODE: 15_600 / 8_000_000 / self.usd_rate,
                },
            ),
            'claude-3-5-haiku-latest': ModelConfig(
                client_class=ChatAnthropic,
                token_counter=self._create_anthropic_counter(),
                pricing={
                    TokenDirection.ENCODE: 0.25 / 1e6,
                    TokenDirection.DECODE: 1.25 / 1e6,
                },
            ),
        }

    async def get_tokens(self, model_name: str, messages: list[BaseMessage]) -> int:
        """Получает нужную функцию счетчика токенов и вызывает ее

        Args:
            model_name (str): Название модели
            messages (list[BaseMessage]): Сообщения

        Returns:
            int: Количество токенов
        """
        if model_name not in self._models:
            raise ValueError(f'Unknown model: {model_name}')
        return await self._models[model_name].token_counter(messages, model_name)

    def init_client(self, config: dict[str, Any]) -> LLMClientInstance:
        """Инициализирует клиента LLM

        Args:
            config (dict): Конфигурация

        Returns:
            LLMClientInstance: Клиент LLM
        """
        model_name = config.get('model')
        if model_name not in self._models:
            raise ValueError(f'Unknown model: {model_name}')
        self.client = self._models[model_name].client_class(**config)
        return self.client

    def get_price(self, model_name: str, direction: TokenDirection) -> float:
        """Получает нужную цену

        Args:
            model_name (str): Название модели
            direction (TokenDirection): Направление

        Returns:
            float: Цена
        """
        if model_name not in self._models:
            raise ValueError(f'Unknown model: {model_name}')
        return self._models[model_name].pricing[direction]

    @staticmethod
    async def _count_tokens_openai(messages: list[BaseMessage], model_name: str) -> int:
        """Подсчитывает количество токенов, для моделей OpenAI

        Args:
            messages (list[BaseMessage]): Сообщения
            model_name (str): Название модели

        Returns:
            int: Количество токенов
        """
        encoding = tiktoken.encoding_for_model(model_name)
        text = ' '.join(str(m.content) for m in messages)
        return len(encoding.encode(text))

    def _create_gigachat_counter(self):
        """Создает функцию счетчика токенов для Gigachat"""

        async def count_tokens(messages: list[BaseMessage], model_name: str) -> int:
            if not self.client:
                raise ValueError('Client not initialized')

            text = ' '.join(str(m.content) for m in messages)
            response = await self.client.atokens_count([text], model_name)
            return response[0].tokens

        return count_tokens

    def _create_anthropic_counter(self):
        """Создает функцию счетчика токенов для Anthropic"""

        async def count_tokens(messages: list[BaseMessage], model_name: str) -> int:
            if not self.client:
                raise ValueError('Client not initialized')

            return self.client.get_num_tokens_from_messages(messages)

        return count_tokens
