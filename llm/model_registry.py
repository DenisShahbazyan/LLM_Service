from dataclasses import dataclass
from typing import Any, Callable, Type

import aiohttp
import tiktoken
from langchain.schema import BaseMessage
from langchain_anthropic import ChatAnthropic
from langchain_gigachat import GigaChat
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langchain_xai import ChatXAI

from llm.direction import TokenDirection

LLMClientInstance = (
    ChatOpenAI | GigaChat | ChatAnthropic | ChatGoogleGenerativeAI | ChatXAI
)
LLMClientClass = (
    Type[ChatOpenAI]
    | Type[GigaChat]
    | Type[ChatAnthropic]
    | Type[ChatGoogleGenerativeAI]
    | Type[ChatXAI]
)


@dataclass
class ModelConfig:
    """Конфигурация для конкретной модели"""

    client_class: LLMClientClass
    token_counter: Callable
    pricing: dict[TokenDirection, float]


class ModelRegistry:
    """Реестр моделей

    - Цены OpenAI: https://platform.openai.com/docs/pricing
    - Цены Gigachat: https://developers.sber.ru/docs/ru/gigachat/tariffs/legal-tariffs
    - Цены Anthropic: https://docs.anthropic.com/en/docs/about-claude/pricing
        - имена моделей https://docs.anthropic.com/en/docs/about-claude/models/overview#model-names
    - Цены Google: https://ai.google.dev/gemini-api/docs/pricing#gemini-2.5-pro-preview
    - Цены XAI: https://docs.x.ai/docs/models
    """  # noqa: E501

    def __init__(self, usd_rate: float) -> None:
        self.usd_rate = usd_rate
        self.client: LLMClientInstance = None
        self._models = self._init_models()

    def _init_models(self) -> dict[str, ModelConfig]:
        return {
            # OpenAI
            'gpt-4o-mini': ModelConfig(
                client_class=ChatOpenAI,
                token_counter=self._count_tokens_openai,
                pricing={
                    TokenDirection.ENCODE: 0.15 / 1_000_000,
                    TokenDirection.DECODE: 0.6 / 1_000_000,
                },
            ),
            'gpt-4o': ModelConfig(
                client_class=ChatOpenAI,
                token_counter=self._count_tokens_openai,
                pricing={
                    TokenDirection.ENCODE: 2.5 / 1_000_000,
                    TokenDirection.DECODE: 10.0 / 1_000_000,
                },
            ),
            'o3-2025-04-16': ModelConfig(
                client_class=ChatOpenAI,
                token_counter=self._count_tokens_openai,
                pricing={
                    TokenDirection.ENCODE: 2.0 / 1_000_000,
                    TokenDirection.DECODE: 8.0 / 1_000_000,
                },
            ),
            'o4-mini-2025-04-16': ModelConfig(
                client_class=ChatOpenAI,
                token_counter=self._count_tokens_openai,
                pricing={
                    TokenDirection.ENCODE: 1.10 / 1_000_000,
                    TokenDirection.DECODE: 4.40 / 1_000_000,
                },
            ),
            # Gigachat
            'GigaChat': ModelConfig(
                client_class=GigaChat,
                token_counter=self._create_gigachat_counter(),
                pricing={
                    # 5_000 рублей / 25_000_000 токенов / курс доллара
                    TokenDirection.ENCODE: 5_000 / 25_000_000 / self.usd_rate,
                    TokenDirection.DECODE: 5_000 / 25_000_000 / self.usd_rate,
                },
            ),
            'GigaChat-2': ModelConfig(
                client_class=GigaChat,
                token_counter=self._create_gigachat_counter(),
                pricing={
                    # 5_000 рублей / 25_000_000 токенов / курс доллара
                    TokenDirection.ENCODE: 5_000 / 25_000_000 / self.usd_rate,
                    TokenDirection.DECODE: 5_000 / 25_000_000 / self.usd_rate,
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
            'GigaChat-2-Pro': ModelConfig(
                client_class=GigaChat,
                token_counter=self._create_gigachat_counter(),
                pricing={
                    # 10_500 рублей / 7_000_000 токенов / курс доллара
                    TokenDirection.ENCODE: 10_500 / 7_000_000 / self.usd_rate,
                    TokenDirection.DECODE: 10_500 / 7_000_000 / self.usd_rate,
                },
            ),
            'GigaChat-Max': ModelConfig(
                client_class=GigaChat,
                token_counter=self._create_gigachat_counter(),
                pricing={
                    # 15_600 рублей / 8_000_000 токенов / курс доллара
                    TokenDirection.ENCODE: 15_600 / 8_000_000 / self.usd_rate,
                    TokenDirection.DECODE: 15_600 / 8_000_000 / self.usd_rate,
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
            # Anthropic
            'claude-3-5-haiku-latest': ModelConfig(
                client_class=ChatAnthropic,
                token_counter=self._create_anthropic_counter(),
                pricing={
                    TokenDirection.ENCODE: 0.8 / 1_000_000,
                    TokenDirection.DECODE: 4.0 / 1_000_000,
                },
            ),
            'claude-3-7-sonnet-latest': ModelConfig(
                client_class=ChatAnthropic,
                token_counter=self._create_anthropic_counter(),
                pricing={
                    TokenDirection.ENCODE: 3.0 / 1_000_000,
                    TokenDirection.DECODE: 15.0 / 1_000_000,
                },
            ),
            'claude-opus-4-20250514': ModelConfig(
                client_class=ChatAnthropic,
                token_counter=self._create_anthropic_counter(),
                pricing={
                    TokenDirection.ENCODE: 15.0 / 1_000_000,
                    TokenDirection.DECODE: 75.0 / 1_000_000,
                },
            ),
            # Google
            'gemini-2.0-flash-001': ModelConfig(
                client_class=ChatGoogleGenerativeAI,
                token_counter=self._create_google_counter(),
                pricing={
                    TokenDirection.ENCODE: 0.1 / 1_000_000,
                    TokenDirection.DECODE: 0.4 / 1_000_000,
                },
            ),
            'gemini-2.5-pro-preview-06-05': ModelConfig(
                client_class=ChatGoogleGenerativeAI,
                token_counter=self._create_google_counter(),
                pricing={
                    TokenDirection.ENCODE: 2.5 / 1_000_000,
                    TokenDirection.DECODE: 15.0 / 1_000_000,
                },
            ),
            # Groq
            'grok-3-mini': ModelConfig(
                client_class=ChatXAI,
                token_counter=self._create_xai_counter(),
                pricing={
                    TokenDirection.ENCODE: 0.3 / 1_000_000,
                    TokenDirection.DECODE: 0.5 / 1_000_000,
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

    def _create_google_counter(self):
        """Создает функцию счетчика токенов для Google"""

        async def count_tokens(messages: list[BaseMessage], model_name: str) -> int:
            if not self.client:
                raise ValueError('Client not initialized')

            return self.client.get_num_tokens_from_messages(messages)

        return count_tokens

    def _create_xai_counter(self):
        """Создает функцию счетчика токенов для xAI"""

        async def count_tokens(messages: list[BaseMessage], model_name: str) -> int:
            if not self.client:
                raise ValueError('Client not initialized')

            x_api_key = self.client.xai_api_key._secret_value

            url = 'https://api.x.ai/v1/tokenize-text'

            headers = {
                'Authorization': f"Bearer {x_api_key}",
                'Content-Type': 'application/json',
            }

            text = ' '.join(str(m.content) for m in messages)
            payload = {
                'text': text,
                'model': model_name,
            }

            async with aiohttp.ClientSession() as session:
                async with session.post(url, headers=headers, json=payload) as response:
                    data = await response.json()
                    return len(data['token_ids'])

        return count_tokens
