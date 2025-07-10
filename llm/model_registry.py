from dataclasses import dataclass
from typing import Any, Callable, Type

from langchain.schema import BaseMessage
from langchain_anthropic import ChatAnthropic
from langchain_deepseek import ChatDeepSeek
from langchain_gigachat import GigaChat
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langchain_xai import ChatXAI

from llm.direction import TokenDirection
from llm.token_counters import TokenCounterFactory

LLMClientInstance = (
    ChatOpenAI
    | GigaChat
    | ChatAnthropic
    | ChatGoogleGenerativeAI
    | ChatXAI
    | ChatDeepSeek
)
LLMClientClass = (
    Type[ChatOpenAI]
    | Type[GigaChat]
    | Type[ChatAnthropic]
    | Type[ChatGoogleGenerativeAI]
    | Type[ChatXAI]
    | Type[ChatDeepSeek]
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
    - Цены DeepSeek: https://api-docs.deepseek.com/quick_start/pricing
    """  # noqa: E501

    def __init__(self, usd_rate: float) -> None:
        self.usd_rate = usd_rate
        self.client: LLMClientInstance = None
        self._models = self._init_models()

    def _init_models(self) -> dict[str, ModelConfig]:
        return {
            # OpenAI
            'gpt-4.1': ModelConfig(
                client_class=ChatOpenAI,
                token_counter=TokenCounterFactory().create_openai_counter(),
                pricing={
                    TokenDirection.ENCODE: 2.00 / 1_000_000,
                    TokenDirection.DECODE: 8.00 / 1_000_000,
                },
            ),
            'gpt-4.1-mini': ModelConfig(
                client_class=ChatOpenAI,
                token_counter=TokenCounterFactory().create_openai_counter(),
                pricing={
                    TokenDirection.ENCODE: 0.40 / 1_000_000,
                    TokenDirection.DECODE: 1.60 / 1_000_000,
                },
            ),
            'gpt-4.1-nano': ModelConfig(
                client_class=ChatOpenAI,
                token_counter=TokenCounterFactory().create_openai_counter(),
                pricing={
                    TokenDirection.ENCODE: 0.10 / 1_000_000,
                    TokenDirection.DECODE: 0.40 / 1_000_000,
                },
            ),
            'gpt-4.5-preview': ModelConfig(
                client_class=ChatOpenAI,
                token_counter=TokenCounterFactory().create_openai_counter(),
                pricing={
                    TokenDirection.ENCODE: 75.00 / 1_000_000,
                    TokenDirection.DECODE: 150.00 / 1_000_000,
                },
            ),
            'gpt-4o-mini': ModelConfig(
                client_class=ChatOpenAI,
                token_counter=TokenCounterFactory().create_openai_counter(),
                pricing={
                    TokenDirection.ENCODE: 0.15 / 1_000_000,
                    TokenDirection.DECODE: 0.60 / 1_000_000,
                },
            ),
            'gpt-4o': ModelConfig(
                client_class=ChatOpenAI,
                token_counter=TokenCounterFactory().create_openai_counter(),
                pricing={
                    TokenDirection.ENCODE: 2.50 / 1_000_000,
                    TokenDirection.DECODE: 10.00 / 1_000_000,
                },
            ),
            'o3-2025-04-16': ModelConfig(
                client_class=ChatOpenAI,
                token_counter=TokenCounterFactory().create_openai_counter(),
                pricing={
                    TokenDirection.ENCODE: 2.00 / 1_000_000,
                    TokenDirection.DECODE: 8.00 / 1_000_000,
                },
            ),
            'o4-mini-2025-04-16': ModelConfig(
                client_class=ChatOpenAI,
                token_counter=TokenCounterFactory().create_openai_counter(),
                pricing={
                    TokenDirection.ENCODE: 1.10 / 1_000_000,
                    TokenDirection.DECODE: 4.40 / 1_000_000,
                },
            ),
            # Gigachat
            'GigaChat': ModelConfig(
                client_class=GigaChat,
                token_counter=TokenCounterFactory().create_gigachat_counter(),
                pricing={
                    # 5_000 рублей / 25_000_000 токенов / курс доллара
                    TokenDirection.ENCODE: 5_000 / 25_000_000 / self.usd_rate,
                    TokenDirection.DECODE: 5_000 / 25_000_000 / self.usd_rate,
                },
            ),
            'GigaChat-2': ModelConfig(
                client_class=GigaChat,
                token_counter=TokenCounterFactory().create_gigachat_counter(),
                pricing={
                    # 5_000 рублей / 25_000_000 токенов / курс доллара
                    TokenDirection.ENCODE: 5_000 / 25_000_000 / self.usd_rate,
                    TokenDirection.DECODE: 5_000 / 25_000_000 / self.usd_rate,
                },
            ),
            'GigaChat-Pro': ModelConfig(
                client_class=GigaChat,
                token_counter=TokenCounterFactory().create_gigachat_counter(),
                pricing={
                    # 10_500 рублей / 7_000_000 токенов / курс доллара
                    TokenDirection.ENCODE: 10_500 / 7_000_000 / self.usd_rate,
                    TokenDirection.DECODE: 10_500 / 7_000_000 / self.usd_rate,
                },
            ),
            'GigaChat-2-Pro': ModelConfig(
                client_class=GigaChat,
                token_counter=TokenCounterFactory().create_gigachat_counter(),
                pricing={
                    # 10_500 рублей / 7_000_000 токенов / курс доллара
                    TokenDirection.ENCODE: 10_500 / 7_000_000 / self.usd_rate,
                    TokenDirection.DECODE: 10_500 / 7_000_000 / self.usd_rate,
                },
            ),
            'GigaChat-Max': ModelConfig(
                client_class=GigaChat,
                token_counter=TokenCounterFactory().create_gigachat_counter(),
                pricing={
                    # 15_600 рублей / 8_000_000 токенов / курс доллара
                    TokenDirection.ENCODE: 15_600 / 8_000_000 / self.usd_rate,
                    TokenDirection.DECODE: 15_600 / 8_000_000 / self.usd_rate,
                },
            ),
            'GigaChat-2-Max': ModelConfig(
                client_class=GigaChat,
                token_counter=TokenCounterFactory().create_gigachat_counter(),
                pricing={
                    # 15_600 рублей / 8_000_000 токенов / курс доллара
                    TokenDirection.ENCODE: 15_600 / 8_000_000 / self.usd_rate,
                    TokenDirection.DECODE: 15_600 / 8_000_000 / self.usd_rate,
                },
            ),
            # Anthropic
            'claude-3-5-haiku-latest': ModelConfig(
                client_class=ChatAnthropic,
                token_counter=TokenCounterFactory().create_anthropic_counter(),
                pricing={
                    TokenDirection.ENCODE: 0.80 / 1_000_000,
                    TokenDirection.DECODE: 4.00 / 1_000_000,
                },
            ),
            'claude-3-7-sonnet-latest': ModelConfig(
                client_class=ChatAnthropic,
                token_counter=TokenCounterFactory().create_anthropic_counter(),
                pricing={
                    TokenDirection.ENCODE: 3.00 / 1_000_000,
                    TokenDirection.DECODE: 15.00 / 1_000_000,
                },
            ),
            'claude-opus-4-0': ModelConfig(
                client_class=ChatAnthropic,
                token_counter=TokenCounterFactory().create_anthropic_counter(),
                pricing={
                    TokenDirection.ENCODE: 15.00 / 1_000_000,
                    TokenDirection.DECODE: 75.00 / 1_000_000,
                },
            ),
            'claude-sonnet-4-0': ModelConfig(
                client_class=ChatAnthropic,
                token_counter=TokenCounterFactory().create_anthropic_counter(),
                pricing={
                    TokenDirection.ENCODE: 3.00 / 1_000_000,
                    TokenDirection.DECODE: 15.00 / 1_000_000,
                },
            ),
            # Google
            'gemini-2.0-flash-001': ModelConfig(
                client_class=ChatGoogleGenerativeAI,
                token_counter=TokenCounterFactory().create_google_counter(),
                pricing={
                    TokenDirection.ENCODE: 0.10 / 1_000_000,
                    TokenDirection.DECODE: 0.40 / 1_000_000,
                },
            ),
            'gemini-2.5-flash': ModelConfig(
                client_class=ChatGoogleGenerativeAI,
                token_counter=TokenCounterFactory().create_google_counter(),
                pricing={
                    TokenDirection.ENCODE: 0.30 / 1_000_000,
                    TokenDirection.DECODE: 1.00 / 1_000_000,
                },
            ),
            'gemini-2.5-pro-preview-06-05': ModelConfig(
                client_class=ChatGoogleGenerativeAI,
                token_counter=TokenCounterFactory().create_google_counter(),
                pricing={
                    TokenDirection.ENCODE: 2.50 / 1_000_000,
                    TokenDirection.DECODE: 15.00 / 1_000_000,
                },
            ),
            # Groq
            'grok-3-mini': ModelConfig(
                client_class=ChatXAI,
                token_counter=TokenCounterFactory().create_xai_counter(),
                pricing={
                    TokenDirection.ENCODE: 0.30 / 1_000_000,
                    TokenDirection.DECODE: 0.50 / 1_000_000,
                },
            ),
            'grok-3': ModelConfig(
                client_class=ChatXAI,
                token_counter=TokenCounterFactory().create_xai_counter(),
                pricing={
                    TokenDirection.ENCODE: 3.00 / 1_000_000,
                    TokenDirection.DECODE: 15.00 / 1_000_000,
                },
            ),
            'grok-3-fast': ModelConfig(
                client_class=ChatXAI,
                token_counter=TokenCounterFactory().create_xai_counter(),
                pricing={
                    TokenDirection.ENCODE: 5.00 / 1_000_000,
                    TokenDirection.DECODE: 25.00 / 1_000_000,
                },
            ),
            # DeepSeek
            'deepseek-chat': ModelConfig(
                client_class=ChatDeepSeek,
                token_counter=TokenCounterFactory().create_deepseek_counter(),
                pricing={
                    TokenDirection.ENCODE: 0.27 / 1_000_000,
                    TokenDirection.DECODE: 1.10 / 1_000_000,
                },
            ),
            'deepseek-reasoner': ModelConfig(
                client_class=ChatDeepSeek,
                token_counter=TokenCounterFactory().create_deepseek_counter(),
                pricing={
                    TokenDirection.ENCODE: 0.55 / 1_000_000,
                    TokenDirection.DECODE: 2.19 / 1_000_000,
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
        return await self._models[model_name].token_counter(
            messages,
            model_name,
            self.client,
        )

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
