from langchain_google_genai import ChatGoogleGenerativeAI

from llm.direction import TokenDirection
from llm.providers._base import BaseProvider, ModelConfig
from llm.test_connections import TestConnections
from llm.token_counters import TokenCounterFactory


class GoogleProvider(BaseProvider):
    """Провайдер для моделей Google"""

    @property
    def name(self) -> str:
        return 'Google'

    def get_models(self) -> dict[str, ModelConfig]:
        """Возвращает конфигурации всех моделей Google"""
        return {
            'gemini-2.0-flash-001': ModelConfig(
                client_class=ChatGoogleGenerativeAI,
                token_counter=TokenCounterFactory().create_google_counter(),
                test_connection=TestConnections().google,
                pricing={
                    TokenDirection.ENCODE: 0.10 / 1_000_000,
                    TokenDirection.DECODE: 0.40 / 1_000_000,
                },
            ),
            'gemini-2.5-flash': ModelConfig(
                client_class=ChatGoogleGenerativeAI,
                token_counter=TokenCounterFactory().create_google_counter(),
                test_connection=TestConnections().google,
                pricing={
                    TokenDirection.ENCODE: 0.30 / 1_000_000,
                    TokenDirection.DECODE: 1.00 / 1_000_000,
                },
            ),
            'gemini-2.5-pro-preview-06-05': ModelConfig(
                client_class=ChatGoogleGenerativeAI,
                token_counter=TokenCounterFactory().create_google_counter(),
                test_connection=TestConnections().google,
                pricing={
                    TokenDirection.ENCODE: 2.50 / 1_000_000,
                    TokenDirection.DECODE: 15.00 / 1_000_000,
                },
            ),
        }
