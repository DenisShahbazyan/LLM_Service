from langchain_xai import ChatXAI

from llm.direction import TokenDirection
from llm.providers._base import BaseProvider, ModelConfig
from llm.test_connections import TestConnections
from llm.token_counters import TokenCounterFactory


class XAIProvider(BaseProvider):
    """Провайдер для моделей XAI (Grok)"""

    @property
    def name(self) -> str:
        return 'XAI'

    def get_models(self) -> dict[str, ModelConfig]:
        """Возвращает конфигурации всех моделей XAI"""
        return {
            'grok-3-mini': ModelConfig(
                client_class=ChatXAI,
                token_counter=TokenCounterFactory().create_xai_counter(),
                test_connection=TestConnections().xai,
                pricing={
                    TokenDirection.ENCODE: 0.30 / 1_000_000,
                    TokenDirection.DECODE: 0.50 / 1_000_000,
                },
            ),
            'grok-3': ModelConfig(
                client_class=ChatXAI,
                token_counter=TokenCounterFactory().create_xai_counter(),
                test_connection=TestConnections().xai,
                pricing={
                    TokenDirection.ENCODE: 3.00 / 1_000_000,
                    TokenDirection.DECODE: 15.00 / 1_000_000,
                },
            ),
            'grok-3-fast': ModelConfig(
                client_class=ChatXAI,
                token_counter=TokenCounterFactory().create_xai_counter(),
                test_connection=TestConnections().xai,
                pricing={
                    TokenDirection.ENCODE: 5.00 / 1_000_000,
                    TokenDirection.DECODE: 25.00 / 1_000_000,
                },
            ),
        }
