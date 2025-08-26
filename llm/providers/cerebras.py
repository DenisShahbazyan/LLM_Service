from langchain_cerebras import ChatCerebras

from llm.direction import TokenDirection
from llm.providers._base import BaseProvider, ModelConfig
from llm.test_connections import TestConnections
from llm.token_counters import TokenCounterFactory


class CerebrasProvider(BaseProvider):
    """Провайдер для моделей Cerebras"""

    @property
    def name(self) -> str:
        return 'Cerebras'

    def get_models(self) -> dict[str, ModelConfig]:
        """Возвращает конфигурации всех моделей Cerebras"""
        return {
            'gpt-oss-120b': ModelConfig(
                client_class=ChatCerebras,
                token_counter=TokenCounterFactory().create_cerebas_counter(),
                test_connection=TestConnections().cerebras,
                pricing={
                    TokenDirection.ENCODE: 0.25 / 1_000_000,
                    TokenDirection.DECODE: 0.69 / 1_000_000,
                },
            ),
            'qwen-3-32b': ModelConfig(
                client_class=ChatCerebras,
                token_counter=TokenCounterFactory().create_cerebas_counter(),
                test_connection=TestConnections().cerebras,
                pricing={
                    TokenDirection.ENCODE: 0.40 / 1_000_000,
                    TokenDirection.DECODE: 0.80 / 1_000_000,
                },
            ),
            'llama-4-scout-17b-16e-instruct': ModelConfig(
                client_class=ChatCerebras,
                token_counter=TokenCounterFactory().create_cerebas_counter(),
                test_connection=TestConnections().cerebras,
                pricing={
                    TokenDirection.ENCODE: 0.65 / 1_000_000,
                    TokenDirection.DECODE: 0.85 / 1_000_000,
                },
            ),
            'llama-4-maverick-17b-128e-instruct': ModelConfig(
                client_class=ChatCerebras,
                token_counter=TokenCounterFactory().create_cerebas_counter(),
                test_connection=TestConnections().cerebras,
                pricing={
                    TokenDirection.ENCODE: 0.20 / 1_000_000,
                    TokenDirection.DECODE: 0.60 / 1_000_000,
                },
            ),
        }
