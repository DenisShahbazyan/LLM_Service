from langchain_deepseek import ChatDeepSeek

from llm.direction import TokenDirection
from llm.providers._base import BaseProvider, ModelConfig
from llm.test_connections import TestConnections
from llm.token_counters import TokenCounterFactory


class DeepSeekProvider(BaseProvider):
    """Провайдер для моделей DeepSeek"""

    @property
    def name(self) -> str:
        return 'DeepSeek'

    def get_models(self) -> dict[str, ModelConfig]:
        """Возвращает конфигурации всех моделей DeepSeek"""
        return {
            'deepseek-chat': ModelConfig(
                client_class=ChatDeepSeek,
                token_counter=TokenCounterFactory().create_deepseek_counter(),
                test_connection=TestConnections().deepseek,
                pricing={
                    TokenDirection.ENCODE: 0.27 / 1_000_000,
                    TokenDirection.DECODE: 1.10 / 1_000_000,
                },
            ),
            'deepseek-reasoner': ModelConfig(
                client_class=ChatDeepSeek,
                token_counter=TokenCounterFactory().create_deepseek_counter(),
                test_connection=TestConnections().deepseek,
                pricing={
                    TokenDirection.ENCODE: 0.55 / 1_000_000,
                    TokenDirection.DECODE: 2.19 / 1_000_000,
                },
            ),
        }
