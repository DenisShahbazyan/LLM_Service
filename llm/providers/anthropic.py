from langchain_anthropic import ChatAnthropic

from llm.direction import TokenDirection
from llm.providers._base import BaseProvider, ModelConfig
from llm.test_connections import TestConnections
from llm.token_counters import TokenCounterFactory


class AnthropicProvider(BaseProvider):
    """Провайдер для моделей Anthropic"""

    @property
    def name(self) -> str:
        return 'Anthropic'

    def get_models(self) -> dict[str, ModelConfig]:
        """Возвращает конфигурации всех моделей Anthropic"""
        return {
            'claude-3-5-haiku-latest': ModelConfig(
                client_class=ChatAnthropic,
                token_counter=TokenCounterFactory().create_anthropic_counter(),
                test_connection=TestConnections().anthropic,
                pricing={
                    TokenDirection.ENCODE: 0.80 / 1_000_000,
                    TokenDirection.DECODE: 4.00 / 1_000_000,
                },
            ),
            'claude-3-7-sonnet-latest': ModelConfig(
                client_class=ChatAnthropic,
                token_counter=TokenCounterFactory().create_anthropic_counter(),
                test_connection=TestConnections().anthropic,
                pricing={
                    TokenDirection.ENCODE: 3.00 / 1_000_000,
                    TokenDirection.DECODE: 15.00 / 1_000_000,
                },
            ),
            'claude-opus-4-0': ModelConfig(
                client_class=ChatAnthropic,
                token_counter=TokenCounterFactory().create_anthropic_counter(),
                test_connection=TestConnections().anthropic,
                pricing={
                    TokenDirection.ENCODE: 15.00 / 1_000_000,
                    TokenDirection.DECODE: 75.00 / 1_000_000,
                },
            ),
            'claude-sonnet-4-0': ModelConfig(
                client_class=ChatAnthropic,
                token_counter=TokenCounterFactory().create_anthropic_counter(),
                test_connection=TestConnections().anthropic,
                pricing={
                    TokenDirection.ENCODE: 3.00 / 1_000_000,
                    TokenDirection.DECODE: 15.00 / 1_000_000,
                },
            ),
        }
