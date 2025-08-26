from langchain_openai import ChatOpenAI

from llm.direction import TokenDirection
from llm.moderations import ModerationPrompt
from llm.providers._base import BaseProvider, ModelConfig
from llm.test_connections import TestConnections
from llm.token_counters import TokenCounterFactory


class OpenAIProvider(BaseProvider):
    """Провайдер для моделей OpenAI"""

    @property
    def name(self) -> str:
        return 'OpenAI'

    def get_models(self) -> dict[str, ModelConfig]:
        """Возвращает конфигурации всех моделей OpenAI"""
        return {
            'gpt-5': ModelConfig(
                client_class=ChatOpenAI,
                token_counter=TokenCounterFactory().create_openai_counter(),
                moderation=ModerationPrompt().openai,
                test_connection=TestConnections().openai,
                pricing={
                    TokenDirection.ENCODE: 1.25 / 1_000_000,
                    TokenDirection.DECODE: 10.00 / 1_000_000,
                },
            ),
            'gpt-5-mini': ModelConfig(
                client_class=ChatOpenAI,
                token_counter=TokenCounterFactory().create_openai_counter(),
                moderation=ModerationPrompt().openai,
                test_connection=TestConnections().openai,
                pricing={
                    TokenDirection.ENCODE: 0.25 / 1_000_000,
                    TokenDirection.DECODE: 2.00 / 1_000_000,
                },
            ),
            'gpt-5-nano': ModelConfig(
                client_class=ChatOpenAI,
                token_counter=TokenCounterFactory().create_openai_counter(),
                moderation=ModerationPrompt().openai,
                test_connection=TestConnections().openai,
                pricing={
                    TokenDirection.ENCODE: 0.05 / 1_000_000,
                    TokenDirection.DECODE: 0.40 / 1_000_000,
                },
            ),
            'gpt-5-chat-latest': ModelConfig(
                client_class=ChatOpenAI,
                token_counter=TokenCounterFactory().create_openai_counter(),
                moderation=ModerationPrompt().openai,
                test_connection=TestConnections().openai,
                pricing={
                    TokenDirection.ENCODE: 1.25 / 1_000_000,
                    TokenDirection.DECODE: 10.00 / 1_000_000,
                },
            ),
            'gpt-4.1': ModelConfig(
                client_class=ChatOpenAI,
                token_counter=TokenCounterFactory().create_openai_counter(),
                moderation=ModerationPrompt().openai,
                test_connection=TestConnections().openai,
                pricing={
                    TokenDirection.ENCODE: 2.00 / 1_000_000,
                    TokenDirection.DECODE: 8.00 / 1_000_000,
                },
            ),
            'gpt-4.1-mini': ModelConfig(
                client_class=ChatOpenAI,
                token_counter=TokenCounterFactory().create_openai_counter(),
                moderation=ModerationPrompt().openai,
                test_connection=TestConnections().openai,
                pricing={
                    TokenDirection.ENCODE: 0.40 / 1_000_000,
                    TokenDirection.DECODE: 1.60 / 1_000_000,
                },
            ),
            'gpt-4.1-nano': ModelConfig(
                client_class=ChatOpenAI,
                token_counter=TokenCounterFactory().create_openai_counter(),
                moderation=ModerationPrompt().openai,
                test_connection=TestConnections().openai,
                pricing={
                    TokenDirection.ENCODE: 0.10 / 1_000_000,
                    TokenDirection.DECODE: 0.40 / 1_000_000,
                },
            ),
            'gpt-4.5-preview': ModelConfig(
                client_class=ChatOpenAI,
                token_counter=TokenCounterFactory().create_openai_counter(),
                moderation=ModerationPrompt().openai,
                test_connection=TestConnections().openai,
                pricing={
                    TokenDirection.ENCODE: 75.00 / 1_000_000,
                    TokenDirection.DECODE: 150.00 / 1_000_000,
                },
            ),
            'gpt-4o-mini': ModelConfig(
                client_class=ChatOpenAI,
                token_counter=TokenCounterFactory().create_openai_counter(),
                moderation=ModerationPrompt().openai,
                test_connection=TestConnections().openai,
                pricing={
                    TokenDirection.ENCODE: 0.15 / 1_000_000,
                    TokenDirection.DECODE: 0.60 / 1_000_000,
                },
            ),
            'gpt-4o': ModelConfig(
                client_class=ChatOpenAI,
                token_counter=TokenCounterFactory().create_openai_counter(),
                moderation=ModerationPrompt().openai,
                test_connection=TestConnections().openai,
                pricing={
                    TokenDirection.ENCODE: 2.50 / 1_000_000,
                    TokenDirection.DECODE: 10.00 / 1_000_000,
                },
            ),
            'o3-2025-04-16': ModelConfig(
                client_class=ChatOpenAI,
                token_counter=TokenCounterFactory().create_openai_counter(),
                moderation=ModerationPrompt().openai,
                test_connection=TestConnections().openai,
                pricing={
                    TokenDirection.ENCODE: 2.00 / 1_000_000,
                    TokenDirection.DECODE: 8.00 / 1_000_000,
                },
            ),
            'o4-mini-2025-04-16': ModelConfig(
                client_class=ChatOpenAI,
                token_counter=TokenCounterFactory().create_openai_counter(),
                moderation=ModerationPrompt().openai,
                test_connection=TestConnections().openai,
                pricing={
                    TokenDirection.ENCODE: 1.10 / 1_000_000,
                    TokenDirection.DECODE: 4.40 / 1_000_000,
                },
            ),
        }
