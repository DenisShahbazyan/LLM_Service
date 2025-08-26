from langchain_gigachat import GigaChat

from llm.direction import TokenDirection
from llm.providers._base import BaseProvider, ModelConfig
from llm.test_connections import TestConnections
from llm.token_counters import TokenCounterFactory


class GigaChatProvider(BaseProvider):
    """Провайдер для моделей GigaChat"""

    @property
    def name(self) -> str:
        return 'GigaChat'

    def get_models(self) -> dict[str, ModelConfig]:
        """Возвращает конфигурации всех моделей GigaChat"""
        return {
            'GigaChat': ModelConfig(
                client_class=GigaChat,
                token_counter=TokenCounterFactory().create_gigachat_counter(),
                test_connection=TestConnections().gigachat,
                pricing={
                    # 5_000 рублей / 25_000_000 токенов / курс доллара
                    TokenDirection.ENCODE: 5_000 / 25_000_000 / self.usd_rate,
                    TokenDirection.DECODE: 5_000 / 25_000_000 / self.usd_rate,
                },
            ),
            'GigaChat-2': ModelConfig(
                client_class=GigaChat,
                token_counter=TokenCounterFactory().create_gigachat_counter(),
                test_connection=TestConnections().gigachat,
                pricing={
                    # 5_000 рублей / 25_000_000 токенов / курс доллара
                    TokenDirection.ENCODE: 5_000 / 25_000_000 / self.usd_rate,
                    TokenDirection.DECODE: 5_000 / 25_000_000 / self.usd_rate,
                },
            ),
            'GigaChat-Pro': ModelConfig(
                client_class=GigaChat,
                token_counter=TokenCounterFactory().create_gigachat_counter(),
                test_connection=TestConnections().gigachat,
                pricing={
                    # 10_500 рублей / 7_000_000 токенов / курс доллара
                    TokenDirection.ENCODE: 10_500 / 7_000_000 / self.usd_rate,
                    TokenDirection.DECODE: 10_500 / 7_000_000 / self.usd_rate,
                },
            ),
            'GigaChat-2-Pro': ModelConfig(
                client_class=GigaChat,
                token_counter=TokenCounterFactory().create_gigachat_counter(),
                test_connection=TestConnections().gigachat,
                pricing={
                    # 10_500 рублей / 7_000_000 токенов / курс доллара
                    TokenDirection.ENCODE: 10_500 / 7_000_000 / self.usd_rate,
                    TokenDirection.DECODE: 10_500 / 7_000_000 / self.usd_rate,
                },
            ),
            'GigaChat-Max': ModelConfig(
                client_class=GigaChat,
                token_counter=TokenCounterFactory().create_gigachat_counter(),
                test_connection=TestConnections().gigachat,
                pricing={
                    # 15_600 рублей / 8_000_000 токенов / курс доллара
                    TokenDirection.ENCODE: 15_600 / 8_000_000 / self.usd_rate,
                    TokenDirection.DECODE: 15_600 / 8_000_000 / self.usd_rate,
                },
            ),
            'GigaChat-2-Max': ModelConfig(
                client_class=GigaChat,
                token_counter=TokenCounterFactory().create_gigachat_counter(),
                test_connection=TestConnections().gigachat,
                pricing={
                    # 15_600 рублей / 8_000_000 токенов / курс доллара
                    TokenDirection.ENCODE: 15_600 / 8_000_000 / self.usd_rate,
                    TokenDirection.DECODE: 15_600 / 8_000_000 / self.usd_rate,
                },
            ),
        }
