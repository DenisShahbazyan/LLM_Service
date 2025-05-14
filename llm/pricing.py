from llm.direction import TokenDirection
from dataclasses import dataclass, field


@dataclass
class TokenPricing:
    """Класс для расчета стоимости токенов.
    Актуальные цены: https://openai.com/api/pricing/
    Актуальные цены: https://developers.sber.ru/docs/ru/gigachat/api/tariffs
    """

    usd_rate: float
    pricing_config: dict = field(init=False)

    def __post_init__(self):
        self.pricing_config = {
            'gpt-4o-mini': {
                TokenDirection.ENCODE: 0.15 / 1e6,
                TokenDirection.DECODE: 0.6 / 1e6,
            },
            'gpt-4o': {
                TokenDirection.ENCODE: 2.5 / 1e6,
                TokenDirection.DECODE: 10.0 / 1e6,
            },
            'GigaChat-Pro': {
                # 10_500 рублей / 7_000_000 токенов / курс доллара
                TokenDirection.ENCODE: 10_500 / 7_000_000 / self.usd_rate,
                TokenDirection.DECODE: 10_500 / 7_000_000 / self.usd_rate,
            },
            'GigaChat-2-Max': {
                # 15_600 рублей / 8_000_000 токенов / курс доллара
                TokenDirection.ENCODE: 15_600 / 8_000_000 / self.usd_rate,
                TokenDirection.DECODE: 15_600 / 8_000_000 / self.usd_rate,
            },
            'claude-3-5-haiku-latest': {
                TokenDirection.ENCODE: 0.0012,
                TokenDirection.DECODE: 0.0012,
            },
        }

    async def get_token_price(
        self, model_name: str, direction: TokenDirection
    ) -> float:
        """Получение стоимости токена.

        Args:
            model_name (str): LLM модель
            direction (TokenDirection): направление подсчета

        Raises:
            Exception: если модель не найдена

        Returns:
            float: стоимость токена
        """
        try:
            return self.pricing_config[model_name][direction]
        except KeyError:
            raise Exception(f'No model found {model_name}')
