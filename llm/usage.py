from dataclasses import dataclass


@dataclass
class TokenUsage:
    """Класс для хранения информации об использовании токенов и стоимости."""

    all_input_tokens: int = 0
    all_output_tokens: int = 0

    last_input_tokens: int = 0
    last_output_tokens: int = 0

    all_input_spendings: float = 0
    all_output_spendings: float = 0

    last_input_spendings: float = 0
    last_output_spendings: float = 0

    @property
    def total_tokens(self):
        return self.all_input_tokens + self.all_output_tokens

    @property
    def total_spendings(self):
        return self.all_input_spendings + self.all_output_spendings

    @property
    def last_tokens(self):
        return self.last_input_tokens + self.last_output_tokens

    @property
    def last_spendings(self):
        return self.last_input_spendings + self.last_output_spendings
