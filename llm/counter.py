from langchain.schema import BaseMessage

from llm.direction import TokenDirection
from llm.model_registry import ModelRegistry
from llm.usage import TokenUsage


class TokenCounter:
    """Счетчик токенов и стоимости."""

    def __init__(
        self,
        model_name: str,
        usage: TokenUsage,
        model_registry: ModelRegistry,
    ) -> None:
        self.model_name = model_name
        self.usage = usage
        self.model_registry = model_registry

    async def count_input_tokens_from_text(self, messages: list[BaseMessage]) -> None:
        """Считает количество токенов из текста, которое отправляются в LLM.

        Args:
            messages (list[BaseMessage]): Сообщения
        """
        tokens = await self.model_registry.get_tokens(self.model_name, messages)
        await self.count_input_tokens(tokens)

    async def count_output_tokens_from_text(self, messages: list[BaseMessage]) -> None:
        """Считает количество токенов из текста, которое получаются из LLM.

        Args:
            messages (list[BaseMessage]): Сообщения
        """
        tokens = await self.model_registry.get_tokens(self.model_name, messages)
        await self.count_output_tokens(tokens)

    async def count_input_tokens(self, tokens: int) -> None:
        """Считает количество токенов, которое отправляются в LLM.

        Args:
            tokens (int): токены
        """
        await self._enc_spendings(tokens)
        self.usage.all_input_tokens += tokens
        self.usage.last_input_tokens = tokens

    async def count_output_tokens(self, tokens: int) -> None:
        """Считает количество токенов, которое получаются из LLM.

        Args:
            tokens (int): токены
        """
        await self._dec_spendings(tokens)
        self.usage.all_output_tokens += tokens
        self.usage.last_output_tokens = tokens

    async def _enc_spendings(self, tokens: int) -> None:
        """Считает расходы в USD при отправке в LLM.

        Args:
            tokens (int): токены
        """
        toc_price = self.model_registry.get_price(
            self.model_name, TokenDirection.ENCODE
        )

        self.usage.all_input_spendings += tokens * toc_price
        self.usage.last_input_spendings = tokens * toc_price

    async def _dec_spendings(self, tokens: int) -> None:
        """Считает расходы в USD при получении из LLM.

        Args:
            tokens (int): токены
        """
        toc_price = self.model_registry.get_price(
            self.model_name, TokenDirection.DECODE
        )

        self.usage.all_output_spendings += tokens * toc_price
        self.usage.last_output_spendings = tokens * toc_price
