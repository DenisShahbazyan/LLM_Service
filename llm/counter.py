from langchain_openai import ChatOpenAI
from langchain_gigachat import GigaChat
from langchain_anthropic import ChatAnthropic
import tiktoken

from llm.direction import TokenDirection
from llm.pricing import TokenPricing
from llm.usage import TokenUsage

from langchain.schema import BaseMessage


class TokenCounter:
    """Счетчик токенов и стоимости."""

    def __init__(
        self,
        client: ChatOpenAI | GigaChat | ChatAnthropic,
        usage: TokenUsage,
        pricing: TokenPricing,
    ) -> None:
        self.client = client
        self.model_name = self.__get_model_name()
        self.usage = usage
        self.pricing = pricing

    def __get_model_name(self) -> str:
        """Получение имени модели

        Raises:
            Exception: если модель не инициализирована
            Exception: если модель не найдена

        Returns:
            str: имя модели
        """
        if self.client is None:
            raise Exception('Client is not initialized')

        if isinstance(self.client, ChatOpenAI):
            return self.client.model_name

        if isinstance(self.client, GigaChat) or isinstance(self.client, ChatAnthropic):
            return self.client.model

        raise Exception('Unknown client type')

    def __get_text(self, messages: list[BaseMessage]) -> str:
        full_text = ''
        for message in messages:
            full_text += message.content
        return full_text

    async def __get_tokens(self, messages: list[BaseMessage]) -> int:
        match self.model_name:
            case 'gpt-4o-mini' | 'gpt-4o':
                encoding = tiktoken.encoding_for_model(self.model_name)
                text = self.__get_text(messages)
                return len(encoding.encode(text))
            case 'GigaChat-Pro' | 'GigaChat-2-Max':
                text = self.__get_text(messages)
                response = await self.client.atokens_count([text], self.model_name)
                return response[0].tokens
            case 'claude-3-5-haiku-latest':
                return self.client.get_num_tokens_from_messages(messages)
            case _:
                raise Exception(f'No model found {self.model_name}')

    async def count_input_tokens_from_text(self, messages: list[BaseMessage]) -> None:
        tokens = await self.__get_tokens(messages)
        await self.count_input_tokens(tokens)

    async def count_output_tokens_from_text(self, messages: list[BaseMessage]) -> None:
        tokens = await self.__get_tokens(messages)
        await self.count_output_tokens(tokens)

    async def count_input_tokens(self, tokens: int) -> None:
        await self._enc_spendings(tokens)
        self.usage.all_input_tokens += tokens
        self.usage.last_input_tokens = tokens

    async def count_output_tokens(self, tokens: int) -> None:
        await self._dec_spendings(tokens)
        self.usage.all_output_tokens += tokens
        self.usage.last_output_tokens = tokens

    async def _enc_spendings(self, tokens: int) -> None:
        """Считает расходы в USD при отправке в LLM.

        Args:
            tokens (int): токены
        """
        toc_price = await self.pricing.get_token_price(
            self.model_name, TokenDirection.ENCODE
        )
        self.usage.all_input_spendings += tokens * toc_price
        self.usage.last_input_spendings = tokens * toc_price

    async def _dec_spendings(self, tokens: int) -> None:
        """Считает расходы в USD при получении из LLM.

        Args:
            tokens (int): токены
        """
        toc_price = await self.pricing.get_token_price(
            self.model_name, TokenDirection.DECODE
        )
        self.usage.all_output_spendings += tokens * toc_price
        self.usage.last_output_spendings = tokens * toc_price
