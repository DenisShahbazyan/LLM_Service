import json
from typing import Any, Awaitable, Callable, TypeVar

from langchain.schema import AIMessage, BaseMessage
from pydantic import BaseModel

from llm.counter import TokenCounter
from llm.pydantic_utils.checker import is_pydantic_instance

PydanticSchema = TypeVar('PydanticSchema', bound=BaseModel)


class BillingDecorator:
    """Декоратор для расчета токенов и расходов в USD при вызове LLM-функции."""

    def __init__(
        self,
        func: Callable[..., Awaitable[Any]],
        counter: TokenCounter,
    ) -> None:
        self.func = func
        self.counter = counter

    def __get__(self, instance, owner):
        return lambda *args, **kwargs: self(instance, *args, **kwargs)

    async def __call__(self, *args, **kwargs) -> BaseMessage:
        result: AIMessage | PydanticSchema = await self.func(*args, **kwargs)

        if isinstance(result, AIMessage):
            await self.counter.count_input_tokens(
                result.usage_metadata['input_tokens'],
            )
            await self.counter.count_output_tokens(
                result.usage_metadata['output_tokens'],
            )
        elif is_pydantic_instance(result) or isinstance(result, dict):
            await self.counter.count_input_tokens_from_text(
                kwargs.get('input'),
            )

            if is_pydantic_instance(result):
                result_dict = result.model_dump()
                output_text = json.dumps(result_dict, ensure_ascii=False)
            else:
                output_text = json.dumps(result, ensure_ascii=False)

            await self.counter.count_output_tokens_from_text(
                [AIMessage(content=output_text)],
            )

        return result


class StreamBillingDecorator:
    """Декоратор для расчета токенов и расходов в USD при стриминге LLM."""

    def __init__(
        self,
        counter: TokenCounter,
    ) -> None:
        self.counter = counter

    async def count_input_tokens(self, input_data):
        """Подсчитывает входные токены"""
        await self.counter.count_input_tokens_from_text(input_data)

    async def count_output_tokens(self, full_output_text: str):
        """Подсчитывает выходные токены"""
        await self.counter.count_output_tokens_from_text(
            [AIMessage(content=full_output_text)],
        )
