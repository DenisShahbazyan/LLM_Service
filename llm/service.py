from contextlib import asynccontextmanager
from functools import partial
from typing import Any, AsyncGenerator, Literal, Self

from langchain.schema import BaseMessage

from llm.billing import BillingDecorator
from llm.cbr.cbr import CBRRate
from llm.counter import TokenCounter
from llm.model_registry import ModelRegistry
from llm.prepare_chat import PrepareChat
from llm.usage import TokenUsage


class LLMService:
    """
    Класс для работы с LLM.

    После инициализации
        `llm = await LLMService.create(get_llm_config('gpt-4o-mini'))`

    будут доступны такие поля:
        `llm.usage.all_input_tokens` - общее количество отправленных токенов
            с момента инициализации
        `llm.usage.all_output_tokens`  - общее количество полученных токенов
            с момента инициализации
        `llm.usage.last_input_tokens` - количество отправленных токенов
            за последний вызов
        `llm.usage.last_output_tokens` - количество полученных токенов
            за последний вызов

        ---

        `llm.usage.all_input_spendings` - общие расходы в USD при отправке
            с момента инициализации
        `llm.usage.all_output_spendings` - общие расходы в USD при получении
            с момента инициализации
        `llm.usage.last_input_spendings` - расходы в USD при отправке
            за последний вызов
        `llm.usage.last_output_spendings` - расходы в USD при получении
            за последний вызов

        ---

        `llm.counter.model_registry.usd_rate` - курс валюты в USD.
    """

    def __init__(self, config: dict, usd_rate: float = None) -> None:
        self.config = config
        self.model_registry = ModelRegistry(usd_rate)

        self.client = self.model_registry.init_client(config)
        self.usage = TokenUsage()
        self._is_structured_output = False

        # Инициализируем pricing с предоставленным курсом доллара
        self.counter = TokenCounter(
            model_name=config.get('model'),
            usage=self.usage,
            model_registry=self.model_registry,
        )

        self.__ainvoke = partial(BillingDecorator(self.__ainvoke, self.counter))

    @classmethod
    async def create(cls, config: dict) -> Self:
        # Получаем курс доллара
        cbr = CBRRate()
        usd_rate = await cbr.get_usd_rate()

        # Создаем экземпляр класса с уже полученным курсом
        instance = cls(config, usd_rate)
        return instance

    async def ainvoke(
        self,
        chat_history: list[dict[str, str] | BaseMessage] | None = None,
        system_prompt: str | BaseMessage | None = None,
        message: str | BaseMessage | None = None,
        **kwargs,
    ) -> str:
        chat_for_model = PrepareChat(chat_history, system_prompt, message)
        result = await self.__ainvoke(chat_for_model=chat_for_model, **kwargs)

        if self._is_structured_output:
            return result

        return result.content

    async def __ainvoke(
        self, *, chat_for_model: list[BaseMessage], **kwargs
    ) -> BaseMessage:
        return await self.client.ainvoke(chat_for_model, **kwargs)

    async def with_structured_output(
        self,
        schema: dict | type,
        *,
        method: Literal[
            'function_calling', 'json_mode', 'format_instructions'
        ] = 'function_calling',
        include_raw: bool = False,
        **kwargs: Any,
    ) -> Self:
        new_instance = await LLMService.create(self.config)

        new_instance.client = self.client.with_structured_output(
            schema, method=method, include_raw=include_raw, **kwargs
        )
        new_instance._is_structured_output = True

        return new_instance

    async def astream(
        self,
        chat_history: list[dict[str, str] | BaseMessage] | None = None,
        system_prompt: str | BaseMessage | None = None,
        message: str | BaseMessage | None = None,
        **kwargs,
    ) -> AsyncGenerator[str, None]:
        chat_for_model = PrepareChat(chat_history, system_prompt, message)
        async for chunk in self.__astream(chat_for_model=chat_for_model, **kwargs):
            if hasattr(chunk, 'content') and chunk.content:
                yield chunk.content

    async def __astream(self, *, chat_for_model: list[BaseMessage], **kwargs):
        async for chunk in self.client.astream(chat_for_model, **kwargs):
            yield chunk

    @asynccontextmanager
    async def astream_mgr(
        self,
        chat_history: list[dict[str, str] | BaseMessage] | None = None,
        system_prompt: str | BaseMessage | None = None,
        message: str | BaseMessage | None = None,
        **kwargs,
    ) -> AsyncGenerator[AsyncGenerator[str, None], None]:
        """
        Контекстный менеджер для потоковой генерации описания.

        Пример использования:
        ```
        async with llm.astream_mgr(system_prompt=prompt) as stream:
            async for chunk in stream:
                # обработка chunk
        ```
        """
        try:
            generator = self.astream(
                chat_history=chat_history,
                system_prompt=system_prompt,
                message=message,
                **kwargs,
            )

            yield generator

        finally:
            pass
