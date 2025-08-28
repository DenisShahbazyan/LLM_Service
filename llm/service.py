from typing import Any, AsyncGenerator, Literal, Self

from langchain.schema import BaseMessage

from llm.billing import BillingDecorator, StreamBillingDecorator
from llm.cbr.cbr import CBRRate
from llm.counter import TokenCounter
from llm.model_registry import ModelRegistry
from llm.prepare_chat import PrepareChat
from llm.types import LLMClientInstance
from llm.usage import TokenUsage


class StreamResult:
    def __init__(self, generator: AsyncGenerator[str, None]):
        self._generator = generator
        self._full_text = ''
        self._completed = False

    def __aiter__(self):
        return self

    async def __anext__(self):
        try:
            chunk = await self._generator.__anext__()
            self._full_text += chunk
            return chunk
        except StopAsyncIteration:
            self._completed = True
            raise

    @property
    def full_text(self) -> str:
        """Возвращает полный текст. Доступен только после завершения итерации."""
        return self._full_text

    @property
    def is_completed(self) -> bool:
        """Проверяет, завершился ли стрим"""
        return self._completed


class LLMService:
    def __init__(self, config: dict, usd_rate: float = None) -> None:
        self.config = config

        self.model_registry = ModelRegistry(usd_rate, config)
        self.client: LLMClientInstance = self.model_registry.init_client()

        self.usage = TokenUsage()
        self.counter = TokenCounter(
            model_name=config.get('model'),
            usage=self.usage,
            model_registry=self.model_registry,
        )

        self.__is_structured_output = False

    @classmethod
    async def create(cls, config: dict) -> Self:
        # Получаем курс доллара
        cbr = CBRRate()
        usd_rate = await cbr.get_usd_rate()

        # Создаем экземпляр класса с уже полученным курсом
        instance = cls(config, usd_rate)
        return instance

    async def _moderation_check(
        self,
        moderation: bool,
        chat_for_model: list[BaseMessage],
    ) -> None:
        if moderation:
            await self.model_registry.get_moderation(
                self.config.get('model'),
                chat_for_model,
            )

    async def test_connection(self) -> bool | None:
        return await self.model_registry.get_test_connections(self.config.get('model'))

    async def ainvoke(
        self,
        chat_history: list[dict[str, str] | BaseMessage] | None = None,
        system_prompt: str | BaseMessage | None = None,
        message: str | BaseMessage | None = None,
        moderation: bool = False,
        **kwargs,
    ) -> str:
        chat_for_model = PrepareChat(chat_history, system_prompt, message)

        await self._moderation_check(moderation, chat_for_model)

        billing_invoke = BillingDecorator(self.client.ainvoke, self.counter)
        result = await billing_invoke(input=chat_for_model, **kwargs)

        if self.__is_structured_output:
            return result

        return result.content

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
        new_instance.__is_structured_output = True

        return new_instance

    async def astream(
        self,
        chat_history: list[dict[str, str] | BaseMessage] | None = None,
        system_prompt: str | BaseMessage | None = None,
        message: str | BaseMessage | None = None,
        moderation: bool = False,
        **kwargs,
    ) -> StreamResult:
        chat_for_model = PrepareChat(chat_history, system_prompt, message)

        await self._moderation_check(moderation, chat_for_model)

        billing_stream = StreamBillingDecorator(self.client.astream, self.counter)

        async def content_generator():
            stream = billing_stream(input=chat_for_model, **kwargs)
            async for chunk in stream:
                yield chunk.content

        return StreamResult(content_generator())
