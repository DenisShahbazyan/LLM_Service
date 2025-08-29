import inspect
import json
from typing import Any, AsyncGenerator, Self

from langchain.schema import AIMessage, BaseMessage
from pydantic import BaseModel

from llm.billing import BillingDecorator, StreamBillingDecorator
from llm.cbr.cbr import CBRRate
from llm.counter import TokenCounter
from llm.model_registry import ModelRegistry
from llm.prepare_chat import PrepareChat
from llm.types import LLMClientInstance
from llm.usage import TokenUsage


class StreamResult:
    def __init__(
        self,
        generator: AsyncGenerator[str, None],
        get_full_text: callable = None,
    ):
        self._generator = generator
        self._get_full_text = get_full_text

    def __aiter__(self):
        return self

    async def __anext__(self):
        try:
            chunk = await self._generator.__anext__()
            return chunk
        except StopAsyncIteration:
            raise

    @property
    def full_text(self) -> str:
        """Возвращает полный текст. Доступен только после завершения итерации."""
        if self._get_full_text:
            return self._get_full_text()
        return ''


# TODO: Исправить подсчет токенов для with_structured_output. Сейчас токены считаются
# просто из выходного текста, а нужно считать по тому, что было вызвано в модели
# (function или tool или что-то еще)
# TODO: Добавить возможность использования всех моделей провайдера, или кастомных
class LLMService:
    def __init__(self, config: dict, usd_rate: float = None) -> None:
        self.config = config
        self.usd_rate = usd_rate

        self.model_registry = ModelRegistry(usd_rate, config)
        self.client: LLMClientInstance = self.model_registry.init_client()

        self.usage = TokenUsage()
        self.counter = TokenCounter(
            model_name=config.get('model'),
            usage=self.usage,
            model_registry=self.model_registry,
        )

        self.__is_structured_output = False

        self.__chat_history: list[BaseMessage] = []
        self.__last_response: AIMessage | None = None

    @classmethod
    async def create(cls, config: dict) -> Self:
        # Получаем курс доллара
        cbr = CBRRate()
        usd_rate = await cbr.get_usd_rate()

        # Создаем экземпляр класса с уже полученным курсом
        instance = cls(config, usd_rate)
        return instance

    async def __moderation_check(
        self,
        moderation: bool,
        chat_for_model: list[BaseMessage],
    ) -> None:
        if moderation:
            await self.model_registry.get_moderation(
                self.config.get('model'),
                chat_for_model,
            )

    def __serialize_structured_result(self, result: BaseModel | dict[str, Any]) -> str:
        if hasattr(result, 'model_dump'):
            return result.model_dump_json()
        elif isinstance(result, dict):
            return json.dumps(result, ensure_ascii=False)
        else:
            return str(result)

    def _save_chat_history(
        self, chat_for_model: list[BaseMessage], answer: AIMessage
    ) -> None:
        """Сохраняет историю чата и ответ модели."""
        self.__chat_history = chat_for_model.copy()
        self.__last_response = answer

    @property
    def chat_json(self) -> str | None:
        """Возвращает историю чата в JSON формате."""
        if not self.__chat_history or not self.__last_response:
            return None

        result = []
        for message in self.__chat_history:
            result.append(
                {
                    'type': message.type,
                    'content': message.content,
                },
            )
        result.append(
            {
                'type': self.__last_response.type,
                'content': self.__last_response.content,
            },
        )
        return json.dumps(result, ensure_ascii=False)

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

        await self.__moderation_check(moderation, chat_for_model)

        billing_invoke = BillingDecorator(self.client.ainvoke, self.counter)
        result = await billing_invoke(input=chat_for_model, **kwargs)

        if self.__is_structured_output:
            content = self.__serialize_structured_result(result)
            ai_message = AIMessage(content=content)
            self._save_chat_history(chat_for_model, ai_message)
            return result

        self._save_chat_history(chat_for_model, result)
        return result.content

    async def with_structured_output(
        self,
        schema: dict | type,
        *,
        method: str = None,
        include_raw: bool = None,
        strict: bool = None,
        tools: list = None,
        **kwargs: Any,
    ) -> Self:
        # Создаем новый экземпляр
        new_instance = self.__class__(self.config, self.usd_rate)

        # Получаем сигнатуру метода провайдера
        provider_method = self.client.with_structured_output
        sig = inspect.signature(provider_method)
        supported_params = set(sig.parameters.keys())

        # Собираем только явно переданные аргументы, которые поддерживает провайдер
        provider_kwargs = {'schema': schema}

        # Добавляем только явно заданные опциональные параметры
        optional_params = {
            'method': method,
            'include_raw': include_raw,
            'strict': strict,
            'tools': tools,
        }

        for param_name, param_value in optional_params.items():
            if param_value is not None and param_name in supported_params:
                provider_kwargs[param_name] = param_value

        # Добавляем остальные kwargs если они поддерживаются
        for key, value in kwargs.items():
            if key in supported_params:
                provider_kwargs[key] = value

        # Создаем structured output клиент
        new_instance.client = provider_method(**provider_kwargs)
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

        await self.__moderation_check(moderation, chat_for_model)

        billing = StreamBillingDecorator(self.counter)
        full_output_text = ''

        async def content_generator():
            nonlocal full_output_text

            stream = self.client.astream(input=chat_for_model, **kwargs)
            async for chunk in stream:
                full_output_text += chunk.content
                yield chunk.content

            # Подсчитываем токены после завершения стрима
            await billing.count_input_tokens(chat_for_model)
            await billing.count_output_tokens(full_output_text)

            # Сохраняем историю чата
            ai_message = AIMessage(content=full_output_text)
            self._save_chat_history(chat_for_model, ai_message)

        return StreamResult(content_generator(), lambda: full_output_text)
