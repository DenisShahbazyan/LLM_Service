from langchain.schema import AIMessage, BaseMessage, HumanMessage, SystemMessage


class PrepareChat:
    """Класс для подготовки чата в формате Langchain для модели.

    1. Данный класс всегда отдает список сообщений в формате Langchain.
    2. Если был отправлен только системный промпт, то после него добавляется пустое
        сообщение пользователя. Иначе некоторые модели не хотят работать с единственным
        системным промптом.

    Словарь chat_history должен иметь следующую структуру:
    ```
    [
        {
            "role": "system" | "user" | "assistant",
            "content": "..."
        },
        {
            "role": "system" | "user" | "assistant",
            "content": "..."
        },
    ]
    ```
    или
    ```
    [
        BaseMessage(content="..."),
        BaseMessage(content="..."),
    ]
    ```
    где BaseMessage - может быть любым классом из SystemMessage, HumanMessage, AIMessage


    Args:
        chat_history list[dict[str, str] | BaseMessage] | None: Список сообщений чата.
            По умолчанию None.
        system_prompt str | BaseMessage | None: Системный промпт. По умолчанию None.
        message str | BaseMessage | None: Сообщение. По умолчанию None.

    Returns:
        list[BaseMessage]: Список сообщений для модели
    """

    def __new__(
        cls,
        chat_history: list[dict[str, str] | BaseMessage] | None = None,
        system_prompt: str | BaseMessage | None = None,
        message: str | BaseMessage | None = None,
    ):
        instance = super().__new__(cls)
        return instance.prepare_chat(chat_history, system_prompt, message)

    def prepare_chat(
        self,
        chat_history: list[dict[str, str] | BaseMessage] | None = None,
        system_prompt: str | BaseMessage | None = None,
        message: str | BaseMessage | None = None,
    ) -> list[BaseMessage]:
        messages = []
        messages.extend(self._prepare_system_prompt(system_prompt))
        messages.extend(self._prepare_chat_history(chat_history))
        messages.extend(self._prepare_message(message))

        messages = self._add_empty_user_message_if_needed(messages)

        return messages

    def _add_empty_user_message_if_needed(
        self, messages: list[BaseMessage]
    ) -> list[BaseMessage]:
        """Добавляет пустое сообщение пользователя, если в списке только один системный
        промпт.

        Args:
            messages (list[BaseMessage]): Список сообщений

        Returns:
            list[BaseMessage]: Обновленный список сообщений
        """
        if len(messages) == 1 and isinstance(messages[0], SystemMessage):
            messages.append(HumanMessage(content=''))
        return messages

    def _prepare_system_prompt(
        self, system_prompt: str | BaseMessage | None
    ) -> list[BaseMessage]:
        messages = []
        if system_prompt:
            if isinstance(system_prompt, BaseMessage):
                messages.append(system_prompt)
            elif isinstance(system_prompt, str):
                messages.append(SystemMessage(content=system_prompt))
            else:
                raise ValueError(
                    'system_prompt должен быть либо строкой, либо объектом BaseMessage'
                )
        return messages

    def _prepare_chat_history(
        self, chat_history: list[dict[str, str] | BaseMessage] | None
    ) -> list[BaseMessage]:
        messages = []
        if chat_history:
            for msg in chat_history:
                if isinstance(msg, BaseMessage):
                    messages.append(msg)
                elif isinstance(msg, dict):
                    role = msg.get('role')
                    content = msg.get('content', '')
                    match role:
                        case 'system':
                            messages.append(SystemMessage(content=content))
                        case 'user':
                            messages.append(HumanMessage(content=content))
                        case 'assistant':
                            messages.append(AIMessage(content=content))
                        case _:
                            raise ValueError(f"Неизвестная роль сообщения: {role}")
                else:
                    raise ValueError(
                        'Элементы chat_history должны быть либо словарями, либо '
                        'объектами BaseMessage'
                    )
        return messages

    def _prepare_message(self, message: str | BaseMessage | None) -> list[BaseMessage]:
        messages = []
        if message:
            if isinstance(message, BaseMessage):
                messages.append(message)
            elif isinstance(message, str):
                messages.append(HumanMessage(content=message))
            else:
                raise ValueError(
                    'message должен быть либо строкой, либо объектом BaseMessage'
                )
        return messages
