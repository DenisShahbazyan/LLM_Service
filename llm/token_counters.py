import aiohttp
import tiktoken
from langchain.schema import BaseMessage
from langchain_anthropic import ChatAnthropic
from langchain_deepseek import ChatDeepSeek
from langchain_gigachat import GigaChat
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langchain_xai import ChatXAI

LLMClientInstance = (
    ChatOpenAI
    | GigaChat
    | ChatAnthropic
    | ChatGoogleGenerativeAI
    | ChatXAI
    | ChatDeepSeek
)


class TokenCounterFactory:
    @staticmethod
    def create_openai_counter():
        """Создает функцию счетчика токенов для OpenAI"""

        async def count_tokens(
            messages: list[BaseMessage],
            model_name: str,
            client: LLMClientInstance | None = None,
        ) -> int:
            """Подсчитывает количество токенов.

            Args:
                messages (list[BaseMessage]): Сообщения
                model_name (str): Название модели
                client (LLMClientInstance | None, optional): Клиент LLM.
                    По умолчанию None.

            Returns:
                int: Количество токенов
            """
            encoding = tiktoken.encoding_for_model(model_name)
            text = ' '.join(str(m.content) for m in messages)
            return len(encoding.encode(text))

        return count_tokens

    @staticmethod
    def create_gigachat_counter():
        """Создает функцию счетчика токенов для GigaChat"""

        async def count_tokens(
            messages: list[BaseMessage],
            model_name: str,
            client: LLMClientInstance | None = None,
        ) -> int:
            """Подсчитывает количество токенов.

            Args:
                messages (list[BaseMessage]): Сообщения
                model_name (str): Название модели
                client (LLMClientInstance | None, optional): Клиент LLM.
                    По умолчанию None.

            Returns:
                int: Количество токенов
            """
            if not client:
                raise ValueError('Client not initialized')

            text = ' '.join(str(m.content) for m in messages)
            response = await client.atokens_count([text], model_name)
            return response[0].tokens

        return count_tokens

    @staticmethod
    def create_anthropic_counter():
        """Создает функцию счетчика токенов для Anthropic"""

        async def count_tokens(
            messages: list[BaseMessage],
            model_name: str,
            client: LLMClientInstance | None = None,
        ) -> int:
            """Подсчитывает количество токенов.

            Args:
                messages (list[BaseMessage]): Сообщения
                model_name (str): Название модели
                client (LLMClientInstance | None, optional): Клиент LLM.
                    По умолчанию None.

            Returns:
                int: Количество токенов
            """
            if not client:
                raise ValueError('Client not initialized')

            return client.get_num_tokens_from_messages(messages)

        return count_tokens

    @staticmethod
    def create_google_counter():
        """Создает функцию счетчика токенов для Google"""

        async def count_tokens(
            messages: list[BaseMessage],
            model_name: str,
            client: LLMClientInstance | None = None,
        ) -> int:
            """Подсчитывает количество токенов.

            Args:
                messages (list[BaseMessage]): Сообщения
                model_name (str): Название модели
                client (LLMClientInstance | None, optional): Клиент LLM.
                    По умолчанию None.

            Returns:
                int: Количество токенов
            """
            if not client:
                raise ValueError('Client not initialized')

            return client.get_num_tokens_from_messages(messages)

        return count_tokens

    @staticmethod
    def create_xai_counter():
        """Создает функцию счетчика токенов для xAI"""

        async def count_tokens(
            messages: list[BaseMessage],
            model_name: str,
            client: LLMClientInstance | None = None,
        ) -> int:
            """Подсчитывает количество токенов.

            Args:
                messages (list[BaseMessage]): Сообщения
                model_name (str): Название модели
                client (LLMClientInstance | None, optional): Клиент LLM.
                    По умолчанию None.

            Returns:
                int: Количество токенов
            """
            if not client:
                raise ValueError('Client not initialized')

            x_api_key = client.xai_api_key._secret_value

            url = 'https://api.x.ai/v1/tokenize-text'

            headers = {
                'Authorization': f"Bearer {x_api_key}",
                'Content-Type': 'application/json',
            }

            text = ' '.join(str(m.content) for m in messages)
            payload = {
                'text': text,
                'model': model_name,
            }

            async with aiohttp.ClientSession() as session:
                async with session.post(url, headers=headers, json=payload) as response:
                    data = await response.json()
                    return len(data['token_ids'])

        return count_tokens

    @staticmethod
    def create_deepseek_counter():
        """Создает функцию счетчика токенов для DeepSeek"""

        async def count_tokens(
            messages: list[BaseMessage],
            model_name: str,
            client: LLMClientInstance | None = None,
        ) -> int:
            """Подсчитывает количество токенов.

            Args:
                messages (list[BaseMessage]): Сообщения
                model_name (str): Название модели
                client (LLMClientInstance | None, optional): Клиент LLM.
                    По умолчанию None.

            Returns:
                int: Количество токенов
            """
            if not client:
                raise ValueError('Client not initialized')

            return client.get_num_tokens_from_messages(messages)

        return count_tokens
