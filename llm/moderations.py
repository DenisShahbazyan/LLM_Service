from langchain.schema import BaseMessage
from openai import AsyncOpenAI

from llm.types import LLMClientInstance


class ModerationError(Exception):
    pass


class ModerationPrompt:
    @staticmethod
    async def openai(
        messages: list[BaseMessage],
        client: LLMClientInstance | None = None,
    ) -> None:
        dict_messages = [message.model_dump().get('content') for message in messages]

        openai_api_key = client.openai_api_key._secret_value

        local_client = AsyncOpenAI(api_key=openai_api_key)

        response = await local_client.moderations.create(
            model='omni-moderation-latest',
            input=dict_messages,
        )

        if response.results[0].flagged:
            raise ModerationError
