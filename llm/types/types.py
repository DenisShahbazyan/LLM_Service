"""Типы для LLM библиотеки"""

from typing import Type, Union

from langchain_anthropic import ChatAnthropic
from langchain_cerebras import ChatCerebras
from langchain_deepseek import ChatDeepSeek
from langchain_gigachat import GigaChat
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langchain_xai import ChatXAI

# Типы для экземпляров клиентов LLM
LLMClientInstance = Union[
    ChatOpenAI,
    GigaChat,
    ChatAnthropic,
    ChatGoogleGenerativeAI,
    ChatXAI,
    ChatDeepSeek,
    ChatCerebras,
]

# Типы для классов клиентов LLM
LLMClientClass = Union[
    Type[ChatOpenAI],
    Type[GigaChat],
    Type[ChatAnthropic],
    Type[ChatGoogleGenerativeAI],
    Type[ChatXAI],
    Type[ChatDeepSeek],
    Type[ChatCerebras],
]
