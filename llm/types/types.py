"""Типы для LLM библиотеки"""

from typing import Type, TypeAlias, Union

# Собираем типы динамически в зависимости от доступных провайдеров
_llm_client_instances = []
_llm_client_classes = []

# OpenAI
try:
    from langchain_openai import ChatOpenAI

    _llm_client_instances.append(ChatOpenAI)
    _llm_client_classes.append(Type[ChatOpenAI])
except ImportError:
    pass

# GigaChat
try:
    from langchain_gigachat import GigaChat

    _llm_client_instances.append(GigaChat)
    _llm_client_classes.append(Type[GigaChat])
except ImportError:
    pass

# Anthropic
try:
    from langchain_anthropic import ChatAnthropic

    _llm_client_instances.append(ChatAnthropic)
    _llm_client_classes.append(Type[ChatAnthropic])
except ImportError:
    pass

# Google
try:
    from langchain_google_genai import ChatGoogleGenerativeAI

    _llm_client_instances.append(ChatGoogleGenerativeAI)
    _llm_client_classes.append(Type[ChatGoogleGenerativeAI])
except ImportError:
    pass

# XAI
try:
    from langchain_xai import ChatXAI

    _llm_client_instances.append(ChatXAI)
    _llm_client_classes.append(Type[ChatXAI])
except ImportError:
    pass

# DeepSeek
try:
    from langchain_deepseek import ChatDeepSeek

    _llm_client_instances.append(ChatDeepSeek)
    _llm_client_classes.append(Type[ChatDeepSeek])
except ImportError:
    pass

# Cerebras
try:
    from langchain_cerebras import ChatCerebras

    _llm_client_instances.append(ChatCerebras)
    _llm_client_classes.append(Type[ChatCerebras])
except ImportError:
    pass

if _llm_client_instances:
    LLMClientInstance: TypeAlias = Union[*_llm_client_instances]  # type: ignore


if _llm_client_classes:
    LLMClientClass: TypeAlias = Union[*_llm_client_classes]  # type: ignore
