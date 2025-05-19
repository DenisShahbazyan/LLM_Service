import os
from typing import Any

from pydantic import BaseModel

CHAT_GPT__KEY = os.getenv('CHAT_GPT__KEY')
GIGACHAT__KEY = os.getenv('GIGACHAT__KEY')
ANTHROPIC__KEY = os.getenv('ANTHROPIC__KEY')


class GPT4ominiConfig(BaseModel):
    model: str = 'gpt-4o-mini'
    api_key: str = CHAT_GPT__KEY


class GigaChat2MaxConfig(BaseModel):
    model: str = 'GigaChat-2-Max'
    credentials: str = GIGACHAT__KEY
    scope: str = 'GIGACHAT_API_CORP'
    verify_ssl_certs: bool = False
    profanity_check: bool = False


class Claude35HaikuConfig(BaseModel):
    model: str = 'claude-3-5-haiku-latest'
    api_key: str = ANTHROPIC__KEY


class LLMConfig:
    gpt_4o_mini: dict[str, Any] = GPT4ominiConfig().model_dump()
    gigachat_2_max: dict[str, Any] = GigaChat2MaxConfig().model_dump()
    claude_3_5_haiku: dict[str, Any] = Claude35HaikuConfig().model_dump()


llm_config = LLMConfig()


def get_llm_config(llm_code: str) -> dict[str, Any]:
    """Отдает конфиг LLM по его коду. Код - который используется для токенизаторов.

    Args:
        llm_code (str): Код LLM

    Raises:
        ValueError: Если код LLM неизвестен

    Returns:
        dict[str, Any]: Конфиг LLM в виде dict
    """
    match llm_code:
        case 'gpt-4o-mini':
            return llm_config.gpt_4o_mini
        case 'GigaChat-2-Max':
            return llm_config.gigachat_2_max
        case 'claude-3-5-haiku-latest':
            return llm_config.claude_3_5_haiku
        case _:
            raise ValueError(f'Unknown LLM code: {llm_code}')
