import json
from typing import TypeVar

from langchain.schema import BaseMessage
from pydantic import BaseModel

from llm.pydantic.checker import is_pydantic_instance

PydanticSchema = TypeVar('PydanticSchema', bound=BaseModel)


def text_extractor(result: BaseMessage | PydanticSchema) -> str:
    """
    Извлекает текст из сообщения от LLM, для подсчета токенов.

    Args:
        result (BaseMessage | PydanticSchema): результат ответа модели, либо
            BaseMessage, либо PydanticSchema, если режим with_structured_output

    Returns:
        str: текст
    """
    if is_pydantic_instance(result):
        result_dict = result.model_dump()
        text = json.dumps(result_dict, ensure_ascii=False)
    else:
        text = result.content

    return text
