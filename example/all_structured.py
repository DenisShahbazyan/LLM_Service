"""Тест разных типов схем для with_structured_output"""

import asyncio

from pydantic import BaseModel
from typing_extensions import TypedDict

from example.common_imports import *  # noqa: F403
from llm.service import LLMService


# 1. Pydantic модель
class PydanticAnswer(BaseModel):
    answer: str
    justification: str


# 2. TypedDict
class TypedDictAnswer(TypedDict):
    """Answer with justification"""

    answer: str
    justification: str


# 3. OpenAI function schema
openai_schema = {
    'name': 'AnswerWithJustification',
    'description': 'Answer with justification',
    'parameters': {
        'type': 'object',
        'properties': {
            'answer': {'type': 'string'},
            'justification': {'type': 'string'},
        },
        'required': ['answer'],
    },
}

# 4. JSON Schema
json_schema = {
    'type': 'object',
    'properties': {'answer': {'type': 'string'}, 'justification': {'type': 'string'}},
    'required': ['answer'],
}


async def test_schema_types():
    llm = await LLMService.create(gpt_4o_mini.to_dict())  # noqa: F405
    question = 'Что тяжелее: килограмм железа или килограмм пуха?'

    try:
        # Тест 1: Pydantic модель
        print('=== Тест 1: Pydantic модель ===')
        structured_llm = await llm.with_structured_output(PydanticAnswer)
        result = await structured_llm.ainvoke(message=question)
        print(f'Тип результата: {type(result)}')
        print(f'Результат: {result}')
        print(structured_llm.chat_json)
        print()

        # Тест 2: TypedDict
        print('=== Тест 2: TypedDict ===')
        structured_llm = await llm.with_structured_output(TypedDictAnswer)
        result = await structured_llm.ainvoke(message=question)
        print(f'Тип результата: {type(result)}')
        print(f'Результат: {result}')
        print(structured_llm.chat_json)
        print()

        # Тест 3: OpenAI schema
        print('=== Тест 3: OpenAI schema ===')
        structured_llm = await llm.with_structured_output(openai_schema)
        result = await structured_llm.ainvoke(message=question)
        print(f'Тип результата: {type(result)}')
        print(f'Результат: {result}')
        print(structured_llm.chat_json)
        print()

        # Тест 4: JSON Schema
        print('=== Тест 4: JSON Schema ===')
        structured_llm = await llm.with_structured_output(json_schema)
        result = await structured_llm.ainvoke(message=question)
        print(f'Тип результата: {type(result)}')
        print(f'Результат: {result}')
        print(structured_llm.chat_json)

    except Exception as e:
        print(f'Ошибка: {e}')


if __name__ == '__main__':
    asyncio.run(test_schema_types())
