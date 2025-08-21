import asyncio

from pydantic import BaseModel, Field

from example.common_imports import *  # noqa: F403
from llm.service import LLMService


class RelatedConceptOutput(BaseModel):
    """Новый термин и его сила связи с исходным термином."""

    title: str = Field(..., description='Название термина')
    length: int = Field(..., description='Сила связи')


class RelatedConceptListOutput(BaseModel):
    """Новые термины и их сила связи с исходным термином."""

    concepts: list[RelatedConceptOutput]


SYSTEM_PROMPT = (
    'Тебе дано понятие школьной программы: "Молекула". Сгенерируй ровно "5" понятий '
    'школьной программы, наиболее близких к этому понятию.'
)


async def test() -> None:
    llm = await LLMService.create(deepseek_chat.to_dict())  # noqa: F405
    structured_llm = await llm.with_structured_output(RelatedConceptListOutput)
    result = await structured_llm.ainvoke(message=SYSTEM_PROMPT)
    print(result)
    print(structured_llm.usage)
    print(structured_llm.counter.model_registry.usd_rate)


async def main() -> None:
    await test()


if __name__ == '__main__':
    asyncio.run(main())
