import asyncio

from pydantic import BaseModel, Field

from example.llm_config import get_llm_config
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


async def chatgpt() -> None:
    llm = await LLMService.create(get_llm_config('gpt-4o-mini'))
    structured_llm = await llm.with_structured_output(RelatedConceptListOutput)
    result = await structured_llm.ainvoke(message=SYSTEM_PROMPT)
    print(result)
    print(structured_llm.usage)
    print(structured_llm.counter.pricing.usd_rate)


async def gigachat() -> None:
    llm = await LLMService.create(get_llm_config('GigaChat-2-Max'))
    structured_llm = await llm.with_structured_output(RelatedConceptListOutput)
    result = await structured_llm.ainvoke(message=SYSTEM_PROMPT)
    print(result)
    print(structured_llm.usage)
    print(structured_llm.counter.pricing.usd_rate)


async def claude() -> None:
    llm = await LLMService.create(get_llm_config('claude-3-5-haiku-latest'))
    structured_llm = await llm.with_structured_output(RelatedConceptListOutput)
    result = await structured_llm.ainvoke(message=SYSTEM_PROMPT)
    print(result)
    print(structured_llm.usage)
    print(structured_llm.counter.pricing.usd_rate)


async def main() -> None:
    await chatgpt()
    await gigachat()
    await claude()


if __name__ == '__main__':
    asyncio.run(main())
