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


async def chatgpt() -> None:
    llm = await LLMService.create(gpt_4o_mini.to_dict())  # noqa: F405
    structured_llm = await llm.with_structured_output(RelatedConceptListOutput)
    result = await structured_llm.ainvoke(message=SYSTEM_PROMPT)
    print(result)
    print(structured_llm.usage)
    print(structured_llm.counter.model_registry.usd_rate)


async def gigachat() -> None:
    llm = await LLMService.create(giga_chat_2_max.to_dict())  # noqa: F405
    structured_llm = await llm.with_structured_output(RelatedConceptListOutput)
    result = await structured_llm.ainvoke(message=SYSTEM_PROMPT)
    print(result)
    print(structured_llm.usage)
    print(structured_llm.counter.model_registry.usd_rate)


async def claude() -> None:
    llm = await LLMService.create(claude_3_5_haiku.to_dict())  # noqa: F405
    structured_llm = await llm.with_structured_output(RelatedConceptListOutput)
    result = await structured_llm.ainvoke(message=SYSTEM_PROMPT)
    print(result)
    print(structured_llm.usage)
    print(structured_llm.counter.model_registry.usd_rate)


async def gemini() -> None:
    llm = await LLMService.create(gemini_2_5_pro.to_dict())  # noqa: F405
    structured_llm = await llm.with_structured_output(RelatedConceptListOutput)
    result = await structured_llm.ainvoke(message=SYSTEM_PROMPT)
    print(result)
    print(structured_llm.usage)
    print(structured_llm.counter.model_registry.usd_rate)


async def grok() -> None:
    llm = await LLMService.create(grok_3_mini.to_dict())  # noqa: F405
    structured_llm = await llm.with_structured_output(RelatedConceptListOutput)
    result = await structured_llm.ainvoke(message=SYSTEM_PROMPT)
    print(result)
    print(structured_llm.usage)
    print(structured_llm.counter.model_registry.usd_rate)


async def deepseek() -> None:
    llm = await LLMService.create(deepseek_chat.to_dict())  # noqa: F405
    structured_llm = await llm.with_structured_output(RelatedConceptListOutput)
    result = await structured_llm.ainvoke(message=SYSTEM_PROMPT)
    print(result)
    print(structured_llm.usage)
    print(structured_llm.counter.model_registry.usd_rate)


async def main() -> None:
    await chatgpt()
    # await gigachat()
    # await claude()
    # await gemini()  # Проверить
    # await grok()
    # await deepseek()  # Проверить


if __name__ == '__main__':
    asyncio.run(main())
