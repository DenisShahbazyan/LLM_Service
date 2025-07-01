import asyncio

from pydantic import BaseModel, Field

from example.llm_config import claude_3_5_haiku  # noqa: F401
from example.llm_config import claude_3_7_sonnet  # noqa: F401
from example.llm_config import claude_opus_4  # noqa: F401
from example.llm_config import gemini_2_0_flash_001  # noqa: F401
from example.llm_config import gemini_2_5_pro  # noqa: F401
from example.llm_config import giga_chat  # noqa: F401
from example.llm_config import giga_chat_2  # noqa: F401
from example.llm_config import giga_chat_2_max  # noqa: F401
from example.llm_config import giga_chat_2_pro  # noqa: F401
from example.llm_config import giga_chat_max  # noqa: F401
from example.llm_config import giga_chat_pro  # noqa: F401
from example.llm_config import gpt_4_1  # noqa: F401
from example.llm_config import gpt_4_1_mini  # noqa: F401
from example.llm_config import gpt_4_1_nano  # noqa: F401
from example.llm_config import gpt_4_5_preview  # noqa: F401
from example.llm_config import gpt_4o  # noqa: F401
from example.llm_config import gpt_4o_mini  # noqa: F401
from example.llm_config import grok_3_mini  # noqa: F401
from example.llm_config import o3_2025_04_16  # noqa: F401
from example.llm_config import o4_mini_2025_04_16  # noqa: F401
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
    llm = await LLMService.create(gpt_4o_mini.to_dict())
    structured_llm = await llm.with_structured_output(RelatedConceptListOutput)
    result = await structured_llm.ainvoke(message=SYSTEM_PROMPT)
    print(result)
    print(structured_llm.usage)
    print(structured_llm.counter.model_registry.usd_rate)


async def gigachat() -> None:
    llm = await LLMService.create(giga_chat_2_max.to_dict())
    structured_llm = await llm.with_structured_output(RelatedConceptListOutput)
    result = await structured_llm.ainvoke(message=SYSTEM_PROMPT)
    print(result)
    print(structured_llm.usage)
    print(structured_llm.counter.model_registry.usd_rate)


async def claude() -> None:
    llm = await LLMService.create(claude_3_5_haiku.to_dict())
    structured_llm = await llm.with_structured_output(RelatedConceptListOutput)
    result = await structured_llm.ainvoke(message=SYSTEM_PROMPT)
    print(result)
    print(structured_llm.usage)
    print(structured_llm.counter.model_registry.usd_rate)


async def gemini() -> None:
    llm = await LLMService.create(gemini_2_5_pro.to_dict())
    structured_llm = await llm.with_structured_output(RelatedConceptListOutput)
    result = await structured_llm.ainvoke(message=SYSTEM_PROMPT)
    print(result)
    print(structured_llm.usage)
    print(structured_llm.counter.model_registry.usd_rate)


async def grok() -> None:
    llm = await LLMService.create(grok_3_mini.to_dict())
    structured_llm = await llm.with_structured_output(RelatedConceptListOutput)
    result = await structured_llm.ainvoke(message=SYSTEM_PROMPT)
    print(result)
    print(structured_llm.usage)
    print(structured_llm.counter.model_registry.usd_rate)


async def main() -> None:
    # await chatgpt()
    # await gigachat()
    # await claude()
    # await gemini()  # Проверить
    await grok()


if __name__ == '__main__':
    asyncio.run(main())
