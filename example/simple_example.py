import asyncio

from example.llm_config import claude_3_5_haiku  # noqa: F401
from example.llm_config import claude_3_7_sonnet  # noqa: F401
from example.llm_config import claude_opus_4  # noqa: F401
from example.llm_config import gemini_2_0_flash_001  # noqa: F401
from example.llm_config import giga_chat  # noqa: F401
from example.llm_config import giga_chat_2  # noqa: F401
from example.llm_config import giga_chat_2_max  # noqa: F401
from example.llm_config import giga_chat_2_pro  # noqa: F401
from example.llm_config import giga_chat_max  # noqa: F401
from example.llm_config import giga_chat_pro  # noqa: F401
from example.llm_config import google_gemini_2_5_pro  # noqa: F401
from example.llm_config import gpt_4o  # noqa: F401
from example.llm_config import gpt_4o_mini  # noqa: F401
from example.llm_config import o3_2025_04_16  # noqa: F401
from example.llm_config import o4_mini_2025_04_16  # noqa: F401
from llm.service import LLMService


async def chatgpt() -> None:
    llm = await LLMService.create(gpt_4o_mini.to_dict())
    result = await llm.ainvoke(message='Сколько будет 2 + 2?')
    print(result)
    print(llm.usage)
    print(llm.counter.model_registry.usd_rate)


async def gigachat() -> None:
    llm = await LLMService.create(giga_chat.to_dict())
    result = await llm.ainvoke(message='Сколько будет 2 + 2?')
    print(result)
    print(llm.usage)
    print(llm.counter.model_registry.usd_rate)


async def claude() -> None:
    llm = await LLMService.create(claude_3_5_haiku.to_dict())
    result = await llm.ainvoke(message='Сколько будет 2 + 2?')
    print(result)
    print(llm.usage)
    print(llm.counter.model_registry.usd_rate)


async def gemini() -> None:
    llm = await LLMService.create(google_gemini_2_5_pro.to_dict())
    result = await llm.ainvoke(message='Сколько будет 2 + 2?')
    print(result)
    print(llm.usage)
    print(llm.counter.model_registry.usd_rate)


async def main() -> None:
    await chatgpt()
    # await gigachat()
    # await claude()
    # await gemini()


if __name__ == '__main__':
    asyncio.run(main())
