import asyncio

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
from example.llm_config import gpt_4o  # noqa: F401
from example.llm_config import gpt_4o_mini  # noqa: F401
from example.llm_config import grok_3_mini  # noqa: F401
from example.llm_config import o3_2025_04_16  # noqa: F401
from example.llm_config import o4_mini_2025_04_16  # noqa: F401
from llm.service import LLMService


async def chatgpt_mgr() -> None:
    llm = await LLMService.create(gpt_4o_mini.to_dict())
    async with llm.astream_mgr(message='Расскажи теорему Пифагора') as stream:
        async for chunk in stream:
            print(chunk, end='', flush=True)


async def gigachat_mgr() -> None:
    llm = await LLMService.create(giga_chat.to_dict())
    async with llm.astream_mgr(message='Расскажи теорему Пифагора') as stream:
        async for chunk in stream:
            print(chunk, end='', flush=True)


async def claude_mgr() -> None:
    llm = await LLMService.create(claude_3_5_haiku.to_dict())
    async with llm.astream_mgr(message='Расскажи теорему Пифагора') as stream:
        async for chunk in stream:
            print(chunk, end='', flush=True)


async def gemini_mgr() -> None:
    llm = await LLMService.create(gemini_2_5_pro.to_dict())
    async with llm.astream_mgr(message='Расскажи теорему Пифагора') as stream:
        async for chunk in stream:
            print(chunk, end='', flush=True)


async def grok_mgr() -> None:
    llm = await LLMService.create(grok_3_mini.to_dict())
    async with llm.astream_mgr(message='Расскажи теорему Пифагора') as stream:
        async for chunk in stream:
            print(chunk, end='', flush=True)


async def main() -> None:
    # await chatgpt_mgr()
    # await gigachat_mgr()
    # await claude_mgr()
    # await gemini_mgr()
    await grok_mgr()


if __name__ == '__main__':
    asyncio.run(main())
