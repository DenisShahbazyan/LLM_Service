import asyncio

from example.common_imports import *  # noqa: F403
from llm.service import LLMService


async def chatgpt_mgr() -> None:
    llm = await LLMService.create(gpt_4o_mini.to_dict())  # noqa: F405
    async with llm.astream_mgr(message='Расскажи теорему Пифагора') as stream:
        async for chunk in stream:
            print(chunk, end='', flush=True)


async def gigachat_mgr() -> None:
    llm = await LLMService.create(giga_chat.to_dict())  # noqa: F405
    async with llm.astream_mgr(message='Расскажи теорему Пифагора') as stream:
        async for chunk in stream:
            print(chunk, end='', flush=True)


async def claude_mgr() -> None:
    llm = await LLMService.create(claude_3_5_haiku.to_dict())  # noqa: F405
    async with llm.astream_mgr(message='Расскажи теорему Пифагора') as stream:
        async for chunk in stream:
            print(chunk, end='', flush=True)


async def gemini_mgr() -> None:
    llm = await LLMService.create(gemini_2_5_pro.to_dict())  # noqa: F405
    async with llm.astream_mgr(message='Расскажи теорему Пифагора') as stream:
        async for chunk in stream:
            print(chunk, end='', flush=True)


async def grok_mgr() -> None:
    llm = await LLMService.create(grok_3_mini.to_dict())  # noqa: F405
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
