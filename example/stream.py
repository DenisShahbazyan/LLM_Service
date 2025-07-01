import asyncio

from example.common_imports import *  # noqa: F403
from llm.service import LLMService


async def chatgpt() -> None:
    llm = await LLMService.create(gpt_4o_mini.to_dict())  # noqa: F405
    result = llm.astream(message='Расскажи теорему Пифагора')
    async for chunk in result:
        print(chunk, end='', flush=True)


async def gigachat() -> None:
    llm = await LLMService.create(giga_chat.to_dict())  # noqa: F405
    result = llm.astream(message='Расскажи теорему Пифагора')
    async for chunk in result:
        print(chunk, end='', flush=True)


async def claude() -> None:
    llm = await LLMService.create(claude_3_5_haiku.to_dict())  # noqa: F405
    result = llm.astream(message='Расскажи теорему Пифагора')
    async for chunk in result:
        print(chunk, end='', flush=True)


async def gemini() -> None:
    llm = await LLMService.create(gemini_2_5_pro.to_dict())  # noqa: F405
    result = llm.astream(message='Расскажи теорему Пифагора')
    async for chunk in result:
        print(chunk, end='', flush=True)


async def grok() -> None:
    llm = await LLMService.create(grok_3_mini.to_dict())  # noqa: F405
    result = llm.astream(message='Расскажи теорему Пифагора')
    async for chunk in result:
        print(chunk, end='', flush=True)


async def main() -> None:
    # await chatgpt()
    # await gigachat()
    # await claude()
    # await gemini()
    await grok()


if __name__ == '__main__':
    asyncio.run(main())
