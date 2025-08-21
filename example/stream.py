import asyncio

from example.common_imports import *  # noqa: F403
from llm.service import LLMService


async def test() -> None:
    llm = await LLMService.create(gpt_4o_mini.to_dict())  # noqa: F405
    result = llm.astream(message='Расскажи теорему Пифагора')
    async for chunk in result:
        print(chunk, end='', flush=True)


async def main() -> None:
    await test()


if __name__ == '__main__':
    asyncio.run(main())
