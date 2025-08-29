import asyncio

from example.common_imports import *  # noqa: F403
from llm.service import LLMService


async def test() -> None:
    llm = await LLMService.create(grok_3_mini.to_dict())  # noqa: F405
    result = await llm.ainvoke(message='Сколько будет 2 + 2?')
    print(result)
    print(llm.usage)
    print(llm.usd_rate)
    print(llm.chat_json)


async def main() -> None:
    await test()


if __name__ == '__main__':
    asyncio.run(main())
