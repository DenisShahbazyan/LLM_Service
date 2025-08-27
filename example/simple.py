import asyncio

from example.common_imports import *  # noqa: F403
from llm.service import LLMService


async def test() -> None:
    llm = await LLMService.create(openrouter.to_dict())  # noqa: F405
    result = await llm.ainvoke(message='Сколько будет 2 + 2?')
    print(result)
    print(llm.usage)
    print(llm.counter.model_registry.usd_rate)


async def main() -> None:
    await test()


if __name__ == '__main__':
    asyncio.run(main())
