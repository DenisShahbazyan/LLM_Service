import asyncio

from example.llm_config import get_llm_config
from llm.service import LLMService


async def chatgpt() -> None:
    llm = await LLMService.create(get_llm_config('gpt-4o-mini'))
    result = await llm.ainvoke(message='Сколько будет 2 + 2?')
    print(result)
    print(llm.usage)
    print(llm.counter.pricing.usd_rate)


async def gigachat() -> None:
    llm = await LLMService.create(get_llm_config('GigaChat-2-Max'))
    result = await llm.ainvoke(message='Сколько будет 2 + 2?')
    print(result)
    print(llm.usage)
    print(llm.counter.pricing.usd_rate)


async def claude() -> None:
    llm = await LLMService.create(get_llm_config('claude-3-5-haiku-latest'))
    result = await llm.ainvoke(message='Сколько будет 2 + 2?')
    print(result)
    print(llm.usage)
    print(llm.counter.pricing.usd_rate)


async def main() -> None:
    await chatgpt()
    await gigachat()
    await claude()


if __name__ == '__main__':
    asyncio.run(main())
