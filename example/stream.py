import asyncio

from example.common_imports import *  # noqa: F403
from llm.service import LLMService


async def test() -> None:
    llm = await LLMService.create(grok_3_mini.to_dict())  # noqa: F405
    stream = await llm.astream(message='Кратко расскажи что такое Python')
    async for chunk in stream:
        print(chunk, end='', flush=True)

    print('\n\n')
    print(len(stream.full_text))
    print(llm.usage)
    print(llm.chat_json)


async def main() -> None:
    await test()


if __name__ == '__main__':
    asyncio.run(main())
