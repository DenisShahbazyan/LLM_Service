import asyncio

from example.common_imports import *  # noqa: F403
from llm.service import LLMService


async def test() -> None:
    llm = await LLMService.create(gpt_4o_mini.to_dict())  # noqa: F405
    stream = await llm.astream(message='Кратко расскажи что такое Python')
    async for chunk in stream:
        print(chunk, end='', flush=True)

    print(f'\n\nДлина текста: {len(stream.full_text)} символов')
    print(f'Usage после стрима: {llm.usage}')


async def main() -> None:
    await test()


if __name__ == '__main__':
    asyncio.run(main())
