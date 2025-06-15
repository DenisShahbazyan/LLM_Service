import asyncio

from example.llm_config import gpt_4o_mini  # noqa: F401
from llm.service import LLMService


async def main() -> None:
    llm = await LLMService.create(gpt_4o_mini.to_dict())
    async with llm.astream_mgr(message='Расскажи теорему Пифагора') as stream:
        async for chunk in stream:
            print(chunk, end='', flush=True)


if __name__ == '__main__':
    asyncio.run(main())
