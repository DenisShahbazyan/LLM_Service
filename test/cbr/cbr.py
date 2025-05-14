from llm.cbr.cbr import CBRRate


async def main() -> None:
    from datetime import datetime

    auth = CBRRate()

    usd_rate = await auth.get_usd_rate()
    print(f'Время: {datetime.now()}\t\tID: {id(usd_rate)}\t\tКурс: {usd_rate}')

    # for _ in range(10):
    #     usd_rate = await auth.get_usd_rate()
    #     print(f'Время: {datetime.now()}\t\tID: {id(usd_rate)}\t\tКурс: {usd_rate}')

    #     await asyncio.sleep(60 * 2)


if __name__ == '__main__':
    import asyncio

    asyncio.run(main())
