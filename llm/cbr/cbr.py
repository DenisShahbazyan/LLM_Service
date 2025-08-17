import time
from http import HTTPStatus
from typing import Optional, Self
from xml.etree import ElementTree

import aiohttp


class CBRRate:
    _instance: Optional['CBRRate'] = None
    USD_CURRENCY_ID = 'R01235'
    CBR_DAILY_RATE_URL = 'https://www.cbr.ru/scripts/XML_daily.asp'
    RATE_REFRESH_BUFFER = 60 * 10

    def __new__(cls) -> Self:
        if cls._instance is None:
            cls._instance = super(CBRRate, cls).__new__(cls)
            cls._instance.cached_rate = None
            cls._instance.cached_time = None
        return cls._instance

    async def get_usd_rate(self) -> float:
        current_time = int(time.time())
        if self._rate_needs_refresh(current_time):
            new_rate = await self._fetch_usd_rate()
            if new_rate is not None:
                self.cached_time = current_time
                self.cached_rate = new_rate
        return self.cached_rate

    def _rate_needs_refresh(self, current_time: int) -> bool:
        return (
            self.cached_rate is None
            or self.cached_time is None
            or current_time >= self.cached_time + self.RATE_REFRESH_BUFFER
        )

    async def process_response(self, response: aiohttp.ClientResponse) -> float | None:
        if response.status == HTTPStatus.OK:
            content = await response.text()
            root = ElementTree.fromstring(content)
            for valute in root.findall('Valute'):
                if valute.get('ID') == self.USD_CURRENCY_ID:
                    rate = valute.find('Value').text
                    return float(rate.replace(',', '.'))
        return None

    async def _fetch_usd_rate(self) -> float | None:
        timeout = aiohttp.ClientTimeout(
            total=60,
            connect=10,
            sock_read=30,
        )
        connector = aiohttp.TCPConnector(
            limit=100,
            limit_per_host=30,
            keepalive_timeout=30,
            enable_cleanup_closed=True,
            force_close=False,
        )
        async with aiohttp.ClientSession(
            timeout=timeout,
            connector=connector,
        ) as session:
            async with session.get(
                url=self.CBR_DAILY_RATE_URL,
                ssl=False,
            ) as response:
                return await self.process_response(response)
