from http import HTTPStatus
from urllib.parse import urljoin

import aiohttp

from llm.types import LLMClientInstance


# TODO: Выполнять запросы самостоятельно, или использовать нативные библиотеки?
class TestConnections:
    @staticmethod
    async def _send_request(url: str, headers: dict[str, str]) -> bool:
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
                url=url,
                headers=headers,
                ssl=False,
            ) as response:
                if response.status == HTTPStatus.OK:
                    return True
                return False

    @staticmethod
    async def openai(client: LLMClientInstance | None = None) -> bool:
        api_key = client.openai_api_key._secret_value

        base_url = str(client.root_async_client.base_url)
        path = 'models'
        full_url = urljoin(base_url, path)

        headers = {'Authorization': f'Bearer {api_key}'}

        return await TestConnections()._send_request(full_url, headers)

    @staticmethod
    async def anthropic(client: LLMClientInstance | None = None) -> bool:
        api_key = client.anthropic_api_key._secret_value

        base_url = client.anthropic_api_url
        path = 'v1/models'
        full_url = urljoin(base_url, path)

        headers = {
            'x-api-key': api_key,
            'anthropic-version': '2023-06-01',
        }

        return await TestConnections()._send_request(full_url, headers)

    @staticmethod
    async def gigachat(client: LLMClientInstance | None = None) -> bool:
        try:
            return bool(await client.aget_models())
        except Exception:
            return False

    @staticmethod
    async def google(client: LLMClientInstance | None = None) -> bool:
        api_key = client.google_api_key._secret_value

        url = 'https://generativelanguage.googleapis.com/v1beta/models'

        headers = {
            'x-goog-api-key': api_key,
            'Content-Type': 'application/json',
        }

        return await TestConnections()._send_request(url, headers)

    @staticmethod
    async def xai(client: LLMClientInstance | None = None) -> bool:
        api_key = client.xai_api_key._secret_value

        url = 'https://api.x.ai/v1/models'

        headers = {
            'Authorization': f'Bearer {api_key}',
            'Content-Type': 'application/json',
        }

        return await TestConnections()._send_request(url, headers)

    @staticmethod
    async def deepseek(client: LLMClientInstance | None = None) -> bool:
        api_key = client.api_key._secret_value

        url = client.api_base
        path = 'models'
        full_url = urljoin(url, path)

        headers = {'Authorization': f'Bearer {api_key}'}

        return await TestConnections()._send_request(full_url, headers)

    @staticmethod
    async def cerebras(client: LLMClientInstance | None = None) -> bool:
        api_key = client.cerebras_api_key._secret_value

        url = client.cerebras_api_base
        path = 'models'
        full_url = urljoin(url, path)

        headers = {'Authorization': f'Bearer {api_key}'}

        return await TestConnections()._send_request(full_url, headers)

    @staticmethod
    async def openrouter(client: LLMClientInstance | None = None) -> bool:
        return await TestConnections().openai(client)
