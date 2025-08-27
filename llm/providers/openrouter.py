# ruff: noqa: E501
from functools import partial

from langchain_openai import ChatOpenAI

from llm.direction import TokenDirection
from llm.providers._base import BaseProvider, ModelConfig
from llm.test_connections import TestConnections
from llm.token_counters import TokenCounterFactory

OpenRouterAI = partial(ChatOpenAI, base_url='https://openrouter.ai/api/v1')


class OpenRouterProvider(BaseProvider):
    """Провайдер для моделей OpenRouter"""

    @property
    def name(self) -> str:
        return 'OpenRouter'

    def get_models(self) -> dict[str, ModelConfig]:
        """Возвращает конфигурации всех моделей OpenRouter"""
        return {
            'x-ai/grok-code-fast-1': ModelConfig(
                client_class=OpenRouterAI,
                token_counter=TokenCounterFactory().create_openrouter_counter(),
                test_connection=TestConnections().openrouter,
                pricing={
                    TokenDirection.ENCODE: 0.0000002,
                    TokenDirection.DECODE: 0.0000015,
                },
            ),
            'nousresearch/hermes-4-70b': ModelConfig(
                client_class=OpenRouterAI,
                token_counter=TokenCounterFactory().create_openrouter_counter(),
                test_connection=TestConnections().openrouter,
                pricing={
                    TokenDirection.ENCODE: 0.00000009329544,
                    TokenDirection.DECODE: 0.0000003733632,
                },
            ),
            'nousresearch/hermes-4-405b': ModelConfig(
                client_class=OpenRouterAI,
                token_counter=TokenCounterFactory().create_openrouter_counter(),
                test_connection=TestConnections().openrouter,
                pricing={
                    TokenDirection.ENCODE: 0.0000001999188,
                    TokenDirection.DECODE: 0.000000800064,
                },
            ),
            'google/gemini-2.5-flash-image-preview:free': ModelConfig(
                client_class=OpenRouterAI,
                token_counter=TokenCounterFactory().create_openrouter_counter(),
                test_connection=TestConnections().openrouter,
                pricing={
                    TokenDirection.ENCODE: 0,
                    TokenDirection.DECODE: 0,
                },
            ),
            'google/gemini-2.5-flash-image-preview': ModelConfig(
                client_class=OpenRouterAI,
                token_counter=TokenCounterFactory().create_openrouter_counter(),
                test_connection=TestConnections().openrouter,
                pricing={
                    TokenDirection.ENCODE: 0.0000003,
                    TokenDirection.DECODE: 0.0000025,
                },
            ),
            'deepseek/deepseek-chat-v3.1': ModelConfig(
                client_class=OpenRouterAI,
                token_counter=TokenCounterFactory().create_openrouter_counter(),
                test_connection=TestConnections().openrouter,
                pricing={
                    TokenDirection.ENCODE: 0.0000002,
                    TokenDirection.DECODE: 0.0000008,
                },
            ),
            'deepseek/deepseek-v3.1-base': ModelConfig(
                client_class=OpenRouterAI,
                token_counter=TokenCounterFactory().create_openrouter_counter(),
                test_connection=TestConnections().openrouter,
                pricing={
                    TokenDirection.ENCODE: 0.0000002,
                    TokenDirection.DECODE: 0.0000008,
                },
            ),
            'openai/gpt-4o-audio-preview': ModelConfig(
                client_class=OpenRouterAI,
                token_counter=TokenCounterFactory().create_openrouter_counter(),
                test_connection=TestConnections().openrouter,
                pricing={
                    TokenDirection.ENCODE: 0.0000025,
                    TokenDirection.DECODE: 0.00001,
                },
            ),
            'mistralai/mistral-medium-3.1': ModelConfig(
                client_class=OpenRouterAI,
                token_counter=TokenCounterFactory().create_openrouter_counter(),
                test_connection=TestConnections().openrouter,
                pricing={
                    TokenDirection.ENCODE: 0.0000004,
                    TokenDirection.DECODE: 0.000002,
                },
            ),
            'baidu/ernie-4.5-21b-a3b': ModelConfig(
                client_class=OpenRouterAI,
                token_counter=TokenCounterFactory().create_openrouter_counter(),
                test_connection=TestConnections().openrouter,
                pricing={
                    TokenDirection.ENCODE: 0.00000007,
                    TokenDirection.DECODE: 0.00000028,
                },
            ),
            'baidu/ernie-4.5-vl-28b-a3b': ModelConfig(
                client_class=OpenRouterAI,
                token_counter=TokenCounterFactory().create_openrouter_counter(),
                test_connection=TestConnections().openrouter,
                pricing={
                    TokenDirection.ENCODE: 0.00000014,
                    TokenDirection.DECODE: 0.00000056,
                },
            ),
            'z-ai/glm-4.5v': ModelConfig(
                client_class=OpenRouterAI,
                token_counter=TokenCounterFactory().create_openrouter_counter(),
                test_connection=TestConnections().openrouter,
                pricing={
                    TokenDirection.ENCODE: 0.0000005,
                    TokenDirection.DECODE: 0.0000018,
                },
            ),
            'ai21/jamba-mini-1.7': ModelConfig(
                client_class=OpenRouterAI,
                token_counter=TokenCounterFactory().create_openrouter_counter(),
                test_connection=TestConnections().openrouter,
                pricing={
                    TokenDirection.ENCODE: 0.0000002,
                    TokenDirection.DECODE: 0.0000004,
                },
            ),
            'ai21/jamba-large-1.7': ModelConfig(
                client_class=OpenRouterAI,
                token_counter=TokenCounterFactory().create_openrouter_counter(),
                test_connection=TestConnections().openrouter,
                pricing={
                    TokenDirection.ENCODE: 0.000002,
                    TokenDirection.DECODE: 0.000008,
                },
            ),
            'openai/gpt-5-chat': ModelConfig(
                client_class=OpenRouterAI,
                token_counter=TokenCounterFactory().create_openrouter_counter(),
                test_connection=TestConnections().openrouter,
                pricing={
                    TokenDirection.ENCODE: 0.00000125,
                    TokenDirection.DECODE: 0.00001,
                },
            ),
            'openai/gpt-5': ModelConfig(
                client_class=OpenRouterAI,
                token_counter=TokenCounterFactory().create_openrouter_counter(),
                test_connection=TestConnections().openrouter,
                pricing={
                    TokenDirection.ENCODE: 0.00000125,
                    TokenDirection.DECODE: 0.00001,
                },
            ),
            'openai/gpt-5-mini': ModelConfig(
                client_class=OpenRouterAI,
                token_counter=TokenCounterFactory().create_openrouter_counter(),
                test_connection=TestConnections().openrouter,
                pricing={
                    TokenDirection.ENCODE: 0.00000025,
                    TokenDirection.DECODE: 0.000002,
                },
            ),
            'openai/gpt-5-nano': ModelConfig(
                client_class=OpenRouterAI,
                token_counter=TokenCounterFactory().create_openrouter_counter(),
                test_connection=TestConnections().openrouter,
                pricing={
                    TokenDirection.ENCODE: 0.00000005,
                    TokenDirection.DECODE: 0.0000004,
                },
            ),
            'openai/gpt-oss-120b': ModelConfig(
                client_class=OpenRouterAI,
                token_counter=TokenCounterFactory().create_openrouter_counter(),
                test_connection=TestConnections().openrouter,
                pricing={
                    TokenDirection.ENCODE: 0.000000072,
                    TokenDirection.DECODE: 0.00000028,
                },
            ),
            'openai/gpt-oss-20b:free': ModelConfig(
                client_class=OpenRouterAI,
                token_counter=TokenCounterFactory().create_openrouter_counter(),
                test_connection=TestConnections().openrouter,
                pricing={
                    TokenDirection.ENCODE: 0,
                    TokenDirection.DECODE: 0,
                },
            ),
            'openai/gpt-oss-20b': ModelConfig(
                client_class=OpenRouterAI,
                token_counter=TokenCounterFactory().create_openrouter_counter(),
                test_connection=TestConnections().openrouter,
                pricing={
                    TokenDirection.ENCODE: 0.00000004,
                    TokenDirection.DECODE: 0.00000015,
                },
            ),
            'anthropic/claude-opus-4.1': ModelConfig(
                client_class=OpenRouterAI,
                token_counter=TokenCounterFactory().create_openrouter_counter(),
                test_connection=TestConnections().openrouter,
                pricing={
                    TokenDirection.ENCODE: 0.000015,
                    TokenDirection.DECODE: 0.000075,
                },
            ),
            'mistralai/codestral-2508': ModelConfig(
                client_class=OpenRouterAI,
                token_counter=TokenCounterFactory().create_openrouter_counter(),
                test_connection=TestConnections().openrouter,
                pricing={
                    TokenDirection.ENCODE: 0.0000003,
                    TokenDirection.DECODE: 0.0000009,
                },
            ),
            'qwen/qwen3-30b-a3b-instruct-2507': ModelConfig(
                client_class=OpenRouterAI,
                token_counter=TokenCounterFactory().create_openrouter_counter(),
                test_connection=TestConnections().openrouter,
                pricing={
                    TokenDirection.ENCODE: 0.0000001,
                    TokenDirection.DECODE: 0.0000003,
                },
            ),
            'z-ai/glm-4.5': ModelConfig(
                client_class=OpenRouterAI,
                token_counter=TokenCounterFactory().create_openrouter_counter(),
                test_connection=TestConnections().openrouter,
                pricing={
                    TokenDirection.ENCODE: 0.0000001999188,
                    TokenDirection.DECODE: 0.000000800064,
                },
            ),
            'z-ai/glm-4.5-air:free': ModelConfig(
                client_class=OpenRouterAI,
                token_counter=TokenCounterFactory().create_openrouter_counter(),
                test_connection=TestConnections().openrouter,
                pricing={
                    TokenDirection.ENCODE: 0,
                    TokenDirection.DECODE: 0,
                },
            ),
            'z-ai/glm-4.5-air': ModelConfig(
                client_class=OpenRouterAI,
                token_counter=TokenCounterFactory().create_openrouter_counter(),
                test_connection=TestConnections().openrouter,
                pricing={
                    TokenDirection.ENCODE: 0.0000002,
                    TokenDirection.DECODE: 0.0000011,
                },
            ),
            'qwen/qwen3-235b-a22b-thinking-2507': ModelConfig(
                client_class=OpenRouterAI,
                token_counter=TokenCounterFactory().create_openrouter_counter(),
                test_connection=TestConnections().openrouter,
                pricing={
                    TokenDirection.ENCODE: 0.000000077968332,
                    TokenDirection.DECODE: 0.00000031202496,
                },
            ),
            'z-ai/glm-4-32b': ModelConfig(
                client_class=OpenRouterAI,
                token_counter=TokenCounterFactory().create_openrouter_counter(),
                test_connection=TestConnections().openrouter,
                pricing={
                    TokenDirection.ENCODE: 0.0000001,
                    TokenDirection.DECODE: 0.0000001,
                },
            ),
            'qwen/qwen3-coder:free': ModelConfig(
                client_class=OpenRouterAI,
                token_counter=TokenCounterFactory().create_openrouter_counter(),
                test_connection=TestConnections().openrouter,
                pricing={
                    TokenDirection.ENCODE: 0,
                    TokenDirection.DECODE: 0,
                },
            ),
            'qwen/qwen3-coder': ModelConfig(
                client_class=OpenRouterAI,
                token_counter=TokenCounterFactory().create_openrouter_counter(),
                test_connection=TestConnections().openrouter,
                pricing={
                    TokenDirection.ENCODE: 0.0000002,
                    TokenDirection.DECODE: 0.0000008,
                },
            ),
            'bytedance/ui-tars-1.5-7b': ModelConfig(
                client_class=OpenRouterAI,
                token_counter=TokenCounterFactory().create_openrouter_counter(),
                test_connection=TestConnections().openrouter,
                pricing={
                    TokenDirection.ENCODE: 0.0000001,
                    TokenDirection.DECODE: 0.0000002,
                },
            ),
            'google/gemini-2.5-flash-lite': ModelConfig(
                client_class=OpenRouterAI,
                token_counter=TokenCounterFactory().create_openrouter_counter(),
                test_connection=TestConnections().openrouter,
                pricing={
                    TokenDirection.ENCODE: 0.0000001,
                    TokenDirection.DECODE: 0.0000004,
                },
            ),
            'qwen/qwen3-235b-a22b-2507': ModelConfig(
                client_class=OpenRouterAI,
                token_counter=TokenCounterFactory().create_openrouter_counter(),
                test_connection=TestConnections().openrouter,
                pricing={
                    TokenDirection.ENCODE: 0.000000077968332,
                    TokenDirection.DECODE: 0.00000031202496,
                },
            ),
            'switchpoint/router': ModelConfig(
                client_class=OpenRouterAI,
                token_counter=TokenCounterFactory().create_openrouter_counter(),
                test_connection=TestConnections().openrouter,
                pricing={
                    TokenDirection.ENCODE: 0.00000085,
                    TokenDirection.DECODE: 0.0000034,
                },
            ),
            'moonshotai/kimi-k2:free': ModelConfig(
                client_class=OpenRouterAI,
                token_counter=TokenCounterFactory().create_openrouter_counter(),
                test_connection=TestConnections().openrouter,
                pricing={
                    TokenDirection.ENCODE: 0,
                    TokenDirection.DECODE: 0,
                },
            ),
            'moonshotai/kimi-k2': ModelConfig(
                client_class=OpenRouterAI,
                token_counter=TokenCounterFactory().create_openrouter_counter(),
                test_connection=TestConnections().openrouter,
                pricing={
                    TokenDirection.ENCODE: 0.00000014,
                    TokenDirection.DECODE: 0.00000249,
                },
            ),
            'thudm/glm-4.1v-9b-thinking': ModelConfig(
                client_class=OpenRouterAI,
                token_counter=TokenCounterFactory().create_openrouter_counter(),
                test_connection=TestConnections().openrouter,
                pricing={
                    TokenDirection.ENCODE: 0.000000035,
                    TokenDirection.DECODE: 0.000000138,
                },
            ),
            'mistralai/devstral-medium': ModelConfig(
                client_class=OpenRouterAI,
                token_counter=TokenCounterFactory().create_openrouter_counter(),
                test_connection=TestConnections().openrouter,
                pricing={
                    TokenDirection.ENCODE: 0.0000004,
                    TokenDirection.DECODE: 0.000002,
                },
            ),
            'mistralai/devstral-small': ModelConfig(
                client_class=OpenRouterAI,
                token_counter=TokenCounterFactory().create_openrouter_counter(),
                test_connection=TestConnections().openrouter,
                pricing={
                    TokenDirection.ENCODE: 0.00000007,
                    TokenDirection.DECODE: 0.00000028,
                },
            ),
            'cognitivecomputations/dolphin-mistral-24b-venice-edition:free': ModelConfig(
                client_class=OpenRouterAI,
                token_counter=TokenCounterFactory().create_openrouter_counter(),
                test_connection=TestConnections().openrouter,
                pricing={
                    TokenDirection.ENCODE: 0,
                    TokenDirection.DECODE: 0,
                },
            ),
            'x-ai/grok-4': ModelConfig(
                client_class=OpenRouterAI,
                token_counter=TokenCounterFactory().create_openrouter_counter(),
                test_connection=TestConnections().openrouter,
                pricing={
                    TokenDirection.ENCODE: 0.000003,
                    TokenDirection.DECODE: 0.000015,
                },
            ),
            'google/gemma-3n-e2b-it:free': ModelConfig(
                client_class=OpenRouterAI,
                token_counter=TokenCounterFactory().create_openrouter_counter(),
                test_connection=TestConnections().openrouter,
                pricing={
                    TokenDirection.ENCODE: 0,
                    TokenDirection.DECODE: 0,
                },
            ),
            'tencent/hunyuan-a13b-instruct:free': ModelConfig(
                client_class=OpenRouterAI,
                token_counter=TokenCounterFactory().create_openrouter_counter(),
                test_connection=TestConnections().openrouter,
                pricing={
                    TokenDirection.ENCODE: 0,
                    TokenDirection.DECODE: 0,
                },
            ),
            'tencent/hunyuan-a13b-instruct': ModelConfig(
                client_class=OpenRouterAI,
                token_counter=TokenCounterFactory().create_openrouter_counter(),
                test_connection=TestConnections().openrouter,
                pricing={
                    TokenDirection.ENCODE: 0.00000003,
                    TokenDirection.DECODE: 0.00000003,
                },
            ),
            'tngtech/deepseek-r1t2-chimera:free': ModelConfig(
                client_class=OpenRouterAI,
                token_counter=TokenCounterFactory().create_openrouter_counter(),
                test_connection=TestConnections().openrouter,
                pricing={
                    TokenDirection.ENCODE: 0,
                    TokenDirection.DECODE: 0,
                },
            ),
            'morph/morph-v3-large': ModelConfig(
                client_class=OpenRouterAI,
                token_counter=TokenCounterFactory().create_openrouter_counter(),
                test_connection=TestConnections().openrouter,
                pricing={
                    TokenDirection.ENCODE: 0.0000009,
                    TokenDirection.DECODE: 0.0000019,
                },
            ),
            'morph/morph-v3-fast': ModelConfig(
                client_class=OpenRouterAI,
                token_counter=TokenCounterFactory().create_openrouter_counter(),
                test_connection=TestConnections().openrouter,
                pricing={
                    TokenDirection.ENCODE: 0.0000009,
                    TokenDirection.DECODE: 0.0000019,
                },
            ),
            'baidu/ernie-4.5-vl-424b-a47b': ModelConfig(
                client_class=OpenRouterAI,
                token_counter=TokenCounterFactory().create_openrouter_counter(),
                test_connection=TestConnections().openrouter,
                pricing={
                    TokenDirection.ENCODE: 0.00000042,
                    TokenDirection.DECODE: 0.00000125,
                },
            ),
            'baidu/ernie-4.5-300b-a47b': ModelConfig(
                client_class=OpenRouterAI,
                token_counter=TokenCounterFactory().create_openrouter_counter(),
                test_connection=TestConnections().openrouter,
                pricing={
                    TokenDirection.ENCODE: 0.00000028,
                    TokenDirection.DECODE: 0.0000011,
                },
            ),
            'thedrummer/anubis-70b-v1.1': ModelConfig(
                client_class=OpenRouterAI,
                token_counter=TokenCounterFactory().create_openrouter_counter(),
                test_connection=TestConnections().openrouter,
                pricing={
                    TokenDirection.ENCODE: 0.0000004,
                    TokenDirection.DECODE: 0.0000007,
                },
            ),
            'inception/mercury': ModelConfig(
                client_class=OpenRouterAI,
                token_counter=TokenCounterFactory().create_openrouter_counter(),
                test_connection=TestConnections().openrouter,
                pricing={
                    TokenDirection.ENCODE: 0.00000025,
                    TokenDirection.DECODE: 0.000001,
                },
            ),
            'mistralai/mistral-small-3.2-24b-instruct:free': ModelConfig(
                client_class=OpenRouterAI,
                token_counter=TokenCounterFactory().create_openrouter_counter(),
                test_connection=TestConnections().openrouter,
                pricing={
                    TokenDirection.ENCODE: 0,
                    TokenDirection.DECODE: 0,
                },
            ),
            'mistralai/mistral-small-3.2-24b-instruct': ModelConfig(
                client_class=OpenRouterAI,
                token_counter=TokenCounterFactory().create_openrouter_counter(),
                test_connection=TestConnections().openrouter,
                pricing={
                    TokenDirection.ENCODE: 0.00000005,
                    TokenDirection.DECODE: 0.0000001,
                },
            ),
            'minimax/minimax-m1': ModelConfig(
                client_class=OpenRouterAI,
                token_counter=TokenCounterFactory().create_openrouter_counter(),
                test_connection=TestConnections().openrouter,
                pricing={
                    TokenDirection.ENCODE: 0.0000003,
                    TokenDirection.DECODE: 0.00000165,
                },
            ),
            'google/gemini-2.5-flash-lite-preview-06-17': ModelConfig(
                client_class=OpenRouterAI,
                token_counter=TokenCounterFactory().create_openrouter_counter(),
                test_connection=TestConnections().openrouter,
                pricing={
                    TokenDirection.ENCODE: 0.0000001,
                    TokenDirection.DECODE: 0.0000004,
                },
            ),
            'google/gemini-2.5-flash': ModelConfig(
                client_class=OpenRouterAI,
                token_counter=TokenCounterFactory().create_openrouter_counter(),
                test_connection=TestConnections().openrouter,
                pricing={
                    TokenDirection.ENCODE: 0.0000003,
                    TokenDirection.DECODE: 0.0000025,
                },
            ),
            'google/gemini-2.5-pro': ModelConfig(
                client_class=OpenRouterAI,
                token_counter=TokenCounterFactory().create_openrouter_counter(),
                test_connection=TestConnections().openrouter,
                pricing={
                    TokenDirection.ENCODE: 0.00000125,
                    TokenDirection.DECODE: 0.00001,
                },
            ),
            'moonshotai/kimi-dev-72b:free': ModelConfig(
                client_class=OpenRouterAI,
                token_counter=TokenCounterFactory().create_openrouter_counter(),
                test_connection=TestConnections().openrouter,
                pricing={
                    TokenDirection.ENCODE: 0,
                    TokenDirection.DECODE: 0,
                },
            ),
            'openai/o3-pro': ModelConfig(
                client_class=OpenRouterAI,
                token_counter=TokenCounterFactory().create_openrouter_counter(),
                test_connection=TestConnections().openrouter,
                pricing={
                    TokenDirection.ENCODE: 0.00002,
                    TokenDirection.DECODE: 0.00008,
                },
            ),
            'x-ai/grok-3-mini': ModelConfig(
                client_class=OpenRouterAI,
                token_counter=TokenCounterFactory().create_openrouter_counter(),
                test_connection=TestConnections().openrouter,
                pricing={
                    TokenDirection.ENCODE: 0.0000003,
                    TokenDirection.DECODE: 0.0000005,
                },
            ),
            'x-ai/grok-3': ModelConfig(
                client_class=OpenRouterAI,
                token_counter=TokenCounterFactory().create_openrouter_counter(),
                test_connection=TestConnections().openrouter,
                pricing={
                    TokenDirection.ENCODE: 0.000003,
                    TokenDirection.DECODE: 0.000015,
                },
            ),
            'mistralai/magistral-small-2506': ModelConfig(
                client_class=OpenRouterAI,
                token_counter=TokenCounterFactory().create_openrouter_counter(),
                test_connection=TestConnections().openrouter,
                pricing={
                    TokenDirection.ENCODE: 0.0000005,
                    TokenDirection.DECODE: 0.0000015,
                },
            ),
            'mistralai/magistral-medium-2506': ModelConfig(
                client_class=OpenRouterAI,
                token_counter=TokenCounterFactory().create_openrouter_counter(),
                test_connection=TestConnections().openrouter,
                pricing={
                    TokenDirection.ENCODE: 0.000002,
                    TokenDirection.DECODE: 0.000005,
                },
            ),
            'mistralai/magistral-medium-2506:thinking': ModelConfig(
                client_class=OpenRouterAI,
                token_counter=TokenCounterFactory().create_openrouter_counter(),
                test_connection=TestConnections().openrouter,
                pricing={
                    TokenDirection.ENCODE: 0.000002,
                    TokenDirection.DECODE: 0.000005,
                },
            ),
            'google/gemini-2.5-pro-preview': ModelConfig(
                client_class=OpenRouterAI,
                token_counter=TokenCounterFactory().create_openrouter_counter(),
                test_connection=TestConnections().openrouter,
                pricing={
                    TokenDirection.ENCODE: 0.00000125,
                    TokenDirection.DECODE: 0.00001,
                },
            ),
            'deepseek/deepseek-r1-0528-qwen3-8b:free': ModelConfig(
                client_class=OpenRouterAI,
                token_counter=TokenCounterFactory().create_openrouter_counter(),
                test_connection=TestConnections().openrouter,
                pricing={
                    TokenDirection.ENCODE: 0,
                    TokenDirection.DECODE: 0,
                },
            ),
            'deepseek/deepseek-r1-0528-qwen3-8b': ModelConfig(
                client_class=OpenRouterAI,
                token_counter=TokenCounterFactory().create_openrouter_counter(),
                test_connection=TestConnections().openrouter,
                pricing={
                    TokenDirection.ENCODE: 0.00000001,
                    TokenDirection.DECODE: 0.00000002,
                },
            ),
            'deepseek/deepseek-r1-0528:free': ModelConfig(
                client_class=OpenRouterAI,
                token_counter=TokenCounterFactory().create_openrouter_counter(),
                test_connection=TestConnections().openrouter,
                pricing={
                    TokenDirection.ENCODE: 0,
                    TokenDirection.DECODE: 0,
                },
            ),
            'deepseek/deepseek-r1-0528': ModelConfig(
                client_class=OpenRouterAI,
                token_counter=TokenCounterFactory().create_openrouter_counter(),
                test_connection=TestConnections().openrouter,
                pricing={
                    TokenDirection.ENCODE: 0.0000001999188,
                    TokenDirection.DECODE: 0.000000800064,
                },
            ),
            'sarvamai/sarvam-m:free': ModelConfig(
                client_class=OpenRouterAI,
                token_counter=TokenCounterFactory().create_openrouter_counter(),
                test_connection=TestConnections().openrouter,
                pricing={
                    TokenDirection.ENCODE: 0,
                    TokenDirection.DECODE: 0,
                },
            ),
            'anthropic/claude-opus-4': ModelConfig(
                client_class=OpenRouterAI,
                token_counter=TokenCounterFactory().create_openrouter_counter(),
                test_connection=TestConnections().openrouter,
                pricing={
                    TokenDirection.ENCODE: 0.000015,
                    TokenDirection.DECODE: 0.000075,
                },
            ),
            'anthropic/claude-sonnet-4': ModelConfig(
                client_class=OpenRouterAI,
                token_counter=TokenCounterFactory().create_openrouter_counter(),
                test_connection=TestConnections().openrouter,
                pricing={
                    TokenDirection.ENCODE: 0.000003,
                    TokenDirection.DECODE: 0.000015,
                },
            ),
            'mistralai/devstral-small-2505:free': ModelConfig(
                client_class=OpenRouterAI,
                token_counter=TokenCounterFactory().create_openrouter_counter(),
                test_connection=TestConnections().openrouter,
                pricing={
                    TokenDirection.ENCODE: 0,
                    TokenDirection.DECODE: 0,
                },
            ),
            'mistralai/devstral-small-2505': ModelConfig(
                client_class=OpenRouterAI,
                token_counter=TokenCounterFactory().create_openrouter_counter(),
                test_connection=TestConnections().openrouter,
                pricing={
                    TokenDirection.ENCODE: 0.00000001999188,
                    TokenDirection.DECODE: 0.0000000800064,
                },
            ),
            'google/gemma-3n-e4b-it:free': ModelConfig(
                client_class=OpenRouterAI,
                token_counter=TokenCounterFactory().create_openrouter_counter(),
                test_connection=TestConnections().openrouter,
                pricing={
                    TokenDirection.ENCODE: 0,
                    TokenDirection.DECODE: 0,
                },
            ),
            'google/gemma-3n-e4b-it': ModelConfig(
                client_class=OpenRouterAI,
                token_counter=TokenCounterFactory().create_openrouter_counter(),
                test_connection=TestConnections().openrouter,
                pricing={
                    TokenDirection.ENCODE: 0.00000002,
                    TokenDirection.DECODE: 0.00000004,
                },
            ),
            'openai/codex-mini': ModelConfig(
                client_class=OpenRouterAI,
                token_counter=TokenCounterFactory().create_openrouter_counter(),
                test_connection=TestConnections().openrouter,
                pricing={
                    TokenDirection.ENCODE: 0.0000015,
                    TokenDirection.DECODE: 0.000006,
                },
            ),
            'meta-llama/llama-3.3-8b-instruct:free': ModelConfig(
                client_class=OpenRouterAI,
                token_counter=TokenCounterFactory().create_openrouter_counter(),
                test_connection=TestConnections().openrouter,
                pricing={
                    TokenDirection.ENCODE: 0,
                    TokenDirection.DECODE: 0,
                },
            ),
            'nousresearch/deephermes-3-mistral-24b-preview': ModelConfig(
                client_class=OpenRouterAI,
                token_counter=TokenCounterFactory().create_openrouter_counter(),
                test_connection=TestConnections().openrouter,
                pricing={
                    TokenDirection.ENCODE: 0.00000009329544,
                    TokenDirection.DECODE: 0.0000003733632,
                },
            ),
            'mistralai/mistral-medium-3': ModelConfig(
                client_class=OpenRouterAI,
                token_counter=TokenCounterFactory().create_openrouter_counter(),
                test_connection=TestConnections().openrouter,
                pricing={
                    TokenDirection.ENCODE: 0.0000004,
                    TokenDirection.DECODE: 0.000002,
                },
            ),
            'google/gemini-2.5-pro-preview-05-06': ModelConfig(
                client_class=OpenRouterAI,
                token_counter=TokenCounterFactory().create_openrouter_counter(),
                test_connection=TestConnections().openrouter,
                pricing={
                    TokenDirection.ENCODE: 0.00000125,
                    TokenDirection.DECODE: 0.00001,
                },
            ),
            'arcee-ai/spotlight': ModelConfig(
                client_class=OpenRouterAI,
                token_counter=TokenCounterFactory().create_openrouter_counter(),
                test_connection=TestConnections().openrouter,
                pricing={
                    TokenDirection.ENCODE: 0.00000018,
                    TokenDirection.DECODE: 0.00000018,
                },
            ),
            'arcee-ai/maestro-reasoning': ModelConfig(
                client_class=OpenRouterAI,
                token_counter=TokenCounterFactory().create_openrouter_counter(),
                test_connection=TestConnections().openrouter,
                pricing={
                    TokenDirection.ENCODE: 0.0000009,
                    TokenDirection.DECODE: 0.0000033,
                },
            ),
            'arcee-ai/virtuoso-large': ModelConfig(
                client_class=OpenRouterAI,
                token_counter=TokenCounterFactory().create_openrouter_counter(),
                test_connection=TestConnections().openrouter,
                pricing={
                    TokenDirection.ENCODE: 0.00000075,
                    TokenDirection.DECODE: 0.0000012,
                },
            ),
            'arcee-ai/coder-large': ModelConfig(
                client_class=OpenRouterAI,
                token_counter=TokenCounterFactory().create_openrouter_counter(),
                test_connection=TestConnections().openrouter,
                pricing={
                    TokenDirection.ENCODE: 0.0000005,
                    TokenDirection.DECODE: 0.0000008,
                },
            ),
            'microsoft/phi-4-reasoning-plus': ModelConfig(
                client_class=OpenRouterAI,
                token_counter=TokenCounterFactory().create_openrouter_counter(),
                test_connection=TestConnections().openrouter,
                pricing={
                    TokenDirection.ENCODE: 0.00000007,
                    TokenDirection.DECODE: 0.00000035,
                },
            ),
            'inception/mercury-coder': ModelConfig(
                client_class=OpenRouterAI,
                token_counter=TokenCounterFactory().create_openrouter_counter(),
                test_connection=TestConnections().openrouter,
                pricing={
                    TokenDirection.ENCODE: 0.00000025,
                    TokenDirection.DECODE: 0.000001,
                },
            ),
            'qwen/qwen3-4b:free': ModelConfig(
                client_class=OpenRouterAI,
                token_counter=TokenCounterFactory().create_openrouter_counter(),
                test_connection=TestConnections().openrouter,
                pricing={
                    TokenDirection.ENCODE: 0,
                    TokenDirection.DECODE: 0,
                },
            ),
            'opengvlab/internvl3-14b': ModelConfig(
                client_class=OpenRouterAI,
                token_counter=TokenCounterFactory().create_openrouter_counter(),
                test_connection=TestConnections().openrouter,
                pricing={
                    TokenDirection.ENCODE: 0.0000002,
                    TokenDirection.DECODE: 0.0000004,
                },
            ),
            'deepseek/deepseek-prover-v2': ModelConfig(
                client_class=OpenRouterAI,
                token_counter=TokenCounterFactory().create_openrouter_counter(),
                test_connection=TestConnections().openrouter,
                pricing={
                    TokenDirection.ENCODE: 0.0000005,
                    TokenDirection.DECODE: 0.00000218,
                },
            ),
            'meta-llama/llama-guard-4-12b': ModelConfig(
                client_class=OpenRouterAI,
                token_counter=TokenCounterFactory().create_openrouter_counter(),
                test_connection=TestConnections().openrouter,
                pricing={
                    TokenDirection.ENCODE: 0.00000018,
                    TokenDirection.DECODE: 0.00000018,
                },
            ),
            'qwen/qwen3-30b-a3b:free': ModelConfig(
                client_class=OpenRouterAI,
                token_counter=TokenCounterFactory().create_openrouter_counter(),
                test_connection=TestConnections().openrouter,
                pricing={
                    TokenDirection.ENCODE: 0,
                    TokenDirection.DECODE: 0,
                },
            ),
            'qwen/qwen3-30b-a3b': ModelConfig(
                client_class=OpenRouterAI,
                token_counter=TokenCounterFactory().create_openrouter_counter(),
                test_connection=TestConnections().openrouter,
                pricing={
                    TokenDirection.ENCODE: 0.00000001999188,
                    TokenDirection.DECODE: 0.0000000800064,
                },
            ),
            'qwen/qwen3-8b:free': ModelConfig(
                client_class=OpenRouterAI,
                token_counter=TokenCounterFactory().create_openrouter_counter(),
                test_connection=TestConnections().openrouter,
                pricing={
                    TokenDirection.ENCODE: 0,
                    TokenDirection.DECODE: 0,
                },
            ),
            'qwen/qwen3-8b': ModelConfig(
                client_class=OpenRouterAI,
                token_counter=TokenCounterFactory().create_openrouter_counter(),
                test_connection=TestConnections().openrouter,
                pricing={
                    TokenDirection.ENCODE: 0.000000035,
                    TokenDirection.DECODE: 0.000000138,
                },
            ),
            'qwen/qwen3-14b:free': ModelConfig(
                client_class=OpenRouterAI,
                token_counter=TokenCounterFactory().create_openrouter_counter(),
                test_connection=TestConnections().openrouter,
                pricing={
                    TokenDirection.ENCODE: 0,
                    TokenDirection.DECODE: 0,
                },
            ),
            'qwen/qwen3-14b': ModelConfig(
                client_class=OpenRouterAI,
                token_counter=TokenCounterFactory().create_openrouter_counter(),
                test_connection=TestConnections().openrouter,
                pricing={
                    TokenDirection.ENCODE: 0.00000006,
                    TokenDirection.DECODE: 0.00000024,
                },
            ),
            'qwen/qwen3-32b': ModelConfig(
                client_class=OpenRouterAI,
                token_counter=TokenCounterFactory().create_openrouter_counter(),
                test_connection=TestConnections().openrouter,
                pricing={
                    TokenDirection.ENCODE: 0.000000017992692,
                    TokenDirection.DECODE: 0.00000007200576,
                },
            ),
            'qwen/qwen3-235b-a22b:free': ModelConfig(
                client_class=OpenRouterAI,
                token_counter=TokenCounterFactory().create_openrouter_counter(),
                test_connection=TestConnections().openrouter,
                pricing={
                    TokenDirection.ENCODE: 0,
                    TokenDirection.DECODE: 0,
                },
            ),
            'qwen/qwen3-235b-a22b': ModelConfig(
                client_class=OpenRouterAI,
                token_counter=TokenCounterFactory().create_openrouter_counter(),
                test_connection=TestConnections().openrouter,
                pricing={
                    TokenDirection.ENCODE: 0.00000013,
                    TokenDirection.DECODE: 0.0000006,
                },
            ),
            'tngtech/deepseek-r1t-chimera:free': ModelConfig(
                client_class=OpenRouterAI,
                token_counter=TokenCounterFactory().create_openrouter_counter(),
                test_connection=TestConnections().openrouter,
                pricing={
                    TokenDirection.ENCODE: 0,
                    TokenDirection.DECODE: 0,
                },
            ),
            'tngtech/deepseek-r1t-chimera': ModelConfig(
                client_class=OpenRouterAI,
                token_counter=TokenCounterFactory().create_openrouter_counter(),
                test_connection=TestConnections().openrouter,
                pricing={
                    TokenDirection.ENCODE: 0.0000001999188,
                    TokenDirection.DECODE: 0.000000800064,
                },
            ),
            'microsoft/mai-ds-r1:free': ModelConfig(
                client_class=OpenRouterAI,
                token_counter=TokenCounterFactory().create_openrouter_counter(),
                test_connection=TestConnections().openrouter,
                pricing={
                    TokenDirection.ENCODE: 0,
                    TokenDirection.DECODE: 0,
                },
            ),
            'microsoft/mai-ds-r1': ModelConfig(
                client_class=OpenRouterAI,
                token_counter=TokenCounterFactory().create_openrouter_counter(),
                test_connection=TestConnections().openrouter,
                pricing={
                    TokenDirection.ENCODE: 0.0000001999188,
                    TokenDirection.DECODE: 0.000000800064,
                },
            ),
            'thudm/glm-z1-32b': ModelConfig(
                client_class=OpenRouterAI,
                token_counter=TokenCounterFactory().create_openrouter_counter(),
                test_connection=TestConnections().openrouter,
                pricing={
                    TokenDirection.ENCODE: 0.00000001999188,
                    TokenDirection.DECODE: 0.0000000800064,
                },
            ),
            'thudm/glm-4-32b': ModelConfig(
                client_class=OpenRouterAI,
                token_counter=TokenCounterFactory().create_openrouter_counter(),
                test_connection=TestConnections().openrouter,
                pricing={
                    TokenDirection.ENCODE: 0.00000055,
                    TokenDirection.DECODE: 0.00000166,
                },
            ),
            'openai/o4-mini-high': ModelConfig(
                client_class=OpenRouterAI,
                token_counter=TokenCounterFactory().create_openrouter_counter(),
                test_connection=TestConnections().openrouter,
                pricing={
                    TokenDirection.ENCODE: 0.0000011,
                    TokenDirection.DECODE: 0.0000044,
                },
            ),
            'openai/o3': ModelConfig(
                client_class=OpenRouterAI,
                token_counter=TokenCounterFactory().create_openrouter_counter(),
                test_connection=TestConnections().openrouter,
                pricing={
                    TokenDirection.ENCODE: 0.000002,
                    TokenDirection.DECODE: 0.000008,
                },
            ),
            'openai/o4-mini': ModelConfig(
                client_class=OpenRouterAI,
                token_counter=TokenCounterFactory().create_openrouter_counter(),
                test_connection=TestConnections().openrouter,
                pricing={
                    TokenDirection.ENCODE: 0.0000011,
                    TokenDirection.DECODE: 0.0000044,
                },
            ),
            'shisa-ai/shisa-v2-llama3.3-70b:free': ModelConfig(
                client_class=OpenRouterAI,
                token_counter=TokenCounterFactory().create_openrouter_counter(),
                test_connection=TestConnections().openrouter,
                pricing={
                    TokenDirection.ENCODE: 0,
                    TokenDirection.DECODE: 0,
                },
            ),
            'shisa-ai/shisa-v2-llama3.3-70b': ModelConfig(
                client_class=OpenRouterAI,
                token_counter=TokenCounterFactory().create_openrouter_counter(),
                test_connection=TestConnections().openrouter,
                pricing={
                    TokenDirection.ENCODE: 0.00000001999188,
                    TokenDirection.DECODE: 0.0000000800064,
                },
            ),
            'openai/gpt-4.1': ModelConfig(
                client_class=OpenRouterAI,
                token_counter=TokenCounterFactory().create_openrouter_counter(),
                test_connection=TestConnections().openrouter,
                pricing={
                    TokenDirection.ENCODE: 0.000002,
                    TokenDirection.DECODE: 0.000008,
                },
            ),
            'openai/gpt-4.1-mini': ModelConfig(
                client_class=OpenRouterAI,
                token_counter=TokenCounterFactory().create_openrouter_counter(),
                test_connection=TestConnections().openrouter,
                pricing={
                    TokenDirection.ENCODE: 0.0000004,
                    TokenDirection.DECODE: 0.0000016,
                },
            ),
            'openai/gpt-4.1-nano': ModelConfig(
                client_class=OpenRouterAI,
                token_counter=TokenCounterFactory().create_openrouter_counter(),
                test_connection=TestConnections().openrouter,
                pricing={
                    TokenDirection.ENCODE: 0.0000001,
                    TokenDirection.DECODE: 0.0000004,
                },
            ),
            'eleutherai/llemma_7b': ModelConfig(
                client_class=OpenRouterAI,
                token_counter=TokenCounterFactory().create_openrouter_counter(),
                test_connection=TestConnections().openrouter,
                pricing={
                    TokenDirection.ENCODE: 0.0000008,
                    TokenDirection.DECODE: 0.0000012,
                },
            ),
            'alfredpros/codellama-7b-instruct-solidity': ModelConfig(
                client_class=OpenRouterAI,
                token_counter=TokenCounterFactory().create_openrouter_counter(),
                test_connection=TestConnections().openrouter,
                pricing={
                    TokenDirection.ENCODE: 0.0000007,
                    TokenDirection.DECODE: 0.0000011,
                },
            ),
            'arliai/qwq-32b-arliai-rpr-v1:free': ModelConfig(
                client_class=OpenRouterAI,
                token_counter=TokenCounterFactory().create_openrouter_counter(),
                test_connection=TestConnections().openrouter,
                pricing={
                    TokenDirection.ENCODE: 0,
                    TokenDirection.DECODE: 0,
                },
            ),
            'arliai/qwq-32b-arliai-rpr-v1': ModelConfig(
                client_class=OpenRouterAI,
                token_counter=TokenCounterFactory().create_openrouter_counter(),
                test_connection=TestConnections().openrouter,
                pricing={
                    TokenDirection.ENCODE: 0.00000001,
                    TokenDirection.DECODE: 0.0000000400032,
                },
            ),
            'agentica-org/deepcoder-14b-preview:free': ModelConfig(
                client_class=OpenRouterAI,
                token_counter=TokenCounterFactory().create_openrouter_counter(),
                test_connection=TestConnections().openrouter,
                pricing={
                    TokenDirection.ENCODE: 0,
                    TokenDirection.DECODE: 0,
                },
            ),
            'agentica-org/deepcoder-14b-preview': ModelConfig(
                client_class=OpenRouterAI,
                token_counter=TokenCounterFactory().create_openrouter_counter(),
                test_connection=TestConnections().openrouter,
                pricing={
                    TokenDirection.ENCODE: 0.000000015,
                    TokenDirection.DECODE: 0.000000015,
                },
            ),
            'moonshotai/kimi-vl-a3b-thinking:free': ModelConfig(
                client_class=OpenRouterAI,
                token_counter=TokenCounterFactory().create_openrouter_counter(),
                test_connection=TestConnections().openrouter,
                pricing={
                    TokenDirection.ENCODE: 0,
                    TokenDirection.DECODE: 0,
                },
            ),
            'moonshotai/kimi-vl-a3b-thinking': ModelConfig(
                client_class=OpenRouterAI,
                token_counter=TokenCounterFactory().create_openrouter_counter(),
                test_connection=TestConnections().openrouter,
                pricing={
                    TokenDirection.ENCODE: 0.00000002498985,
                    TokenDirection.DECODE: 0.000000100008,
                },
            ),
            'x-ai/grok-3-mini-beta': ModelConfig(
                client_class=OpenRouterAI,
                token_counter=TokenCounterFactory().create_openrouter_counter(),
                test_connection=TestConnections().openrouter,
                pricing={
                    TokenDirection.ENCODE: 0.0000003,
                    TokenDirection.DECODE: 0.0000005,
                },
            ),
            'x-ai/grok-3-beta': ModelConfig(
                client_class=OpenRouterAI,
                token_counter=TokenCounterFactory().create_openrouter_counter(),
                test_connection=TestConnections().openrouter,
                pricing={
                    TokenDirection.ENCODE: 0.000003,
                    TokenDirection.DECODE: 0.000015,
                },
            ),
            'nvidia/llama-3.3-nemotron-super-49b-v1': ModelConfig(
                client_class=OpenRouterAI,
                token_counter=TokenCounterFactory().create_openrouter_counter(),
                test_connection=TestConnections().openrouter,
                pricing={
                    TokenDirection.ENCODE: 0.00000013,
                    TokenDirection.DECODE: 0.0000004,
                },
            ),
            'nvidia/llama-3.1-nemotron-ultra-253b-v1:free': ModelConfig(
                client_class=OpenRouterAI,
                token_counter=TokenCounterFactory().create_openrouter_counter(),
                test_connection=TestConnections().openrouter,
                pricing={
                    TokenDirection.ENCODE: 0,
                    TokenDirection.DECODE: 0,
                },
            ),
            'nvidia/llama-3.1-nemotron-ultra-253b-v1': ModelConfig(
                client_class=OpenRouterAI,
                token_counter=TokenCounterFactory().create_openrouter_counter(),
                test_connection=TestConnections().openrouter,
                pricing={
                    TokenDirection.ENCODE: 0.0000006,
                    TokenDirection.DECODE: 0.0000018,
                },
            ),
            'meta-llama/llama-4-maverick:free': ModelConfig(
                client_class=OpenRouterAI,
                token_counter=TokenCounterFactory().create_openrouter_counter(),
                test_connection=TestConnections().openrouter,
                pricing={
                    TokenDirection.ENCODE: 0,
                    TokenDirection.DECODE: 0,
                },
            ),
            'meta-llama/llama-4-maverick': ModelConfig(
                client_class=OpenRouterAI,
                token_counter=TokenCounterFactory().create_openrouter_counter(),
                test_connection=TestConnections().openrouter,
                pricing={
                    TokenDirection.ENCODE: 0.00000015,
                    TokenDirection.DECODE: 0.0000006,
                },
            ),
            'meta-llama/llama-4-scout:free': ModelConfig(
                client_class=OpenRouterAI,
                token_counter=TokenCounterFactory().create_openrouter_counter(),
                test_connection=TestConnections().openrouter,
                pricing={
                    TokenDirection.ENCODE: 0,
                    TokenDirection.DECODE: 0,
                },
            ),
            'meta-llama/llama-4-scout': ModelConfig(
                client_class=OpenRouterAI,
                token_counter=TokenCounterFactory().create_openrouter_counter(),
                test_connection=TestConnections().openrouter,
                pricing={
                    TokenDirection.ENCODE: 0.00000008,
                    TokenDirection.DECODE: 0.0000003,
                },
            ),
            'scb10x/llama3.1-typhoon2-70b-instruct': ModelConfig(
                client_class=OpenRouterAI,
                token_counter=TokenCounterFactory().create_openrouter_counter(),
                test_connection=TestConnections().openrouter,
                pricing={
                    TokenDirection.ENCODE: 0.00000088,
                    TokenDirection.DECODE: 0.00000088,
                },
            ),
            'google/gemini-2.5-pro-exp-03-25': ModelConfig(
                client_class=OpenRouterAI,
                token_counter=TokenCounterFactory().create_openrouter_counter(),
                test_connection=TestConnections().openrouter,
                pricing={
                    TokenDirection.ENCODE: 0,
                    TokenDirection.DECODE: 0,
                },
            ),
            'qwen/qwen2.5-vl-32b-instruct:free': ModelConfig(
                client_class=OpenRouterAI,
                token_counter=TokenCounterFactory().create_openrouter_counter(),
                test_connection=TestConnections().openrouter,
                pricing={
                    TokenDirection.ENCODE: 0,
                    TokenDirection.DECODE: 0,
                },
            ),
            'qwen/qwen2.5-vl-32b-instruct': ModelConfig(
                client_class=OpenRouterAI,
                token_counter=TokenCounterFactory().create_openrouter_counter(),
                test_connection=TestConnections().openrouter,
                pricing={
                    TokenDirection.ENCODE: 0.00000001999188,
                    TokenDirection.DECODE: 0.0000000800064,
                },
            ),
            'deepseek/deepseek-chat-v3-0324:free': ModelConfig(
                client_class=OpenRouterAI,
                token_counter=TokenCounterFactory().create_openrouter_counter(),
                test_connection=TestConnections().openrouter,
                pricing={
                    TokenDirection.ENCODE: 0,
                    TokenDirection.DECODE: 0,
                },
            ),
            'deepseek/deepseek-chat-v3-0324': ModelConfig(
                client_class=OpenRouterAI,
                token_counter=TokenCounterFactory().create_openrouter_counter(),
                test_connection=TestConnections().openrouter,
                pricing={
                    TokenDirection.ENCODE: 0.0000001999188,
                    TokenDirection.DECODE: 0.000000800064,
                },
            ),
            'openai/o1-pro': ModelConfig(
                client_class=OpenRouterAI,
                token_counter=TokenCounterFactory().create_openrouter_counter(),
                test_connection=TestConnections().openrouter,
                pricing={
                    TokenDirection.ENCODE: 0.00015,
                    TokenDirection.DECODE: 0.0006,
                },
            ),
            'mistralai/mistral-small-3.1-24b-instruct:free': ModelConfig(
                client_class=OpenRouterAI,
                token_counter=TokenCounterFactory().create_openrouter_counter(),
                test_connection=TestConnections().openrouter,
                pricing={
                    TokenDirection.ENCODE: 0,
                    TokenDirection.DECODE: 0,
                },
            ),
            'mistralai/mistral-small-3.1-24b-instruct': ModelConfig(
                client_class=OpenRouterAI,
                token_counter=TokenCounterFactory().create_openrouter_counter(),
                test_connection=TestConnections().openrouter,
                pricing={
                    TokenDirection.ENCODE: 0.00000001999188,
                    TokenDirection.DECODE: 0.0000000800064,
                },
            ),
            'google/gemma-3-4b-it:free': ModelConfig(
                client_class=OpenRouterAI,
                token_counter=TokenCounterFactory().create_openrouter_counter(),
                test_connection=TestConnections().openrouter,
                pricing={
                    TokenDirection.ENCODE: 0,
                    TokenDirection.DECODE: 0,
                },
            ),
            'google/gemma-3-4b-it': ModelConfig(
                client_class=OpenRouterAI,
                token_counter=TokenCounterFactory().create_openrouter_counter(),
                test_connection=TestConnections().openrouter,
                pricing={
                    TokenDirection.ENCODE: 0.00000002,
                    TokenDirection.DECODE: 0.00000004,
                },
            ),
            'google/gemma-3-12b-it:free': ModelConfig(
                client_class=OpenRouterAI,
                token_counter=TokenCounterFactory().create_openrouter_counter(),
                test_connection=TestConnections().openrouter,
                pricing={
                    TokenDirection.ENCODE: 0,
                    TokenDirection.DECODE: 0,
                },
            ),
            'google/gemma-3-12b-it': ModelConfig(
                client_class=OpenRouterAI,
                token_counter=TokenCounterFactory().create_openrouter_counter(),
                test_connection=TestConnections().openrouter,
                pricing={
                    TokenDirection.ENCODE: 0.0000000481286,
                    TokenDirection.DECODE: 0.000000192608,
                },
            ),
            'cohere/command-a': ModelConfig(
                client_class=OpenRouterAI,
                token_counter=TokenCounterFactory().create_openrouter_counter(),
                test_connection=TestConnections().openrouter,
                pricing={
                    TokenDirection.ENCODE: 0.000002,
                    TokenDirection.DECODE: 0.000008,
                },
            ),
            'openai/gpt-4o-mini-search-preview': ModelConfig(
                client_class=OpenRouterAI,
                token_counter=TokenCounterFactory().create_openrouter_counter(),
                test_connection=TestConnections().openrouter,
                pricing={
                    TokenDirection.ENCODE: 0.00000015,
                    TokenDirection.DECODE: 0.0000006,
                },
            ),
            'openai/gpt-4o-search-preview': ModelConfig(
                client_class=OpenRouterAI,
                token_counter=TokenCounterFactory().create_openrouter_counter(),
                test_connection=TestConnections().openrouter,
                pricing={
                    TokenDirection.ENCODE: 0.0000025,
                    TokenDirection.DECODE: 0.00001,
                },
            ),
            'rekaai/reka-flash-3:free': ModelConfig(
                client_class=OpenRouterAI,
                token_counter=TokenCounterFactory().create_openrouter_counter(),
                test_connection=TestConnections().openrouter,
                pricing={
                    TokenDirection.ENCODE: 0,
                    TokenDirection.DECODE: 0,
                },
            ),
            'google/gemma-3-27b-it:free': ModelConfig(
                client_class=OpenRouterAI,
                token_counter=TokenCounterFactory().create_openrouter_counter(),
                test_connection=TestConnections().openrouter,
                pricing={
                    TokenDirection.ENCODE: 0,
                    TokenDirection.DECODE: 0,
                },
            ),
            'google/gemma-3-27b-it': ModelConfig(
                client_class=OpenRouterAI,
                token_counter=TokenCounterFactory().create_openrouter_counter(),
                test_connection=TestConnections().openrouter,
                pricing={
                    TokenDirection.ENCODE: 0.0000000666396,
                    TokenDirection.DECODE: 0.000000266688,
                },
            ),
            'thedrummer/anubis-pro-105b-v1': ModelConfig(
                client_class=OpenRouterAI,
                token_counter=TokenCounterFactory().create_openrouter_counter(),
                test_connection=TestConnections().openrouter,
                pricing={
                    TokenDirection.ENCODE: 0.0000005,
                    TokenDirection.DECODE: 0.000001,
                },
            ),
            'thedrummer/skyfall-36b-v2': ModelConfig(
                client_class=OpenRouterAI,
                token_counter=TokenCounterFactory().create_openrouter_counter(),
                test_connection=TestConnections().openrouter,
                pricing={
                    TokenDirection.ENCODE: 0.0000000481286,
                    TokenDirection.DECODE: 0.000000192608,
                },
            ),
            'microsoft/phi-4-multimodal-instruct': ModelConfig(
                client_class=OpenRouterAI,
                token_counter=TokenCounterFactory().create_openrouter_counter(),
                test_connection=TestConnections().openrouter,
                pricing={
                    TokenDirection.ENCODE: 0.00000005,
                    TokenDirection.DECODE: 0.0000001,
                },
            ),
            'perplexity/sonar-reasoning-pro': ModelConfig(
                client_class=OpenRouterAI,
                token_counter=TokenCounterFactory().create_openrouter_counter(),
                test_connection=TestConnections().openrouter,
                pricing={
                    TokenDirection.ENCODE: 0.000002,
                    TokenDirection.DECODE: 0.000008,
                },
            ),
            'perplexity/sonar-pro': ModelConfig(
                client_class=OpenRouterAI,
                token_counter=TokenCounterFactory().create_openrouter_counter(),
                test_connection=TestConnections().openrouter,
                pricing={
                    TokenDirection.ENCODE: 0.000003,
                    TokenDirection.DECODE: 0.000015,
                },
            ),
            'perplexity/sonar-deep-research': ModelConfig(
                client_class=OpenRouterAI,
                token_counter=TokenCounterFactory().create_openrouter_counter(),
                test_connection=TestConnections().openrouter,
                pricing={
                    TokenDirection.ENCODE: 0.000002,
                    TokenDirection.DECODE: 0.000008,
                },
            ),
            'qwen/qwq-32b:free': ModelConfig(
                client_class=OpenRouterAI,
                token_counter=TokenCounterFactory().create_openrouter_counter(),
                test_connection=TestConnections().openrouter,
                pricing={
                    TokenDirection.ENCODE: 0,
                    TokenDirection.DECODE: 0,
                },
            ),
            'qwen/qwq-32b': ModelConfig(
                client_class=OpenRouterAI,
                token_counter=TokenCounterFactory().create_openrouter_counter(),
                test_connection=TestConnections().openrouter,
                pricing={
                    TokenDirection.ENCODE: 0.000000075,
                    TokenDirection.DECODE: 0.00000015,
                },
            ),
            'nousresearch/deephermes-3-llama-3-8b-preview:free': ModelConfig(
                client_class=OpenRouterAI,
                token_counter=TokenCounterFactory().create_openrouter_counter(),
                test_connection=TestConnections().openrouter,
                pricing={
                    TokenDirection.ENCODE: 0,
                    TokenDirection.DECODE: 0,
                },
            ),
            'google/gemini-2.0-flash-lite-001': ModelConfig(
                client_class=OpenRouterAI,
                token_counter=TokenCounterFactory().create_openrouter_counter(),
                test_connection=TestConnections().openrouter,
                pricing={
                    TokenDirection.ENCODE: 0.000000075,
                    TokenDirection.DECODE: 0.0000003,
                },
            ),
            'anthropic/claude-3.7-sonnet': ModelConfig(
                client_class=OpenRouterAI,
                token_counter=TokenCounterFactory().create_openrouter_counter(),
                test_connection=TestConnections().openrouter,
                pricing={
                    TokenDirection.ENCODE: 0.000003,
                    TokenDirection.DECODE: 0.000015,
                },
            ),
            'anthropic/claude-3.7-sonnet:thinking': ModelConfig(
                client_class=OpenRouterAI,
                token_counter=TokenCounterFactory().create_openrouter_counter(),
                test_connection=TestConnections().openrouter,
                pricing={
                    TokenDirection.ENCODE: 0.000003,
                    TokenDirection.DECODE: 0.000015,
                },
            ),
            'perplexity/r1-1776': ModelConfig(
                client_class=OpenRouterAI,
                token_counter=TokenCounterFactory().create_openrouter_counter(),
                test_connection=TestConnections().openrouter,
                pricing={
                    TokenDirection.ENCODE: 0.000002,
                    TokenDirection.DECODE: 0.000008,
                },
            ),
            'mistralai/mistral-saba': ModelConfig(
                client_class=OpenRouterAI,
                token_counter=TokenCounterFactory().create_openrouter_counter(),
                test_connection=TestConnections().openrouter,
                pricing={
                    TokenDirection.ENCODE: 0.0000002,
                    TokenDirection.DECODE: 0.0000006,
                },
            ),
            'cognitivecomputations/dolphin3.0-r1-mistral-24b:free': ModelConfig(
                client_class=OpenRouterAI,
                token_counter=TokenCounterFactory().create_openrouter_counter(),
                test_connection=TestConnections().openrouter,
                pricing={
                    TokenDirection.ENCODE: 0,
                    TokenDirection.DECODE: 0,
                },
            ),
            'cognitivecomputations/dolphin3.0-r1-mistral-24b': ModelConfig(
                client_class=OpenRouterAI,
                token_counter=TokenCounterFactory().create_openrouter_counter(),
                test_connection=TestConnections().openrouter,
                pricing={
                    TokenDirection.ENCODE: 0.00000001,
                    TokenDirection.DECODE: 0.0000000340768,
                },
            ),
            'cognitivecomputations/dolphin3.0-mistral-24b:free': ModelConfig(
                client_class=OpenRouterAI,
                token_counter=TokenCounterFactory().create_openrouter_counter(),
                test_connection=TestConnections().openrouter,
                pricing={
                    TokenDirection.ENCODE: 0,
                    TokenDirection.DECODE: 0,
                },
            ),
            'cognitivecomputations/dolphin3.0-mistral-24b': ModelConfig(
                client_class=OpenRouterAI,
                token_counter=TokenCounterFactory().create_openrouter_counter(),
                test_connection=TestConnections().openrouter,
                pricing={
                    TokenDirection.ENCODE: 0.000000037022,
                    TokenDirection.DECODE: 0.00000014816,
                },
            ),
            'meta-llama/llama-guard-3-8b': ModelConfig(
                client_class=OpenRouterAI,
                token_counter=TokenCounterFactory().create_openrouter_counter(),
                test_connection=TestConnections().openrouter,
                pricing={
                    TokenDirection.ENCODE: 0.00000002,
                    TokenDirection.DECODE: 0.00000006,
                },
            ),
            'openai/o3-mini-high': ModelConfig(
                client_class=OpenRouterAI,
                token_counter=TokenCounterFactory().create_openrouter_counter(),
                test_connection=TestConnections().openrouter,
                pricing={
                    TokenDirection.ENCODE: 0.0000011,
                    TokenDirection.DECODE: 0.0000044,
                },
            ),
            'deepseek/deepseek-r1-distill-llama-8b': ModelConfig(
                client_class=OpenRouterAI,
                token_counter=TokenCounterFactory().create_openrouter_counter(),
                test_connection=TestConnections().openrouter,
                pricing={
                    TokenDirection.ENCODE: 0.00000004,
                    TokenDirection.DECODE: 0.00000004,
                },
            ),
            'google/gemini-2.0-flash-001': ModelConfig(
                client_class=OpenRouterAI,
                token_counter=TokenCounterFactory().create_openrouter_counter(),
                test_connection=TestConnections().openrouter,
                pricing={
                    TokenDirection.ENCODE: 0.0000001,
                    TokenDirection.DECODE: 0.0000004,
                },
            ),
            'qwen/qwen-vl-plus': ModelConfig(
                client_class=OpenRouterAI,
                token_counter=TokenCounterFactory().create_openrouter_counter(),
                test_connection=TestConnections().openrouter,
                pricing={
                    TokenDirection.ENCODE: 0.00000021,
                    TokenDirection.DECODE: 0.00000063,
                },
            ),
            'aion-labs/aion-1.0': ModelConfig(
                client_class=OpenRouterAI,
                token_counter=TokenCounterFactory().create_openrouter_counter(),
                test_connection=TestConnections().openrouter,
                pricing={
                    TokenDirection.ENCODE: 0.000004,
                    TokenDirection.DECODE: 0.000008,
                },
            ),
            'aion-labs/aion-1.0-mini': ModelConfig(
                client_class=OpenRouterAI,
                token_counter=TokenCounterFactory().create_openrouter_counter(),
                test_connection=TestConnections().openrouter,
                pricing={
                    TokenDirection.ENCODE: 0.0000007,
                    TokenDirection.DECODE: 0.0000014,
                },
            ),
            'aion-labs/aion-rp-llama-3.1-8b': ModelConfig(
                client_class=OpenRouterAI,
                token_counter=TokenCounterFactory().create_openrouter_counter(),
                test_connection=TestConnections().openrouter,
                pricing={
                    TokenDirection.ENCODE: 0.0000002,
                    TokenDirection.DECODE: 0.0000002,
                },
            ),
            'qwen/qwen-vl-max': ModelConfig(
                client_class=OpenRouterAI,
                token_counter=TokenCounterFactory().create_openrouter_counter(),
                test_connection=TestConnections().openrouter,
                pricing={
                    TokenDirection.ENCODE: 0.0000008,
                    TokenDirection.DECODE: 0.0000032,
                },
            ),
            'qwen/qwen-turbo': ModelConfig(
                client_class=OpenRouterAI,
                token_counter=TokenCounterFactory().create_openrouter_counter(),
                test_connection=TestConnections().openrouter,
                pricing={
                    TokenDirection.ENCODE: 0.00000005,
                    TokenDirection.DECODE: 0.0000002,
                },
            ),
            'qwen/qwen2.5-vl-72b-instruct:free': ModelConfig(
                client_class=OpenRouterAI,
                token_counter=TokenCounterFactory().create_openrouter_counter(),
                test_connection=TestConnections().openrouter,
                pricing={
                    TokenDirection.ENCODE: 0,
                    TokenDirection.DECODE: 0,
                },
            ),
            'qwen/qwen2.5-vl-72b-instruct': ModelConfig(
                client_class=OpenRouterAI,
                token_counter=TokenCounterFactory().create_openrouter_counter(),
                test_connection=TestConnections().openrouter,
                pricing={
                    TokenDirection.ENCODE: 0.0000000999594,
                    TokenDirection.DECODE: 0.000000400032,
                },
            ),
            'qwen/qwen-plus': ModelConfig(
                client_class=OpenRouterAI,
                token_counter=TokenCounterFactory().create_openrouter_counter(),
                test_connection=TestConnections().openrouter,
                pricing={
                    TokenDirection.ENCODE: 0.0000004,
                    TokenDirection.DECODE: 0.0000012,
                },
            ),
            'qwen/qwen-max': ModelConfig(
                client_class=OpenRouterAI,
                token_counter=TokenCounterFactory().create_openrouter_counter(),
                test_connection=TestConnections().openrouter,
                pricing={
                    TokenDirection.ENCODE: 0.0000016,
                    TokenDirection.DECODE: 0.0000064,
                },
            ),
            'openai/o3-mini': ModelConfig(
                client_class=OpenRouterAI,
                token_counter=TokenCounterFactory().create_openrouter_counter(),
                test_connection=TestConnections().openrouter,
                pricing={
                    TokenDirection.ENCODE: 0.0000011,
                    TokenDirection.DECODE: 0.0000044,
                },
            ),
            'deepseek/deepseek-r1-distill-qwen-1.5b': ModelConfig(
                client_class=OpenRouterAI,
                token_counter=TokenCounterFactory().create_openrouter_counter(),
                test_connection=TestConnections().openrouter,
                pricing={
                    TokenDirection.ENCODE: 0.00000018,
                    TokenDirection.DECODE: 0.00000018,
                },
            ),
            'mistralai/mistral-small-24b-instruct-2501:free': ModelConfig(
                client_class=OpenRouterAI,
                token_counter=TokenCounterFactory().create_openrouter_counter(),
                test_connection=TestConnections().openrouter,
                pricing={
                    TokenDirection.ENCODE: 0,
                    TokenDirection.DECODE: 0,
                },
            ),
            'mistralai/mistral-small-24b-instruct-2501': ModelConfig(
                client_class=OpenRouterAI,
                token_counter=TokenCounterFactory().create_openrouter_counter(),
                test_connection=TestConnections().openrouter,
                pricing={
                    TokenDirection.ENCODE: 0.00000001999188,
                    TokenDirection.DECODE: 0.0000000800064,
                },
            ),
            'deepseek/deepseek-r1-distill-qwen-32b': ModelConfig(
                client_class=OpenRouterAI,
                token_counter=TokenCounterFactory().create_openrouter_counter(),
                test_connection=TestConnections().openrouter,
                pricing={
                    TokenDirection.ENCODE: 0.000000075,
                    TokenDirection.DECODE: 0.00000015,
                },
            ),
            'deepseek/deepseek-r1-distill-qwen-14b:free': ModelConfig(
                client_class=OpenRouterAI,
                token_counter=TokenCounterFactory().create_openrouter_counter(),
                test_connection=TestConnections().openrouter,
                pricing={
                    TokenDirection.ENCODE: 0,
                    TokenDirection.DECODE: 0,
                },
            ),
            'deepseek/deepseek-r1-distill-qwen-14b': ModelConfig(
                client_class=OpenRouterAI,
                token_counter=TokenCounterFactory().create_openrouter_counter(),
                test_connection=TestConnections().openrouter,
                pricing={
                    TokenDirection.ENCODE: 0.00000015,
                    TokenDirection.DECODE: 0.00000015,
                },
            ),
            'perplexity/sonar-reasoning': ModelConfig(
                client_class=OpenRouterAI,
                token_counter=TokenCounterFactory().create_openrouter_counter(),
                test_connection=TestConnections().openrouter,
                pricing={
                    TokenDirection.ENCODE: 0.000001,
                    TokenDirection.DECODE: 0.000005,
                },
            ),
            'perplexity/sonar': ModelConfig(
                client_class=OpenRouterAI,
                token_counter=TokenCounterFactory().create_openrouter_counter(),
                test_connection=TestConnections().openrouter,
                pricing={
                    TokenDirection.ENCODE: 0.000001,
                    TokenDirection.DECODE: 0.000001,
                },
            ),
            'liquid/lfm-7b': ModelConfig(
                client_class=OpenRouterAI,
                token_counter=TokenCounterFactory().create_openrouter_counter(),
                test_connection=TestConnections().openrouter,
                pricing={
                    TokenDirection.ENCODE: 0.00000001,
                    TokenDirection.DECODE: 0.00000001,
                },
            ),
            'liquid/lfm-3b': ModelConfig(
                client_class=OpenRouterAI,
                token_counter=TokenCounterFactory().create_openrouter_counter(),
                test_connection=TestConnections().openrouter,
                pricing={
                    TokenDirection.ENCODE: 0.00000002,
                    TokenDirection.DECODE: 0.00000002,
                },
            ),
            'deepseek/deepseek-r1-distill-llama-70b:free': ModelConfig(
                client_class=OpenRouterAI,
                token_counter=TokenCounterFactory().create_openrouter_counter(),
                test_connection=TestConnections().openrouter,
                pricing={
                    TokenDirection.ENCODE: 0,
                    TokenDirection.DECODE: 0,
                },
            ),
            'deepseek/deepseek-r1-distill-llama-70b': ModelConfig(
                client_class=OpenRouterAI,
                token_counter=TokenCounterFactory().create_openrouter_counter(),
                test_connection=TestConnections().openrouter,
                pricing={
                    TokenDirection.ENCODE: 0.0000000259154,
                    TokenDirection.DECODE: 0.000000103712,
                },
            ),
            'deepseek/deepseek-r1:free': ModelConfig(
                client_class=OpenRouterAI,
                token_counter=TokenCounterFactory().create_openrouter_counter(),
                test_connection=TestConnections().openrouter,
                pricing={
                    TokenDirection.ENCODE: 0,
                    TokenDirection.DECODE: 0,
                },
            ),
            'deepseek/deepseek-r1': ModelConfig(
                client_class=OpenRouterAI,
                token_counter=TokenCounterFactory().create_openrouter_counter(),
                test_connection=TestConnections().openrouter,
                pricing={
                    TokenDirection.ENCODE: 0.0000004,
                    TokenDirection.DECODE: 0.000002,
                },
            ),
            'minimax/minimax-01': ModelConfig(
                client_class=OpenRouterAI,
                token_counter=TokenCounterFactory().create_openrouter_counter(),
                test_connection=TestConnections().openrouter,
                pricing={
                    TokenDirection.ENCODE: 0.0000002,
                    TokenDirection.DECODE: 0.0000011,
                },
            ),
            'mistralai/codestral-2501': ModelConfig(
                client_class=OpenRouterAI,
                token_counter=TokenCounterFactory().create_openrouter_counter(),
                test_connection=TestConnections().openrouter,
                pricing={
                    TokenDirection.ENCODE: 0.0000003,
                    TokenDirection.DECODE: 0.0000009,
                },
            ),
            'microsoft/phi-4': ModelConfig(
                client_class=OpenRouterAI,
                token_counter=TokenCounterFactory().create_openrouter_counter(),
                test_connection=TestConnections().openrouter,
                pricing={
                    TokenDirection.ENCODE: 0.00000006,
                    TokenDirection.DECODE: 0.00000014,
                },
            ),
            'deepseek/deepseek-chat': ModelConfig(
                client_class=OpenRouterAI,
                token_counter=TokenCounterFactory().create_openrouter_counter(),
                test_connection=TestConnections().openrouter,
                pricing={
                    TokenDirection.ENCODE: 0.0000001999188,
                    TokenDirection.DECODE: 0.000000800064,
                },
            ),
            'sao10k/l3.3-euryale-70b': ModelConfig(
                client_class=OpenRouterAI,
                token_counter=TokenCounterFactory().create_openrouter_counter(),
                test_connection=TestConnections().openrouter,
                pricing={
                    TokenDirection.ENCODE: 0.00000065,
                    TokenDirection.DECODE: 0.00000075,
                },
            ),
            'openai/o1': ModelConfig(
                client_class=OpenRouterAI,
                token_counter=TokenCounterFactory().create_openrouter_counter(),
                test_connection=TestConnections().openrouter,
                pricing={
                    TokenDirection.ENCODE: 0.000015,
                    TokenDirection.DECODE: 0.00006,
                },
            ),
            'x-ai/grok-2-vision-1212': ModelConfig(
                client_class=OpenRouterAI,
                token_counter=TokenCounterFactory().create_openrouter_counter(),
                test_connection=TestConnections().openrouter,
                pricing={
                    TokenDirection.ENCODE: 0.000002,
                    TokenDirection.DECODE: 0.00001,
                },
            ),
            'x-ai/grok-2-1212': ModelConfig(
                client_class=OpenRouterAI,
                token_counter=TokenCounterFactory().create_openrouter_counter(),
                test_connection=TestConnections().openrouter,
                pricing={
                    TokenDirection.ENCODE: 0.000002,
                    TokenDirection.DECODE: 0.00001,
                },
            ),
            'cohere/command-r7b-12-2024': ModelConfig(
                client_class=OpenRouterAI,
                token_counter=TokenCounterFactory().create_openrouter_counter(),
                test_connection=TestConnections().openrouter,
                pricing={
                    TokenDirection.ENCODE: 0.0000000375,
                    TokenDirection.DECODE: 0.00000015,
                },
            ),
            'google/gemini-2.0-flash-exp:free': ModelConfig(
                client_class=OpenRouterAI,
                token_counter=TokenCounterFactory().create_openrouter_counter(),
                test_connection=TestConnections().openrouter,
                pricing={
                    TokenDirection.ENCODE: 0,
                    TokenDirection.DECODE: 0,
                },
            ),
            'meta-llama/llama-3.3-70b-instruct:free': ModelConfig(
                client_class=OpenRouterAI,
                token_counter=TokenCounterFactory().create_openrouter_counter(),
                test_connection=TestConnections().openrouter,
                pricing={
                    TokenDirection.ENCODE: 0,
                    TokenDirection.DECODE: 0,
                },
            ),
            'meta-llama/llama-3.3-70b-instruct': ModelConfig(
                client_class=OpenRouterAI,
                token_counter=TokenCounterFactory().create_openrouter_counter(),
                test_connection=TestConnections().openrouter,
                pricing={
                    TokenDirection.ENCODE: 0.000000038,
                    TokenDirection.DECODE: 0.00000012,
                },
            ),
            'amazon/nova-lite-v1': ModelConfig(
                client_class=OpenRouterAI,
                token_counter=TokenCounterFactory().create_openrouter_counter(),
                test_connection=TestConnections().openrouter,
                pricing={
                    TokenDirection.ENCODE: 0.00000006,
                    TokenDirection.DECODE: 0.00000024,
                },
            ),
            'amazon/nova-micro-v1': ModelConfig(
                client_class=OpenRouterAI,
                token_counter=TokenCounterFactory().create_openrouter_counter(),
                test_connection=TestConnections().openrouter,
                pricing={
                    TokenDirection.ENCODE: 0.000000035,
                    TokenDirection.DECODE: 0.00000014,
                },
            ),
            'amazon/nova-pro-v1': ModelConfig(
                client_class=OpenRouterAI,
                token_counter=TokenCounterFactory().create_openrouter_counter(),
                test_connection=TestConnections().openrouter,
                pricing={
                    TokenDirection.ENCODE: 0.0000008,
                    TokenDirection.DECODE: 0.0000032,
                },
            ),
            'qwen/qwq-32b-preview': ModelConfig(
                client_class=OpenRouterAI,
                token_counter=TokenCounterFactory().create_openrouter_counter(),
                test_connection=TestConnections().openrouter,
                pricing={
                    TokenDirection.ENCODE: 0.0000002,
                    TokenDirection.DECODE: 0.0000002,
                },
            ),
            'openai/gpt-4o-2024-11-20': ModelConfig(
                client_class=OpenRouterAI,
                token_counter=TokenCounterFactory().create_openrouter_counter(),
                test_connection=TestConnections().openrouter,
                pricing={
                    TokenDirection.ENCODE: 0.0000025,
                    TokenDirection.DECODE: 0.00001,
                },
            ),
            'mistralai/mistral-large-2411': ModelConfig(
                client_class=OpenRouterAI,
                token_counter=TokenCounterFactory().create_openrouter_counter(),
                test_connection=TestConnections().openrouter,
                pricing={
                    TokenDirection.ENCODE: 0.000002,
                    TokenDirection.DECODE: 0.000006,
                },
            ),
            'mistralai/mistral-large-2407': ModelConfig(
                client_class=OpenRouterAI,
                token_counter=TokenCounterFactory().create_openrouter_counter(),
                test_connection=TestConnections().openrouter,
                pricing={
                    TokenDirection.ENCODE: 0.000002,
                    TokenDirection.DECODE: 0.000006,
                },
            ),
            'mistralai/pixtral-large-2411': ModelConfig(
                client_class=OpenRouterAI,
                token_counter=TokenCounterFactory().create_openrouter_counter(),
                test_connection=TestConnections().openrouter,
                pricing={
                    TokenDirection.ENCODE: 0.000002,
                    TokenDirection.DECODE: 0.000006,
                },
            ),
            'x-ai/grok-vision-beta': ModelConfig(
                client_class=OpenRouterAI,
                token_counter=TokenCounterFactory().create_openrouter_counter(),
                test_connection=TestConnections().openrouter,
                pricing={
                    TokenDirection.ENCODE: 0.000005,
                    TokenDirection.DECODE: 0.000015,
                },
            ),
            'infermatic/mn-inferor-12b': ModelConfig(
                client_class=OpenRouterAI,
                token_counter=TokenCounterFactory().create_openrouter_counter(),
                test_connection=TestConnections().openrouter,
                pricing={
                    TokenDirection.ENCODE: 0.0000006,
                    TokenDirection.DECODE: 0.000001,
                },
            ),
            'qwen/qwen-2.5-coder-32b-instruct:free': ModelConfig(
                client_class=OpenRouterAI,
                token_counter=TokenCounterFactory().create_openrouter_counter(),
                test_connection=TestConnections().openrouter,
                pricing={
                    TokenDirection.ENCODE: 0,
                    TokenDirection.DECODE: 0,
                },
            ),
            'qwen/qwen-2.5-coder-32b-instruct': ModelConfig(
                client_class=OpenRouterAI,
                token_counter=TokenCounterFactory().create_openrouter_counter(),
                test_connection=TestConnections().openrouter,
                pricing={
                    TokenDirection.ENCODE: 0.0000000499797,
                    TokenDirection.DECODE: 0.000000200016,
                },
            ),
            'raifle/sorcererlm-8x22b': ModelConfig(
                client_class=OpenRouterAI,
                token_counter=TokenCounterFactory().create_openrouter_counter(),
                test_connection=TestConnections().openrouter,
                pricing={
                    TokenDirection.ENCODE: 0.0000045,
                    TokenDirection.DECODE: 0.0000045,
                },
            ),
            'thedrummer/unslopnemo-12b': ModelConfig(
                client_class=OpenRouterAI,
                token_counter=TokenCounterFactory().create_openrouter_counter(),
                test_connection=TestConnections().openrouter,
                pricing={
                    TokenDirection.ENCODE: 0.0000004,
                    TokenDirection.DECODE: 0.0000004,
                },
            ),
            'anthropic/claude-3.5-haiku': ModelConfig(
                client_class=OpenRouterAI,
                token_counter=TokenCounterFactory().create_openrouter_counter(),
                test_connection=TestConnections().openrouter,
                pricing={
                    TokenDirection.ENCODE: 0.0000008,
                    TokenDirection.DECODE: 0.000004,
                },
            ),
            'anthropic/claude-3.5-haiku-20241022': ModelConfig(
                client_class=OpenRouterAI,
                token_counter=TokenCounterFactory().create_openrouter_counter(),
                test_connection=TestConnections().openrouter,
                pricing={
                    TokenDirection.ENCODE: 0.0000008,
                    TokenDirection.DECODE: 0.000004,
                },
            ),
            'anthracite-org/magnum-v4-72b': ModelConfig(
                client_class=OpenRouterAI,
                token_counter=TokenCounterFactory().create_openrouter_counter(),
                test_connection=TestConnections().openrouter,
                pricing={
                    TokenDirection.ENCODE: 0.000002,
                    TokenDirection.DECODE: 0.000005,
                },
            ),
            'anthropic/claude-3.5-sonnet': ModelConfig(
                client_class=OpenRouterAI,
                token_counter=TokenCounterFactory().create_openrouter_counter(),
                test_connection=TestConnections().openrouter,
                pricing={
                    TokenDirection.ENCODE: 0.000003,
                    TokenDirection.DECODE: 0.000015,
                },
            ),
            'mistralai/ministral-8b': ModelConfig(
                client_class=OpenRouterAI,
                token_counter=TokenCounterFactory().create_openrouter_counter(),
                test_connection=TestConnections().openrouter,
                pricing={
                    TokenDirection.ENCODE: 0.0000001,
                    TokenDirection.DECODE: 0.0000001,
                },
            ),
            'mistralai/ministral-3b': ModelConfig(
                client_class=OpenRouterAI,
                token_counter=TokenCounterFactory().create_openrouter_counter(),
                test_connection=TestConnections().openrouter,
                pricing={
                    TokenDirection.ENCODE: 0.00000004,
                    TokenDirection.DECODE: 0.00000004,
                },
            ),
            'qwen/qwen-2.5-7b-instruct': ModelConfig(
                client_class=OpenRouterAI,
                token_counter=TokenCounterFactory().create_openrouter_counter(),
                test_connection=TestConnections().openrouter,
                pricing={
                    TokenDirection.ENCODE: 0.00000004,
                    TokenDirection.DECODE: 0.0000001,
                },
            ),
            'nvidia/llama-3.1-nemotron-70b-instruct': ModelConfig(
                client_class=OpenRouterAI,
                token_counter=TokenCounterFactory().create_openrouter_counter(),
                test_connection=TestConnections().openrouter,
                pricing={
                    TokenDirection.ENCODE: 0.00000012,
                    TokenDirection.DECODE: 0.0000003,
                },
            ),
            'inflection/inflection-3-pi': ModelConfig(
                client_class=OpenRouterAI,
                token_counter=TokenCounterFactory().create_openrouter_counter(),
                test_connection=TestConnections().openrouter,
                pricing={
                    TokenDirection.ENCODE: 0.0000025,
                    TokenDirection.DECODE: 0.00001,
                },
            ),
            'inflection/inflection-3-productivity': ModelConfig(
                client_class=OpenRouterAI,
                token_counter=TokenCounterFactory().create_openrouter_counter(),
                test_connection=TestConnections().openrouter,
                pricing={
                    TokenDirection.ENCODE: 0.0000025,
                    TokenDirection.DECODE: 0.00001,
                },
            ),
            'google/gemini-flash-1.5-8b': ModelConfig(
                client_class=OpenRouterAI,
                token_counter=TokenCounterFactory().create_openrouter_counter(),
                test_connection=TestConnections().openrouter,
                pricing={
                    TokenDirection.ENCODE: 0.0000000375,
                    TokenDirection.DECODE: 0.00000015,
                },
            ),
            'anthracite-org/magnum-v2-72b': ModelConfig(
                client_class=OpenRouterAI,
                token_counter=TokenCounterFactory().create_openrouter_counter(),
                test_connection=TestConnections().openrouter,
                pricing={
                    TokenDirection.ENCODE: 0.000003,
                    TokenDirection.DECODE: 0.000003,
                },
            ),
            'thedrummer/rocinante-12b': ModelConfig(
                client_class=OpenRouterAI,
                token_counter=TokenCounterFactory().create_openrouter_counter(),
                test_connection=TestConnections().openrouter,
                pricing={
                    TokenDirection.ENCODE: 0.00000017,
                    TokenDirection.DECODE: 0.00000043,
                },
            ),
            'meta-llama/llama-3.2-1b-instruct': ModelConfig(
                client_class=OpenRouterAI,
                token_counter=TokenCounterFactory().create_openrouter_counter(),
                test_connection=TestConnections().openrouter,
                pricing={
                    TokenDirection.ENCODE: 0.000000005,
                    TokenDirection.DECODE: 0.00000001,
                },
            ),
            'meta-llama/llama-3.2-11b-vision-instruct:free': ModelConfig(
                client_class=OpenRouterAI,
                token_counter=TokenCounterFactory().create_openrouter_counter(),
                test_connection=TestConnections().openrouter,
                pricing={
                    TokenDirection.ENCODE: 0,
                    TokenDirection.DECODE: 0,
                },
            ),
            'meta-llama/llama-3.2-11b-vision-instruct': ModelConfig(
                client_class=OpenRouterAI,
                token_counter=TokenCounterFactory().create_openrouter_counter(),
                test_connection=TestConnections().openrouter,
                pricing={
                    TokenDirection.ENCODE: 0.000000049,
                    TokenDirection.DECODE: 0.000000049,
                },
            ),
            'meta-llama/llama-3.2-3b-instruct:free': ModelConfig(
                client_class=OpenRouterAI,
                token_counter=TokenCounterFactory().create_openrouter_counter(),
                test_connection=TestConnections().openrouter,
                pricing={
                    TokenDirection.ENCODE: 0,
                    TokenDirection.DECODE: 0,
                },
            ),
            'meta-llama/llama-3.2-3b-instruct': ModelConfig(
                client_class=OpenRouterAI,
                token_counter=TokenCounterFactory().create_openrouter_counter(),
                test_connection=TestConnections().openrouter,
                pricing={
                    TokenDirection.ENCODE: 0.000000003,
                    TokenDirection.DECODE: 0.000000006,
                },
            ),
            'meta-llama/llama-3.2-90b-vision-instruct': ModelConfig(
                client_class=OpenRouterAI,
                token_counter=TokenCounterFactory().create_openrouter_counter(),
                test_connection=TestConnections().openrouter,
                pricing={
                    TokenDirection.ENCODE: 0.0000012,
                    TokenDirection.DECODE: 0.0000012,
                },
            ),
            'qwen/qwen-2.5-72b-instruct:free': ModelConfig(
                client_class=OpenRouterAI,
                token_counter=TokenCounterFactory().create_openrouter_counter(),
                test_connection=TestConnections().openrouter,
                pricing={
                    TokenDirection.ENCODE: 0,
                    TokenDirection.DECODE: 0,
                },
            ),
            'qwen/qwen-2.5-72b-instruct': ModelConfig(
                client_class=OpenRouterAI,
                token_counter=TokenCounterFactory().create_openrouter_counter(),
                test_connection=TestConnections().openrouter,
                pricing={
                    TokenDirection.ENCODE: 0.0000000518308,
                    TokenDirection.DECODE: 0.000000207424,
                },
            ),
            'neversleep/llama-3.1-lumimaid-8b': ModelConfig(
                client_class=OpenRouterAI,
                token_counter=TokenCounterFactory().create_openrouter_counter(),
                test_connection=TestConnections().openrouter,
                pricing={
                    TokenDirection.ENCODE: 0.00000009,
                    TokenDirection.DECODE: 0.0000006,
                },
            ),
            'openai/o1-mini': ModelConfig(
                client_class=OpenRouterAI,
                token_counter=TokenCounterFactory().create_openrouter_counter(),
                test_connection=TestConnections().openrouter,
                pricing={
                    TokenDirection.ENCODE: 0.0000011,
                    TokenDirection.DECODE: 0.0000044,
                },
            ),
            'openai/o1-mini-2024-09-12': ModelConfig(
                client_class=OpenRouterAI,
                token_counter=TokenCounterFactory().create_openrouter_counter(),
                test_connection=TestConnections().openrouter,
                pricing={
                    TokenDirection.ENCODE: 0.0000011,
                    TokenDirection.DECODE: 0.0000044,
                },
            ),
            'mistralai/pixtral-12b': ModelConfig(
                client_class=OpenRouterAI,
                token_counter=TokenCounterFactory().create_openrouter_counter(),
                test_connection=TestConnections().openrouter,
                pricing={
                    TokenDirection.ENCODE: 0.0000001,
                    TokenDirection.DECODE: 0.0000001,
                },
            ),
            'cohere/command-r-08-2024': ModelConfig(
                client_class=OpenRouterAI,
                token_counter=TokenCounterFactory().create_openrouter_counter(),
                test_connection=TestConnections().openrouter,
                pricing={
                    TokenDirection.ENCODE: 0.00000015,
                    TokenDirection.DECODE: 0.0000006,
                },
            ),
            'cohere/command-r-plus-08-2024': ModelConfig(
                client_class=OpenRouterAI,
                token_counter=TokenCounterFactory().create_openrouter_counter(),
                test_connection=TestConnections().openrouter,
                pricing={
                    TokenDirection.ENCODE: 0.0000025,
                    TokenDirection.DECODE: 0.00001,
                },
            ),
            'sao10k/l3.1-euryale-70b': ModelConfig(
                client_class=OpenRouterAI,
                token_counter=TokenCounterFactory().create_openrouter_counter(),
                test_connection=TestConnections().openrouter,
                pricing={
                    TokenDirection.ENCODE: 0.00000065,
                    TokenDirection.DECODE: 0.00000075,
                },
            ),
            'qwen/qwen-2.5-vl-7b-instruct': ModelConfig(
                client_class=OpenRouterAI,
                token_counter=TokenCounterFactory().create_openrouter_counter(),
                test_connection=TestConnections().openrouter,
                pricing={
                    TokenDirection.ENCODE: 0.0000002,
                    TokenDirection.DECODE: 0.0000002,
                },
            ),
            'microsoft/phi-3.5-mini-128k-instruct': ModelConfig(
                client_class=OpenRouterAI,
                token_counter=TokenCounterFactory().create_openrouter_counter(),
                test_connection=TestConnections().openrouter,
                pricing={
                    TokenDirection.ENCODE: 0.0000001,
                    TokenDirection.DECODE: 0.0000001,
                },
            ),
            'nousresearch/hermes-3-llama-3.1-70b': ModelConfig(
                client_class=OpenRouterAI,
                token_counter=TokenCounterFactory().create_openrouter_counter(),
                test_connection=TestConnections().openrouter,
                pricing={
                    TokenDirection.ENCODE: 0.0000001,
                    TokenDirection.DECODE: 0.00000028,
                },
            ),
            'nousresearch/hermes-3-llama-3.1-405b': ModelConfig(
                client_class=OpenRouterAI,
                token_counter=TokenCounterFactory().create_openrouter_counter(),
                test_connection=TestConnections().openrouter,
                pricing={
                    TokenDirection.ENCODE: 0.0000007,
                    TokenDirection.DECODE: 0.0000008,
                },
            ),
            'openai/chatgpt-4o-latest': ModelConfig(
                client_class=OpenRouterAI,
                token_counter=TokenCounterFactory().create_openrouter_counter(),
                test_connection=TestConnections().openrouter,
                pricing={
                    TokenDirection.ENCODE: 0.000005,
                    TokenDirection.DECODE: 0.000015,
                },
            ),
            'sao10k/l3-lunaris-8b': ModelConfig(
                client_class=OpenRouterAI,
                token_counter=TokenCounterFactory().create_openrouter_counter(),
                test_connection=TestConnections().openrouter,
                pricing={
                    TokenDirection.ENCODE: 0.00000002,
                    TokenDirection.DECODE: 0.00000005,
                },
            ),
            'openai/gpt-4o-2024-08-06': ModelConfig(
                client_class=OpenRouterAI,
                token_counter=TokenCounterFactory().create_openrouter_counter(),
                test_connection=TestConnections().openrouter,
                pricing={
                    TokenDirection.ENCODE: 0.0000025,
                    TokenDirection.DECODE: 0.00001,
                },
            ),
            'meta-llama/llama-3.1-405b': ModelConfig(
                client_class=OpenRouterAI,
                token_counter=TokenCounterFactory().create_openrouter_counter(),
                test_connection=TestConnections().openrouter,
                pricing={
                    TokenDirection.ENCODE: 0.000002,
                    TokenDirection.DECODE: 0.000002,
                },
            ),
            'meta-llama/llama-3.1-8b-instruct': ModelConfig(
                client_class=OpenRouterAI,
                token_counter=TokenCounterFactory().create_openrouter_counter(),
                test_connection=TestConnections().openrouter,
                pricing={
                    TokenDirection.ENCODE: 0.000000015,
                    TokenDirection.DECODE: 0.00000002,
                },
            ),
            'meta-llama/llama-3.1-70b-instruct': ModelConfig(
                client_class=OpenRouterAI,
                token_counter=TokenCounterFactory().create_openrouter_counter(),
                test_connection=TestConnections().openrouter,
                pricing={
                    TokenDirection.ENCODE: 0.0000001,
                    TokenDirection.DECODE: 0.00000028,
                },
            ),
            'meta-llama/llama-3.1-405b-instruct:free': ModelConfig(
                client_class=OpenRouterAI,
                token_counter=TokenCounterFactory().create_openrouter_counter(),
                test_connection=TestConnections().openrouter,
                pricing={
                    TokenDirection.ENCODE: 0,
                    TokenDirection.DECODE: 0,
                },
            ),
            'meta-llama/llama-3.1-405b-instruct': ModelConfig(
                client_class=OpenRouterAI,
                token_counter=TokenCounterFactory().create_openrouter_counter(),
                test_connection=TestConnections().openrouter,
                pricing={
                    TokenDirection.ENCODE: 0.0000008,
                    TokenDirection.DECODE: 0.0000008,
                },
            ),
            'mistralai/mistral-nemo:free': ModelConfig(
                client_class=OpenRouterAI,
                token_counter=TokenCounterFactory().create_openrouter_counter(),
                test_connection=TestConnections().openrouter,
                pricing={
                    TokenDirection.ENCODE: 0,
                    TokenDirection.DECODE: 0,
                },
            ),
            'mistralai/mistral-nemo': ModelConfig(
                client_class=OpenRouterAI,
                token_counter=TokenCounterFactory().create_openrouter_counter(),
                test_connection=TestConnections().openrouter,
                pricing={
                    TokenDirection.ENCODE: 0.0000000075,
                    TokenDirection.DECODE: 0.00000005,
                },
            ),
            'openai/gpt-4o-mini': ModelConfig(
                client_class=OpenRouterAI,
                token_counter=TokenCounterFactory().create_openrouter_counter(),
                test_connection=TestConnections().openrouter,
                pricing={
                    TokenDirection.ENCODE: 0.00000015,
                    TokenDirection.DECODE: 0.0000006,
                },
            ),
            'openai/gpt-4o-mini-2024-07-18': ModelConfig(
                client_class=OpenRouterAI,
                token_counter=TokenCounterFactory().create_openrouter_counter(),
                test_connection=TestConnections().openrouter,
                pricing={
                    TokenDirection.ENCODE: 0.00000015,
                    TokenDirection.DECODE: 0.0000006,
                },
            ),
            'google/gemma-2-27b-it': ModelConfig(
                client_class=OpenRouterAI,
                token_counter=TokenCounterFactory().create_openrouter_counter(),
                test_connection=TestConnections().openrouter,
                pricing={
                    TokenDirection.ENCODE: 0.00000065,
                    TokenDirection.DECODE: 0.00000065,
                },
            ),
            'google/gemma-2-9b-it:free': ModelConfig(
                client_class=OpenRouterAI,
                token_counter=TokenCounterFactory().create_openrouter_counter(),
                test_connection=TestConnections().openrouter,
                pricing={
                    TokenDirection.ENCODE: 0,
                    TokenDirection.DECODE: 0,
                },
            ),
            'google/gemma-2-9b-it': ModelConfig(
                client_class=OpenRouterAI,
                token_counter=TokenCounterFactory().create_openrouter_counter(),
                test_connection=TestConnections().openrouter,
                pricing={
                    TokenDirection.ENCODE: 0.00000001,
                    TokenDirection.DECODE: 0.0000000100008,
                },
            ),
            'anthropic/claude-3.5-sonnet-20240620': ModelConfig(
                client_class=OpenRouterAI,
                token_counter=TokenCounterFactory().create_openrouter_counter(),
                test_connection=TestConnections().openrouter,
                pricing={
                    TokenDirection.ENCODE: 0.000003,
                    TokenDirection.DECODE: 0.000015,
                },
            ),
            'sao10k/l3-euryale-70b': ModelConfig(
                client_class=OpenRouterAI,
                token_counter=TokenCounterFactory().create_openrouter_counter(),
                test_connection=TestConnections().openrouter,
                pricing={
                    TokenDirection.ENCODE: 0.00000148,
                    TokenDirection.DECODE: 0.00000148,
                },
            ),
            'cognitivecomputations/dolphin-mixtral-8x22b': ModelConfig(
                client_class=OpenRouterAI,
                token_counter=TokenCounterFactory().create_openrouter_counter(),
                test_connection=TestConnections().openrouter,
                pricing={
                    TokenDirection.ENCODE: 0.0000009,
                    TokenDirection.DECODE: 0.0000009,
                },
            ),
            'qwen/qwen-2-72b-instruct': ModelConfig(
                client_class=OpenRouterAI,
                token_counter=TokenCounterFactory().create_openrouter_counter(),
                test_connection=TestConnections().openrouter,
                pricing={
                    TokenDirection.ENCODE: 0.0000009,
                    TokenDirection.DECODE: 0.0000009,
                },
            ),
            'mistralai/mistral-7b-instruct-v0.3': ModelConfig(
                client_class=OpenRouterAI,
                token_counter=TokenCounterFactory().create_openrouter_counter(),
                test_connection=TestConnections().openrouter,
                pricing={
                    TokenDirection.ENCODE: 0.000000028,
                    TokenDirection.DECODE: 0.000000054,
                },
            ),
            'mistralai/mistral-7b-instruct:free': ModelConfig(
                client_class=OpenRouterAI,
                token_counter=TokenCounterFactory().create_openrouter_counter(),
                test_connection=TestConnections().openrouter,
                pricing={
                    TokenDirection.ENCODE: 0,
                    TokenDirection.DECODE: 0,
                },
            ),
            'mistralai/mistral-7b-instruct': ModelConfig(
                client_class=OpenRouterAI,
                token_counter=TokenCounterFactory().create_openrouter_counter(),
                test_connection=TestConnections().openrouter,
                pricing={
                    TokenDirection.ENCODE: 0.000000028,
                    TokenDirection.DECODE: 0.000000054,
                },
            ),
            'nousresearch/hermes-2-pro-llama-3-8b': ModelConfig(
                client_class=OpenRouterAI,
                token_counter=TokenCounterFactory().create_openrouter_counter(),
                test_connection=TestConnections().openrouter,
                pricing={
                    TokenDirection.ENCODE: 0.000000025,
                    TokenDirection.DECODE: 0.00000004,
                },
            ),
            'microsoft/phi-3-mini-128k-instruct': ModelConfig(
                client_class=OpenRouterAI,
                token_counter=TokenCounterFactory().create_openrouter_counter(),
                test_connection=TestConnections().openrouter,
                pricing={
                    TokenDirection.ENCODE: 0.0000001,
                    TokenDirection.DECODE: 0.0000001,
                },
            ),
            'microsoft/phi-3-medium-128k-instruct': ModelConfig(
                client_class=OpenRouterAI,
                token_counter=TokenCounterFactory().create_openrouter_counter(),
                test_connection=TestConnections().openrouter,
                pricing={
                    TokenDirection.ENCODE: 0.000001,
                    TokenDirection.DECODE: 0.000001,
                },
            ),
            'neversleep/llama-3-lumimaid-70b': ModelConfig(
                client_class=OpenRouterAI,
                token_counter=TokenCounterFactory().create_openrouter_counter(),
                test_connection=TestConnections().openrouter,
                pricing={
                    TokenDirection.ENCODE: 0.000004,
                    TokenDirection.DECODE: 0.000006,
                },
            ),
            'google/gemini-flash-1.5': ModelConfig(
                client_class=OpenRouterAI,
                token_counter=TokenCounterFactory().create_openrouter_counter(),
                test_connection=TestConnections().openrouter,
                pricing={
                    TokenDirection.ENCODE: 0.000000075,
                    TokenDirection.DECODE: 0.0000003,
                },
            ),
            'openai/gpt-4o': ModelConfig(
                client_class=OpenRouterAI,
                token_counter=TokenCounterFactory().create_openrouter_counter(),
                test_connection=TestConnections().openrouter,
                pricing={
                    TokenDirection.ENCODE: 0.0000025,
                    TokenDirection.DECODE: 0.00001,
                },
            ),
            'openai/gpt-4o:extended': ModelConfig(
                client_class=OpenRouterAI,
                token_counter=TokenCounterFactory().create_openrouter_counter(),
                test_connection=TestConnections().openrouter,
                pricing={
                    TokenDirection.ENCODE: 0.000006,
                    TokenDirection.DECODE: 0.000018,
                },
            ),
            'meta-llama/llama-guard-2-8b': ModelConfig(
                client_class=OpenRouterAI,
                token_counter=TokenCounterFactory().create_openrouter_counter(),
                test_connection=TestConnections().openrouter,
                pricing={
                    TokenDirection.ENCODE: 0.0000002,
                    TokenDirection.DECODE: 0.0000002,
                },
            ),
            'openai/gpt-4o-2024-05-13': ModelConfig(
                client_class=OpenRouterAI,
                token_counter=TokenCounterFactory().create_openrouter_counter(),
                test_connection=TestConnections().openrouter,
                pricing={
                    TokenDirection.ENCODE: 0.000005,
                    TokenDirection.DECODE: 0.000015,
                },
            ),
            'meta-llama/llama-3-70b-instruct': ModelConfig(
                client_class=OpenRouterAI,
                token_counter=TokenCounterFactory().create_openrouter_counter(),
                test_connection=TestConnections().openrouter,
                pricing={
                    TokenDirection.ENCODE: 0.0000003,
                    TokenDirection.DECODE: 0.0000004,
                },
            ),
            'meta-llama/llama-3-8b-instruct': ModelConfig(
                client_class=OpenRouterAI,
                token_counter=TokenCounterFactory().create_openrouter_counter(),
                test_connection=TestConnections().openrouter,
                pricing={
                    TokenDirection.ENCODE: 0.00000003,
                    TokenDirection.DECODE: 0.00000006,
                },
            ),
            'mistralai/mixtral-8x22b-instruct': ModelConfig(
                client_class=OpenRouterAI,
                token_counter=TokenCounterFactory().create_openrouter_counter(),
                test_connection=TestConnections().openrouter,
                pricing={
                    TokenDirection.ENCODE: 0.0000009,
                    TokenDirection.DECODE: 0.0000009,
                },
            ),
            'microsoft/wizardlm-2-8x22b': ModelConfig(
                client_class=OpenRouterAI,
                token_counter=TokenCounterFactory().create_openrouter_counter(),
                test_connection=TestConnections().openrouter,
                pricing={
                    TokenDirection.ENCODE: 0.00000048,
                    TokenDirection.DECODE: 0.00000048,
                },
            ),
            'google/gemini-pro-1.5': ModelConfig(
                client_class=OpenRouterAI,
                token_counter=TokenCounterFactory().create_openrouter_counter(),
                test_connection=TestConnections().openrouter,
                pricing={
                    TokenDirection.ENCODE: 0.00000125,
                    TokenDirection.DECODE: 0.000005,
                },
            ),
            'openai/gpt-4-turbo': ModelConfig(
                client_class=OpenRouterAI,
                token_counter=TokenCounterFactory().create_openrouter_counter(),
                test_connection=TestConnections().openrouter,
                pricing={
                    TokenDirection.ENCODE: 0.00001,
                    TokenDirection.DECODE: 0.00003,
                },
            ),
            'cohere/command-r-plus': ModelConfig(
                client_class=OpenRouterAI,
                token_counter=TokenCounterFactory().create_openrouter_counter(),
                test_connection=TestConnections().openrouter,
                pricing={
                    TokenDirection.ENCODE: 0.000003,
                    TokenDirection.DECODE: 0.000015,
                },
            ),
            'cohere/command-r-plus-04-2024': ModelConfig(
                client_class=OpenRouterAI,
                token_counter=TokenCounterFactory().create_openrouter_counter(),
                test_connection=TestConnections().openrouter,
                pricing={
                    TokenDirection.ENCODE: 0.000003,
                    TokenDirection.DECODE: 0.000015,
                },
            ),
            'sophosympatheia/midnight-rose-70b': ModelConfig(
                client_class=OpenRouterAI,
                token_counter=TokenCounterFactory().create_openrouter_counter(),
                test_connection=TestConnections().openrouter,
                pricing={
                    TokenDirection.ENCODE: 0.0000008,
                    TokenDirection.DECODE: 0.0000008,
                },
            ),
            'cohere/command': ModelConfig(
                client_class=OpenRouterAI,
                token_counter=TokenCounterFactory().create_openrouter_counter(),
                test_connection=TestConnections().openrouter,
                pricing={
                    TokenDirection.ENCODE: 0.000001,
                    TokenDirection.DECODE: 0.000002,
                },
            ),
            'cohere/command-r': ModelConfig(
                client_class=OpenRouterAI,
                token_counter=TokenCounterFactory().create_openrouter_counter(),
                test_connection=TestConnections().openrouter,
                pricing={
                    TokenDirection.ENCODE: 0.0000005,
                    TokenDirection.DECODE: 0.0000015,
                },
            ),
            'anthropic/claude-3-haiku': ModelConfig(
                client_class=OpenRouterAI,
                token_counter=TokenCounterFactory().create_openrouter_counter(),
                test_connection=TestConnections().openrouter,
                pricing={
                    TokenDirection.ENCODE: 0.00000025,
                    TokenDirection.DECODE: 0.00000125,
                },
            ),
            'anthropic/claude-3-opus': ModelConfig(
                client_class=OpenRouterAI,
                token_counter=TokenCounterFactory().create_openrouter_counter(),
                test_connection=TestConnections().openrouter,
                pricing={
                    TokenDirection.ENCODE: 0.000015,
                    TokenDirection.DECODE: 0.000075,
                },
            ),
            'cohere/command-r-03-2024': ModelConfig(
                client_class=OpenRouterAI,
                token_counter=TokenCounterFactory().create_openrouter_counter(),
                test_connection=TestConnections().openrouter,
                pricing={
                    TokenDirection.ENCODE: 0.0000005,
                    TokenDirection.DECODE: 0.0000015,
                },
            ),
            'mistralai/mistral-large': ModelConfig(
                client_class=OpenRouterAI,
                token_counter=TokenCounterFactory().create_openrouter_counter(),
                test_connection=TestConnections().openrouter,
                pricing={
                    TokenDirection.ENCODE: 0.000002,
                    TokenDirection.DECODE: 0.000006,
                },
            ),
            'openai/gpt-4-turbo-preview': ModelConfig(
                client_class=OpenRouterAI,
                token_counter=TokenCounterFactory().create_openrouter_counter(),
                test_connection=TestConnections().openrouter,
                pricing={
                    TokenDirection.ENCODE: 0.00001,
                    TokenDirection.DECODE: 0.00003,
                },
            ),
            'openai/gpt-3.5-turbo-0613': ModelConfig(
                client_class=OpenRouterAI,
                token_counter=TokenCounterFactory().create_openrouter_counter(),
                test_connection=TestConnections().openrouter,
                pricing={
                    TokenDirection.ENCODE: 0.000001,
                    TokenDirection.DECODE: 0.000002,
                },
            ),
            'nousresearch/nous-hermes-2-mixtral-8x7b-dpo': ModelConfig(
                client_class=OpenRouterAI,
                token_counter=TokenCounterFactory().create_openrouter_counter(),
                test_connection=TestConnections().openrouter,
                pricing={
                    TokenDirection.ENCODE: 0.0000006,
                    TokenDirection.DECODE: 0.0000006,
                },
            ),
            'mistralai/mistral-small': ModelConfig(
                client_class=OpenRouterAI,
                token_counter=TokenCounterFactory().create_openrouter_counter(),
                test_connection=TestConnections().openrouter,
                pricing={
                    TokenDirection.ENCODE: 0.0000002,
                    TokenDirection.DECODE: 0.0000006,
                },
            ),
            'mistralai/mistral-tiny': ModelConfig(
                client_class=OpenRouterAI,
                token_counter=TokenCounterFactory().create_openrouter_counter(),
                test_connection=TestConnections().openrouter,
                pricing={
                    TokenDirection.ENCODE: 0.00000025,
                    TokenDirection.DECODE: 0.00000025,
                },
            ),
            'mistralai/mixtral-8x7b-instruct': ModelConfig(
                client_class=OpenRouterAI,
                token_counter=TokenCounterFactory().create_openrouter_counter(),
                test_connection=TestConnections().openrouter,
                pricing={
                    TokenDirection.ENCODE: 0.00000008,
                    TokenDirection.DECODE: 0.00000024,
                },
            ),
            'neversleep/noromaid-20b': ModelConfig(
                client_class=OpenRouterAI,
                token_counter=TokenCounterFactory().create_openrouter_counter(),
                test_connection=TestConnections().openrouter,
                pricing={
                    TokenDirection.ENCODE: 0.000001,
                    TokenDirection.DECODE: 0.00000175,
                },
            ),
            'alpindale/goliath-120b': ModelConfig(
                client_class=OpenRouterAI,
                token_counter=TokenCounterFactory().create_openrouter_counter(),
                test_connection=TestConnections().openrouter,
                pricing={
                    TokenDirection.ENCODE: 0.000004,
                    TokenDirection.DECODE: 0.0000055,
                },
            ),
            'openrouter/auto': ModelConfig(
                client_class=OpenRouterAI,
                token_counter=TokenCounterFactory().create_openrouter_counter(),
                test_connection=TestConnections().openrouter,
                pricing={
                    TokenDirection.ENCODE: -1,
                    TokenDirection.DECODE: -1,
                },
            ),
            'openai/gpt-4-1106-preview': ModelConfig(
                client_class=OpenRouterAI,
                token_counter=TokenCounterFactory().create_openrouter_counter(),
                test_connection=TestConnections().openrouter,
                pricing={
                    TokenDirection.ENCODE: 0.00001,
                    TokenDirection.DECODE: 0.00003,
                },
            ),
            'openai/gpt-3.5-turbo-instruct': ModelConfig(
                client_class=OpenRouterAI,
                token_counter=TokenCounterFactory().create_openrouter_counter(),
                test_connection=TestConnections().openrouter,
                pricing={
                    TokenDirection.ENCODE: 0.0000015,
                    TokenDirection.DECODE: 0.000002,
                },
            ),
            'mistralai/mistral-7b-instruct-v0.1': ModelConfig(
                client_class=OpenRouterAI,
                token_counter=TokenCounterFactory().create_openrouter_counter(),
                test_connection=TestConnections().openrouter,
                pricing={
                    TokenDirection.ENCODE: 0.00000011,
                    TokenDirection.DECODE: 0.00000019,
                },
            ),
            'pygmalionai/mythalion-13b': ModelConfig(
                client_class=OpenRouterAI,
                token_counter=TokenCounterFactory().create_openrouter_counter(),
                test_connection=TestConnections().openrouter,
                pricing={
                    TokenDirection.ENCODE: 0.0000007,
                    TokenDirection.DECODE: 0.0000011,
                },
            ),
            'openai/gpt-3.5-turbo-16k': ModelConfig(
                client_class=OpenRouterAI,
                token_counter=TokenCounterFactory().create_openrouter_counter(),
                test_connection=TestConnections().openrouter,
                pricing={
                    TokenDirection.ENCODE: 0.000003,
                    TokenDirection.DECODE: 0.000004,
                },
            ),
            'mancer/weaver': ModelConfig(
                client_class=OpenRouterAI,
                token_counter=TokenCounterFactory().create_openrouter_counter(),
                test_connection=TestConnections().openrouter,
                pricing={
                    TokenDirection.ENCODE: 0.000001125,
                    TokenDirection.DECODE: 0.000001125,
                },
            ),
            'undi95/remm-slerp-l2-13b': ModelConfig(
                client_class=OpenRouterAI,
                token_counter=TokenCounterFactory().create_openrouter_counter(),
                test_connection=TestConnections().openrouter,
                pricing={
                    TokenDirection.ENCODE: 0.00000045,
                    TokenDirection.DECODE: 0.00000065,
                },
            ),
            'gryphe/mythomax-l2-13b': ModelConfig(
                client_class=OpenRouterAI,
                token_counter=TokenCounterFactory().create_openrouter_counter(),
                test_connection=TestConnections().openrouter,
                pricing={
                    TokenDirection.ENCODE: 0.00000006,
                    TokenDirection.DECODE: 0.00000006,
                },
            ),
            'openai/gpt-4': ModelConfig(
                client_class=OpenRouterAI,
                token_counter=TokenCounterFactory().create_openrouter_counter(),
                test_connection=TestConnections().openrouter,
                pricing={
                    TokenDirection.ENCODE: 0.00003,
                    TokenDirection.DECODE: 0.00006,
                },
            ),
            'openai/gpt-4-0314': ModelConfig(
                client_class=OpenRouterAI,
                token_counter=TokenCounterFactory().create_openrouter_counter(),
                test_connection=TestConnections().openrouter,
                pricing={
                    TokenDirection.ENCODE: 0.00003,
                    TokenDirection.DECODE: 0.00006,
                },
            ),
            'openai/gpt-3.5-turbo': ModelConfig(
                client_class=OpenRouterAI,
                token_counter=TokenCounterFactory().create_openrouter_counter(),
                test_connection=TestConnections().openrouter,
                pricing={
                    TokenDirection.ENCODE: 0.0000005,
                    TokenDirection.DECODE: 0.0000015,
                },
            ),
        }
