import os

from llm.constructor import BaseLLM

CHAT_GPT__KEY = os.getenv('CHAT_GPT__KEY')
GIGACHAT__KEY = os.getenv('GIGACHAT__KEY')
ANTHROPIC__KEY = os.getenv('ANTHROPIC__KEY')


gpt_4o_mini = BaseLLM(
    model='gpt-4o-mini',
    api_key=CHAT_GPT__KEY,
)

gpt_4o = BaseLLM(
    model='gpt-4o',
    api_key=CHAT_GPT__KEY,
)

o3_2025_04_16 = BaseLLM(
    model='o3-2025-04-16',
    api_key=CHAT_GPT__KEY,
)

o4_mini_2025_04_16 = BaseLLM(
    model='o4-mini-2025-04-16',
    api_key=CHAT_GPT__KEY,
)

giga_chat = BaseLLM(
    model='GigaChat',
    credentials=GIGACHAT__KEY,
    scope='GIGACHAT_API_CORP',
    verify_ssl_certs=False,
    profanity_check=False,
)

giga_chat_2 = BaseLLM(
    model='GigaChat-2',
    credentials=GIGACHAT__KEY,
    scope='GIGACHAT_API_CORP',
    verify_ssl_certs=False,
    profanity_check=False,
)

giga_chat_pro = BaseLLM(
    model='GigaChat-Pro',
    credentials=GIGACHAT__KEY,
    scope='GIGACHAT_API_CORP',
    verify_ssl_certs=False,
    profanity_check=False,
)

giga_chat_2_pro = BaseLLM(
    model='GigaChat-2-Pro',
    credentials=GIGACHAT__KEY,
    scope='GIGACHAT_API_CORP',
    verify_ssl_certs=False,
    profanity_check=False,
)

giga_chat_max = BaseLLM(
    model='GigaChat-Max',
    credentials=GIGACHAT__KEY,
    scope='GIGACHAT_API_CORP',
    verify_ssl_certs=False,
    profanity_check=False,
)

giga_chat_2_max = BaseLLM(
    model='GigaChat-2-Max',
    credentials=GIGACHAT__KEY,
    scope='GIGACHAT_API_CORP',
    verify_ssl_certs=False,
    profanity_check=False,
)

claude_3_5_haiku = BaseLLM(
    model='claude-3-5-haiku-latest',
    api_key=ANTHROPIC__KEY,
)

claude_3_7_sonnet = BaseLLM(
    model='claude-3-7-sonnet-latest',
    api_key=ANTHROPIC__KEY,
)

claude_opus_4 = BaseLLM(
    model='claude-opus-4-20250514',
    api_key=ANTHROPIC__KEY,
)
