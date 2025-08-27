import os

from dotenv import load_dotenv

from llm.constructor import BaseLLM

load_dotenv()


CHAT_GPT__KEY = os.getenv('CHAT_GPT__KEY')
GIGACHAT__KEY = os.getenv('GIGACHAT__KEY')
ANTHROPIC__KEY = os.getenv('ANTHROPIC__KEY')
GEMINI__KEY = os.getenv('GEMINI__KEY')
XAI__KEY = os.getenv('XAI__KEY')
DEEPSEEK__KEY = os.getenv('DEEPSEEK__KEY')
CEREBAS__KEY = os.getenv('CEREBAS__KEY')
OPENROUTER__KEY = os.getenv('OPENROUTER__KEY')

gpt_5 = BaseLLM(
    model='gpt-5',
    api_key=CHAT_GPT__KEY,
)

gpt_5_mini = BaseLLM(
    model='gpt-5-mini',
    api_key=CHAT_GPT__KEY,
)

gpt_5_nano = BaseLLM(
    model='gpt-5-nano',
    api_key=CHAT_GPT__KEY,
)

gpt_5_chat_latest = BaseLLM(
    model='gpt-5-chat-latest',
    api_key=CHAT_GPT__KEY,
)

gpt_4_1 = BaseLLM(
    model='gpt-4.1',
    api_key=CHAT_GPT__KEY,
)

gpt_4_1_mini = BaseLLM(
    model='gpt-4.1-mini',
    api_key=CHAT_GPT__KEY,
)

gpt_4_1_nano = BaseLLM(
    model='gpt-4.1-nano',
    api_key=CHAT_GPT__KEY,
)

gpt_4_5_preview = BaseLLM(
    model='gpt-4.5-preview',
    api_key=CHAT_GPT__KEY,
)

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
    model='claude-opus-4-0',
    api_key=ANTHROPIC__KEY,
)

claude_sonnet_4 = BaseLLM(
    model='claude-sonnet-4-0',
    api_key=ANTHROPIC__KEY,
)

gemini_2_0_flash_001 = BaseLLM(
    model='gemini-2.0-flash-001',
    api_key=GEMINI__KEY,
)

gemini_2_5_flash = BaseLLM(
    model='gemini-2.5-flash',
    api_key=GEMINI__KEY,
)

gemini_2_5_pro = BaseLLM(
    model='gemini-2.5-pro-preview-06-05',
    api_key=GEMINI__KEY,
)

grok_3_mini = BaseLLM(
    model='grok-3-mini',
    api_key=XAI__KEY,
)

grok_3 = BaseLLM(
    model='grok-3',
    api_key=XAI__KEY,
)

grok_3_fast = BaseLLM(
    model='grok-3-fast',
    api_key=XAI__KEY,
)

deepseek_chat = BaseLLM(
    model='deepseek-chat',
    api_key=DEEPSEEK__KEY,
)

deepseek_reasoner = BaseLLM(
    model='deepseek-reasoner',
    api_key=DEEPSEEK__KEY,
)

gpt_oss_120b = BaseLLM(
    model='gpt-oss-120b',
    api_key=CEREBAS__KEY,
)

qwen_3_32b = BaseLLM(
    model='qwen-3-32b',
    api_key=CEREBAS__KEY,
)

llama_4_scout_17b_16e_instruct = BaseLLM(
    model='llama-4-scout-17b-16e-instruct',
    api_key=CEREBAS__KEY,
)

llama_4_maverick_17b_128e_instruct = BaseLLM(
    model='llama-4-maverick-17b-128e-instruct',
    api_key=CEREBAS__KEY,
)

openrouter = BaseLLM(
    model='openai/gpt-4o',
    api_key=OPENROUTER__KEY,
)
