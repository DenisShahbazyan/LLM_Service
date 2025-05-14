from langchain_gigachat import GigaChat
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic


class LLMFactory:
    @staticmethod
    def init_client(config: dict) -> ChatOpenAI | GigaChat:
        match config.get('model', None):
            case 'gpt-4o-mini' | 'gpt-4o':
                return ChatOpenAI(**config)
            case 'GigaChat-Pro' | 'GigaChat-2-Max':
                return GigaChat(**config)
            case 'claude-3-5-haiku-latest':
                return ChatAnthropic(**config)
            case _:
                raise ValueError('Unknown model')
