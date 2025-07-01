# LLM Service

[![PyPI version](https://img.shields.io/pypi/v/universal-llm-service.svg)](https://pypi.org/project/universal-llm-service/)
[![Build Status](https://github.com/DenisShahbazyan/LLM_Service/actions/workflows/publish.yml/badge.svg)](https://github.com/DenisShahbazyan/LLM_Service/actions)
[![Python Versions](https://img.shields.io/pypi/pyversions/universal-llm-service.svg)](https://pypi.org/project/universal-llm-service/)
[![License](https://img.shields.io/pypi/l/universal-llm-service.svg)](https://github.com/DenisShahbazyan/LLM_Service/blob/master/LICENSE)

# Библиотека для упрощенного использования LLM

Библиотека предоставляет упрощенный интерфейс для работы с различными LLM моделями, автоматический подсчет токенов и стоимости запросов, а также удобную интеграцию с существующими проектами. Создана на основе Langchain, являясь практичной оберткой для стандартизации взаимодействия с моделями.

## Основные возможности:
- Простой и понятный интерфейс для работы с LLM моделями
- Автоматический подсчет стоимости запросов
- Поддержка асинхронного API
- Мониторинг использования токенов
- Расчет затрат в USD с учетом текущего курса валюты по ЦБ РФ

## Установка:
```sh
pip install universal-llm-service
```

## Использование:

### Перед началом нужно создать экземпляр модели, langchain - подобный:
Если в langchain мы создаем такой экземпляр:
```py
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    model='gpt-4o-mini',
    api_key='sk-proj-1234567890',
    temperature=0,
)
```

То здесь мы должны создавать такой:
```py
from llm.constructor import BaseLLM

llm = BaseLLM(
    model='gpt-4o-mini',
    api_key='sk-proj-1234567890',
    temperature=0,
)
```

И его передаем при создании `LLMService`. Имена полей должны совпадать.



### Обычный диалог с LLM:
```py
import asyncio
from llm import LLMService

from llm_config import gpt_4o_mini


async def main():
    llm = await LLMService.create(gpt_4o_mini.to_dict())
    result = await llm.ainvoke(message='Сколько будет 2 + 2?')


if __name__ == "__main__":
    asyncio.run(main())
```
- `result` - ответ модели (строка)

### Структурированный вывод:
```py
import asyncio

from pydantic import BaseModel, Field

from llm_config import gpt_4o_mini
from llm import LLMService


class RelatedConceptOutput(BaseModel):
    """Новый термин и его сила связи с исходным термином."""

    title: str = Field(..., description='Название термина')
    length: int = Field(..., description='Сила связи')


class RelatedConceptListOutput(BaseModel):
    """Новые термины и их сила связи с исходным термином."""

    concepts: list[RelatedConceptOutput]


SYSTEM_PROMPT = (
    'Тебе дано понятие школьной программы: "Молекула". Сгенерируй ровно "5" понятий '
    'школьной программы, наиболее близких к этому понятию.'
)


async def main() -> None:
    llm = await LLMService.create(gpt_4o_mini.to_dict())
    structured_llm = await llm.with_structured_output(RelatedConceptListOutput)
    result = await structured_llm.ainvoke(message=SYSTEM_PROMPT)


if __name__ == '__main__':
    asyncio.run(main())
```
- `result` - ответ модели (запрошенная Pydantic схема)

### Потоковый вывод:
```py
import asyncio


from example.llm_config import gpt_4o_mini  # noqa: F401
from llm.service import LLMService


async def main() -> None:
    llm = await LLMService.create(gpt_4o_mini.to_dict())
    result = llm.astream(message='Расскажи теорему Пифагора')
    async for chunk in result:
        print(chunk, end='', flush=True)


if __name__ == '__main__':
    asyncio.run(main())
```

### Потоковый вывод с использованием контекстного менеджера:
```py
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
```

## Подробнее о возможностях:
- `get_llm_config` - получает конфиг для LLM модели, можно увидеть в примерах, имена полей должны совпадать с именами полей при инициализации моделей для Langchain.

- `llm.ainvoke` - метод принимает как отдельно системный промпт или сообщение, так и историю полность. Параметры принимаются как в стиле langchain так и в виде словарей. Перед отправкой любого сообщения в LLM отрабатывает класс подготовки контекста - PrepareChat:
Класс для подготовки чата в формате Langchain для модели.

1. Данный класс всегда отдает список сообщений в формате Langchain.
2. Если был отправлен только системный промпт, то после него добавляется пустое
    сообщение пользователя. Иначе некоторые модели не хотят работать с единственным
    системным промптом.


- `llm.usage.all_input_tokens` - общее количество отправленных токенов с момента инициализации
- `llm.usage.all_output_tokens`  - общее количество полученных токенов с момента инициализации
- `llm.usage.last_input_tokens` - количество отправленных токенов за последний вызов
- `llm.usage.last_output_tokens` - количество полученных токенов за последний вызов

---

- `llm.usage.all_input_spendings` - общие расходы в USD при отправке с момента инициализации
- `llm.usage.all_output_spendings` - общие расходы в USD при получении с момента инициализации
- `llm.usage.last_input_spendings` - расходы в USD при отправке за последний вызов
- `llm.usage.last_output_spendings` - расходы в USD при получении за последний вызов

---

- `llm.counter.model_registry.usd_rate` - курс валюты в USD.

## Запуск локально (для разработки):
Установка зависимостей
```sh
pip install -r requirements.txt
```

Создать файл `.env` на основе шаблона `.env.template` и вписать ключи

Запуск примеров:
```sh
python -m example.simple  # Пример обычного общения с LLM
python -m example.structured  # Пример общения с LLM со структурированным выводом
```

## TODO:
- Добавить популярные модели для использования
- Добавить подсчет токенов для стримминговой передачи

# Список поддерживаемых моделей:
- `gpt-4.1`
- `gpt-4.1-mini`
- `gpt-4.1-nano`
- `gpt-4.5-preview`
- `gpt-4o-mini`
- `gpt-4o`
- `o3-2025-04-16`
- `o4-mini-2025-04-16`
- `GigaChat`
- `GigaChat-2`
- `GigaChat-Pro`
- `GigaChat-2-Pro`
- `GigaChat-Max`
- `GigaChat-2-Max`
- `claude-3-5-haiku-latest`
- `claude-3-7-sonnet-latest`
- `claude-opus-4-20250514`
- `gemini-2.0-flash-001`
- `gemini-2.5-pro-preview-06-05`
- `grok-3-mini`
