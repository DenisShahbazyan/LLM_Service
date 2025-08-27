import json
import os

import requests
from dotenv import load_dotenv

load_dotenv()

OPENROUTER__KEY = os.getenv('OPENROUTER__KEY')

response = requests.get(
    'https://openrouter.ai/api/v1/models',
    headers={'Authorization': f'Bearer {OPENROUTER__KEY}'},
)

data = json.loads(response.text)

code = ''
for i in data['data']:
    model_name = i['id']
    price_input = i['pricing']['prompt']
    price_output = i['pricing']['completion']

    code += (
        f'"{model_name}": ModelConfig(\n'
        f'    client_class=OpenRouterAI,\n'
        f'    token_counter=TokenCounterFactory().create_openrouter_counter(),\n'
        f'    test_connection=TestConnections().openrouter,\n'
        f'    pricing=\n'
        f'    {{\n'
        f'        TokenDirection.ENCODE: {price_input},\n'
        f'        TokenDirection.DECODE: {price_output},\n'
        f'    }},\n'
        f'),\n'
    )


with open('./result_openrouter.txt', 'w', encoding='utf-8') as file:
    file.write(code)
