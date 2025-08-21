import importlib.resources

from tokenizers import Tokenizer


class DeepSeekTokenizer:
    def __init__(self, tokenizer_path=None):
        if tokenizer_path is None:
            with importlib.resources.path(
                package='llm.tokenizer.deepseek',
                resource='tokenizer.json',
            ) as path:
                tokenizer_path = str(path)
        self.tokenizer = Tokenizer.from_file(tokenizer_path)

    def count_tokens(self, text):
        return len(self.tokenizer.encode(text).ids)

    def encode(self, text):
        result = self.tokenizer.encode(text)
        return {'ids': result.ids, 'tokens': result.tokens, 'count': len(result.ids)}


# Использование
# tokenizer = DeepSeekTokenizer()
# print(tokenizer.count_tokens("Hello!"))
# print(tokenizer.encode("Hello!"))
