from typing import List, Optional

import numpy as np
from transformers import AutoTokenizer


class Tokenizer:
    """Wrapper per tokenizer HuggingFace con caching."""

    _instance: Optional["Tokenizer"] = None
    _tokenizer = None

    def __init__(self, model_name: str = "deepseek-ai/DeepSeek-V3"):
        self.model_name = model_name
        self._tokenizer = AutoTokenizer.from_pretrained(model_name)

    @classmethod
    def get_instance(cls, model_name: str = "deepseek-ai/DeepSeek-V3") -> "Tokenizer":
        """Singleton pattern per evitare ricaricamenti multipli."""
        if cls._instance is None or cls._instance.model_name != model_name:
            cls._instance = cls(model_name)
        return cls._instance

    @property
    def vocab_size(self) -> int:
        return self._tokenizer.vocab_size

    def encode(self, text: str) -> List[int]:
        """Tokenizza testo in lista di token IDs."""
        return self._tokenizer.encode(text, add_special_tokens=False)

    def decode(self, token_ids: List[int]) -> str:
        """Decodifica token IDs in testo."""
        return self._tokenizer.decode(token_ids)

    def to_tokens(self, text: str) -> List[str]:
        """Converte testo in lista di token come stringhe."""
        token_ids = self.encode(text)
        return self._tokenizer.convert_ids_to_tokens(token_ids)
