from typing import List, Optional

from transformers import AutoTokenizer


class Tokenizer:
    """Wrapper for HuggingFace tokenizer with caching and batch support."""

    _instance: Optional["Tokenizer"] = None
    _tokenizer = None

    def __init__(self, model_name: str = "deepseek-ai/DeepSeek-V3"):
        self.model_name = model_name
        self._tokenizer = AutoTokenizer.from_pretrained(model_name)

    @classmethod
    def get_instance(cls, model_name: str = "deepseek-ai/DeepSeek-V3") -> "Tokenizer":
        """Singleton pattern to avoid multiple reloads."""
        if cls._instance is None or cls._instance.model_name != model_name:
            cls._instance = cls(model_name)
        return cls._instance

    @property
    def vocab_size(self) -> int:
        return self._tokenizer.vocab_size

    def encode(self, text: str) -> List[int]:
        """Tokenize text to list of token IDs."""
        return self._tokenizer.encode(text, add_special_tokens=False)

    def encode_batch(self, texts: List[str]) -> List[List[int]]:
        """Batch tokenize multiple texts at once (faster)."""
        encoded = self._tokenizer(
            texts,
            add_special_tokens=False,
            padding=False,
            truncation=False,
            return_attention_mask=False,
        )
        return encoded["input_ids"]

    def decode(self, token_ids: List[int]) -> str:
        """Decode token IDs to text."""
        return self._tokenizer.decode(token_ids)

    def to_tokens(self, text: str) -> List[str]:
        """Convert text to list of token strings."""
        token_ids = self.encode(text)
        return self._tokenizer.convert_ids_to_tokens(token_ids)
