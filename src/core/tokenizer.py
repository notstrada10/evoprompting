from typing import List, Optional

from transformers import AutoTokenizer


class Tokenizer:
    """Wrapper for HuggingFace tokenizer with singleton and batch support."""

    instance: Optional["Tokenizer"] = None

    def __init__(self, model_name: str = "deepseek-ai/DeepSeek-V3"):
        self.model_name = model_name
        self.hf_tokenizer = AutoTokenizer.from_pretrained(model_name)

    @classmethod
    def get_instance(cls, model_name: str = "deepseek-ai/DeepSeek-V3") -> "Tokenizer":
        if cls.instance is None or cls.instance.model_name != model_name:
            cls.instance = cls(model_name)
        return cls.instance

    @property
    def vocab_size(self) -> int:
        return self.hf_tokenizer.vocab_size

    def encode(self, text: str) -> List[int]:
        return self.hf_tokenizer.encode(text, add_special_tokens=False)

    def encode_batch(self, texts: List[str]) -> List[List[int]]:
        encoded = self.hf_tokenizer(
            texts,
            add_special_tokens=False,
            padding=False,
            truncation=False,
            return_attention_mask=False,
        )
        return encoded["input_ids"]

    def decode(self, token_ids: List[int]) -> str:
        return self.hf_tokenizer.decode(token_ids)
