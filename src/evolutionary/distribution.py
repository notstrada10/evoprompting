"""Sparse probability distributions over token vocabulary."""
import math
from collections import Counter
from typing import Dict, List

from .tokenizer import Tokenizer

SparseProb = Dict[int, float]


class Distribution:
    """Sparse probability distribution {token_id: probability}."""

    def __init__(self, probs: SparseProb):
        self.probs = probs

    @classmethod
    def from_text(cls, text: str, tokenizer: Tokenizer) -> "Distribution":
        token_ids = tokenizer.encode(text)
        if not token_ids:
            return cls({})
        counts = Counter(token_ids)
        total = len(token_ids)
        return cls({tid: count / total for tid, count in counts.items()})

    @classmethod
    def combine(cls, distributions: List["Distribution"]) -> "Distribution":
        if not distributions:
            return cls({})
        combined: SparseProb = {}
        for dist in distributions:
            for tid, prob in dist.probs.items():
                combined[tid] = combined.get(tid, 0.0) + prob
        total = sum(combined.values())
        if total > 0:
            combined = {k: v / total for k, v in combined.items()}
        return cls(combined)


    def kl_divergence(self, other: "Distribution", epsilon: float = 1e-9) -> float:
        if not self.probs:
            return 0.0
        kl = 0.0
        for tid, p in self.probs.items():
            q = other.probs.get(tid, epsilon)
            if p > 0:
                kl += p * math.log(p / q)
        return kl

    def js_divergence(self, other: "Distribution") -> float:
        m_probs: SparseProb = {}
        all_tokens = set(self.probs.keys()) | set(other.probs.keys())
        for tid in all_tokens:
            p = self.probs.get(tid, 0.0)
            q = other.probs.get(tid, 0.0)
            m_probs[tid] = (p + q) / 2
        m = Distribution(m_probs)
        return 0.5 * self.kl_divergence(m) + 0.5 * other.kl_divergence(m)

    def __len__(self) -> int:
        return len(self.probs)


class ChunkDistributions:
    """Pre-computed token distributions for all chunks."""

    def __init__(self, chunks: List[str], tokenizer: Tokenizer, batch_size: int = 1000):
        self.chunks = chunks
        self.tokenizer = tokenizer
        self.distributions: List[Distribution] = []
        self.build(batch_size)

    def build(self, batch_size: int):
        self.distributions = []
        for i in range(0, len(self.chunks), batch_size):
            batch = self.chunks[i : i + batch_size]
            token_ids_batch = self.tokenizer.encode_batch(batch)
            for token_ids in token_ids_batch:
                if not token_ids:
                    self.distributions.append(Distribution({}))
                    continue
                counts = Counter(token_ids)
                total = len(token_ids)
                self.distributions.append(
                    Distribution({tid: c / total for tid, c in counts.items()})
                )

    def get_combined(self, bitmask) -> Distribution:
        """Combine distributions of selected chunks (bitmask: array where 1 = selected)."""
        selected = [d for d, m in zip(self.distributions, bitmask) if m == 1]
        if not selected:
            return Distribution({})
        return Distribution.combine(selected)

    def __len__(self) -> int:
        return len(self.chunks)
