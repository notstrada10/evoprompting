"""
Sparse probability distributions over vocabulary.

Uses sparse dicts {token_id: probability} for memory efficiency.
"""
import math
from collections import Counter
from typing import Dict, List

from .tokenizer import Tokenizer

# Type alias for sparse probability distribution
SparseProb = Dict[int, float]


class Distribution:
    """Sparse probability distribution over vocabulary."""

    def __init__(self, probs: SparseProb):
        """
        Args:
            probs: Sparse dict {token_id: probability}
        """
        self.probs = probs

    @classmethod
    def from_text(cls, text: str, tokenizer: Tokenizer) -> "Distribution":
        """Create distribution from text (normalized token frequencies)."""
        token_ids = tokenizer.encode(text)
        if not token_ids:
            return cls({})

        counts = Counter(token_ids)
        total = len(token_ids)

        probs = {token_id: count / total for token_id, count in counts.items()}
        return cls(probs)

    @classmethod
    def combine(cls, distributions: List["Distribution"]) -> "Distribution":
        """Combine distributions (average mixture model)."""
        if not distributions:
            return cls({})

        combined: SparseProb = {}
        for dist in distributions:
            for token_id, prob in dist.probs.items():
                combined[token_id] = combined.get(token_id, 0.0) + prob

        # Normalize
        total = sum(combined.values())
        if total > 0:
            combined = {k: v / total for k, v in combined.items()}

        return cls(combined)

    def entropy(self) -> float:
        """Calculate entropy H(P) = -sum(p * log(p))."""
        if not self.probs:
            return 0.0

        h = 0.0
        for prob in self.probs.values():
            if prob > 0:
                h -= prob * math.log(prob)
        return h

    def kl_divergence(self, other: "Distribution", epsilon: float = 1e-9) -> float:
        """
        Calculate KL divergence D_KL(self || other).

        Args:
            other: The Q distribution
            epsilon: Smoothing for missing tokens

        Returns:
            KL divergence value
        """
        if not self.probs:
            return 0.0

        kl = 0.0
        for token_id, p in self.probs.items():
            q = other.probs.get(token_id, epsilon)
            if p > 0:
                kl += p * math.log(p / q)
        return kl

    def js_divergence(self, other: "Distribution") -> float:
        """Calculate Jensen-Shannon divergence (symmetric)."""
        # Create mixture M = (P + Q) / 2
        m_probs: SparseProb = {}
        all_tokens = set(self.probs.keys()) | set(other.probs.keys())

        for token_id in all_tokens:
            p = self.probs.get(token_id, 0.0)
            q = other.probs.get(token_id, 0.0)
            m_probs[token_id] = (p + q) / 2

        m = Distribution(m_probs)

        # JS = 0.5 * KL(P || M) + 0.5 * KL(Q || M)
        return 0.5 * self.kl_divergence(m) + 0.5 * other.kl_divergence(m)

    def __len__(self) -> int:
        return len(self.probs)


class ChunkDistributions:
    """Manages pre-computed distributions for all chunks."""

    def __init__(self, chunks: List[str], tokenizer: Tokenizer):
        self.chunks = chunks
        self.tokenizer = tokenizer
        self.distributions: List[Distribution] = []
        self._build_distributions()

    def _build_distributions(self):
        """Convert all chunks to sparse distributions."""
        self.distributions = [
            Distribution.from_text(chunk, self.tokenizer)
            for chunk in self.chunks
        ]

    def get_combined(self, indices: List[int]) -> Distribution:
        """Combine distributions of selected chunk indices."""
        if not indices:
            return Distribution({})

        selected = [self.distributions[i] for i in indices]
        return Distribution.combine(selected)

    def __len__(self) -> int:
        return len(self.chunks)
