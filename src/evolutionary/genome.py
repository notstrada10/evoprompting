from dataclasses import dataclass

import numpy as np


@dataclass
class Genome:
    """Bitmask over chunks. bits[i] = 1 means chunk i is selected."""

    bits: np.ndarray
    fitness: float = 0.0

    @classmethod
    def random(cls, n_chunks: int, k: int) -> "Genome":
        bits = np.zeros(n_chunks, dtype=np.int8)
        indices = np.random.choice(n_chunks, size=min(k, n_chunks), replace=False)
        bits[indices] = 1
        return cls(bits=bits)

    @classmethod
    def empty(cls, n_chunks: int) -> "Genome":
        return cls(bits=np.zeros(n_chunks, dtype=np.int8))

    def copy(self) -> "Genome":
        return Genome(bits=self.bits.copy(), fitness=self.fitness)

    @property
    def n_selected(self) -> int:
        return int(self.bits.sum())

    @property
    def selected_indices(self) -> np.ndarray:
        return np.where(self.bits == 1)[0]

    def __len__(self) -> int:
        return len(self.bits)

    def __repr__(self) -> str:
        return f"Genome(selected={self.n_selected}/{len(self)}, fitness={self.fitness:.4f})"
