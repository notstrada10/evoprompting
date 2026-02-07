from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class Genome:
    """
    Genoma rappresentato come bitmask sui chunk del vector DB.
    bits[i] = 1 significa che il chunk i è selezionato.
    """
    bits: np.ndarray
    fitness: float = 0.0

    @classmethod
    def random(cls, n_chunks: int, k: int) -> "Genome":
        """Crea genoma random con esattamente k chunk selezionati."""
        bits = np.zeros(n_chunks, dtype=np.int8)
        selected_indices = np.random.choice(n_chunks, size=min(k, n_chunks), replace=False)
        bits[selected_indices] = 1
        return cls(bits=bits)

    @classmethod
    def empty(cls, n_chunks: int) -> "Genome":
        """Crea genoma vuoto (nessun chunk selezionato)."""
        return cls(bits=np.zeros(n_chunks, dtype=np.int8))

    def copy(self) -> "Genome":
        """Crea una copia del genoma."""
        return Genome(bits=self.bits.copy(), fitness=self.fitness)

    @property
    def n_selected(self) -> int:
        """Numero di chunk selezionati."""
        return int(self.bits.sum())

    @property
    def selected_indices(self) -> np.ndarray:
        """Indici dei chunk selezionati."""
        return np.where(self.bits == 1)[0]

    def __len__(self) -> int:
        return len(self.bits)

    def __repr__(self) -> str:
        return f"Genome(selected={self.n_selected}/{len(self)}, fitness={self.fitness:.4f})"
