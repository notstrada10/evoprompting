"""Test del sistema di evolutionary prompting - standalone."""

import random
from dataclasses import dataclass
from typing import Callable, List, Optional

import numpy as np
from transformers import AutoTokenizer


# === TOKENIZER ===
class Tokenizer:
    _instance = None

    def __init__(self, model_name: str = "deepseek-ai/DeepSeek-V3"):
        self.model_name = model_name
        self._tokenizer = AutoTokenizer.from_pretrained(model_name)

    @classmethod
    def get_instance(cls, model_name: str = "deepseek-ai/DeepSeek-V3"):
        if cls._instance is None:
            cls._instance = cls(model_name)
        return cls._instance

    @property
    def vocab_size(self) -> int:
        return self._tokenizer.vocab_size

    def encode(self, text: str) -> List[int]:
        return self._tokenizer.encode(text, add_special_tokens=False)


# === DISTRIBUTION ===
class Distribution:
    def __init__(self, probs: np.ndarray):
        self.probs = probs

    @classmethod
    def from_text(cls, text: str, tokenizer: Tokenizer):
        token_ids = tokenizer.encode(text)
        probs = np.zeros(tokenizer.vocab_size, dtype=np.float32)
        for token_id in token_ids:
            probs[token_id] += 1
        total = probs.sum()
        if total > 0:
            probs /= total
        return cls(probs)

    @classmethod
    def combine(cls, distributions: List["Distribution"]):
        combined = np.sum([d.probs for d in distributions], axis=0)
        total = combined.sum()
        if total > 0:
            combined /= total
        return cls(combined)


class ChunkDistributions:
    def __init__(self, chunks: List[str], tokenizer: Tokenizer):
        self.chunks = chunks
        self.tokenizer = tokenizer
        self.distributions = [Distribution.from_text(c, tokenizer) for c in chunks]

    def get_combined(self, mask: np.ndarray) -> Distribution:
        selected = [d for d, m in zip(self.distributions, mask) if m == 1]
        if not selected:
            uniform = np.ones(self.tokenizer.vocab_size, dtype=np.float32)
            return Distribution(uniform / uniform.sum())
        return Distribution.combine(selected)


# === GENOME ===
class Genome:
    def __init__(self, bits: np.ndarray, fitness: float = 0.0):
        self.bits = bits
        self.fitness = fitness

    @classmethod
    def random(cls, n_chunks: int, k: int):
        bits = np.zeros(n_chunks, dtype=np.int8)
        indices = np.random.choice(n_chunks, size=min(k, n_chunks), replace=False)
        bits[indices] = 1
        return cls(bits=bits)

    @property
    def n_selected(self) -> int:
        return int(self.bits.sum())

    @property
    def selected_indices(self) -> np.ndarray:
        return np.where(self.bits == 1)[0]


# === FITNESS ===
def kl_divergence(p: Distribution, q: Distribution, epsilon: float = 1e-10) -> float:
    p_smooth = p.probs + epsilon
    q_smooth = q.probs + epsilon
    p_smooth /= p_smooth.sum()
    q_smooth /= q_smooth.sum()
    return float(np.sum(p_smooth * np.log(p_smooth / q_smooth)))


def information_gain(query_dist: Distribution, genome_dist: Distribution) -> float:
    kl = kl_divergence(query_dist, genome_dist)
    return 1.0 / (1.0 + kl)


# === GA ===
def crossover(p1: Genome, p2: Genome, k: int) -> Genome:
    """Crossover che mantiene esattamente k chunk selezionati."""
    point = random.randint(1, len(p1.bits) - 1)
    child_bits = np.concatenate([p1.bits[:point], p2.bits[point:]])

    # Aggiusta a esattamente k chunk
    child_bits = _adjust_to_k(child_bits, k)
    return Genome(bits=child_bits)


def mutate(genome: Genome, rate: float = 0.1, k: int = None):
    """Mutation bilanciata: se accendi uno, spegni un altro (mantiene k)."""
    if k is None:
        k = genome.n_selected

    for _ in range(len(genome.bits)):
        if random.random() < rate:
            # Trova un bit=1 e un bit=0
            ones = np.where(genome.bits == 1)[0]
            zeros = np.where(genome.bits == 0)[0]

            if len(ones) > 0 and len(zeros) > 0:
                # Swap: spegni uno, accendi un altro
                to_off = np.random.choice(ones)
                to_on = np.random.choice(zeros)
                genome.bits[to_off] = 0
                genome.bits[to_on] = 1


def _adjust_to_k(bits: np.ndarray, k: int) -> np.ndarray:
    """Aggiusta il bitmask per avere esattamente k bit=1."""
    current = int(bits.sum())

    if current == k:
        return bits

    if current > k:
        # Troppi chunk: spegni alcuni random
        ones = np.where(bits == 1)[0]
        to_remove = np.random.choice(ones, size=current - k, replace=False)
        bits[to_remove] = 0
    else:
        # Pochi chunk: accendi alcuni random
        zeros = np.where(bits == 0)[0]
        to_add = np.random.choice(zeros, size=k - current, replace=False)
        bits[to_add] = 1

    return bits


# === MAIN TEST ===
def main():
    print("=" * 50)
    print("Test Evolutionary Prompting")
    print("=" * 50)

    # Chunk simulati
    chunks = [
        "Python è un linguaggio di programmazione interpretato",
        "Il machine learning richiede grandi quantità di dati",
        "Le reti neurali sono ispirate al cervello umano",
        "La pizza margherita è un piatto italiano",
        "Gli algoritmi genetici simulano l'evoluzione naturale",
        "Il deep learning usa reti neurali profonde",
        "Roma è la capitale d'Italia",
        "L'ottimizzazione cerca il minimo di una funzione",
        "I transformer hanno rivoluzionato il NLP",
        "La pasta è un alimento base della cucina italiana"
    ]

    query = "Come funzionano gli algoritmi evolutivi nel machine learning?"

    print(f"\nQuery: {query}")
    print(f"Chunks: {len(chunks)}")
    print("\nCaricamento tokenizer...")

    # Setup
    tokenizer = Tokenizer.get_instance()
    chunk_dists = ChunkDistributions(chunks, tokenizer)
    query_dist = Distribution.from_text(query, tokenizer)

    print(f"Vocab size: {tokenizer.vocab_size}")

    # Popolazione iniziale
    pop_size, k_init, max_gen = 20, 5, 50
    population = [Genome.random(len(chunks), k_init) for _ in range(pop_size)]

    # Valuta fitness iniziale
    for g in population:
        g.fitness = information_gain(query_dist, chunk_dists.get_combined(g.bits))

    print(f"\n{'Gen':<5} {'Best':<10} {'Avg':<10} {'Chunks':<8}")
    print("-" * 35)

    # Evoluzione
    for gen in range(max_gen):
        # Seleziona 2 genitori
        p1, p2 = random.sample(population, 2)

        # Crossover + mutazione (k fisso!)
        child = crossover(p1, p2, k=k_init)
        mutate(child, 0.15, k=k_init)
        child.fitness = information_gain(query_dist, chunk_dists.get_combined(child.bits))

        # Sostituzione
        worst = min(p1, p2, key=lambda g: g.fitness)
        if child.fitness > worst.fitness:
            population[population.index(worst)] = child

        # Stats
        if gen % 10 == 0:
            fits = [g.fitness for g in population]
            avg_sel = np.mean([g.n_selected for g in population])
            print(f"{gen:<5} {max(fits):<10.4f} {np.mean(fits):<10.4f} {avg_sel:<8.1f}")

    # Risultato
    best = max(population, key=lambda g: g.fitness)

    print("\n" + "=" * 50)
    print("RISULTATO")
    print("=" * 50)
    print(f"\nChunk selezionati ({best.n_selected}):")
    for idx in best.selected_indices:
        print(f"  [{idx}] {chunks[idx]}")
    print(f"\nFitness: {best.fitness:.4f}")


if __name__ == "__main__":
    main()
