"""
BM25 Evolutionary RAG: BM25 Pre-filter + Bit-String GA Selection.

Stage 1: BM25 lexical search for top-N candidate chunks
Stage 2: GA evolves bit-string genomes to select optimal k-chunk subset

Genome = binary vector of length N (1=include, 0=exclude), constrained to sum=k
Fitness = alpha * (-KL) + beta * Density - gamma * Redundancy
  - KL: token distribution alignment between query and selected set
  - Density: mutual rare-token bridging between selected chunks (multi-hop signal)
  - Redundancy: penalize near-duplicate chunks
Selection = roulette (fitness-proportionate)
Crossover = uniform bit-swap + repair to maintain k ones
"""
import logging
import random
import time
from collections import Counter
from typing import Optional

import numpy as np
from rank_bm25 import BM25Okapi
from scipy.sparse import csr_matrix

from ..config import EvolutionConfig
from .rag import RAGSystem
from .tokenizer import Tokenizer

logger = logging.getLogger(__name__)


class BM25EvolutionaryRAGSystem(RAGSystem):

    def __init__(
        self,
        model: str | None = None,
        table_name: str | None = None,
        evolution_config: EvolutionConfig | None = None,
    ):
        super().__init__(model=model, table_name=table_name)

        self.evolution_config = evolution_config or EvolutionConfig(
            population_size=50,
            k_initial=10,
            max_generations=100,
            mutation_rate=0.2,
        )

        self.n_candidates = 200
        self.early_stop_patience = 15
        self.alpha = 1.0       # KL alignment weight
        self.beta = 0.1        # Density (multi-hop bridging) weight
        self.gamma = 0.3       # Redundancy penalty weight
        self.tokenizer_instance: Optional[Tokenizer] = None
        self._bm25: Optional[BM25Okapi] = None
        self._corpus_texts: Optional[list[str]] = None
        self._corpus_tokenized: Optional[list[list[str]]] = None

    @property
    def tokenizer(self) -> Tokenizer:
        if self.tokenizer_instance is None:
            self.tokenizer_instance = Tokenizer.get_instance(self.evolution_config.model_name)
        return self.tokenizer_instance

    def _ensure_bm25(self) -> None:
        """Build BM25 index from all chunks in the database (lazy, built once)."""
        if self._bm25 is not None:
            return

        rows = self.vector_search.db.get_all_chunks()
        self._corpus_texts = [text for _, text in rows]
        self._corpus_tokenized = [text.lower().split() for text in self._corpus_texts]
        self._bm25 = BM25Okapi(self._corpus_tokenized)
        logger.info(f"BM25 index built over {len(self._corpus_texts)} chunks")

    def bm25_search(self, query: str, n: int) -> list[str]:
        """Return top-n chunks by BM25 score."""
        self._ensure_bm25()
        tokenized_query = query.lower().split()
        scores = self._bm25.get_scores(tokenized_query)
        top_indices = np.argsort(scores)[::-1][:n]
        return [self._corpus_texts[i] for i in top_indices]

    def build_matrices(self, query_text: str, candidate_texts: list[str]) -> tuple[np.ndarray, csr_matrix]:
        """Batch-tokenize query + candidates, return (query_vec, chunk_matrix) as sparse distributions."""
        all_texts = [query_text] + candidate_texts
        all_token_ids = self.tokenizer.encode_batch(all_texts)

        query_ids = all_token_ids[0]
        candidate_ids_list = all_token_ids[1:]

        query_counts = Counter(query_ids)
        query_total = len(query_ids)

        all_token_set = set(query_ids)
        candidate_counters = []
        candidate_totals = []
        for ids in candidate_ids_list:
            all_token_set.update(ids)
            candidate_counters.append(Counter(ids))
            candidate_totals.append(len(ids))

        token_to_col = {tid: col for col, tid in enumerate(sorted(all_token_set))}
        n_tokens = len(token_to_col)
        n_candidates = len(candidate_texts)

        query_vec = np.zeros(n_tokens, dtype=np.float32)
        for tid, count in query_counts.items():
            query_vec[token_to_col[tid]] = count / query_total

        rows, cols, data = [], [], []
        for i, (counts, total) in enumerate(zip(candidate_counters, candidate_totals)):
            if total == 0:
                continue
            for tid, count in counts.items():
                rows.append(i)
                cols.append(token_to_col[tid])
                data.append(count / total)

        chunk_matrix = csr_matrix(
            (np.array(data, dtype=np.float32), (rows, cols)),
            shape=(n_candidates, n_tokens),
        )
        return query_vec, chunk_matrix

    def build_similarity_matrix(self, chunk_matrix: csr_matrix) -> np.ndarray:
        """Pairwise cosine similarity between all candidate chunk distributions."""
        norms = np.sqrt(np.asarray(chunk_matrix.power(2).sum(axis=1)).flatten())
        norms[norms == 0] = 1.0
        inv_norms = csr_matrix(np.diag(1.0 / norms))
        normalized = inv_norms @ chunk_matrix
        sim = (normalized @ normalized.T).toarray()
        np.fill_diagonal(sim, 0.0)
        return sim

    def build_idf_weights(self, chunk_matrix: csr_matrix) -> np.ndarray:
        """Compute IDF weights per token across the candidate pool.

        IDF(t) = log(N / df(t)), where df(t) = number of chunks containing token t.
        Rare tokens get high IDF, common tokens get low IDF.
        """
        N = chunk_matrix.shape[0]
        df = np.asarray((chunk_matrix > 0).sum(axis=0)).flatten().astype(np.float32)
        df = np.clip(df, 1, None)
        return np.log(N / df).astype(np.float32)

    def build_rare_token_bridge_matrix(self, chunk_matrix: csr_matrix, idf_weights: np.ndarray) -> np.ndarray:
        """Pairwise rare-token bridging score between all candidates, normalized to [0, 1].

        For each pair (i, j): sum of IDF weights for tokens present in BOTH chunks,
        normalized by the max possible bridge score.
        High score = chunks share rare tokens = potential multi-hop bridge.
        """
        presence = (chunk_matrix > 0).astype(np.float32)
        weighted = presence.multiply(csr_matrix(idf_weights))
        bridge = (weighted @ weighted.T).toarray()
        np.fill_diagonal(bridge, 0.0)
        max_val = bridge.max()
        if max_val > 0:
            bridge /= max_val
        return bridge

    def kl_divergence(self, query_vec: np.ndarray, genome_dist: np.ndarray, epsilon: float = 1e-10) -> float:
        """KL(query || genome_dist), only over tokens where query > 0."""
        mask = query_vec > 0
        q = query_vec[mask]
        p = genome_dist[mask] + epsilon
        return float(np.sum(q * np.log(q / p)))

    def genome_distribution(self, indices: np.ndarray, chunk_matrix: csr_matrix) -> np.ndarray:
        """Sum and normalize token distributions of selected chunks."""
        combined = np.asarray(chunk_matrix[indices].sum(axis=0)).flatten()
        total = combined.sum()
        if total > 0:
            combined /= total
        return combined

    def fitness(self, genome: np.ndarray, query_vec: np.ndarray, chunk_matrix: csr_matrix, sim_matrix: np.ndarray, bridge_matrix: np.ndarray) -> float:
        """Multi-hop fitness: alpha * (-KL) + beta * Density - gamma * Redundancy.

        genome is a bit-string (0/1 array of length M). Selected indices = where genome == 1.

        - KL: how well the selected set's token distribution covers the query
        - Density: avg rare-token bridging between selected chunks (multi-hop signal)
        - Redundancy: penalize near-duplicate chunks (cosine similarity)
        """
        indices = np.where(genome == 1)[0]
        if len(indices) == 0:
            return -1e6

        dist = self.genome_distribution(indices, chunk_matrix)
        kl = self.kl_divergence(query_vec, dist)

        k = len(indices)
        if k < 2:
            density = 0.0
            redundancy = 0.0
        else:
            n_pairs = k * (k - 1) / 2

            sub_bridge = bridge_matrix[np.ix_(indices, indices)]
            density = np.triu(sub_bridge, k=1).sum() / n_pairs

            sub_sim = sim_matrix[np.ix_(indices, indices)]
            redundancy = np.triu(sub_sim, k=1).sum() / n_pairs

        return self.alpha * (-kl) + self.beta * density - self.gamma * redundancy

    def fitness_batch(self, population: np.ndarray, query_vec: np.ndarray, chunk_matrix: csr_matrix, sim_matrix: np.ndarray, bridge_matrix: np.ndarray) -> np.ndarray:
        return np.array([
            self.fitness(genome, query_vec, chunk_matrix, sim_matrix, bridge_matrix)
            for genome in population
        ], dtype=np.float32)

    def random_bitstring(self, M: int, k: int) -> np.ndarray:
        """Create a random bit-string genome with exactly k ones."""
        genome = np.zeros(M, dtype=np.int8)
        genome[random.sample(range(M), k)] = 1
        return genome

    def crossover(self, p1: np.ndarray, p2: np.ndarray, k: int, mutation_rate: float) -> np.ndarray:
        """Uniform crossover for bit-strings with repair to maintain exactly k ones.

        For each bit: inherit from a random parent. Then repair:
        - If too many ones: randomly flip excess 1->0
        - If too few ones: randomly flip deficit 0->1
        Finally, mutation: swap a random 1-bit with a random 0-bit.
        """
        M = len(p1)
        mask = np.random.randint(0, 2, size=M, dtype=np.int8)
        child = np.where(mask, p1, p2)

        ones = np.where(child == 1)[0]
        zeros = np.where(child == 0)[0]
        n_ones = len(ones)

        if n_ones > k:
            to_off = np.random.choice(ones, size=n_ones - k, replace=False)
            child[to_off] = 0
        elif n_ones < k:
            to_on = np.random.choice(zeros, size=k - n_ones, replace=False)
            child[to_on] = 1

        if random.random() < mutation_rate:
            ones = np.where(child == 1)[0]
            zeros = np.where(child == 0)[0]
            if len(ones) > 0 and len(zeros) > 0:
                bit_off = np.random.choice(ones)
                bit_on = np.random.choice(zeros)
                child[bit_off] = 0
                child[bit_on] = 1

        return child

    def roulette_select(self, fit: np.ndarray) -> int:
        """Fitness-proportionate selection. Shift fitness to positive values."""
        shifted = fit - fit.min() + 1e-6
        probs = shifted / shifted.sum()
        return int(np.random.choice(len(fit), p=probs))

    def bm25_seeded_population(self, bm25_scores: np.ndarray, M: int, k: int, pop_size: int, seed_fraction: float = 0.3) -> np.ndarray:
        """Create initial population: some seeded from BM25 top-k, rest random.

        Seeded genomes select the top-k BM25 chunks (with small perturbation).
        """
        n_seeded = max(1, int(pop_size * seed_fraction))
        population = []

        top_k_indices = np.argsort(bm25_scores)[::-1][:k]
        for _ in range(n_seeded):
            genome = np.zeros(M, dtype=np.int8)
            genome[top_k_indices] = 1
            n_flips = random.randint(0, min(2, k))
            for _ in range(n_flips):
                ones = np.where(genome == 1)[0]
                zeros = np.where(genome == 0)[0]
                if len(ones) > 0 and len(zeros) > 0:
                    genome[np.random.choice(ones)] = 0
                    genome[np.random.choice(zeros)] = 1
            population.append(genome)

        for _ in range(pop_size - n_seeded):
            population.append(self.random_bitstring(M, k))

        return np.array(population, dtype=np.int8)

    def run_ga(self, query_vec: np.ndarray, chunk_matrix: csr_matrix, sim_matrix: np.ndarray, bridge_matrix: np.ndarray, k: int, bm25_scores: np.ndarray | None = None, track_convergence: bool = False) -> list[int] | tuple[list[int], list[dict]]:
        """Generational GA with elitism, roulette selection, and bit-string genomes."""
        M = chunk_matrix.shape[0]
        if M <= k:
            result = list(range(M))
            if track_convergence:
                return result, []
            return result

        pop_size = self.evolution_config.population_size
        generations = self.evolution_config.max_generations
        mutation_rate = self.evolution_config.mutation_rate
        elite_size = max(1, pop_size // 10)

        if bm25_scores is not None:
            population = self.bm25_seeded_population(bm25_scores, M, k, pop_size)
        else:
            population = np.array([
                self.random_bitstring(M, k) for _ in range(pop_size)
            ], dtype=np.int8)

        fit = self.fitness_batch(population, query_vec, chunk_matrix, sim_matrix, bridge_matrix)

        best_idx = int(np.argmax(fit))
        best_fitness = fit[best_idx]
        best_genome = population[best_idx].copy()
        stale_generations = 0

        fitness_history = None
        if track_convergence:
            fitness_history = [{"best": float(best_fitness), "mean": float(fit.mean())}]

        for gen in range(generations):
            elite_indices = np.argsort(fit)[::-1][:elite_size]
            new_population = [population[i].copy() for i in elite_indices]
            new_fit = [fit[i] for i in elite_indices]

            while len(new_population) < pop_size:
                p1_idx = self.roulette_select(fit)
                p2_idx = self.roulette_select(fit)

                child = self.crossover(
                    population[p1_idx], population[p2_idx],
                    k, mutation_rate,
                )
                child_fitness = self.fitness(child, query_vec, chunk_matrix, sim_matrix, bridge_matrix)
                new_population.append(child)
                new_fit.append(child_fitness)

            population = np.array(new_population, dtype=np.int8)
            fit = np.array(new_fit, dtype=np.float32)

            gen_best_idx = int(np.argmax(fit))
            gen_best_fitness = fit[gen_best_idx]

            if gen_best_fitness > best_fitness:
                best_fitness = gen_best_fitness
                best_genome = population[gen_best_idx].copy()
                stale_generations = 0
            else:
                stale_generations += 1

            if track_convergence:
                fitness_history.append({"best": float(best_fitness), "mean": float(fit.mean())})

            if stale_generations >= self.early_stop_patience:
                logger.debug(f"GA early stop at gen {gen} (no improvement for {self.early_stop_patience} gens)")
                break

        best_indices = np.where(best_genome == 1)[0].tolist()

        if track_convergence:
            return best_indices, fitness_history
        return best_indices

    def retrieve(self, query: str, limit: int = 5) -> list[str]:
        """Two-stage: BM25 candidate retrieval -> GA bit-string selection."""
        t0 = time.perf_counter()

        candidate_texts = self.bm25_search(query, self.n_candidates)
        if not candidate_texts:
            return []

        k = self.evolution_config.k_initial
        if len(candidate_texts) <= k:
            return candidate_texts

        t1 = time.perf_counter()

        query_vec, chunk_matrix = self.build_matrices(query, candidate_texts)
        sim_matrix = self.build_similarity_matrix(chunk_matrix)
        idf_weights = self.build_idf_weights(chunk_matrix)
        bridge_matrix = self.build_rare_token_bridge_matrix(chunk_matrix, idf_weights)

        # Get BM25 scores for the candidates to seed the population
        self._ensure_bm25()
        tokenized_query = query.lower().split()
        all_scores = self._bm25.get_scores(tokenized_query)
        candidate_scores = np.zeros(len(candidate_texts), dtype=np.float32)
        corpus_text_to_score = {}
        top_indices = np.argsort(all_scores)[::-1][:self.n_candidates]
        for idx in top_indices:
            corpus_text_to_score[self._corpus_texts[idx]] = all_scores[idx]
        for i, text in enumerate(candidate_texts):
            candidate_scores[i] = corpus_text_to_score.get(text, 0.0)

        t2 = time.perf_counter()

        best_indices = self.run_ga(query_vec, chunk_matrix, sim_matrix, bridge_matrix, k, bm25_scores=candidate_scores)

        t3 = time.perf_counter()

        selected = [candidate_texts[i] for i in best_indices]
        logger.info(
            f"BM25-Evolutionary: BM25 {len(candidate_texts)} candidates -> GA {len(selected)} selected | "
            f"bm25={t1-t0:.3f}s build={t2-t1:.3f}s GA={t3-t2:.3f}s"
        )
        return selected
