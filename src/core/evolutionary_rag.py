"""
Evolutionary RAG: Vector Search + GA Selection.

Stage 1: Vector search for top-N candidates
Stage 2: GA selects optimal k-chunk combination

Fitness = -KL(query || genome_dist) - diversity_weight * redundancy
Crossover = uniform (pool parents' genes, pick k, mutate)
Tournament: 2 parents + 1 child, worst of 3 eliminated.
"""
import logging
import random
import time
from collections import Counter
from typing import Optional

import numpy as np
from scipy.sparse import csr_matrix

from ..config import EvolutionConfig
from .rag import RAGSystem
from .tokenizer import Tokenizer

logger = logging.getLogger(__name__)


class EvolutionaryRAGSystem(RAGSystem):

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

        self.n_candidates = 100
        self.early_stop_patience = 15
        self.diversity_weight = 0.3
        self.tokenizer_instance: Optional[Tokenizer] = None

    @property
    def tokenizer(self) -> Tokenizer:
        if self.tokenizer_instance is None:
            self.tokenizer_instance = Tokenizer.get_instance(self.evolution_config.model_name)
        return self.tokenizer_instance

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

    def kl_divergence(self, query_vec: np.ndarray, genome_dist: np.ndarray, epsilon: float = 1e-10) -> float:
        """KL(query || genome_dist), only over tokens where query > 0."""
        mask = query_vec > 0
        q = query_vec[mask]
        p = genome_dist[mask] + epsilon
        return float(np.sum(q * np.log(q / p)))

    def genome_distribution(self, genome: np.ndarray, chunk_matrix: csr_matrix) -> np.ndarray:
        """Sum and normalize token distributions of selected chunks."""
        combined = np.asarray(chunk_matrix[genome].sum(axis=0)).flatten()
        total = combined.sum()
        if total > 0:
            combined /= total
        return combined

    def fitness(self, genome: np.ndarray, query_vec: np.ndarray, chunk_matrix: csr_matrix, sim_matrix: np.ndarray) -> float:
        """Fitness = -KL(query || genome_dist) - diversity_weight * avg_pairwise_similarity."""
        dist = self.genome_distribution(genome, chunk_matrix)
        kl = self.kl_divergence(query_vec, dist)

        k = len(genome)
        if k < 2:
            redundancy = 0.0
        else:
            sub_sim = sim_matrix[np.ix_(genome, genome)]
            redundancy = np.triu(sub_sim, k=1).sum() / (k * (k - 1) / 2)

        return -kl - self.diversity_weight * redundancy

    def fitness_batch(self, population: np.ndarray, query_vec: np.ndarray, chunk_matrix: csr_matrix, sim_matrix: np.ndarray) -> np.ndarray:
        return np.array([
            self.fitness(genome, query_vec, chunk_matrix, sim_matrix)
            for genome in population
        ], dtype=np.float32)

    def crossover(self, p1: np.ndarray, p2: np.ndarray, k: int, M: int, mutation_rate: float) -> np.ndarray:
        """Uniform crossover: pool both parents' genes, pick k unique indices, mutate."""
        pool = list(set(p1.tolist()) | set(p2.tolist()))
        random.shuffle(pool)

        if len(pool) >= k:
            child = pool[:k]
        else:
            remaining = [i for i in range(M) if i not in set(pool)]
            random.shuffle(remaining)
            child = pool + remaining[:k - len(pool)]

        # Mutation: swap random genes with non-selected candidates
        child_set = set(child)
        non_selected = [i for i in range(M) if i not in child_set]
        for pos in range(k):
            if random.random() < mutation_rate and non_selected:
                swap_in = random.choice(non_selected)
                non_selected.remove(swap_in)
                non_selected.append(child[pos])
                child[pos] = swap_in

        return np.array(child, dtype=np.int32)

    def roulette_select(self, fit: np.ndarray) -> int:
        """Fitness-proportionate selection. Shift fitness to positive values."""
        shifted = fit - fit.min() + 1e-6
        probs = shifted / shifted.sum()
        return int(np.random.choice(len(fit), p=probs))

    def run_ga(self, query_vec: np.ndarray, chunk_matrix: csr_matrix, sim_matrix: np.ndarray, k: int, track_convergence: bool = False) -> list[int] | tuple[list[int], list[float]]:
        """Generational GA with elitism and roulette selection."""
        M = chunk_matrix.shape[0]
        if M <= k:
            if track_convergence:
                return list(range(M)), []
            return list(range(M))

        pop_size = self.evolution_config.population_size
        generations = self.evolution_config.max_generations
        mutation_rate = self.evolution_config.mutation_rate
        elite_size = max(1, pop_size // 10)

        population = np.array([
            random.sample(range(M), k) for _ in range(pop_size)
        ], dtype=np.int32)

        fit = self.fitness_batch(population, query_vec, chunk_matrix, sim_matrix)

        best_idx = np.argmax(fit)
        best_fitness = fit[best_idx]
        best_genome = population[best_idx].tolist()
        stale_generations = 0

        fitness_history = None
        if track_convergence:
            fitness_history = [{"best": float(best_fitness), "mean": float(fit.mean())}]

        for gen in range(generations):
            # Elitism: keep top individuals
            elite_indices = np.argsort(fit)[::-1][:elite_size]
            new_population = list(population[elite_indices])
            new_fit = list(fit[elite_indices])

            # Fill rest with offspring via roulette selection
            while len(new_population) < pop_size:
                p1_idx = self.roulette_select(fit)
                p2_idx = self.roulette_select(fit)

                child = self.crossover(
                    population[p1_idx], population[p2_idx],
                    k, M, mutation_rate,
                )
                child_fitness = self.fitness(child, query_vec, chunk_matrix, sim_matrix)
                new_population.append(child)
                new_fit.append(child_fitness)

            population = np.array(new_population, dtype=np.int32)
            fit = np.array(new_fit, dtype=np.float32)

            gen_best_idx = np.argmax(fit)
            gen_best_fitness = fit[gen_best_idx]

            if gen_best_fitness > best_fitness:
                best_fitness = gen_best_fitness
                best_genome = population[gen_best_idx].tolist()
                stale_generations = 0
            else:
                stale_generations += 1

            if track_convergence:
                fitness_history.append({"best": float(best_fitness), "mean": float(fit.mean())})

            if stale_generations >= self.early_stop_patience:
                logger.debug(f"GA early stop at gen {gen} (no improvement for {self.early_stop_patience} gens)")
                break

        if track_convergence:
            return best_genome, fitness_history
        return best_genome

    def retrieve(self, query: str, limit: int = 5) -> list[str]:
        """Two-stage: vector search candidates -> GA selection."""
        t0 = time.perf_counter()

        candidates = self.vector_search.search(query, limit=self.n_candidates)
        if not candidates:
            return []

        candidate_texts = [c[1] for c in candidates]
        if len(candidates) <= limit:
            return candidate_texts

        t1 = time.perf_counter()

        query_vec, chunk_matrix = self.build_matrices(query, candidate_texts)
        sim_matrix = self.build_similarity_matrix(chunk_matrix)

        t2 = time.perf_counter()

        k = self.evolution_config.k_initial
        best_indices = self.run_ga(query_vec, chunk_matrix, sim_matrix, k)

        t3 = time.perf_counter()

        selected = [candidate_texts[i] for i in best_indices]
        logger.info(
            f"Evolutionary: {len(candidates)} candidates -> {len(selected)} selected | "
            f"search={t1-t0:.3f}s build={t2-t1:.3f}s GA={t3-t2:.3f}s"
        )
        return selected
