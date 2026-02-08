"""
Evolutionary RAG System - Two-Stage Pipeline.

Stage 1: Vector search to get top-N candidates (e.g., 100)
Stage 2: Genetic Algorithm to select optimal k chunks from candidates

Fitness = Coverage(query, chunks) - Redundancy(chunks)

Optimized with numpy vectorization and batch tokenization.
"""
import logging
import random
import time
from collections import Counter
from typing import Dict, List, Optional

import numpy as np
from scipy.sparse import csr_matrix

from ..evolutionary.evolution import EvolutionConfig
from ..evolutionary.tokenizer import Tokenizer
from .rag import RAGSystem

logger = logging.getLogger(__name__)


class EvolutionaryRAGSystem(RAGSystem):
    """
    Two-Stage Evolutionary RAG:

    1. Vector search: 72k -> top 100 candidates (fast, ~50ms)
    2. GA optimization: 100 -> best k chunks (vectorized, ~20-50ms)

    Fitness = coverage - redundancy (no LLM calls)
    """

    def __init__(
        self,
        model: str | None = None,
        table_name: str | None = None,
        evolution_config: EvolutionConfig | None = None,
    ):
        super().__init__(model=model, table_name=table_name)

        self.evolution_config = evolution_config or EvolutionConfig(
            population_size=50,
            k_initial=15,
            max_generations=100,
            mutation_rate=0.2,
            alpha=1.0,
            beta=0.1,
        )

        self.n_candidates = 100
        self.early_stop_patience = 15
        self._tokenizer: Optional[Tokenizer] = None

    @property
    def tokenizer(self) -> Tokenizer:
        if self._tokenizer is None:
            self._tokenizer = Tokenizer.get_instance(self.evolution_config.model_name)
        return self._tokenizer

    def _build_matrices(
        self, query_text: str, candidate_texts: List[str]
    ):
        """
        Batch-tokenize and build numpy structures for fast fitness evaluation.

        Returns:
            coverage_per_chunk: np.ndarray of shape (n_candidates,) — precomputed coverage score per chunk
            chunk_matrix: csr_matrix of shape (n_candidates, n_tokens) — sparse prob distributions
        """
        # Batch tokenize: 1 call instead of 100+
        all_texts = [query_text] + candidate_texts
        all_token_ids = self.tokenizer.encode_batch(all_texts)

        query_ids = all_token_ids[0]
        candidate_ids_list = all_token_ids[1:]

        # Build query distribution
        query_counts = Counter(query_ids)
        query_total = len(query_ids)
        query_dist = {tid: c / query_total for tid, c in query_counts.items()}

        # Collect all unique token IDs across query + candidates to build compact column mapping
        all_token_set = set(query_ids)
        candidate_counters = []
        candidate_totals = []
        for ids in candidate_ids_list:
            all_token_set.update(ids)
            candidate_counters.append(Counter(ids))
            candidate_totals.append(len(ids))

        # Token ID -> compact column index
        token_to_col = {tid: col for col, tid in enumerate(sorted(all_token_set))}
        n_tokens = len(token_to_col)
        n_candidates = len(candidate_texts)

        # Build query vector (dense, small)
        query_vec = np.zeros(n_tokens, dtype=np.float32)
        for tid, prob in query_dist.items():
            query_vec[token_to_col[tid]] = prob

        # Build candidate sparse matrix
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

        # Pre-compute per-chunk coverage: chunk_matrix @ query_vec
        # Coverage is additive per chunk, so we can precompute it once
        coverage_per_chunk = chunk_matrix.dot(query_vec)  # shape (n_candidates,)

        return coverage_per_chunk, chunk_matrix

    def _compute_redundancy_batch(
        self, population: np.ndarray, chunk_matrix: csr_matrix
    ) -> np.ndarray:
        """
        Compute redundancy for all genomes in the population.

        Replicates the original formula: for each token, the first chunk to
        contain it sets seen_tokens[token] = prob. Every subsequent chunk
        with that token adds prob_subsequent * prob_first to redundancy.

        Vectorized as: for each token column, first_prob = prob of first chunk
        (by genome order) that has it. Redundancy = sum over remaining chunks
        of (their prob * first_prob).

        Args:
            population: np.ndarray of shape (pop_size, k)
            chunk_matrix: csr_matrix of shape (n_candidates, n_tokens)

        Returns:
            redundancy scores: np.ndarray of shape (pop_size,)
        """
        pop_size, k = population.shape
        redundancies = np.empty(pop_size, dtype=np.float32)

        for i in range(pop_size):
            selected = chunk_matrix[population[i]].toarray()  # (k, n_tokens)
            # For each token column, find the first non-zero row (first chunk that has it)
            nonzero_mask = selected > 0  # (k, n_tokens)
            # First occurrence index per column (argmax on bool gives first True)
            first_idx = np.argmax(nonzero_mask, axis=0)  # (n_tokens,)
            first_prob = selected[first_idx, np.arange(selected.shape[1])]  # (n_tokens,)
            # Zero out the first occurrence so we only sum subsequent ones
            subsequent = selected.copy()
            subsequent[first_idx, np.arange(selected.shape[1])] = 0
            # Redundancy = sum of (subsequent_prob * first_prob) across all tokens
            redundancies[i] = (subsequent * first_prob[np.newaxis, :]).sum()

        return redundancies

    def _run_ga(
        self,
        coverage_per_chunk: np.ndarray,
        chunk_matrix: csr_matrix,
        k: int,
    ) -> List[int]:
        """
        Run genetic algorithm with vectorized fitness evaluation.

        Args:
            coverage_per_chunk: Precomputed coverage score per chunk (n_candidates,)
            chunk_matrix: Sparse matrix of chunk token distributions
            k: Number of chunks to select

        Returns:
            List of best chunk indices
        """
        M = len(coverage_per_chunk)
        if M <= k:
            return list(range(M))

        pop_size = self.evolution_config.population_size
        generations = self.evolution_config.max_generations
        mutation_rate = self.evolution_config.mutation_rate
        alpha = self.evolution_config.alpha
        beta = self.evolution_config.beta

        # Initialize population as numpy array: (pop_size, k)
        population = np.array([
            random.sample(range(M), k) for _ in range(pop_size)
        ], dtype=np.int32)

        best_genome = None
        best_fitness = float('-inf')
        stale_generations = 0

        for gen in range(generations):
            # Vectorized coverage: sum precomputed per-chunk scores for each genome
            coverage_scores = coverage_per_chunk[population].sum(axis=1)  # (pop_size,)

            # Vectorized redundancy
            redundancy_scores = self._compute_redundancy_batch(population, chunk_matrix)

            # Fitness = alpha * coverage - beta * redundancy
            fitness = alpha * coverage_scores - beta * redundancy_scores  # (pop_size,)

            # Track best
            gen_best_idx = np.argmax(fitness)
            gen_best_fitness = fitness[gen_best_idx]

            if gen_best_fitness > best_fitness:
                best_fitness = gen_best_fitness
                best_genome = population[gen_best_idx].tolist()
                stale_generations = 0
            else:
                stale_generations += 1

            # Early stopping
            if stale_generations >= self.early_stop_patience:
                logger.debug(f"GA early stop at generation {gen} (no improvement for {self.early_stop_patience} gens)")
                break

            # Selection: top 50% survive
            sorted_indices = np.argsort(fitness)[::-1]
            n_survivors = pop_size // 2
            survivors = population[sorted_indices[:n_survivors]].copy()

            # New population: elites + mutated offspring
            new_pop = survivors.copy()
            n_offspring = pop_size - n_survivors
            offspring = np.empty((n_offspring, k), dtype=np.int32)

            for i in range(n_offspring):
                parent_idx = random.randint(0, n_survivors - 1)
                child = survivors[parent_idx].copy()

                if random.random() < mutation_rate:
                    pos = random.randint(0, k - 1)
                    new_val = random.randint(0, M - 1)
                    child_set = set(child.tolist())
                    attempts = 0
                    while new_val in child_set and attempts < 10:
                        new_val = random.randint(0, M - 1)
                        attempts += 1
                    if new_val not in child_set:
                        child[pos] = new_val

                offspring[i] = child

            population = np.vstack([new_pop, offspring])

        return best_genome if best_genome else population[0].tolist()

    def retrieve(self, query: str, limit: int = 5) -> List[str]:
        """
        Two-stage retrieval:
        1. Vector search: get top-N candidates
        2. GA: select optimal k from candidates (vectorized)
        """
        t0 = time.perf_counter()

        # Stage 1: Vector search for candidates
        candidates = self.vector_search.search(query, limit=self.n_candidates)

        if not candidates:
            return []

        candidate_texts = [c[1] for c in candidates]

        if len(candidates) <= limit:
            return candidate_texts

        t1 = time.perf_counter()

        # Build numpy structures (batch tokenization + sparse matrix)
        coverage_per_chunk, chunk_matrix = self._build_matrices(query, candidate_texts)

        t2 = time.perf_counter()

        # Stage 2: GA optimization
        k = self.evolution_config.k_initial
        best_indices = self._run_ga(coverage_per_chunk, chunk_matrix, k)

        t3 = time.perf_counter()

        selected = [candidate_texts[i] for i in best_indices]

        logger.info(
            f"Two-stage: {len(candidates)} -> {len(selected)} chunks | "
            f"search={t1-t0:.3f}s tokenize={t2-t1:.3f}s GA={t3-t2:.3f}s"
        )

        return selected
