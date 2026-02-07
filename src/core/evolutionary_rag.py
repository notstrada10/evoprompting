"""
Evolutionary RAG System - Two-Stage Pipeline.

Stage 1: Vector search to get top-N candidates (e.g., 100)
Stage 2: Genetic Algorithm to select optimal k chunks from candidates

Fitness = Coverage(query, chunks) - Redundancy(chunks)
"""
import logging
import random
from collections import Counter
from typing import Dict, List, Optional, Tuple

from ..evolutionary.evolution import EvolutionConfig
from ..evolutionary.tokenizer import Tokenizer
from .rag import RAGSystem

logger = logging.getLogger(__name__)

# Sparse probability distribution type
SparseProb = Dict[int, float]


class EvolutionaryRAGSystem(RAGSystem):
    """
    Two-Stage Evolutionary RAG:

    1. Vector search: 72k -> top 100 candidates (fast, ~50ms)
    2. GA optimization: 100 -> best 5 chunks (fast, ~100ms)

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
            population_size=30,
            k_initial=10,  # Final number of chunks to select (more for multi-hop)
            max_generations=30,
            mutation_rate=0.3,
            alpha=1.0,  # coverage weight
            beta=0.3,   # redundancy penalty weight (reduced)
        )

        # Number of candidates to retrieve from vector search
        self.n_candidates = 100

        self._tokenizer: Optional[Tokenizer] = None

    @property
    def tokenizer(self) -> Tokenizer:
        if self._tokenizer is None:
            self._tokenizer = Tokenizer.get_instance(self.evolution_config.model_name)
        return self._tokenizer

    def _text_to_sparse_dist(self, text: str) -> SparseProb:
        """Convert text to sparse probability distribution."""
        if not text:
            return {}
        token_ids = self.tokenizer.encode(text)
        if not token_ids:
            return {}
        counts = Counter(token_ids)
        total = len(token_ids)
        return {tid: count / total for tid, count in counts.items()}

    def _compute_fitness(
        self,
        genome: List[int],
        query_dist: SparseProb,
        candidate_dists: List[SparseProb]
    ) -> float:
        """
        Fitness = Coverage - Redundancy

        Coverage: How well do selected chunks cover query tokens?
        Redundancy: How much do chunks repeat each other?
        """
        if not genome:
            return float('-inf')

        alpha = self.evolution_config.alpha
        beta = self.evolution_config.beta

        # Coverage: overlap between chunks and query
        coverage_score = 0.0

        # Track seen tokens for redundancy calculation
        seen_tokens: Dict[int, float] = {}
        redundancy_penalty = 0.0

        for idx in genome:
            chunk_dist = candidate_dists[idx]

            for token, prob in chunk_dist.items():
                # Coverage: reward overlap with query
                if token in query_dist:
                    coverage_score += prob * query_dist[token]

                # Redundancy: penalize repeated tokens across chunks
                if token in seen_tokens:
                    redundancy_penalty += prob * seen_tokens[token]
                else:
                    seen_tokens[token] = prob

        return alpha * coverage_score - beta * redundancy_penalty

    def _run_ga(
        self,
        query_dist: SparseProb,
        candidate_dists: List[SparseProb],
        k: int,
    ) -> List[int]:
        """
        Run genetic algorithm on candidate pool.

        Args:
            query_dist: Query token distribution
            candidate_dists: List of candidate chunk distributions
            k: Number of chunks to select

        Returns:
            List of indices (0 to len(candidates)-1) of best chunks
        """
        M = len(candidate_dists)
        if M <= k:
            return list(range(M))

        pop_size = self.evolution_config.population_size
        generations = self.evolution_config.max_generations
        mutation_rate = self.evolution_config.mutation_rate

        # Initialize population: random k indices
        population = [random.sample(range(M), k) for _ in range(pop_size)]

        best_genome = None
        best_fitness = float('-inf')

        for gen in range(generations):
            # Evaluate
            scored = []
            for genome in population:
                score = self._compute_fitness(genome, query_dist, candidate_dists)
                scored.append((score, genome))

                if score > best_fitness:
                    best_fitness = score
                    best_genome = genome[:]

            # Sort by fitness
            scored.sort(key=lambda x: x[0], reverse=True)

            # Selection: top 50% survive
            survivors = [s[1] for s in scored[:pop_size // 2]]

            # Create new population with elitism
            new_pop = [g[:] for g in survivors]

            # Fill with mutated offspring
            while len(new_pop) < pop_size:
                parent = random.choice(survivors)
                child = parent[:]

                # Mutation: swap one chunk
                if random.random() < mutation_rate:
                    idx_to_replace = random.randint(0, k - 1)
                    new_candidate = random.randint(0, M - 1)

                    attempts = 0
                    while new_candidate in child and attempts < 10:
                        new_candidate = random.randint(0, M - 1)
                        attempts += 1

                    if new_candidate not in child:
                        child[idx_to_replace] = new_candidate

                new_pop.append(child)

            population = new_pop

        return best_genome if best_genome else population[0]

    def retrieve(self, query: str, limit: int = 5) -> List[str]:
        """
        Two-stage retrieval:
        1. Vector search: get top-N candidates
        2. GA: select optimal k from candidates

        Args:
            query: User query
            limit: Number of chunks to return

        Returns:
            Selected chunks
        """
        # Stage 1: Vector search for candidates
        candidates = self.vector_search.search(query, limit=self.n_candidates)

        if not candidates:
            return []

        if len(candidates) <= limit:
            return [c[1] for c in candidates]  # text is at index 1

        # Build sparse distributions
        # candidates format: (id, text, score, metadata)
        query_dist = self._text_to_sparse_dist(query)
        candidate_texts = [c[1] for c in candidates]  # text is at index 1
        candidate_dists = [self._text_to_sparse_dist(text) for text in candidate_texts]

        # Stage 2: GA optimization
        k = self.evolution_config.k_initial
        best_indices = self._run_ga(query_dist, candidate_dists, k)

        selected = [candidate_texts[i] for i in best_indices]

        logger.info(
            f"Two-stage: {self.n_candidates} candidates -> {len(selected)} selected via GA"
        )

        return selected
