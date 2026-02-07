"""
Evolutionary RAG System.

Offline genetic algorithm for chunk selection using sparse probability distributions.
Fitness = α * coverage (negative KL) + β * entropy (diversity)
"""
import logging
import random
from typing import List, Optional

from ..evolutionary.distribution import ChunkDistributions, Distribution
from ..evolutionary.evolution import EvolutionConfig
from ..evolutionary.tokenizer import Tokenizer
from .rag import RAGSystem

logger = logging.getLogger(__name__)


class EvolutionaryRAGSystem(RAGSystem):
    """
    Evolutionary RAG using sparse token distributions.

    Fitness function:
    - Coverage: -KL(query || mixture) - how well chunks cover the query
    - Diversity: H(mixture) - entropy of combined chunks

    Fitness = α * coverage + β * diversity
    """

    def __init__(
        self,
        model: str | None = None,
        table_name: str | None = None,
        evolution_config: EvolutionConfig | None = None,
    ):
        super().__init__(model=model, table_name=table_name)

        self.evolution_config = evolution_config or EvolutionConfig(
            population_size=100,
            k_initial=20,
            max_generations=100,
            mutation_rate=0.7,  # High mutation for exploration
            alpha=0.7,  # coverage weight
            beta=0.3,   # diversity/entropy weight
        )

        self._tokenizer: Optional[Tokenizer] = None
        self._chunk_distributions: Optional[ChunkDistributions] = None
        self._query_distribution: Optional[Distribution] = None

    @property
    def tokenizer(self) -> Tokenizer:
        if self._tokenizer is None:
            self._tokenizer = Tokenizer.get_instance(self.evolution_config.model_name)
        return self._tokenizer

    def _compute_fitness(self, genome: List[int]) -> float:
        """
        Compute fitness: α * coverage + β * entropy.

        - Coverage: -KL(query || mixture) (negative because we minimize divergence)
        - Diversity: H(mixture) - entropy of the combined distribution
        """
        if not genome:
            return float('-inf')

        # Get mixture distribution of selected chunks
        mixture = self._chunk_distributions.get_combined(genome)

        if not mixture.probs:
            return float('-inf')

        # Coverage: negative KL divergence (we want to minimize distance to query)
        kl = self._query_distribution.kl_divergence(mixture)
        coverage = -kl

        # Diversity: entropy of the mixture (high entropy = diverse vocabulary)
        entropy = mixture.entropy()

        alpha = self.evolution_config.alpha
        beta = self.evolution_config.beta

        return alpha * coverage + beta * entropy

    def evolve_chunks(self, chunks: List[str], query: str, k: int) -> List[str]:
        """
        Evolve population to find best k chunks.

        Args:
            chunks: All candidate chunks.
            query: User query.
            k: Number of chunks to select.

        Returns:
            Selected chunks after evolution.
        """
        n_chunks = len(chunks)
        if n_chunks <= k:
            return chunks

        # Build sparse distributions
        logger.info(f"Building sparse distributions for {n_chunks} chunks...")
        self._chunk_distributions = ChunkDistributions(chunks, self.tokenizer)
        self._query_distribution = Distribution.from_text(query, self.tokenizer)

        pop_size = self.evolution_config.population_size
        generations = self.evolution_config.max_generations
        mutation_rate = self.evolution_config.mutation_rate

        # Initialize population: random k indices per genome
        population = [random.sample(range(n_chunks), k) for _ in range(pop_size)]

        best_genome = None
        best_fitness = float('-inf')

        for gen in range(generations):
            # Evaluate all genomes
            scored_pop = []
            for genome in population:
                score = self._compute_fitness(genome)
                scored_pop.append((score, genome))

                if score > best_fitness:
                    best_fitness = score
                    best_genome = genome

            # Sort by fitness (descending)
            scored_pop.sort(key=lambda x: x[0], reverse=True)

            # Selection: top 50% survive
            survivors = [x[1] for x in scored_pop[:pop_size // 2]]

            # Create new population with elitism
            new_pop = [genome[:] for genome in survivors]  # Keep survivors

            # Fill rest with mutated offspring
            while len(new_pop) < pop_size:
                parent = random.choice(survivors)
                child = parent[:]

                # Mutation: swap one chunk with a random new one
                if random.random() < mutation_rate:
                    idx_to_replace = random.randint(0, k - 1)
                    new_candidate = random.randint(0, n_chunks - 1)

                    # Ensure uniqueness
                    attempts = 0
                    while new_candidate in child and attempts < 10:
                        new_candidate = random.randint(0, n_chunks - 1)
                        attempts += 1

                    if new_candidate not in child:
                        child[idx_to_replace] = new_candidate

                new_pop.append(child)

            population = new_pop

        # Return best chunks
        if best_genome is None:
            best_genome = population[0]

        selected_chunks = [chunks[i] for i in best_genome]

        logger.info(
            f"Evolution: {n_chunks} chunks, {generations} gens -> "
            f"{len(selected_chunks)} selected (fitness={best_fitness:.4f})"
        )

        return selected_chunks

    def retrieve(self, query: str, limit: int = 5) -> List[str]:
        """
        Evolutionary retrieval over entire KB.

        Args:
            query: User query.
            limit: Not used (k_initial from config is used).

        Returns:
            Selected chunks.
        """
        db = self.vector_search.db
        all_chunks = db.get_all_chunks()

        if not all_chunks:
            return []

        chunks = [text for (id, text) in all_chunks]

        selected = self.evolve_chunks(chunks, query, k=self.evolution_config.k_initial)

        return selected
