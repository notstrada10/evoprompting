from typing import List

import numpy as np

from .distribution import ChunkDistributions, Distribution
from .genome import Genome


def kl_divergence(p: Distribution, q: Distribution, epsilon: float = 1e-10) -> float:
    """KL(P || Q). Lower = more similar."""
    p_smooth = p.probs + epsilon
    q_smooth = q.probs + epsilon
    p_smooth /= p_smooth.sum()
    q_smooth /= q_smooth.sum()
    return float(np.sum(p_smooth * np.log(p_smooth / q_smooth)))


def information_gain(query_dist: Distribution, genome_dist: Distribution, epsilon: float = 1e-10) -> float:
    """Inverse KL. Higher = better match to query."""
    kl = kl_divergence(query_dist, genome_dist, epsilon)
    return 1.0 / (1.0 + kl)


def pairwise_diversity(distributions: List[Distribution], epsilon: float = 1e-10) -> float:
    """Average Jensen-Shannon divergence between all pairs. Higher = more diverse."""
    if len(distributions) < 2:
        return 0.0

    total = 0.0
    n_pairs = 0
    for i in range(len(distributions)):
        for j in range(i + 1, len(distributions)):
            p = distributions[i].probs + epsilon
            q = distributions[j].probs + epsilon
            p /= p.sum()
            q /= q.sum()
            m = (p + q) / 2
            js = 0.5 * np.sum(p * np.log(p / m)) + 0.5 * np.sum(q * np.log(q / m))
            total += js
            n_pairs += 1

    return total / n_pairs if n_pairs > 0 else 0.0


class FitnessEvaluator:
    """fitness = alpha * relevance + beta * diversity"""

    def __init__(
        self,
        chunk_distributions: ChunkDistributions,
        query_distribution: Distribution,
        alpha: float = 0.7,
        beta: float = 0.3,
    ):
        self.chunk_distributions = chunk_distributions
        self.query_distribution = query_distribution
        self.alpha = alpha
        self.beta = beta

    def evaluate(self, genome: Genome) -> float:
        genome_dist = self.chunk_distributions.get_combined(genome.bits)
        relevance = information_gain(self.query_distribution, genome_dist)

        selected_dists = [
            d for d, m in zip(self.chunk_distributions.distributions, genome.bits) if m == 1
        ]
        diversity = pairwise_diversity(selected_dists)
        diversity_normalized = min(diversity / 0.693, 1.0)  # JS max = ln(2)

        fitness = self.alpha * relevance + self.beta * diversity_normalized
        genome.fitness = fitness
        return fitness

    def evaluate_population(self, genomes: list[Genome]) -> None:
        for g in genomes:
            self.evaluate(g)
