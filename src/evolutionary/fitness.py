from typing import List

import numpy as np

from .distribution import ChunkDistributions, Distribution
from .genome import Genome


def kl_divergence(p: Distribution, q: Distribution, epsilon: float = 1e-10) -> float:
    """
    Calcola KL Divergence: D_KL(P || Q) = sum(P * log(P / Q))

    Misura quanta informazione si perde usando Q per approssimare P.
    Valori più alti = distribuzioni più diverse.

    Args:
        p: Distribuzione "vera" (query)
        q: Distribuzione approssimata (chunk combinati)
        epsilon: Smoothing per evitare log(0)

    Returns:
        KL divergence (>= 0, più basso = più simili)
    """
    # Smoothing per evitare divisioni per zero
    p_smooth = p.probs + epsilon
    q_smooth = q.probs + epsilon

    # Ri-normalizza dopo smoothing
    p_smooth /= p_smooth.sum()
    q_smooth /= q_smooth.sum()

    # KL divergence
    kl = np.sum(p_smooth * np.log(p_smooth / q_smooth))

    return float(kl)


def information_gain(query_dist: Distribution, genome_dist: Distribution, epsilon: float = 1e-10) -> float:
    """
    Calcola l'Information Gain come inverso della KL divergence.

    Usiamo -KL o 1/(1+KL) per avere una metrica da massimizzare:
    - KL basso = distribuzioni simili = alto information gain
    - KL alto = distribuzioni diverse = basso information gain

    Returns:
        Information gain (più alto = migliore)
    """
    kl = kl_divergence(query_dist, genome_dist, epsilon)
    # Trasforma in metrica da massimizzare
    return 1.0 / (1.0 + kl)


def pairwise_diversity(distributions: List[Distribution], epsilon: float = 1e-10) -> float:
    """
    Calcola la diversità media tra coppie di distribuzioni.

    Usa la Jensen-Shannon divergence (simmetrica) per misurare
    quanto sono diverse le distribuzioni tra loro.

    Args:
        distributions: Lista di distribuzioni dei chunk selezionati
        epsilon: Smoothing per evitare log(0)

    Returns:
        Diversità media (più alto = chunk più diversi tra loro)
    """
    if len(distributions) < 2:
        return 0.0

    total_divergence = 0.0
    n_pairs = 0

    for i in range(len(distributions)):
        for j in range(i + 1, len(distributions)):
            # Jensen-Shannon divergence (simmetrica)
            p = distributions[i].probs + epsilon
            q = distributions[j].probs + epsilon
            p /= p.sum()
            q /= q.sum()

            # M = (P + Q) / 2
            m = (p + q) / 2

            # JS = 0.5 * KL(P||M) + 0.5 * KL(Q||M)
            kl_pm = np.sum(p * np.log(p / m))
            kl_qm = np.sum(q * np.log(q / m))
            js = 0.5 * kl_pm + 0.5 * kl_qm

            total_divergence += js
            n_pairs += 1

    return total_divergence / n_pairs if n_pairs > 0 else 0.0


class FitnessEvaluator:
    """Valuta la fitness dei genomi rispetto a una query."""

    def __init__(
        self,
        chunk_distributions: ChunkDistributions,
        query_distribution: Distribution,
        alpha: float = 0.7,
        beta: float = 0.3,
    ):
        """
        Args:
            chunk_distributions: Distribuzioni di tutti i chunk
            query_distribution: Distribuzione della query
            alpha: Peso per la relevance (quanto i chunk matchano la query)
            beta: Peso per la diversity (quanto i chunk sono diversi tra loro)
        """
        self.chunk_distributions = chunk_distributions
        self.query_distribution = query_distribution
        self.alpha = alpha
        self.beta = beta

    def _get_selected_distributions(self, genome: Genome) -> List[Distribution]:
        """Restituisce le distribuzioni dei chunk selezionati."""
        return [
            d for d, m in zip(self.chunk_distributions.distributions, genome.bits)
            if m == 1
        ]

    def evaluate(self, genome: Genome) -> float:
        """
        Calcola fitness di un genoma come combinazione di relevance e diversity.

        fitness = alpha * relevance + beta * diversity

        - relevance: quanto i chunk combinati matchano la query
        - diversity: quanto i chunk selezionati sono diversi tra loro
        """
        # Relevance: quanto i chunk matchano la query
        genome_dist = self.chunk_distributions.get_combined(genome.bits)
        relevance = information_gain(self.query_distribution, genome_dist)

        # Diversity: quanto i chunk sono diversi tra loro
        selected_dists = self._get_selected_distributions(genome)
        diversity = pairwise_diversity(selected_dists)

        # Normalizza diversity a [0, 1] (JS divergence è in [0, ln(2)])
        # ln(2) ≈ 0.693
        diversity_normalized = min(diversity / 0.693, 1.0)

        # Fitness combinata
        fitness = self.alpha * relevance + self.beta * diversity_normalized
        genome.fitness = fitness
        return fitness

    def evaluate_population(self, genomes: list[Genome]) -> None:
        """Valuta fitness di tutta la popolazione."""
        for genome in genomes:
            self.evaluate(genome)
