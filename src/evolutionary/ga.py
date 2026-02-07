import random
from typing import List, Tuple

import numpy as np

from .genome import Genome


def crossover_single_point(parent1: Genome, parent2: Genome) -> Genome:
    """Single-point crossover: taglia in un punto e combina."""
    n = len(parent1)
    point = random.randint(1, n - 1)

    child_bits = np.concatenate([
        parent1.bits[:point],
        parent2.bits[point:]
    ])

    return Genome(bits=child_bits)


def crossover_uniform(parent1: Genome, parent2: Genome) -> Genome:
    """Uniform crossover: ogni bit scelto random da un genitore."""
    n = len(parent1)
    mask = np.random.randint(0, 2, size=n)

    child_bits = np.where(mask == 0, parent1.bits, parent2.bits)

    return Genome(bits=child_bits.astype(np.int8))


def mutate_bitflip(genome: Genome, mutation_rate: float = 0.1) -> None:
    """Bit-flip mutation: ogni bit ha probabilità mutation_rate di essere flippato."""
    for i in range(len(genome)):
        if random.random() < mutation_rate:
            genome.bits[i] = 1 - genome.bits[i]


def select_parents(population: List[Genome]) -> Tuple[Genome, Genome]:
    """Seleziona 2 genitori random dalla popolazione."""
    return tuple(random.sample(population, 2))


def tournament_select(population: List[Genome], tournament_size: int = 3) -> Genome:
    """Tournament selection: scegli il migliore tra k random."""
    contestants = random.sample(population, min(tournament_size, len(population)))
    return max(contestants, key=lambda g: g.fitness)


class GeneticAlgorithm:
    """Genetic Algorithm con sostituzione genitori."""

    def __init__(
        self,
        population_size: int = 20,
        k_initial: int = 5,
        mutation_rate: float = 0.1,
        crossover_type: str = "single_point"
    ):
        self.population_size = population_size
        self.k_initial = k_initial
        self.mutation_rate = mutation_rate
        self.crossover_type = crossover_type

        self.crossover_fn = (
            crossover_single_point if crossover_type == "single_point"
            else crossover_uniform
        )

    def initialize_population(self, n_chunks: int) -> List[Genome]:
        """Crea popolazione iniziale con k chunk random per genoma."""
        return [
            Genome.random(n_chunks, self.k_initial)
            for _ in range(self.population_size)
        ]

    def evolve_step(self, population: List[Genome], fitness_fn) -> bool:
        """
        Un passo di evoluzione:
        1. Seleziona 2 genitori
        2. Crea figlio via crossover
        3. Muta figlio
        4. Valuta fitness figlio
        5. Sostituisci genitore peggiore se figlio è migliore

        Returns:
            True se c'è stata sostituzione, False altrimenti
        """
        # Seleziona genitori
        parent1, parent2 = select_parents(population)

        # Crossover
        child = self.crossover_fn(parent1, parent2)

        # Mutazione
        mutate_bitflip(child, self.mutation_rate)

        # Valuta fitness
        fitness_fn(child)

        # Confronta con genitore peggiore
        worst_parent = min(parent1, parent2, key=lambda g: g.fitness)

        if child.fitness > worst_parent.fitness:
            # Sostituisci
            idx = population.index(worst_parent)
            population[idx] = child
            return True

        return False

    def get_best(self, population: List[Genome]) -> Genome:
        """Restituisce il genoma con fitness più alta."""
        return max(population, key=lambda g: g.fitness)

    def get_stats(self, population: List[Genome]) -> dict:
        """Statistiche sulla popolazione."""
        fitnesses = [g.fitness for g in population]
        return {
            "best_fitness": max(fitnesses),
            "avg_fitness": np.mean(fitnesses),
            "worst_fitness": min(fitnesses),
            "std_fitness": np.std(fitnesses),
            "avg_selected": np.mean([g.n_selected for g in population])
        }
