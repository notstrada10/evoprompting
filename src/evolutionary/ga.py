import random
from typing import List

import numpy as np

from .genome import Genome


def crossover_single_point(p1: Genome, p2: Genome) -> Genome:
    n = len(p1)
    point = random.randint(1, n - 1)
    child_bits = np.concatenate([p1.bits[:point], p2.bits[point:]])
    return Genome(bits=child_bits)


def crossover_uniform(p1: Genome, p2: Genome) -> Genome:
    n = len(p1)
    mask = np.random.randint(0, 2, size=n)
    child_bits = np.where(mask == 0, p1.bits, p2.bits)
    return Genome(bits=child_bits.astype(np.int8))


def mutate_bitflip(genome: Genome, mutation_rate: float = 0.1) -> None:
    for i in range(len(genome)):
        if random.random() < mutation_rate:
            genome.bits[i] = 1 - genome.bits[i]


def tournament_select(population: List[Genome], size: int = 3) -> Genome:
    contestants = random.sample(population, min(size, len(population)))
    return max(contestants, key=lambda g: g.fitness)


class GeneticAlgorithm:

    def __init__(
        self,
        population_size: int = 20,
        k_initial: int = 5,
        mutation_rate: float = 0.1,
        crossover_type: str = "single_point",
    ):
        self.population_size = population_size
        self.k_initial = k_initial
        self.mutation_rate = mutation_rate
        self.crossover_fn = (
            crossover_single_point if crossover_type == "single_point" else crossover_uniform
        )

    def initialize_population(self, n_chunks: int) -> List[Genome]:
        return [Genome.random(n_chunks, self.k_initial) for _ in range(self.population_size)]

    def evolve_step(self, population: List[Genome], fitness_fn) -> bool:
        """Select 2 parents, crossover, mutate, replace worse parent if child is better."""
        p1, p2 = random.sample(population, 2)
        child = self.crossover_fn(p1, p2)
        mutate_bitflip(child, self.mutation_rate)
        fitness_fn(child)

        worst = min(p1, p2, key=lambda g: g.fitness)
        if child.fitness > worst.fitness:
            population[population.index(worst)] = child
            return True
        return False

    def get_best(self, population: List[Genome]) -> Genome:
        return max(population, key=lambda g: g.fitness)

    def get_stats(self, population: List[Genome]) -> dict:
        fitnesses = [g.fitness for g in population]
        return {
            "best_fitness": max(fitnesses),
            "avg_fitness": np.mean(fitnesses),
            "worst_fitness": min(fitnesses),
            "std_fitness": np.std(fitnesses),
            "avg_selected": np.mean([g.n_selected for g in population]),
        }
