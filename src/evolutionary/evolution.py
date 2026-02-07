from dataclasses import dataclass
from typing import Callable, List, Optional

import numpy as np

from .distribution import ChunkDistributions, Distribution
from .fitness import FitnessEvaluator
from .ga import GeneticAlgorithm
from .genome import Genome
from .tokenizer import Tokenizer


@dataclass
class EvolutionConfig:
    """Configurazione per l'evoluzione."""
    population_size: int = 20
    k_initial: int = 5
    max_generations: int = 100
    mutation_rate: float = 0.1
    crossover_type: str = "single_point"
    model_name: str = "deepseek-ai/DeepSeek-V3"
    # Fitness weights
    alpha: float = 0.7  # Weight for relevance (query matching)
    beta: float = 0.3   # Weight for diversity (chunk diversity)


@dataclass
class EvolutionResult:
    """Risultato dell'evoluzione."""
    best_genome: Genome
    selected_chunks: List[str]
    selected_indices: List[int]
    fitness_history: List[float]
    generations_run: int


class Evolution:
    """Orchestratore principale per l'evolutionary prompting."""

    def __init__(self, config: EvolutionConfig = None):
        self.config = config or EvolutionConfig()
        self.tokenizer = Tokenizer.get_instance(self.config.model_name)
        self.ga = GeneticAlgorithm(
            population_size=self.config.population_size,
            k_initial=self.config.k_initial,
            mutation_rate=self.config.mutation_rate,
            crossover_type=self.config.crossover_type
        )

        self.chunk_distributions: Optional[ChunkDistributions] = None
        self.query_distribution: Optional[Distribution] = None
        self.fitness_evaluator: Optional[FitnessEvaluator] = None
        self.population: List[Genome] = []
        self.fitness_history: List[float] = []

    def setup(self, chunks: List[str], query: str) -> None:
        """Inizializza il sistema con chunks e query."""
        # Crea distribuzioni per tutti i chunk
        self.chunk_distributions = ChunkDistributions(chunks, self.tokenizer)

        # Crea distribuzione per la query
        self.query_distribution = Distribution.from_text(query, self.tokenizer)

        # Crea valutatore fitness
        self.fitness_evaluator = FitnessEvaluator(
            self.chunk_distributions,
            self.query_distribution,
            alpha=self.config.alpha,
            beta=self.config.beta,
        )

        # Inizializza popolazione
        self.population = self.ga.initialize_population(len(chunks))

        # Valuta fitness iniziale
        self.fitness_evaluator.evaluate_population(self.population)

        # Reset history
        self.fitness_history = []

    def run(
        self,
        callback: Optional[Callable[[int, dict], None]] = None,
        early_stop_threshold: float = 0.999
    ) -> EvolutionResult:
        """
        Esegue l'evoluzione per max_generations.

        Args:
            callback: Funzione chiamata ogni generazione con (gen_number, stats)
            early_stop_threshold: Ferma se fitness > threshold

        Returns:
            EvolutionResult con il miglior genoma trovato
        """
        if self.chunk_distributions is None:
            raise RuntimeError("Chiamare setup() prima di run()")

        for gen in range(self.config.max_generations):
            # Evolvi
            self.ga.evolve_step(
                self.population,
                self.fitness_evaluator.evaluate
            )

            # Statistiche
            stats = self.ga.get_stats(self.population)
            self.fitness_history.append(stats["best_fitness"])

            # Callback
            if callback:
                callback(gen, stats)

            # Early stopping
            if stats["best_fitness"] >= early_stop_threshold:
                break

        # Risultato
        best = self.ga.get_best(self.population)
        selected_indices = best.selected_indices.tolist()
        selected_chunks = [
            self.chunk_distributions.chunks[i]
            for i in selected_indices
        ]

        return EvolutionResult(
            best_genome=best,
            selected_chunks=selected_chunks,
            selected_indices=selected_indices,
            fitness_history=self.fitness_history,
            generations_run=len(self.fitness_history)
        )


def evolve_chunks(
    chunks: List[str],
    query: str,
    config: EvolutionConfig = None,
    verbose: bool = True
) -> EvolutionResult:
    """
    Funzione di utilità per eseguire l'evoluzione.

    Args:
        chunks: Lista di chunk dal vector DB
        query: Query di riferimento
        config: Configurazione (opzionale)
        verbose: Se True, stampa progresso

    Returns:
        EvolutionResult
    """
    evo = Evolution(config)
    evo.setup(chunks, query)

    def print_progress(gen: int, stats: dict):
        if verbose and gen % 10 == 0:
            print(f"Gen {gen:3d}: best={stats['best_fitness']:.4f}, "
                  f"avg={stats['avg_fitness']:.4f}, "
                  f"chunks={stats['avg_selected']:.1f}")

    result = evo.run(callback=print_progress if verbose else None)

    if verbose:
        print(f"\nEvoluzione completata in {result.generations_run} generazioni")
        print(f"Best fitness: {result.best_genome.fitness:.4f}")
        print(f"Chunk selezionati: {len(result.selected_chunks)}")

    return result
