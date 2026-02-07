"""
Evolutionary Prompting System

Sistema di ottimizzazione chunk tramite Genetic Algorithm.
Massimizza l'Information Gain rispetto a una query.
"""

from .distribution import ChunkDistributions, Distribution
from .evolution import Evolution, EvolutionConfig, EvolutionResult, evolve_chunks
from .fitness import FitnessEvaluator, information_gain, kl_divergence
from .ga import GeneticAlgorithm
from .genome import Genome
from .tokenizer import Tokenizer

__all__ = [
    "Tokenizer",
    "Distribution",
    "ChunkDistributions",
    "Genome",
    "FitnessEvaluator",
    "information_gain",
    "kl_divergence",
    "GeneticAlgorithm",
    "Evolution",
    "EvolutionConfig",
    "EvolutionResult",
    "evolve_chunks",
]
