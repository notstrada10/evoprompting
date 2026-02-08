"""Standalone test for evolutionary prompting."""
import random
from typing import List

import numpy as np

from .distribution import ChunkDistributions, Distribution
from .fitness import information_gain
from .genome import Genome
from .tokenizer import Tokenizer


def main():
    chunks = [
        "Python is an interpreted programming language",
        "Machine learning requires large amounts of data",
        "Neural networks are inspired by the human brain",
        "Margherita pizza is an Italian dish",
        "Genetic algorithms simulate natural evolution",
        "Deep learning uses deep neural networks",
        "Rome is the capital of Italy",
        "Optimization seeks the minimum of a function",
        "Transformers revolutionized NLP",
        "Pasta is a staple food in Italian cuisine",
    ]

    query = "How do evolutionary algorithms work in machine learning?"
    print(f"Query: {query}")
    print(f"Chunks: {len(chunks)}")

    tokenizer = Tokenizer.get_instance()
    chunk_dists = ChunkDistributions(chunks, tokenizer)
    query_dist = Distribution.from_text(query, tokenizer)

    pop_size, k, max_gen = 20, 5, 50
    population = [Genome.random(len(chunks), k) for _ in range(pop_size)]

    for g in population:
        g.fitness = information_gain(query_dist, chunk_dists.get_combined(g.bits))

    print(f"\n{'Gen':<5} {'Best':<10} {'Avg':<10} {'Chunks':<8}")
    print("-" * 35)

    for gen in range(max_gen):
        p1, p2 = random.sample(population, 2)

        # Crossover
        point = random.randint(1, len(p1.bits) - 1)
        child_bits = np.concatenate([p1.bits[:point], p2.bits[point:]])

        # Adjust to exactly k chunks
        current = int(child_bits.sum())
        if current > k:
            ones = np.where(child_bits == 1)[0]
            child_bits[np.random.choice(ones, size=current - k, replace=False)] = 0
        elif current < k:
            zeros = np.where(child_bits == 0)[0]
            child_bits[np.random.choice(zeros, size=k - current, replace=False)] = 1

        child = Genome(bits=child_bits)

        # Balanced mutation: swap one on for one off
        for _ in range(len(child.bits)):
            if random.random() < 0.15:
                ones = np.where(child.bits == 1)[0]
                zeros = np.where(child.bits == 0)[0]
                if len(ones) > 0 and len(zeros) > 0:
                    child.bits[np.random.choice(ones)] = 0
                    child.bits[np.random.choice(zeros)] = 1

        child.fitness = information_gain(query_dist, chunk_dists.get_combined(child.bits))

        worst = min(p1, p2, key=lambda g: g.fitness)
        if child.fitness > worst.fitness:
            population[population.index(worst)] = child

        if gen % 10 == 0:
            fits = [g.fitness for g in population]
            avg_sel = np.mean([g.n_selected for g in population])
            print(f"{gen:<5} {max(fits):<10.4f} {np.mean(fits):<10.4f} {avg_sel:<8.1f}")

    best = max(population, key=lambda g: g.fitness)
    print(f"\nSelected chunks ({best.n_selected}):")
    for idx in best.selected_indices:
        print(f"  [{idx}] {chunks[idx]}")
    print(f"\nFitness: {best.fitness:.4f}")


if __name__ == "__main__":
    main()
