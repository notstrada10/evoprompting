"""Run evolutionary prompting on the vector database."""
from ..core.db import VectorDatabase
from .evolution import EvolutionConfig
from .integration import evolve_from_db


def main():
    config = EvolutionConfig(
        population_size=20,
        k_initial=5,
        max_generations=100,
        mutation_rate=0.15,
        crossover_type="single_point",
    )

    db = VectorDatabase()
    db.connect()

    n_chunks = db.count()
    print(f"Chunks in DB: {n_chunks}")
    if n_chunks == 0:
        print("Database is empty.")
        return

    query = input("Enter query: ").strip() or "How does machine learning work?"

    result = evolve_from_db(db, query, config, verbose=True)

    print(f"\nSelected chunks ({len(result.selected_chunks)}):")
    for i, (idx, chunk) in enumerate(zip(result.selected_indices, result.selected_chunks), 1):
        preview = chunk[:200] + "..." if len(chunk) > 200 else chunk
        print(f"\n{i}. [ID {idx}] {preview}")

    print(f"\nFitness: {result.best_genome.fitness:.4f}")
    print(f"Generations: {result.generations_run}")
    db.close()


if __name__ == "__main__":
    main()
