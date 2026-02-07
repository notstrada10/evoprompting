"""
Esempio: esegui evolutionary prompting sul tuo vector database.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.db import VectorDatabase
from evolutionary.evolution import EvolutionConfig
from evolutionary.integration import evolve_from_db, evolve_from_search_results


def main():
    # Configurazione
    config = EvolutionConfig(
        population_size=20,
        k_initial=5,
        max_generations=100,
        mutation_rate=0.15,
        crossover_type="single_point"
    )

    # Connetti al database
    print("Connessione al vector database...")
    db = VectorDatabase()
    db.connect()

    # Verifica numero chunk disponibili
    n_chunks = db.count()
    print(f"Chunk disponibili nel DB: {n_chunks}\n")

    if n_chunks == 0:
        print("⚠️  Database vuoto. Popola prima il vector DB.")
        return

    # Query
    query = input("Inserisci la tua query: ").strip()
    if not query:
        query = "Come funziona il machine learning?"

    print("\n" + "="*60)
    print("EVOLUTIONARY PROMPTING - Da tutto il DB")
    print("="*60 + "\n")

    # Metodo 1: Evoluzione su TUTTO il database
    result = evolve_from_db(db, query, config, verbose=True)

    print("\n" + "="*60)
    print("RISULTATI")
    print("="*60)
    print(f"\nChunk selezionati ({len(result.selected_chunks)}):")
    for i, (idx, chunk) in enumerate(zip(result.selected_indices, result.selected_chunks), 1):
        print(f"\n{i}. [ID {idx}]")
        print(f"   {chunk[:200]}..." if len(chunk) > 200 else f"   {chunk}")

    print(f"\nFitness finale: {result.best_genome.fitness:.4f}")
    print(f"Generazioni: {result.generations_run}")

    # Chiudi connessione
    db.close()


def demo_from_search():
    """Esempio alternativo: evoluzione sui top-K risultati di una search."""
    config = EvolutionConfig(
        population_size=15,
        k_initial=3,
        max_generations=50
    )

    db = VectorDatabase()
    db.connect()

    # Prima fai una search per ottenere candidati iniziali
    from core.embeddings import EmbeddingService

    query = "algoritmi genetici e machine learning"

    embedder = EmbeddingService()
    query_embedding = embedder.embed_text(query)

    # Top 20 chunk più simili
    search_results = db.search_similar(query_embedding, limit=20)

    print(f"\nTop-20 chunk dalla search, ora ottimizziamo con GA...")

    # Evolvi sui top-20
    result = evolve_from_search_results(search_results, query, config)

    print(f"\nChunk selezionati: {len(result.selected_chunks)}")
    for chunk in result.selected_chunks:
        print(f"  - {chunk[:100]}...")

    db.close()


if __name__ == "__main__":
    # Scegli quale eseguire:

    # Opzione 1: Evoluzione su tutto il DB
    main()

    # Opzione 2: Evoluzione sui risultati di una search
    # demo_from_search()
