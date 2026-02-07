"""
Integrazione tra VectorDatabase e sistema evolutivo.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import List, Tuple

from core.db import VectorDatabase

from .evolution import EvolutionConfig, EvolutionResult, evolve_chunks


def get_all_chunks(db: VectorDatabase) -> List[str]:
    """
    Recupera tutti i chunk (testi) dal vector database.

    Args:
        db: Istanza di VectorDatabase connessa

    Returns:
        Lista di testi (chunk)
    """
    conn = db._ensure_connection()
    with conn.cursor() as cur:
        cur.execute(f"SELECT text FROM {db.table_name} ORDER BY id;")
        results = cur.fetchall()

    return [row[0] for row in results]


def evolve_from_db(
    db: VectorDatabase,
    query: str,
    config: EvolutionConfig = None,
    verbose: bool = True
) -> EvolutionResult:
    """
    Esegue evolutionary prompting sui chunk del vector database.

    Args:
        db: VectorDatabase connessa
        query: Query di riferimento
        config: Configurazione evoluzione (opzionale)
        verbose: Stampa progresso

    Returns:
        EvolutionResult con i chunk migliori selezionati
    """
    # Recupera tutti i chunk
    chunks = get_all_chunks(db)

    if not chunks:
        raise ValueError("Vector database vuoto, nessun chunk disponibile")

    if verbose:
        print(f"Recuperati {len(chunks)} chunk dal database")
        print(f"Query: {query}\n")

    # Esegui evoluzione
    result = evolve_chunks(chunks, query, config, verbose)

    return result


def evolve_from_search_results(
    search_results: List[Tuple],
    query: str,
    config: EvolutionConfig = None,
    verbose: bool = True
) -> EvolutionResult:
    """
    Esegue evolutionary prompting sui risultati di una search.

    Args:
        search_results: Risultati da db.search_similar() - List[(id, text, similarity, metadata)]
        query: Query di riferimento
        config: Configurazione evoluzione
        verbose: Stampa progresso

    Returns:
        EvolutionResult
    """
    # Estrai solo i testi dai risultati
    chunks = [result[1] for result in search_results]  # result[1] = text

    if not chunks:
        raise ValueError("Nessun risultato dalla search")

    if verbose:
        print(f"Ottimizzazione su {len(chunks)} chunk dalla search")
        print(f"Query: {query}\n")

    return evolve_chunks(chunks, query, config, verbose)
