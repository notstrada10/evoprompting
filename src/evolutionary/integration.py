"""Integration between VectorDatabase and the evolutionary system."""
from typing import List, Tuple

from ..core.db import VectorDatabase
from .evolution import EvolutionConfig, EvolutionResult, evolve_chunks


def get_all_chunks(db: VectorDatabase) -> List[str]:
    conn = db.ensure_connection()
    with conn.cursor() as cur:
        cur.execute(f"SELECT text FROM {db.table_name} ORDER BY id;")
        return [row[0] for row in cur.fetchall()]


def evolve_from_db(
    db: VectorDatabase, query: str, config: EvolutionConfig = None, verbose: bool = True
) -> EvolutionResult:
    chunks = get_all_chunks(db)
    if not chunks:
        raise ValueError("Database is empty")
    if verbose:
        print(f"Retrieved {len(chunks)} chunks from database")
    return evolve_chunks(chunks, query, config, verbose)


def evolve_from_search_results(
    search_results: List[Tuple], query: str, config: EvolutionConfig = None, verbose: bool = True
) -> EvolutionResult:
    chunks = [r[1] for r in search_results]
    if not chunks:
        raise ValueError("No search results")
    if verbose:
        print(f"Optimizing {len(chunks)} chunks from search")
    return evolve_chunks(chunks, query, config, verbose)
