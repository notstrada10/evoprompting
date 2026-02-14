"""
Information-Gain Evolutionary RAG: BM25 Pre-filter + Token-Level IG GA Selection.

Extends BayesianEvolutionaryRAGSystem with token-level Information Gain that
measures how much NEW information each chunk brings given what we already have.

The key idea — precomputed ONCE before the GA loop:

1. Per-chunk IG: for each candidate chunk, how many new IDF-weighted tokens
   does it introduce that the query doesn't already contain? Weighted by
   semantic relevance so irrelevant chunks with exotic tokens don't score high.

   ig[i] = sim(chunk_i, query) * sum(idf[t] for t in chunk_i if t not in query)

2. Pairwise IG overlap: for each pair (i,j), how much of their new-token
   information is shared? If two chunks introduce the same rare tokens,
   picking both is redundant — one already covers the other.

   overlap[i][j] = cosine(new_tokens_i, new_tokens_j)  (IDF-weighted)

The GA fitness simulates a chain for each genome:
  - Sort chunks by query relevance (most relevant first)
  - First chunk gets its full IG
  - Each subsequent chunk: marginal_ig = ig[chunk] - avg_overlap(chunk, prev)
  - Total chain IG = sum of marginal IGs

This is O(k^2) per genome — just precomputed array lookups. Fast.

Fitness = ig_weight * ChainIG + SemanticRelevance + beta * Bridging
         - gamma * EmbedRedundancy
"""
import logging
import time

import numpy as np

from .bayesian_evolutionary_rag import BayesianEvolutionaryRAGSystem

logger = logging.getLogger(__name__)


class IGEvolutionaryRAGSystem(BayesianEvolutionaryRAGSystem):
    """Evolutionary RAG with token-level Information Gain fitness."""

    def __init__(self, model=None, table_name=None, evolution_config=None):
        super().__init__(model=model, table_name=table_name, evolution_config=evolution_config)

        # Precomputed IG data (set by retrieve, read by fitness_batch)
        self._chunk_ig = None           # (n_candidates,) per-chunk IG score
        self._ig_overlap = None         # (n_candidates, n_candidates) shared new-token info

    # =========================================================================
    # Precompute IG tables (runs ONCE before GA)
    # =========================================================================

    def precompute_ig_tables(
        self,
        query_vec: np.ndarray,
        dense_chunk_matrix: np.ndarray,
        idf_weights: np.ndarray,
        query_cand_sims: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute per-chunk IG and pairwise new-token overlap.

        Per-chunk IG:
          For each chunk, find tokens it has that the query DOESN'T have.
          Weight those new tokens by IDF (rare tokens = more informative).
          Multiply by semantic relevance (chunk must be on-topic to count).

        Pairwise overlap:
          For each pair (i,j), cosine similarity of their IDF-weighted
          new-token vectors. High overlap = they bring the same new info.

        Returns:
            chunk_ig: (n_candidates,) normalized [0, 1]
            ig_overlap: (n_candidates, n_candidates) normalized [0, 1]
        """
        n = dense_chunk_matrix.shape[0]

        # Which tokens does the query already have?
        query_mask = (query_vec > 0).astype(np.float32)  # (D,)

        # For each chunk: presence of tokens NOT in the query
        chunk_presence = (dense_chunk_matrix > 0).astype(np.float32)  # (n, D)
        new_tokens = chunk_presence * (1.0 - query_mask[None, :])      # (n, D)

        # IDF-weighted new-token vectors
        new_weighted = new_tokens * idf_weights[None, :]  # (n, D)

        # Per-chunk IG = sum of IDF-weighted new tokens * semantic relevance
        raw_ig = new_weighted.sum(axis=1)  # (n,)
        # Weight by relevance: relevant chunks with new info score highest
        chunk_ig = raw_ig * np.maximum(query_cand_sims, 0.0)

        # Normalize to [0, 1]
        ig_max = chunk_ig.max()
        if ig_max > 1e-8:
            chunk_ig /= ig_max
        else:
            chunk_ig = np.zeros(n, dtype=np.float32)

        # Pairwise overlap: cosine similarity of new-token vectors
        norms = np.linalg.norm(new_weighted, axis=1, keepdims=True) + 1e-10
        normed = new_weighted / norms
        ig_overlap = (normed @ normed.T).astype(np.float32)
        np.fill_diagonal(ig_overlap, 0.0)

        return chunk_ig.astype(np.float32), ig_overlap

    # =========================================================================
    # Fitness (overrides parent)
    # =========================================================================

    def fitness_batch(
        self,
        population: np.ndarray,
        query_cand_sims: np.ndarray,
        embed_sim_matrix: np.ndarray,
        bridge_matrix: np.ndarray,
    ) -> np.ndarray:
        """Fitness with chain-based marginal IG.

        For each genome, simulates adding chunks one by one (sorted by
        query relevance). Each chunk's marginal IG = its base IG minus
        average overlap with chunks already in the set.

        O(k^2) per genome — just array lookups, no heavy math.
        """
        pop_size, k = population.shape

        # ---- Semantic relevance ----
        sims = query_cand_sims[population]  # (pop_size, k)
        relevance = sims.mean(axis=1)

        # ---- Bridging & embedding redundancy (from parent) ----
        if k < 2:
            bridging = np.zeros(pop_size, dtype=np.float32)
            embed_redundancy = np.zeros(pop_size, dtype=np.float32)
        else:
            n_pairs = k * (k - 1) / 2
            row_idx = population[:, :, None]
            col_idx = population[:, None, :]
            sub_bridge = bridge_matrix[row_idx, col_idx]
            sub_embed_sim = embed_sim_matrix[row_idx, col_idx]
            triu_mask = np.triu(np.ones((k, k), dtype=bool), k=1)
            bridging = sub_bridge[:, triu_mask].sum(axis=1) / n_pairs
            embed_redundancy = sub_embed_sim[:, triu_mask].sum(axis=1) / n_pairs

        # ---- Chain-based marginal IG ----
        if self._chunk_ig is not None and self._ig_overlap is not None and k >= 2:
            # Sort each genome by query relevance (most relevant first)
            sort_order = np.argsort(-sims, axis=1)
            rows = np.arange(pop_size)[:, None]
            sorted_pop = population[rows, sort_order]  # (pop_size, k)

            # Base IG scores for all chunks in each genome
            base_ig = self._chunk_ig[sorted_pop]  # (pop_size, k)

            # Compute marginal IG along the chain
            marginal_ig = np.zeros((pop_size, k), dtype=np.float32)
            marginal_ig[:, 0] = base_ig[:, 0]  # first chunk gets full IG

            for step in range(1, k):
                current_chunk = sorted_pop[:, step]  # (pop_size,)
                # Average overlap with all previously selected chunks
                prev_chunks = sorted_pop[:, :step]  # (pop_size, step)
                overlaps = self._ig_overlap[
                    current_chunk[:, None].repeat(step, axis=1),
                    prev_chunks,
                ]  # (pop_size, step)
                avg_overlap = overlaps.mean(axis=1)  # (pop_size,)
                marginal_ig[:, step] = np.maximum(base_ig[:, step] - avg_overlap, 0.0)

            chain_ig = marginal_ig.sum(axis=1)  # (pop_size,)

            # Normalize across population
            ig_min, ig_max = chain_ig.min(), chain_ig.max()
            if ig_max - ig_min > 1e-8:
                chain_ig = (chain_ig - ig_min) / (ig_max - ig_min)
            else:
                chain_ig = np.zeros(pop_size, dtype=np.float32)
        else:
            chain_ig = np.zeros(pop_size, dtype=np.float32)

        return (
            relevance
            + self.beta * bridging
            + self.ig_weight * chain_ig
            - self.gamma * embed_redundancy
        ).astype(np.float32)

    # =========================================================================
    # Retrieve
    # =========================================================================

    def retrieve(self, query: str, limit: int = 5) -> list[str]:
        """Two-stage: BM25 candidates -> chain-IG GA selection."""
        t0 = time.perf_counter()

        candidate_texts, candidate_embs = self.bm25_search(query, self.n_candidates)
        if not candidate_texts:
            return []

        k = self.evolution_config.k_initial
        if len(candidate_texts) <= k:
            return candidate_texts

        t1 = time.perf_counter()

        # Embeddings
        query_emb = self.embed_query(query)
        query_cand_sims = self.build_query_candidate_similarities(query_emb, candidate_embs)
        embed_sim_matrix = self.build_candidate_similarity_matrix(candidate_embs)

        # Token matrices
        chunk_matrix, query_vec = self.build_token_matrices(query, candidate_texts)
        idf_weights = self.build_idf_weights(chunk_matrix)
        bridge_matrix = self.build_rare_token_bridge_matrix(chunk_matrix, idf_weights)

        # Precompute IG tables ONCE
        dense_chunk_matrix = chunk_matrix.toarray().astype(np.float32)
        self._chunk_ig, self._ig_overlap = self.precompute_ig_tables(
            query_vec, dense_chunk_matrix, idf_weights, query_cand_sims,
        )

        t2 = time.perf_counter()

        best_indices = self.run_ga(
            query_cand_sims, embed_sim_matrix, bridge_matrix, k,
        )

        t3 = time.perf_counter()

        # Clean up
        self._chunk_ig = None
        self._ig_overlap = None

        selected = [candidate_texts[i] for i in best_indices]
        logger.info(
            f"IG-Evolutionary: BM25 {len(candidate_texts)} -> GA {len(selected)} | "
            f"bm25={t1-t0:.3f}s build={t2-t1:.3f}s GA={t3-t2:.3f}s"
        )
        return selected
