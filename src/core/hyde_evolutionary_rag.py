"""
HyDE-Evolutionary Hybrid RAG: Hypothesis-Guided BM25 + IG-GA Selection.

Combines HyDE's answer-shaped retrieval with the evolutionary GA's
diversity/information-gain optimization. The hypothesis expands the
candidate pool AND steers the GA fitness.

Flow:
  1. LLM generates hypothetical answer document
  2. Dual BM25: query candidates + hypothesis candidates -> large pool (~500)
  3. GA evolves chunk subsets over this expanded pool
  4. Fitness = query_sim + hyp_sim + chain_ig + bridging - redundancy

Why bigger pool + GA matters:
  - BM25(query) pulls 200 chunks matching query terms
  - BM25(hypothesis) pulls 200 chunks matching predicted answer terms
  - Union gives ~350-400 unique candidates — much more than k=10
  - The GA must actively search this large pool for the best diverse subset
  - Hypothesis similarity in fitness steers toward answer-containing chunks
  - IG + bridging ensure multi-hop coverage

For multi-hop questions, the hypothesis mentions both entities, so both
biographical chunks appear in the candidate pool. The GA's job is to
pick the right combination that covers both hops with minimal redundancy.
"""
import asyncio
import logging
import random
import time

import numpy as np

from .ig_evolutionary_rag import IGEvolutionaryRAGSystem

logger = logging.getLogger(__name__)


class HyDEEvolutionaryRAGSystem(IGEvolutionaryRAGSystem):
    """HyDE + IG evolutionary GA with expanded candidate pool."""

    def __init__(self, model=None, table_name=None, evolution_config=None):
        super().__init__(model=model, table_name=table_name, evolution_config=evolution_config)
        self.hyp_weight = 0.3
        self.n_query_candidates = 150
        self.n_hyp_candidates = 150
        self.max_candidates = 300

        # Inherit parent weights: alpha=0.3, beta=0.3, gamma=0.2, ig_weight=0.2

        # Per-query state
        self._hyp_cand_sims = None
        self._blended_sims = None

    # =========================================================================
    # Hypothesis generation
    # =========================================================================

    async def async_generate_hypothesis(self, query: str) -> str:
        """Generate a hypothetical answer document via LLM."""
        system_prompt = (
            "You generate hypothetical document passages that would answer a question. "
            "Write a detailed, factual-sounding passage as it would appear in a knowledge base or encyclopedia. "
            "Be specific with names, dates, and details. Write directly without preamble."
        )
        user_prompt = f"Question: {query}\n\nPassage:"
        try:
            response = await self.async_llm.chat.completions.create(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                model=self.model,
                temperature=0.2,
                max_tokens=350,
            )
            return response.choices[0].message.content or query
        except Exception as e:
            logger.error(f"Hypothesis generation failed: {e}")
            return query

    def generate_hypothesis(self, query: str) -> str:
        """Sync wrapper for hypothesis generation."""
        return asyncio.run(self.async_generate_hypothesis(query))

    # =========================================================================
    # Dual BM25 retrieval (expanded pool)
    # =========================================================================

    def dual_bm25_search(
        self, query: str, hypothesis: str
    ) -> tuple[list[str], np.ndarray]:
        """BM25 with both query and hypothesis — larger candidate pool."""
        self._ensure_bm25()

        query_tokens = query.lower().split()
        hyp_tokens = hypothesis.lower().split()

        query_scores = self._bm25.get_scores(query_tokens)
        hyp_scores = self._bm25.get_scores(hyp_tokens)

        # Top indices from each
        query_top = set(np.argsort(query_scores)[::-1][:self.n_query_candidates].tolist())
        hyp_top = set(np.argsort(hyp_scores)[::-1][:self.n_hyp_candidates].tolist())

        # Union
        combined = list(query_top | hyp_top)

        # Cap at max_candidates by combined BM25 score
        if len(combined) > self.max_candidates:
            combined_arr = np.array(combined)
            blended = query_scores[combined_arr] + hyp_scores[combined_arr]
            top_idx = np.argpartition(blended, -self.max_candidates)[-self.max_candidates:]
            combined = combined_arr[top_idx].tolist()

        texts = [self._corpus_texts[i] for i in combined]
        embeddings = self._corpus_embeddings[np.array(combined)]
        return texts, embeddings

    # =========================================================================
    # Fitness (adds hypothesis term to IG parent)
    # =========================================================================

    def fitness_batch(
        self,
        population: np.ndarray,
        query_cand_sims: np.ndarray,
        embed_sim_matrix: np.ndarray,
        bridge_matrix: np.ndarray,
    ) -> np.ndarray:
        """Fitness with chain IG + hypothesis relevance."""
        base_fitness = super().fitness_batch(
            population, query_cand_sims, embed_sim_matrix, bridge_matrix,
        )

        if self._hyp_cand_sims is not None:
            pop_size, k = population.shape
            hyp_sims = self._hyp_cand_sims[population]
            hyp_relevance = hyp_sims.mean(axis=1)
            base_fitness += self.hyp_weight * hyp_relevance

        return base_fitness

    # =========================================================================
    # Crossover (uses blended query+hypothesis sims)
    # =========================================================================

    def crossover(
        self,
        p1: np.ndarray,
        p2: np.ndarray,
        k: int,
        M: int,
        query_cand_sims: np.ndarray,
        mutation_rate: float,
    ) -> np.ndarray:
        """Crossover using blended query+hypothesis similarity for ranking."""
        sims = self._blended_sims if self._blended_sims is not None else query_cand_sims

        pool = np.union1d(p1, p2)

        if len(pool) <= k:
            mask = np.ones(M, dtype=bool)
            mask[pool] = False
            remaining = np.where(mask)[0]
            np.random.shuffle(remaining)
            pool = np.concatenate([pool, remaining[:k - len(pool)]])
            child = pool[:k].astype(np.int32)
        else:
            pool_sims = sims[pool]
            top_k_pos = np.argpartition(pool_sims, -k)[-k:]
            child = pool[top_k_pos].astype(np.int32)

        # Mutation: replace lowest-relevance chunk with random outsider
        if random.random() < mutation_rate:
            mask = np.ones(M, dtype=bool)
            mask[child] = False
            outsiders = np.where(mask)[0]
            if len(outsiders) > 0:
                worst_pos = int(np.argmin(sims[child]))
                child[worst_pos] = outsiders[np.random.randint(len(outsiders))]

        return child

    # =========================================================================
    # Population seeding (hypothesis-aware)
    # =========================================================================

    def diverse_seeded_population(
        self,
        query_cand_sims: np.ndarray,
        M: int,
        k: int,
        pop_size: int,
    ) -> np.ndarray:
        """Seed population using blended sims (query + hypothesis)."""
        sims = self._blended_sims if self._blended_sims is not None else query_cand_sims
        ranked = np.argsort(sims)[::-1]
        population = []

        top_pool = ranked[:min(15, M)]
        mid_pool = ranked[15:min(50, M)]
        deep_pool = ranked[50:min(150, M)]

        # 25% top blended-sim seeds
        n_top = max(1, int(pop_size * 0.25))
        for _ in range(n_top):
            if len(top_pool) >= k:
                indices = np.random.choice(top_pool, size=k, replace=False)
            else:
                extra = np.random.choice(M, size=k - len(top_pool), replace=False)
                indices = np.concatenate([top_pool, extra])
            # Perturb 1-2 genes
            n_swaps = random.randint(1, min(2, k))
            idx_set = set(indices.tolist())
            avail = [j for j in range(M) if j not in idx_set]
            for _ in range(n_swaps):
                if avail:
                    pos = random.randint(0, k - 1)
                    swap_in = random.choice(avail)
                    avail.append(indices[pos])
                    avail.remove(swap_in)
                    indices[pos] = swap_in
            population.append(np.array(indices, dtype=np.int32))

        # 20% top + mid mix (bridge candidates)
        n_mid = max(1, int(pop_size * 0.20))
        for _ in range(n_mid):
            n_from_mid = k // 2
            n_from_top = k - n_from_mid
            mid_picks = np.random.choice(mid_pool, size=min(n_from_mid, len(mid_pool)), replace=False) if len(mid_pool) > 0 else np.array([], dtype=np.int64)
            top_picks = np.random.choice(top_pool, size=min(n_from_top, len(top_pool)), replace=False) if len(top_pool) > 0 else np.array([], dtype=np.int64)
            indices = np.concatenate([mid_picks, top_picks])
            if len(indices) < k:
                idx_set = set(indices.tolist())
                remaining = [j for j in range(M) if j not in idx_set]
                extra = random.sample(remaining, k - len(indices))
                indices = np.concatenate([indices, np.array(extra, dtype=np.int64)])
            population.append(np.array(indices[:k], dtype=np.int32))

        # 15% top + deep mix (second-hop discovery)
        n_deep = max(1, int(pop_size * 0.15))
        for _ in range(n_deep):
            n_from_deep = k // 2
            n_from_top = k - n_from_deep
            deep_picks = np.random.choice(deep_pool, size=min(n_from_deep, len(deep_pool)), replace=False) if len(deep_pool) > 0 else np.array([], dtype=np.int64)
            top_picks = np.random.choice(top_pool, size=min(n_from_top, len(top_pool)), replace=False) if len(top_pool) > 0 else np.array([], dtype=np.int64)
            indices = np.concatenate([deep_picks, top_picks])
            if len(indices) < k:
                idx_set = set(indices.tolist())
                remaining = [j for j in range(M) if j not in idx_set]
                extra = random.sample(remaining, k - len(indices))
                indices = np.concatenate([indices, np.array(extra, dtype=np.int64)])
            population.append(np.array(indices[:k], dtype=np.int32))

        # 40% random (exploration across the full 500-candidate pool)
        n_random = pop_size - len(population)
        for _ in range(n_random):
            population.append(np.array(random.sample(range(M), k), dtype=np.int32))

        return np.array(population, dtype=np.int32)

    # =========================================================================
    # Retrieve
    # =========================================================================

    def _retrieve_with_hypothesis(self, query: str, hypothesis: str, t_hyp: float) -> list[str]:
        """Core retrieval: dual BM25 -> GA with hypothesis fitness."""
        t1 = time.perf_counter()

        # Dual BM25 retrieval
        candidate_texts, candidate_embs = self.dual_bm25_search(query, hypothesis)
        if not candidate_texts:
            return []

        k = self.evolution_config.k_initial
        if len(candidate_texts) <= k:
            return candidate_texts

        t2 = time.perf_counter()

        # Embed query and hypothesis
        query_emb = self.embed_query(query)
        hyp_emb = self.embed_query(hypothesis)

        # Build similarity matrices
        query_cand_sims = self.build_query_candidate_similarities(query_emb, candidate_embs)
        self._hyp_cand_sims = self.build_query_candidate_similarities(hyp_emb, candidate_embs)
        self._blended_sims = 0.6 * query_cand_sims + 0.4 * self._hyp_cand_sims

        embed_sim_matrix = self.build_candidate_similarity_matrix(candidate_embs)

        # Token matrices + IG tables
        chunk_matrix, query_vec = self.build_token_matrices(query, candidate_texts)
        idf_weights = self.build_idf_weights(chunk_matrix)
        bridge_matrix = self.build_rare_token_bridge_matrix(chunk_matrix, idf_weights)

        dense_chunk_matrix = chunk_matrix.toarray().astype(np.float32)
        self._chunk_ig, self._ig_overlap = self.precompute_ig_tables(
            query_vec, dense_chunk_matrix, idf_weights, query_cand_sims,
        )

        t3 = time.perf_counter()

        # Run GA (overridden fitness_batch, crossover, and seeding)
        best_indices = self.run_ga(
            query_cand_sims, embed_sim_matrix, bridge_matrix, k,
        )

        t4 = time.perf_counter()

        # Clean up
        self._chunk_ig = None
        self._ig_overlap = None
        self._hyp_cand_sims = None
        self._blended_sims = None

        selected = [candidate_texts[i] for i in best_indices]
        logger.info(
            f"HyDE-Evolutionary: hypothesis={t_hyp:.3f}s bm25={t2-t1:.3f}s "
            f"build={t3-t2:.3f}s GA={t4-t3:.3f}s | "
            f"candidates={len(candidate_texts)} -> selected={len(selected)}"
        )
        return selected

    def retrieve(self, query: str, limit: int = 5) -> list[str]:
        """Sync retrieve."""
        t0 = time.perf_counter()
        hypothesis = self.generate_hypothesis(query)
        t_hyp = time.perf_counter() - t0
        return self._retrieve_with_hypothesis(query, hypothesis, t_hyp)

    # =========================================================================
    # Async API (used by benchmark pipeline)
    # =========================================================================

    async def async_ask(self, query: str, limit: int = 3) -> dict:
        """Full async pipeline: hypothesis -> retrieve -> answer."""
        t0 = time.perf_counter()
        hypothesis = await self.async_generate_hypothesis(query)
        t_hyp = time.perf_counter() - t0

        documents = self._retrieve_with_hypothesis(query, hypothesis, t_hyp)
        answer = await self.async_generate(query, documents)
        return {"query": query, "answer": answer, "sources": documents}
