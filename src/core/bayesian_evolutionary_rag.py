"""
Bayesian Information Foraging RAG: BM25 Pre-filter + Embedding-Aware GA Selection.

Stage 1: BM25 lexical search for top-N candidate chunks (dumb pre-filter)
Stage 2: GA evolves index-based genomes to select optimal k-chunk subset

Key insight: token-level distribution matching cannot distinguish semantically
different chunks that share surface tokens (e.g. "Ed Wood" vs "Woodson, Arkansas").
Embeddings provide the semantic signal that tokens can't.

Genome = array of k chunk indices

Fitness = SemanticRelevance + alpha * MarginalGain + beta * Bridging - gamma * Redundancy

  - SemanticRelevance: avg cosine similarity between query embedding and
      selected chunk embeddings. This is the primary signal — chunks must be
      semantically related to the question, not just share tokens.

  - MarginalGain: avg leave-one-out semantic relevance drop.
      Each chunk must individually contribute semantic relevance.
      A document is valuable if removing it hurts.

  - Bridging: avg rare-token bridging between selected chunks (multi-hop).
      Kept from token-level because rare-token overlap between chunks is
      still a valid multi-hop signal (chunks about related topics share
      domain-specific terms).

  - Redundancy: avg pairwise embedding cosine similarity between selected
      chunks. Penalize semantically near-duplicate chunks.

Selection = roulette (fitness-proportionate) with 10% elitism
Crossover = pool parents, rank by individual semantic relevance, take top-k
Mutation = replace lowest-relevance chunk with random non-selected chunk
"""
import logging
import random
import time
from collections import Counter
from typing import Optional

import numpy as np
from rank_bm25 import BM25Okapi
from scipy.sparse import csr_matrix

from ..config import EvolutionConfig
from .embeddings import EmbeddingService
from .rag import RAGSystem
from .tokenizer import Tokenizer

logger = logging.getLogger(__name__)


class BayesianEvolutionaryRAGSystem(RAGSystem):

    def __init__(
        self,
        model: str | None = None,
        table_name: str | None = None,
        evolution_config: EvolutionConfig | None = None,
    ):
        super().__init__(model=model, table_name=table_name)

        self.evolution_config = evolution_config or EvolutionConfig(
            population_size=50,
            k_initial=10,
            max_generations=100,
            mutation_rate=0.2,
        )

        self.n_candidates = 200
        self.early_stop_patience = 15
        self.alpha = 0.3       # Marginal gain weight
        self.beta = 0.3        # Bridging weight (rare-token multi-hop signal)
        self.gamma = 0.3       # Redundancy penalty weight (embedding-based)
        self.tokenizer_instance: Optional[Tokenizer] = None
        self.embedding_service = EmbeddingService()
        self._bm25: Optional[BM25Okapi] = None
        self._corpus_texts: Optional[list[str]] = None
        self._corpus_tokenized: Optional[list[list[str]]] = None

    @property
    def tokenizer(self) -> Tokenizer:
        if self.tokenizer_instance is None:
            self.tokenizer_instance = Tokenizer.get_instance(self.evolution_config.model_name)
        return self.tokenizer_instance

    # =========================================================================
    # BM25 Stage 1 (dumb pre-filter — just narrows search space)
    # =========================================================================

    def _ensure_bm25(self) -> None:
        """Build BM25 index from all chunks in the database (lazy, built once)."""
        if self._bm25 is not None:
            return
        rows = self.vector_search.db.get_all_chunks()
        self._corpus_texts = [text for _, text in rows]
        self._corpus_tokenized = [text.lower().split() for text in self._corpus_texts]
        self._bm25 = BM25Okapi(self._corpus_tokenized)
        logger.info(f"BM25 index built over {len(self._corpus_texts)} chunks")

    def bm25_search(self, query: str, n: int) -> list[str]:
        """Return top-n chunks by BM25 score."""
        self._ensure_bm25()
        tokenized_query = query.lower().split()
        scores = self._bm25.get_scores(tokenized_query)
        top_indices = np.argsort(scores)[::-1][:n]
        return [self._corpus_texts[i] for i in top_indices]

    # =========================================================================
    # Embedding Construction
    # =========================================================================

    def build_embeddings(self, query_text: str, candidate_texts: list[str]) -> tuple[np.ndarray, np.ndarray]:
        """Embed query + candidates in one batch. Returns (query_emb, candidate_embs).

        query_emb: shape (768,)
        candidate_embs: shape (n_candidates, 768)
        """
        all_texts = [query_text] + candidate_texts
        embeddings = self.embedding_service.get_embeddings_batch(all_texts)

        query_emb = np.array(embeddings[0], dtype=np.float32)
        candidate_embs = np.array(embeddings[1:], dtype=np.float32)

        return query_emb, candidate_embs

    def build_query_candidate_similarities(self, query_emb: np.ndarray, candidate_embs: np.ndarray) -> np.ndarray:
        """Cosine similarity between query and each candidate. Shape (n_candidates,)."""
        q_norm = query_emb / (np.linalg.norm(query_emb) + 1e-10)
        c_norms = candidate_embs / (np.linalg.norm(candidate_embs, axis=1, keepdims=True) + 1e-10)
        return c_norms @ q_norm

    def build_candidate_similarity_matrix(self, candidate_embs: np.ndarray) -> np.ndarray:
        """Pairwise cosine similarity between all candidates (embedding-based)."""
        norms = np.linalg.norm(candidate_embs, axis=1, keepdims=True) + 1e-10
        normalized = candidate_embs / norms
        sim = normalized @ normalized.T
        np.fill_diagonal(sim, 0.0)
        return sim

    # =========================================================================
    # Token-level matrices (for bridging only)
    # =========================================================================

    def build_token_matrices(self, query_text: str, candidate_texts: list[str]) -> csr_matrix:
        """Batch-tokenize candidates, return chunk_matrix for bridging computation."""
        all_texts = [query_text] + candidate_texts
        all_token_ids = self.tokenizer.encode_batch(all_texts)

        candidate_ids_list = all_token_ids[1:]

        all_token_set: set[int] = set()
        candidate_counters = []
        candidate_totals = []
        for ids in candidate_ids_list:
            all_token_set.update(ids)
            candidate_counters.append(Counter(ids))
            candidate_totals.append(len(ids))

        token_to_col = {tid: col for col, tid in enumerate(sorted(all_token_set))}
        n_tokens = len(token_to_col)
        n_candidates = len(candidate_texts)

        rows, cols, data = [], [], []
        for i, (counts, total) in enumerate(zip(candidate_counters, candidate_totals)):
            if total == 0:
                continue
            for tid, count in counts.items():
                rows.append(i)
                cols.append(token_to_col[tid])
                data.append(count / total)

        return csr_matrix(
            (np.array(data, dtype=np.float32), (rows, cols)),
            shape=(n_candidates, n_tokens),
        )

    def build_idf_weights(self, chunk_matrix: csr_matrix) -> np.ndarray:
        """IDF(t) = log(N / df(t)). Rare tokens get high IDF."""
        N = chunk_matrix.shape[0]
        df = np.asarray((chunk_matrix > 0).sum(axis=0)).flatten().astype(np.float32)
        df = np.clip(df, 1, None)
        return np.log(N / df).astype(np.float32)

    def build_rare_token_bridge_matrix(self, chunk_matrix: csr_matrix, idf_weights: np.ndarray) -> np.ndarray:
        """Pairwise rare-token bridging score, normalized to [0, 1]."""
        presence = (chunk_matrix > 0).astype(np.float32)
        weighted = presence.multiply(csr_matrix(idf_weights))
        bridge = (weighted @ weighted.T).toarray()
        np.fill_diagonal(bridge, 0.0)
        max_val = bridge.max()
        if max_val > 0:
            bridge /= max_val
        return bridge

    # =========================================================================
    # Fitness Function
    # =========================================================================

    def fitness(
        self,
        genome: np.ndarray,
        query_cand_sims: np.ndarray,
        embed_sim_matrix: np.ndarray,
        bridge_matrix: np.ndarray,
    ) -> float:
        """Fitness = SemanticRelevance + alpha * MarginalGain + beta * Bridging - gamma * Redundancy.

        All precomputed — no per-evaluation embedding calls.
        """
        k = len(genome)
        if k == 0:
            return -1e6

        # Semantic relevance: avg cosine(query_emb, chunk_emb) for selected chunks
        relevance = float(query_cand_sims[genome].mean())

        # Marginal gain: avg leave-one-out relevance drop
        if k <= 1:
            mg = 0.0
        else:
            total_mg = 0.0
            for i in range(k):
                reduced = np.concatenate([genome[:i], genome[i+1:]])
                reduced_rel = float(query_cand_sims[reduced].mean())
                total_mg += relevance - reduced_rel
            mg = total_mg / k

        # Bridging and redundancy
        if k < 2:
            bridging = 0.0
            redundancy = 0.0
        else:
            n_pairs = k * (k - 1) / 2

            sub_bridge = bridge_matrix[np.ix_(genome, genome)]
            bridging = np.triu(sub_bridge, k=1).sum() / n_pairs

            sub_sim = embed_sim_matrix[np.ix_(genome, genome)]
            redundancy = np.triu(sub_sim, k=1).sum() / n_pairs

        return relevance + self.alpha * mg + self.beta * bridging - self.gamma * redundancy

    def fitness_batch(
        self,
        population: np.ndarray,
        query_cand_sims: np.ndarray,
        embed_sim_matrix: np.ndarray,
        bridge_matrix: np.ndarray,
    ) -> np.ndarray:
        return np.array([
            self.fitness(genome, query_cand_sims, embed_sim_matrix, bridge_matrix)
            for genome in population
        ], dtype=np.float32)

    # =========================================================================
    # GA Operators
    # =========================================================================

    def roulette_select(self, fit: np.ndarray) -> int:
        """Fitness-proportionate selection. Shift fitness to positive values."""
        shifted = fit - fit.min() + 1e-6
        probs = shifted / shifted.sum()
        return int(np.random.choice(len(fit), p=probs))

    def crossover(
        self,
        p1: np.ndarray,
        p2: np.ndarray,
        k: int,
        M: int,
        query_cand_sims: np.ndarray,
        mutation_rate: float,
    ) -> np.ndarray:
        """Semantic-aware crossover: pool parents, rank by query similarity, take top-k.

        1. Merge both parents' chunks
        2. Rank by individual semantic relevance to query
        3. Take top-k
        4. Mutation: replace lowest-relevance chunk with random outsider
        """
        pool = list(set(p1.tolist()) | set(p2.tolist()))

        if len(pool) <= k:
            remaining = [i for i in range(M) if i not in set(pool)]
            random.shuffle(remaining)
            pool = pool + remaining[:k - len(pool)]
            child = np.array(pool[:k], dtype=np.int32)
        else:
            # Rank by semantic relevance
            scored = [(idx, query_cand_sims[idx]) for idx in pool]
            scored.sort(key=lambda x: x[1], reverse=True)
            child = np.array([s[0] for s in scored[:k]], dtype=np.int32)

        # Mutation: replace lowest-relevance chunk
        if random.random() < mutation_rate:
            child_set = set(child.tolist())
            non_selected = [i for i in range(M) if i not in child_set]
            if non_selected:
                sims = query_cand_sims[child]
                worst_pos = int(np.argmin(sims))
                child[worst_pos] = random.choice(non_selected)

        return child

    def diverse_seeded_population(
        self,
        query_cand_sims: np.ndarray,
        M: int,
        k: int,
        pop_size: int,
    ) -> np.ndarray:
        """Create initial population seeded by embedding similarity ranks.

        Uses semantic similarity to query (not BM25) for seeding, so seeds
        are actually about the right topics.
          - 20% from top-10 by embedding sim (first hop)
          - 15% mixing top-10 with rank 10-30 (bridge candidates)
          - 15% mixing top-10 with rank 30-80 (deep second-hop candidates)
          - 50% random (exploration)
        """
        ranked = np.argsort(query_cand_sims)[::-1]
        population = []

        top_pool = ranked[:min(10, M)]
        mid_pool = ranked[10:min(30, M)]
        deep_pool = ranked[30:min(80, M)]

        # Slice 1: top embedding-sim seeds (20%)
        n_top = max(1, int(pop_size * 0.20))
        for _ in range(n_top):
            if len(top_pool) >= k:
                indices = np.random.choice(top_pool, size=k, replace=False)
            else:
                extra = np.random.choice(M, size=k - len(top_pool), replace=False)
                indices = np.concatenate([top_pool, extra])
            # Perturb 1-3 genes
            n_swaps = random.randint(1, min(3, k))
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

        # Slice 2: top + mid mix (15%)
        n_mid = max(1, int(pop_size * 0.15))
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

        # Slice 3: top + deep mix (15%)
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

        # Slice 4: random (50%)
        n_random = pop_size - len(population)
        for _ in range(n_random):
            population.append(np.array(random.sample(range(M), k), dtype=np.int32))

        return np.array(population, dtype=np.int32)

    # =========================================================================
    # GA Main Loop
    # =========================================================================

    def run_ga(
        self,
        query_cand_sims: np.ndarray,
        embed_sim_matrix: np.ndarray,
        bridge_matrix: np.ndarray,
        k: int,
        track_convergence: bool = False,
    ) -> list[int] | tuple[list[int], list[dict]]:
        """Generational GA with embedding-based fitness."""
        M = len(query_cand_sims)
        if M <= k:
            result = list(range(M))
            if track_convergence:
                return result, []
            return result

        pop_size = self.evolution_config.population_size
        generations = self.evolution_config.max_generations
        mutation_rate = self.evolution_config.mutation_rate
        elite_size = max(1, pop_size // 10)

        # Initialize with embedding-similarity-based diverse seeding
        population = self.diverse_seeded_population(query_cand_sims, M, k, pop_size)

        # Evaluate initial population
        fit = self.fitness_batch(population, query_cand_sims, embed_sim_matrix, bridge_matrix)

        best_idx = int(np.argmax(fit))
        best_fitness = fit[best_idx]
        best_genome = population[best_idx].copy()
        stale_generations = 0

        fitness_history = None
        if track_convergence:
            fitness_history = [{"best": float(best_fitness), "mean": float(fit.mean())}]

        for gen in range(generations):
            elite_order = np.argsort(fit)[::-1][:elite_size]
            new_population = [population[i].copy() for i in elite_order]
            new_fit = [fit[i] for i in elite_order]

            while len(new_population) < pop_size:
                p1_idx = self.roulette_select(fit)
                p2_idx = self.roulette_select(fit)

                child = self.crossover(
                    population[p1_idx], population[p2_idx],
                    k, M, query_cand_sims, mutation_rate,
                )

                child_fitness = self.fitness(
                    child, query_cand_sims, embed_sim_matrix, bridge_matrix,
                )
                new_population.append(child)
                new_fit.append(child_fitness)

            population = np.array(new_population, dtype=np.int32)
            fit = np.array(new_fit, dtype=np.float32)

            gen_best_idx = int(np.argmax(fit))
            gen_best_fitness = fit[gen_best_idx]

            if gen_best_fitness > best_fitness:
                best_fitness = gen_best_fitness
                best_genome = population[gen_best_idx].copy()
                stale_generations = 0
            else:
                stale_generations += 1

            if track_convergence:
                fitness_history.append({"best": float(best_fitness), "mean": float(fit.mean())})

            if stale_generations >= self.early_stop_patience:
                logger.debug(
                    f"GA early stop at gen {gen} (no improvement for {self.early_stop_patience} gens)"
                )
                break

        result_indices = best_genome.tolist()

        if track_convergence:
            return result_indices, fitness_history
        return result_indices

    # =========================================================================
    # Retrieve (public API)
    # =========================================================================

    def retrieve(self, query: str, limit: int = 5) -> list[str]:
        """Two-stage: BM25 candidate pre-filter -> embedding-aware GA selection."""
        t0 = time.perf_counter()

        candidate_texts = self.bm25_search(query, self.n_candidates)
        if not candidate_texts:
            return []

        k = self.evolution_config.k_initial
        if len(candidate_texts) <= k:
            return candidate_texts

        t1 = time.perf_counter()

        # Embed query + candidates (one batch call)
        query_emb, candidate_embs = self.build_embeddings(query, candidate_texts)
        query_cand_sims = self.build_query_candidate_similarities(query_emb, candidate_embs)
        embed_sim_matrix = self.build_candidate_similarity_matrix(candidate_embs)

        # Token-level bridge matrix (rare-token sharing for multi-hop)
        chunk_matrix = self.build_token_matrices(query, candidate_texts)
        idf_weights = self.build_idf_weights(chunk_matrix)
        bridge_matrix = self.build_rare_token_bridge_matrix(chunk_matrix, idf_weights)

        t2 = time.perf_counter()

        best_indices = self.run_ga(
            query_cand_sims, embed_sim_matrix, bridge_matrix, k,
        )

        t3 = time.perf_counter()

        selected = [candidate_texts[i] for i in best_indices]
        logger.info(
            f"Bayesian-Evolutionary: BM25 {len(candidate_texts)} candidates -> GA {len(selected)} selected | "
            f"bm25={t1-t0:.3f}s build={t2-t1:.3f}s GA={t3-t2:.3f}s"
        )
        return selected
