"""
Run benchmark comparison: Standard RAG vs Evolutionary RAG.

Usage:
    python run_benchmark.py                     # 30 samples, hotpotqa
    python run_benchmark.py --samples 100       # 100 samples
    python run_benchmark.py --table embeddings_small_test  # small dataset
"""
import argparse
import asyncio
import logging
import time

logging.basicConfig(level=logging.INFO)

from src.benchmarks.hotpotqa import (
    HotPotQAConfig,
    load_hotpotqa_dataset,
    run_hotpotqa_benchmark_async,
)
from src.core.evolutionary_rag import EvolutionaryRAGSystem
from src.core.rag import RAGSystem


def main():
    parser = argparse.ArgumentParser(description="Run RAG benchmark")
    parser.add_argument("--samples", type=int, default=30, help="Number of samples")
    parser.add_argument("--table", type=str, default="embeddings_hotpotqa_official", help="Table name")
    parser.add_argument("--limit", type=int, default=10, help="Retrieval limit (chunks to LLM)")
    parser.add_argument("--only-standard", action="store_true", help="Only run standard RAG")
    parser.add_argument("--only-evolution", action="store_true", help="Only run evolutionary RAG")
    args = parser.parse_args()

    config = HotPotQAConfig(table_name=args.table)
    dataset = load_hotpotqa_dataset(split="validation")

    print(f"\n{'='*60}")
    print(f"Benchmark: {args.samples} samples, table={args.table}, limit={args.limit}")
    print(f"{'='*60}\n")

    # Standard RAG
    if not args.only_evolution:
        print("=== STANDARD RAG ===")
        rag = RAGSystem(table_name=args.table)
        rag.setup()
        start = time.time()
        results = asyncio.run(
            run_hotpotqa_benchmark_async(
                dataset, config, max_samples=args.samples, rag=rag, retrieval_limit=args.limit
            )
        )
        elapsed = time.time() - start
        em = results["exact_match"]
        total = results["total"]
        f1 = sum(results["f1_scores"]) / len(results["f1_scores"])
        print(f"Time: {elapsed:.1f}s")
        print(f"Exact Match: {em}/{total} ({100*em/total:.1f}%)")
        print(f"Avg F1: {f1:.3f}")
        rag.close()
        print()

    # Evolutionary RAG
    if not args.only_standard:
        print("=== EVOLUTIONARY RAG ===")
        rag2 = EvolutionaryRAGSystem(table_name=args.table)
        rag2.setup()
        start = time.time()
        results2 = asyncio.run(
            run_hotpotqa_benchmark_async(
                dataset, config, max_samples=args.samples, rag=rag2, retrieval_limit=args.limit
            )
        )
        elapsed = time.time() - start
        em2 = results2["exact_match"]
        total2 = results2["total"]
        f12 = sum(results2["f1_scores"]) / len(results2["f1_scores"])
        print(f"Time: {elapsed:.1f}s")
        print(f"Exact Match: {em2}/{total2} ({100*em2/total2:.1f}%)")
        print(f"Avg F1: {f12:.3f}")
        rag2.close()

    print("\nDone.")


if __name__ == "__main__":
    main()
