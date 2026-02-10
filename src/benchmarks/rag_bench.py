import asyncio
import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional

from datasets import concatenate_datasets, load_dataset

from ..config import Config
from ..core.bm25_evolutionary_rag import BM25EvolutionaryRAGSystem
from ..core.evolutionary_rag import EvolutionaryRAGSystem
from ..core.hyde_rag import HyDERAGSystem
from ..core.rag import RAGSystem
from .metrics import (
    calculate_adherence,
    calculate_context_relevance,
    calculate_context_utilization,
    calculate_exact_match,
    calculate_f1,
    llm_judge_correctness,
    normalize_answer,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Config
# =============================================================================

@dataclass
class DatasetConfig:
    """Configuration for a benchmark dataset."""
    name: str
    source: str  # HuggingFace dataset path
    subset: Optional[str] = None

    # Field extractors
    question_field: str = "question"
    answer_field: str = "response"
    documents_field: str = "documents"


# =============================================================================
# Dataset Loading
# =============================================================================


def load_benchmark_dataset(config: DatasetConfig, split: str = "test"):
    """Load a benchmark dataset."""
    logger.info(f"Loading {config.name} dataset ({split} split)...")
    if config.subset:
        dataset = load_dataset(config.source, config.subset, split=split)
    else:
        dataset = load_dataset(config.source, split=split)
    logger.info(f"Loaded {len(dataset)} samples")
    return dataset


def extract_documents_from_item(item, config: DatasetConfig) -> list[str]:
    """Extract document texts from a dataset item."""
    docs_field = item.get(config.documents_field, [])

    if isinstance(docs_field, list):
        return [doc if isinstance(doc, str) else str(doc) for doc in docs_field if doc]

    if isinstance(docs_field, str) and docs_field.strip():
        return [docs_field.strip()]

    return []



def prepare_knowledge_base(rag, dataset, config: DatasetConfig, force_reload: bool = False) -> None:
    """Extract and add documents to RAG knowledge base."""
    current_count = rag.vector_search.count()

    if current_count > 0 and not force_reload:
        logger.info(f"Knowledge base already contains {current_count} documents, skipping load")
        logger.info("Use --force-reload to clear and reload the database")
        return

    if force_reload and current_count > 0:
        logger.info(f"Clearing existing {current_count} documents...")
        rag.vector_search.delete_all()

    logger.info(f"Preparing knowledge base from {config.name}...")

    # Collect unique documents using extract_documents_from_item for all formats
    documents_added = set()
    for item in dataset:
        docs = extract_documents_from_item(item, config)
        for doc in docs:
            if doc.strip():
                documents_added.add(doc.strip())

    logger.info(f"Embedding {len(documents_added)} unique documents...")
    texts_with_metadata = [(doc, {"source": config.name}) for doc in documents_added]
    rag.vector_search.add_texts(texts_with_metadata)
    logger.info(f"Added {len(documents_added)} documents to knowledge base")


# =============================================================================
# Benchmark Runner
# =============================================================================

async def process_single_sample_llm_only(async_llm, model: str, item, config: DatasetConfig, idx: int):
    """Process a single sample using LLM only (no RAG) - baseline."""
    question = item.get(config.question_field, '')
    ground_truth = item.get(config.answer_field, '')
    if not question or not ground_truth:
        return None

    system_prompt = "You are a helpful assistant. Answer the question concisely and accurately."
    user_prompt = f"Question: {question}\n\nAnswer:"

    try:
        response = await async_llm.chat.completions.create(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            model=model,
            temperature=Config.LLM_TEMPERATURE,
            max_tokens=Config.MAX_TOKENS
        )
        predicted = response.choices[0].message.content or ""
    except Exception as e:
        predicted = f"Error: {e}"

    exact = calculate_exact_match(predicted, ground_truth)
    contains = normalize_answer(ground_truth) in normalize_answer(predicted)
    f1, precision, recall = calculate_f1(predicted, ground_truth)

    return {
        "id": item.get('id', idx),
        "query": question,
        "predicted": predicted,
        "ground_truth": ground_truth,
        "exact_match": exact,
        "contains": contains,
        "f1": f1,
        "precision": precision,
        "recall": recall,
    }


async def process_single_sample(rag, item, config: DatasetConfig, idx: int, retrieval_limit: int, use_llm_judge: bool):
    """Process a single benchmark sample asynchronously."""
    question = item.get(config.question_field, '')
    ground_truth = item.get(config.answer_field, '')
    if not question or not ground_truth:
        return None

    gold_docs = item.get(config.documents_field, [])

    result = await rag.async_ask(question, limit=retrieval_limit)
    predicted = result['answer']
    retrieved_docs = result['sources']

    adherence = calculate_adherence(predicted, retrieved_docs)
    exact = calculate_exact_match(predicted, ground_truth)
    contains = normalize_answer(ground_truth) in normalize_answer(predicted)
    f1, precision, recall = calculate_f1(predicted, ground_truth)
    relevance = calculate_context_relevance(retrieved_docs, gold_docs)
    utilization = calculate_context_utilization(predicted, retrieved_docs)

    prediction_entry = {
        "id": item.get('id', idx),
        "query": question,
        "predicted": predicted,
        "ground_truth": ground_truth,
        "exact_match": exact,
        "contains": contains,
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "relevance": relevance,
        "utilization": utilization,
        "adherence": adherence,
        "sources": retrieved_docs
    }

    if use_llm_judge:
        judge_result = llm_judge_correctness(predicted, ground_truth, question, rag.llm, rag.model)
        prediction_entry["llm_judge"] = judge_result

    return prediction_entry


async def run_benchmark_async(
    dataset,
    config: DatasetConfig,
    max_samples: int = None,
    rag=None,
    retrieval_limit: int = None,
    use_llm_judge: bool = False,
) -> Dict:

    """
    Run benchmark evaluation. If rag is provided, runs RAG mode; otherwise LLM-only baseline.

    Args:
        dataset: The evaluation dataset.
        config: Dataset configuration.
        max_samples: Max samples to evaluate.
        rag: RAG system instance. If None, runs LLM-only baseline.
        retrieval_limit: Number of documents to retrieve (RAG mode only).
        use_llm_judge: Whether to use LLM judge (RAG mode only).
    """
    is_rag = rag is not None
    score_keys = ["f1", "precision", "recall"]
    if is_rag:
        score_keys += ["relevance", "utilization", "adherence"]
        retrieval_limit = retrieval_limit or Config.DEFAULT_RETRIEVAL_LIMIT

    logger.info(f"Running {config.name} benchmark with {Config.BATCH_SIZE} concurrent requests...")

    # Setup LLM-only client if needed
    if not is_rag:
        from openai import AsyncOpenAI
        async_llm = AsyncOpenAI(
            api_key=Config.DEEPSEEK_API_KEY,
            base_url=Config.DEEPSEEK_BASE_URL
        )
        model = Config.DEEPSEEK_MODEL

    results = {
        "dataset": config.name,
        "total": 0,
        "exact_match": 0,
        "contains": 0,
        "predictions": [],
        "metadata": {
            "max_samples": max_samples,
            "llm_model": Config.DEEPSEEK_MODEL,
            "mode": "rag" if is_rag else "llm-only-baseline",
        },
    }
    if is_rag:
        results["metadata"].update({
            "retrieval_limit": retrieval_limit,
            "use_llm_judge": use_llm_judge,
            "embedding_model": Config.EMBEDDING_MODEL,
            "chunk_size": Config.CHUNK_SIZE,
            "chunk_overlap": Config.CHUNK_OVERLAP,
        })

    for key in score_keys:
        results[f"{key}_scores"] = []

    if use_llm_judge:
        results["llm_judge_scores"] = []
        results["llm_correct_count"] = 0

    samples_to_test = min(len(dataset), max_samples) if max_samples else len(dataset)
    samples = list(dataset)[:samples_to_test]

    for batch_start in range(0, len(samples), Config.BATCH_SIZE):
        batch_end = min(batch_start + Config.BATCH_SIZE, len(samples))
        batch = samples[batch_start:batch_end]

        if is_rag:
            tasks = [
                process_single_sample(rag, item, config, batch_start + i, retrieval_limit, use_llm_judge)
                for i, item in enumerate(batch)
            ]
        else:
            tasks = [
                process_single_sample_llm_only(async_llm, model, item, config, batch_start + i)
                for i, item in enumerate(batch)
            ]

        batch_results = await asyncio.gather(*tasks)

        for entry in batch_results:
            if entry is None:
                continue

            results["exact_match"] += int(entry["exact_match"])
            results["contains"] += int(entry["contains"])
            for key in score_keys:
                results[f"{key}_scores"].append(entry[key])
            results["predictions"].append(entry)
            results["total"] += 1

            if use_llm_judge and "llm_judge" in entry:
                results["llm_judge_scores"].append(entry["llm_judge"]["score"])
                results["llm_correct_count"] += int(entry["llm_judge"]["is_correct"])

        # progress logs
        if results["total"] > 0:
            avg_f1 = sum(results["f1_scores"]) / len(results["f1_scores"])
            progress_msg = f"Progress: {batch_end}/{samples_to_test} | F1: {avg_f1:.3f}"
            if "adherence_scores" in results and results["adherence_scores"]:
                avg_adherence = sum(results["adherence_scores"]) / len(results["adherence_scores"])
                progress_msg += f" | Adherence: {avg_adherence:.3f}"
            if use_llm_judge:
                progress_msg += f" | LLM Judge: {results['llm_correct_count'] / results['total']:.1%}"
            logger.info(progress_msg)

    return results


def run_benchmark(dataset, config, **kwargs) -> Dict:
    """Sync wrapper for run_benchmark_async."""
    return asyncio.run(run_benchmark_async(dataset, config, **kwargs))


# =============================================================================
# Results Output
# =============================================================================

def compute_summary(results: Dict) -> Dict:
    """Compute summary statistics from results."""
    total = results["total"]
    if total == 0:
        return {}

    summary = {
        "dataset": results["dataset"],
        "exact_match_rate": results["exact_match"] / total,
        "contains_rate": results["contains"] / total,
        "avg_f1": sum(results["f1_scores"]) / len(results["f1_scores"]),
        "avg_precision": sum(results["precision_scores"]) / len(results["precision_scores"]),
        "avg_recall": sum(results["recall_scores"]) / len(results["recall_scores"]),
    }

    # RAG-specific metrics (not present in LLM-only baseline)
    if "relevance_scores" in results and results["relevance_scores"]:
        summary["avg_relevance"] = sum(results["relevance_scores"]) / len(results["relevance_scores"])
        summary["avg_utilization"] = sum(results["utilization_scores"]) / len(results["utilization_scores"])
        summary["avg_adherence"] = sum(results["adherence_scores"]) / len(results["adherence_scores"])

    if "llm_judge_scores" in results:
        summary["llm_judge_accuracy"] = results["llm_correct_count"] / total

    return summary


def print_results(results: Dict) -> None:
    """Print benchmark results."""
    summary = compute_summary(results)
    total = results["total"]

    if total == 0:
        logger.warning("No samples processed")
        return

    logger.info("\n" + "="*60)
    logger.info(f"{results['dataset'].upper()} BENCHMARK RESULTS" + "\n" + "="*60)

    logger.info(f"\n  Total samples: {total}")
    logger.info(f"  Exact Match: {summary['exact_match_rate']:.2%}")
    logger.info(f"  Contains Answer: {summary['contains_rate']:.2%}")
    logger.info(f"  F1: {summary['avg_f1']:.3f}")
    logger.info(f"  Precision: {summary['avg_precision']:.3f}")
    logger.info(f"  Recall: {summary['avg_recall']:.3f}")

    # RAG-specific metrics (only for RAG benchmarks)
    if "avg_relevance" in summary:
        logger.info(f"  Context Relevance: {summary['avg_relevance']:.3f}")
        logger.info(f"  Context Utilization: {summary['avg_utilization']:.3f}")
        logger.info(f"  Adherence: {summary['avg_adherence']:.3f}")

    if "llm_judge_accuracy" in summary:
        logger.info(f"  LLM Judge Accuracy: {summary['llm_judge_accuracy']:.2%}")



def save_results(results: Dict, output_dir: str | None = None) -> None:
    """Save results to JSON file."""
    output_dir = output_dir or Config.RESULTS_DIR
    os.makedirs(output_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dataset_name = results["dataset"]
    filename = os.path.join(output_dir, f"benchmark_{dataset_name}_{timestamp}.json")

    summary = compute_summary(results)
    output = {"summary": summary, **results}

    with open(filename, 'w') as f:
        json.dump(output, f, indent=2)

    logger.info(f"\nResults saved to: {filename}")


# =============================================================================
# Main
# =============================================================================

def _make_ragbench_config(subset: str) -> DatasetConfig:
    """Create a RAGBench dataset config."""
    return DatasetConfig(
        name=f"ragbench-{subset}",
        source=Config.RAGBENCH_DATASET,
        subset=subset,
        question_field="question",
        answer_field="response",
        documents_field="documents",
    )


def run_benchmark_pipeline(
    subset: str = "hotpotqa",
    force_reload: bool = False,
    max_samples: int = 50,
    retrieval_limit: int | None = None,
    use_llm_judge: bool = False,
    eval_split: str = "test",
    use_hyde: bool = False,
    use_evolution: bool = False,
    use_bm25_evolution: bool = False,
):
    """Run RAG benchmark pipeline."""
    config = _make_ragbench_config(subset)
    retrieval_limit = retrieval_limit or Config.DEFAULT_RETRIEVAL_LIMIT
    logger.info(f"{config.name.upper()} Evaluation")
    logger.info("="*60)

    if use_bm25_evolution:
        rag = BM25EvolutionaryRAGSystem()
        rag_type = "BM25-evolutionary"
    elif use_evolution:
        rag = EvolutionaryRAGSystem()
        rag_type = "evolutionary"
    elif use_hyde:
        rag = HyDERAGSystem()
        rag_type = "HyDE"
    else:
        rag = RAGSystem()
        rag_type = "standard"
    logger.info(f"Using {rag_type} RAG system")
    rag.setup()

    # check to see if the kb is loaded and full
    try:
        if force_reload or rag.vector_search.count() == 0:
            logger.info("Loading all splits into knowledge base...")
            all_datasets = []
            for split in ["train", "validation", "test"]:
                try:
                    split_data = load_benchmark_dataset(config, split=split)
                    all_datasets.append(split_data)
                    logger.info(f"  - {split}: {len(split_data)} samples")
                except Exception as e:
                    logger.warning(f"Could not load {split} split: {e}")

            if all_datasets:
                combined_dataset = concatenate_datasets(all_datasets)
                logger.info(f"Total samples: {len(combined_dataset)}")
                prepare_knowledge_base(rag, combined_dataset, config, force_reload=force_reload)
        else:
            logger.info(f"Knowledge base already loaded ({rag.vector_search.count()} documents)")

        eval_dataset = load_benchmark_dataset(config, split=eval_split)
        logger.info(f"Evaluating on {eval_split} split ({len(eval_dataset)} samples)")

        # run benchmark
        results = run_benchmark(
            eval_dataset, config,
            max_samples=max_samples,
            rag=rag,
            retrieval_limit=retrieval_limit,
            use_llm_judge=use_llm_judge,
        )
        print_results(results)
        save_results(results)

    finally:
        rag.close()
        logger.info("\nBenchmark complete")


def run_llm_only_pipeline(
    subset: str = "hotpotqa",
    max_samples: int = 50,
    eval_split: str = "test",
):
    """Run LLM-only baseline benchmark (no RAG)."""
    config = _make_ragbench_config(subset)
    config.name += "-llm-only"
    logger.info(f"{config.name.upper()} LLM-ONLY BASELINE")
    logger.info("="*60)

    eval_dataset = load_benchmark_dataset(config, split=eval_split)
    logger.info(f"Evaluating on {eval_split} split ({len(eval_dataset)} samples)")

    # run benchmark
    results = run_benchmark(eval_dataset, config, max_samples=max_samples)
    print_results(results)
    save_results(results)

    logger.info("\nBenchmark complete")
