"""
Official HotPotQA dataset loader and benchmark.

This module loads the official HotPotQA dev-distractor dataset and stores it
in a separate table from the ragbench subset for clean comparison.
"""
import asyncio
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional

from datasets import load_dataset
from openai import AsyncOpenAI

from ..config import Config
from ..core.bayesian_evolutionary_rag import BayesianEvolutionaryRAGSystem
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
    normalize_answer,
)
from .rag_bench import (
    print_results,
    save_results,
)

logger = logging.getLogger(__name__)

# Table name for HotPotQA data
HOTPOTQA_TABLE = "embeddings_hotpotqa_official"


@dataclass
class HotPotQAConfig:
    """Configuration for official HotPotQA dataset."""
    name: str = "hotpotqa-dev-distractor"
    table_name: str = HOTPOTQA_TABLE


# =============================================================================
# Dataset Loading
# =============================================================================

def load_hotpotqa_dataset(split: str = "validation"):
    """Load official HotPotQA dataset from HuggingFace."""
    logger.info(f"Loading official HotPotQA dataset ({split} split)...")
    dataset = load_dataset("hotpot_qa", "distractor", split=split)
    logger.info(f"Loaded {len(dataset)} samples")
    return dataset


def extract_paragraphs_from_item(item) -> List[str]:
    """Extract all paragraphs from a HotPotQA item (title + sentences)."""
    paragraphs = []
    context = item.get("context", {})
    titles = context.get("title", [])
    sentences_list = context.get("sentences", [])
    for title, sentences in zip(titles, sentences_list):
        paragraphs.append(f"{title}: {' '.join(sentences)}")
    return paragraphs


def get_gold_paragraphs(item) -> List[str]:
    """Extract gold (supporting) paragraphs from a HotPotQA item."""
    gold_paragraphs = []
    context = item.get("context", {})
    titles = context.get("title", [])
    sentences_list = context.get("sentences", [])
    supporting_facts = item.get("supporting_facts", {})
    gold_titles = set(supporting_facts.get("title", []))
    for title, sentences in zip(titles, sentences_list):
        if title in gold_titles:
            gold_paragraphs.append(f"{title}: {' '.join(sentences)}")
    return gold_paragraphs


# =============================================================================
# Knowledge Base
# =============================================================================

def prepare_hotpotqa_knowledge_base(vector_search, dataset, force_reload: bool = False) -> None:
    """Extract and add documents to knowledge base for official HotPotQA."""
    current_count = vector_search.count()

    if current_count > 0 and not force_reload:
        logger.info(f"Knowledge base already contains {current_count} documents, skipping load")
        return

    if force_reload and current_count > 0:
        logger.info(f"Clearing existing {current_count} documents...")
        vector_search.delete_all()

    # from dataset to text
    logger.info("Preparing knowledge base from official HotPotQA...")
    documents_added = set()
    for item in dataset:
        for para in extract_paragraphs_from_item(item):
            if para.strip():
                documents_added.add(para.strip())

    logger.info(f"Embedding {len(documents_added)} unique paragraphs...")
    texts_with_metadata = [(doc, {"source": "hotpotqa-official"}) for doc in documents_added]
    # pass the paragraphs to the embedder
    vector_search.add_texts(texts_with_metadata)
    logger.info(f"Added {len(documents_added)} documents to knowledge base")


# =============================================================================
# Sample Processing
# =============================================================================

async def process_sample(rag, item, idx: int, retrieval_limit: int):
    """Process a single HotPotQA sample with RAG."""

    # get question and right answer
    question = item.get("question", "")
    ground_truth = item.get("answer", "")
    if not question or not ground_truth:
        return None

    gold_docs = get_gold_paragraphs(item)

    # rag answers
    result = await rag.async_ask(question, limit=retrieval_limit)
    predicted = result["answer"]
    retrieved_docs = result["sources"]

    exact = calculate_exact_match(predicted, ground_truth)
    contains = normalize_answer(ground_truth) in normalize_answer(predicted)
    f1, precision, recall = calculate_f1(predicted, ground_truth)

    return {
        "id": item.get("id", idx),
        "query": question,
        "predicted": predicted,
        "ground_truth": ground_truth,
        "exact_match": exact,
        "contains": contains,
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "relevance": calculate_context_relevance(retrieved_docs, gold_docs),
        "utilization": calculate_context_utilization(predicted, retrieved_docs),
        "adherence": calculate_adherence(predicted, retrieved_docs),
        "sources": retrieved_docs,
        "type": item.get("type", "unknown"),
        "level": item.get("level", "unknown"),
    }


async def process_sample_llm_only(async_llm, model: str, item, idx: int):
    """Process a single HotPotQA sample using LLM only (no RAG)."""
    question = item.get("question", "")
    ground_truth = item.get("answer", "")
    if not question or not ground_truth:
        return None

    system_prompt = """You are a precise question-answering system.

        Instructions:
        - Be concise: use the minimum words necessary
        - If the answer is a name, date, number, or short phrase, respond with just that
        - Never explain your reasoning or add context"""

    try:
        response = await async_llm.chat.completions.create(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Question: {question}\n\nAnswer:"},
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
        "id": item.get("id", idx),
        "query": question,
        "predicted": predicted,
        "ground_truth": ground_truth,
        "exact_match": exact,
        "contains": contains,
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "type": item.get("type", "unknown"),
        "level": item.get("level", "unknown"),
    }


# =============================================================================
# Benchmark Runner
# =============================================================================

async def run_hotpotqa_benchmark_async(
    dataset,
    config: HotPotQAConfig,
    max_samples: Optional[int] = None,
    rag=None,
    retrieval_limit: Optional[int] = None,
    batch_size: Optional[int] = None,
) -> Dict:
    """
    Run HotPotQA benchmark. If rag is provided, runs RAG mode; otherwise LLM-only baseline.
    """
    # prepares the results format
    is_rag = rag is not None
    score_keys = ["f1", "precision", "recall"]
    if is_rag:
        score_keys += ["relevance", "utilization", "adherence"]
        retrieval_limit = retrieval_limit or Config.DEFAULT_RETRIEVAL_LIMIT

    batch_size = batch_size or Config.BATCH_SIZE
    logger.info(f"Running {config.name} benchmark with {batch_size} concurrent requests...")


    if not is_rag:
        async_llm = AsyncOpenAI(
            api_key=Config.DEEPSEEK_API_KEY,
            base_url=Config.DEEPSEEK_BASE_URL
        )
        model = Config.DEEPSEEK_MODEL

    results = {
        "dataset": config.name if is_rag else config.name + "-llm-only",
        "total": 0,
        "exact_match": 0,
        "contains": 0,
        "predictions": [],
        "by_type": {"bridge": [], "comparison": []},
        "by_level": {"easy": [], "medium": [], "hard": []},
        "metadata": {
            "max_samples": max_samples,
            "llm_model": Config.DEEPSEEK_MODEL,
            "mode": "rag" if is_rag else "llm-only-baseline",
            "table_name": config.table_name,
        },
    }
    if is_rag:
        results["metadata"].update({
            "retrieval_limit": retrieval_limit,
            "embedding_model": Config.EMBEDDING_MODEL,
            "chunk_size": Config.CHUNK_SIZE,
            "chunk_overlap": Config.CHUNK_OVERLAP,
        })

    for key in score_keys:
        results[f"{key}_scores"] = []

    samples_to_test = min(len(dataset), max_samples) if max_samples else len(dataset)
    samples = list(dataset)[:samples_to_test]

    for batch_start in range(0, len(samples), batch_size):
        batch_end = min(batch_start + batch_size, len(samples))
        batch = samples[batch_start:batch_end]

        if is_rag:
            tasks = [
                process_sample(rag, item, batch_start + i, retrieval_limit)
                for i, item in enumerate(batch)
            ]
        else:
            tasks = [
                process_sample_llm_only(async_llm, model, item, batch_start + i)
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

            q_type = entry.get("type", "unknown")
            q_level = entry.get("level", "unknown")
            if q_type in results["by_type"]:
                results["by_type"][q_type].append(entry["f1"])
            if q_level in results["by_level"]:
                results["by_level"][q_level].append(entry["f1"])

        if results["total"] > 0:
            avg_f1 = sum(results["f1_scores"]) / len(results["f1_scores"])
            progress_msg = f"Progress: {batch_end}/{samples_to_test} | F1: {avg_f1:.3f}"
            if "adherence_scores" in results and results["adherence_scores"]:
                avg_adherence = sum(results["adherence_scores"]) / len(results["adherence_scores"])
                progress_msg += f" | Adherence: {avg_adherence:.3f}"
            logger.info(progress_msg)

    return results


def run_hotpotqa_benchmark(dataset, config: HotPotQAConfig, **kwargs) -> Dict:
    """Sync wrapper for run_hotpotqa_benchmark_async."""
    return asyncio.run(run_hotpotqa_benchmark_async(dataset, config, **kwargs))


# =============================================================================
# Pipelines
# =============================================================================

def run_hotpotqa_pipeline(
    force_reload: bool = False,
    max_samples: int = 50,
    retrieval_limit: Optional[int] = None,
    use_hyde: bool = False,
    use_evolution: bool = False,
    use_bm25_evolution: bool = False,
    use_bayesian_evolution: bool = False,
):
    """Run official HotPotQA benchmark pipeline."""
    config = HotPotQAConfig()

    logger.info(f"{config.name.upper()} Evaluation")
    logger.info("=" * 60)

    if use_bayesian_evolution:
        rag = BayesianEvolutionaryRAGSystem(table_name=config.table_name)
        rag_type = "Bayesian-evolutionary"
    elif use_bm25_evolution:
        rag = BM25EvolutionaryRAGSystem(table_name=config.table_name)
        rag_type = "BM25-evolutionary"
    elif use_evolution:
        rag = EvolutionaryRAGSystem(table_name=config.table_name)
        rag_type = "evolutionary"
    elif use_hyde:
        rag = HyDERAGSystem(table_name=config.table_name)
        rag_type = "HyDE"
    else:
        rag = RAGSystem(table_name=config.table_name)
        rag_type = "standard"

    logger.info(f"Using {rag_type} RAG system with table '{config.table_name}'")
    rag.setup()

    try:
        if force_reload or rag.vector_search.count() == 0:
            dataset = load_hotpotqa_dataset(split="validation")
            prepare_hotpotqa_knowledge_base(rag.vector_search, dataset, force_reload=force_reload)
        else:
            logger.info(f"Knowledge base already loaded ({rag.vector_search.count()} documents)")

        eval_dataset = load_hotpotqa_dataset(split="validation")
        logger.info(f"Evaluating on validation split ({len(eval_dataset)} samples)")

        # Evolution is CPU-heavy (GA + tokenization), reduce concurrency
        evo_batch_size = 10 if (use_evolution or use_bm25_evolution or use_bayesian_evolution) else None

        results = asyncio.run(run_hotpotqa_benchmark_async(
            eval_dataset, config,
            max_samples=max_samples,
            rag=rag,
            retrieval_limit=retrieval_limit,
            batch_size=evo_batch_size,
        ))

        print_results(results)
        save_results(results)

    finally:
        rag.close()
        logger.info("\nBenchmark complete")


def run_hotpotqa_llm_only_pipeline(max_samples: int = 50):
    """Run HotPotQA LLM-only baseline (no RAG)."""
    config = HotPotQAConfig()

    logger.info(f"{config.name.upper()} LLM-ONLY BASELINE")
    logger.info("=" * 60)

    eval_dataset = load_hotpotqa_dataset(split="validation")
    logger.info(f"Evaluating on validation split ({len(eval_dataset)} samples)")

    results = run_hotpotqa_benchmark(eval_dataset, config, max_samples=max_samples)

    print_results(results)
    save_results(results)

    logger.info("\nBenchmark ccomplete")
