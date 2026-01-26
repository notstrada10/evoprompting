"""
Official HotPotQA dataset loader and benchmark.

This module loads the official HotPotQA dev-distractor dataset and stores it
in a separate table from the ragbench subset for clean comparison.
"""
import asyncio
import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional

from datasets import load_dataset

from ..config import Config
from ..core.db import VectorDatabase
from ..core.embeddings import EmbeddingService
from ..core.hyde_rag import HyDERAGSystem
from ..core.rag import RAGSystem
from ..core.vector_search import VectorSearch
from .rag_bench import (
    calculate_adherence,
    calculate_context_relevance,
    calculate_context_utilization,
    calculate_exact_match,
    calculate_f1,
    compute_summary,
    normalize_answer,
    print_results,
    save_results,
)

logger = logging.getLogger(__name__)

# Table name for official HotPotQA data
HOTPOTQA_TABLE = "embeddings_hotpotqa_official"


@dataclass
class HotPotQAConfig:
    """Configuration for official HotPotQA dataset."""
    name: str = "hotpotqa-dev-distractor"
    table_name: str = HOTPOTQA_TABLE


def load_hotpotqa_dataset(split: str = "validation"):
    """
    Load official HotPotQA dataset from HuggingFace.

    Args:
        split: Dataset split to load. Use "validation" for dev set.
               Options: "train", "validation"

    Returns:
        HuggingFace dataset object
    """
    logger.info(f"Loading official HotPotQA dataset ({split} split)...")

    # Load the distractor setting (standard evaluation)
    dataset = load_dataset("hotpot_qa", "distractor", split=split)

    logger.info(f"Loaded {len(dataset)} samples")
    return dataset


def extract_paragraphs_from_item(item) -> List[str]:
    """
    Extract supporting paragraphs from a HotPotQA item.

    The official HotPotQA format has:
    - context: dict with 'title' and 'sentences' lists
    - supporting_facts: dict with 'title' and 'sent_id' for gold facts

    Args:
        item: A single HotPotQA dataset item

    Returns:
        List of paragraph strings (title + sentences combined)
    """
    paragraphs = []

    context = item.get("context", {})
    titles = context.get("title", [])
    sentences_list = context.get("sentences", [])

    for title, sentences in zip(titles, sentences_list):
        # Combine title and sentences into a single paragraph
        paragraph = f"{title}: {' '.join(sentences)}"
        paragraphs.append(paragraph)

    return paragraphs


def get_gold_paragraphs(item) -> List[str]:
    """
    Extract gold (supporting) paragraphs from a HotPotQA item.

    Args:
        item: A single HotPotQA dataset item

    Returns:
        List of gold paragraph strings
    """
    gold_paragraphs = []

    context = item.get("context", {})
    titles = context.get("title", [])
    sentences_list = context.get("sentences", [])

    supporting_facts = item.get("supporting_facts", {})
    gold_titles = set(supporting_facts.get("title", []))

    for title, sentences in zip(titles, sentences_list):
        if title in gold_titles:
            paragraph = f"{title}: {' '.join(sentences)}"
            gold_paragraphs.append(paragraph)

    return gold_paragraphs


def prepare_hotpotqa_knowledge_base(
    vector_search: VectorSearch,
    dataset,
    force_reload: bool = False
) -> None:
    """
    Extract and add documents to knowledge base for official HotPotQA.

    Args:
        vector_search: VectorSearch instance (using hotpotqa table)
        dataset: HotPotQA dataset
        force_reload: Whether to clear existing data first
    """
    current_count = vector_search.count()

    if current_count > 0 and not force_reload:
        logger.info(f"Knowledge base already contains {current_count} documents, skipping load")
        logger.info("Use --force-reload to clear and reload the database")
        return

    if force_reload and current_count > 0:
        logger.info(f"Clearing existing {current_count} documents...")
        vector_search.delete_all()

    logger.info("Preparing knowledge base from official HotPotQA...")

    # Collect unique paragraphs
    documents_added = set()
    for item in dataset:
        paragraphs = extract_paragraphs_from_item(item)
        for para in paragraphs:
            if para.strip():
                documents_added.add(para.strip())

    logger.info(f"Embedding {len(documents_added)} unique paragraphs...")
    texts_with_metadata = [(doc, {"source": "hotpotqa-official"}) for doc in documents_added]
    vector_search.add_texts(texts_with_metadata)
    logger.info(f"Added {len(documents_added)} documents to knowledge base")


class HotPotQARAGSystem(RAGSystem):
    """RAG system configured for official HotPotQA table."""

    def __init__(self, table_name: str = HOTPOTQA_TABLE):
        self.table_name = table_name
        super().__init__()

    def setup(self) -> None:
        """Initialize the RAG system with HotPotQA-specific table."""
        self.db = VectorDatabase(table_name=self.table_name)
        self.db.connect()
        self.db.setup_database()

        self.embedding_service = EmbeddingService()
        self.vector_search = VectorSearch(self.db, self.embedding_service)

        from openai import AsyncOpenAI, OpenAI
        self.llm = OpenAI(
            api_key=Config.DEEPSEEK_API_KEY,
            base_url=Config.DEEPSEEK_BASE_URL
        )
        self.async_llm = AsyncOpenAI(
            api_key=Config.DEEPSEEK_API_KEY,
            base_url=Config.DEEPSEEK_BASE_URL
        )
        self.model = Config.DEEPSEEK_MODEL

        logger.info(f"HotPotQA RAG system initialized with table '{self.table_name}'")


class HotPotQAHyDERAGSystem(HyDERAGSystem):
    """HyDE RAG system configured for official HotPotQA table."""

    def __init__(self, table_name: str = HOTPOTQA_TABLE):
        self.table_name = table_name
        super().__init__()

    def setup(self) -> None:
        """Initialize the HyDE RAG system with HotPotQA-specific table."""
        self.db = VectorDatabase(table_name=self.table_name)
        self.db.connect()
        self.db.setup_database()

        self.embedding_service = EmbeddingService()
        self.vector_search = VectorSearch(self.db, self.embedding_service)

        from openai import AsyncOpenAI, OpenAI
        self.llm = OpenAI(
            api_key=Config.DEEPSEEK_API_KEY,
            base_url=Config.DEEPSEEK_BASE_URL
        )
        self.async_llm = AsyncOpenAI(
            api_key=Config.DEEPSEEK_API_KEY,
            base_url=Config.DEEPSEEK_BASE_URL
        )
        self.model = Config.DEEPSEEK_MODEL

        logger.info(f"HotPotQA HyDE RAG system initialized with table '{self.table_name}'")


async def process_single_hotpotqa_sample(
    rag,
    item,
    idx: int,
    retrieval_limit: int
):
    """Process a single HotPotQA sample asynchronously."""
    question = item.get("question", "")
    ground_truth = item.get("answer", "")

    if not question or not ground_truth:
        return None

    gold_docs = get_gold_paragraphs(item)

    result = await rag.async_ask(question, limit=retrieval_limit)
    predicted = result["answer"]
    retrieved_docs = result["sources"]

    # Calculate metrics
    exact = calculate_exact_match(predicted, ground_truth)
    contains = normalize_answer(ground_truth) in normalize_answer(predicted)
    f1, precision, recall = calculate_f1(predicted, ground_truth)
    relevance = calculate_context_relevance(retrieved_docs, gold_docs)
    utilization = calculate_context_utilization(predicted, retrieved_docs)
    adherence = calculate_adherence(predicted, retrieved_docs)

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
        "relevance": relevance,
        "utilization": utilization,
        "adherence": adherence,
        "sources": retrieved_docs,
        "type": item.get("type", "unknown"),  # bridge or comparison
        "level": item.get("level", "unknown"),  # easy, medium, hard
    }


async def run_hotpotqa_benchmark_async(
    rag,
    dataset,
    config: HotPotQAConfig,
    max_samples: Optional[int] = None,
    retrieval_limit: Optional[int] = None,
    use_hyde: bool = False,
) -> Dict:
    """Run official HotPotQA benchmark with concurrent LLM requests."""
    retrieval_limit = retrieval_limit or Config.DEFAULT_RETRIEVAL_LIMIT
    logger.info(f"Running {config.name} benchmark with {Config.BATCH_SIZE} concurrent requests...")

    results = {
        "dataset": config.name,
        "total": 0,
        "exact_match": 0,
        "contains": 0,
        "f1_scores": [],
        "precision_scores": [],
        "recall_scores": [],
        "relevance_scores": [],
        "utilization_scores": [],
        "adherence_scores": [],
        "predictions": [],
        "by_type": {"bridge": [], "comparison": []},
        "by_level": {"easy": [], "medium": [], "hard": []},
        "metadata": {
            "retrieval_limit": retrieval_limit,
            "max_samples": max_samples,
            "use_hyde": use_hyde,
            "embedding_model": Config.EMBEDDING_MODEL,
            "llm_model": Config.DEEPSEEK_MODEL,
            "chunk_size": Config.CHUNK_SIZE,
            "chunk_overlap": Config.CHUNK_OVERLAP,
            "table_name": config.table_name,
        }
    }

    samples_to_test = min(len(dataset), max_samples) if max_samples else len(dataset)
    samples = list(dataset)[:samples_to_test]

    for batch_start in range(0, len(samples), Config.BATCH_SIZE):
        batch_end = min(batch_start + Config.BATCH_SIZE, len(samples))
        batch = samples[batch_start:batch_end]

        tasks = [
            process_single_hotpotqa_sample(rag, item, batch_start + i, retrieval_limit)
            for i, item in enumerate(batch)
        ]

        batch_results = await asyncio.gather(*tasks)

        for prediction_entry in batch_results:
            if prediction_entry is None:
                continue

            results["exact_match"] += int(prediction_entry["exact_match"])
            results["contains"] += int(prediction_entry["contains"])
            results["f1_scores"].append(prediction_entry["f1"])
            results["precision_scores"].append(prediction_entry["precision"])
            results["recall_scores"].append(prediction_entry["recall"])
            results["relevance_scores"].append(prediction_entry["relevance"])
            results["utilization_scores"].append(prediction_entry["utilization"])
            results["adherence_scores"].append(prediction_entry["adherence"])
            results["predictions"].append(prediction_entry)
            results["total"] += 1

            # Track by question type and difficulty
            q_type = prediction_entry.get("type", "unknown")
            q_level = prediction_entry.get("level", "unknown")
            if q_type in results["by_type"]:
                results["by_type"][q_type].append(prediction_entry["f1"])
            if q_level in results["by_level"]:
                results["by_level"][q_level].append(prediction_entry["f1"])

        if results["total"] > 0:
            avg_f1 = sum(results["f1_scores"]) / len(results["f1_scores"])
            avg_adherence = sum(results["adherence_scores"]) / len(results["adherence_scores"])
            logger.info(f"Progress: {batch_end}/{samples_to_test} | F1: {avg_f1:.3f} | Adherence: {avg_adherence:.3f}")

    return results


def run_hotpotqa_benchmark(
    rag,
    dataset,
    config: HotPotQAConfig,
    max_samples: Optional[int] = None,
    retrieval_limit: Optional[int] = None,
    use_hyde: bool = False,
) -> Dict:
    """Run official HotPotQA benchmark."""
    return asyncio.run(run_hotpotqa_benchmark_async(
        rag, dataset, config, max_samples, retrieval_limit, use_hyde
    ))


async def process_single_hotpotqa_sample_llm_only(
    async_llm,
    model: str,
    item,
    idx: int,
):
    """Process a single HotPotQA sample using LLM only (no RAG)."""
    question = item.get("question", "")
    ground_truth = item.get("answer", "")

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


async def run_hotpotqa_llm_only_benchmark_async(
    dataset,
    config: HotPotQAConfig,
    max_samples: Optional[int] = None,
) -> Dict:
    """Run HotPotQA LLM-only benchmark (baseline without RAG)."""
    from openai import AsyncOpenAI

    async_llm = AsyncOpenAI(
        api_key=Config.DEEPSEEK_API_KEY,
        base_url=Config.DEEPSEEK_BASE_URL
    )
    model = Config.DEEPSEEK_MODEL

    logger.info(f"Running {config.name} LLM-ONLY baseline with {Config.BATCH_SIZE} concurrent requests...")

    results = {
        "dataset": config.name + "-llm-only",
        "total": 0,
        "exact_match": 0,
        "contains": 0,
        "f1_scores": [],
        "precision_scores": [],
        "recall_scores": [],
        "predictions": [],
        "by_type": {"bridge": [], "comparison": []},
        "by_level": {"easy": [], "medium": [], "hard": []},
        "metadata": {
            "max_samples": max_samples,
            "llm_model": model,
            "mode": "llm-only-baseline",
        }
    }

    samples_to_test = min(len(dataset), max_samples) if max_samples else len(dataset)
    samples = list(dataset)[:samples_to_test]

    for batch_start in range(0, len(samples), Config.BATCH_SIZE):
        batch_end = min(batch_start + Config.BATCH_SIZE, len(samples))
        batch = samples[batch_start:batch_end]

        tasks = [
            process_single_hotpotqa_sample_llm_only(async_llm, model, item, batch_start + i)
            for i, item in enumerate(batch)
        ]

        batch_results = await asyncio.gather(*tasks)

        for prediction_entry in batch_results:
            if prediction_entry is None:
                continue

            results["exact_match"] += int(prediction_entry["exact_match"])
            results["contains"] += int(prediction_entry["contains"])
            results["f1_scores"].append(prediction_entry["f1"])
            results["precision_scores"].append(prediction_entry["precision"])
            results["recall_scores"].append(prediction_entry["recall"])
            results["predictions"].append(prediction_entry)
            results["total"] += 1

            q_type = prediction_entry.get("type", "unknown")
            q_level = prediction_entry.get("level", "unknown")
            if q_type in results["by_type"]:
                results["by_type"][q_type].append(prediction_entry["f1"])
            if q_level in results["by_level"]:
                results["by_level"][q_level].append(prediction_entry["f1"])

        if results["total"] > 0:
            avg_f1 = sum(results["f1_scores"]) / len(results["f1_scores"])
            logger.info(f"Progress: {batch_end}/{samples_to_test} | F1: {avg_f1:.3f}")

    return results


def run_hotpotqa_llm_only_pipeline(max_samples: int = 50):
    """Run HotPotQA LLM-only baseline (no RAG)."""
    config = HotPotQAConfig()

    logger.info(f"{config.name.upper()} LLM-ONLY BASELINE")
    logger.info("=" * 60)

    eval_dataset = load_hotpotqa_dataset(split="validation")
    logger.info(f"Evaluating on validation split ({len(eval_dataset)} samples)")

    results = asyncio.run(run_hotpotqa_llm_only_benchmark_async(
        eval_dataset, config, max_samples=max_samples
    ))

    logger.info("\nResults by question type:")
    for q_type, scores in results["by_type"].items():
        if scores:
            avg = sum(scores) / len(scores)
            logger.info(f"  {q_type}: F1={avg:.3f} (n={len(scores)})")

    logger.info("\nResults by difficulty level:")
    for level, scores in results["by_level"].items():
        if scores:
            avg = sum(scores) / len(scores)
            logger.info(f"  {level}: F1={avg:.3f} (n={len(scores)})")

    print_results(results)
    save_results(results)

    logger.info("\nBenchmark complete")


def run_hotpotqa_pipeline(
    force_reload: bool = False,
    max_samples: int = 50,
    retrieval_limit: Optional[int] = None,
    use_hyde: bool = False,
):
    """
    Run official HotPotQA benchmark pipeline.

    Args:
        force_reload: Clear and reload the knowledge base
        max_samples: Maximum number of samples to evaluate
        retrieval_limit: Number of documents to retrieve
        use_hyde: Use HyDE RAG instead of standard RAG
    """
    config = HotPotQAConfig()

    logger.info(f"{config.name.upper()} Evaluation")
    logger.info("=" * 60)

    # Initialize RAG system with HotPotQA table
    if use_hyde:
        rag = HotPotQAHyDERAGSystem(table_name=config.table_name)
        rag_type = "HyDE"
    else:
        rag = HotPotQARAGSystem(table_name=config.table_name)
        rag_type = "standard"

    logger.info(f"Using {rag_type} RAG system with table '{config.table_name}'")
    rag.setup()

    try:
        # Load and prepare knowledge base
        if force_reload or rag.vector_search.count() == 0:
            logger.info("Loading official HotPotQA dev-distractor into knowledge base...")
            dataset = load_hotpotqa_dataset(split="validation")
            prepare_hotpotqa_knowledge_base(rag.vector_search, dataset, force_reload=force_reload)
        else:
            logger.info(f"Knowledge base already loaded ({rag.vector_search.count()} documents)")

        # Load evaluation dataset
        eval_dataset = load_hotpotqa_dataset(split="validation")
        logger.info(f"Evaluating on validation split ({len(eval_dataset)} samples)")

        # Run benchmark
        results = run_hotpotqa_benchmark(
            rag, eval_dataset, config,
            max_samples=max_samples,
            retrieval_limit=retrieval_limit,
            use_hyde=use_hyde,
        )

        # Print breakdown by type and level
        logger.info("\nResults by question type:")
        for q_type, scores in results["by_type"].items():
            if scores:
                avg = sum(scores) / len(scores)
                logger.info(f"  {q_type}: F1={avg:.3f} (n={len(scores)})")

        logger.info("\nResults by difficulty level:")
        for level, scores in results["by_level"].items():
            if scores:
                avg = sum(scores) / len(scores)
                logger.info(f"  {level}: F1={avg:.3f} (n={len(scores)})")

        print_results(results)
        save_results(results)

    finally:
        rag.close()
        logger.info("\nBenchmark complete")
