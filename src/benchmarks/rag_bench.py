import asyncio
import json
import logging
import os
import re
import string
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional

from datasets import concatenate_datasets, load_dataset

from ..config import Config
from ..core.hyde_rag import HyDERAGSystem
from ..core.rag import RAGSystem

logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
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

def extract_documents_from_item(item, config: DatasetConfig) -> list[str]:
    """Extract document texts from a dataset item."""
    docs_field = item.get(config.documents_field, [])

    if isinstance(docs_field, list):
        return [doc if isinstance(doc, str) else str(doc) for doc in docs_field if doc]

    if isinstance(docs_field, str) and docs_field.strip():
        return [docs_field.strip()]

    return []


def load_benchmark_dataset(config: DatasetConfig, split: str = "test"):
    """Load a benchmark dataset."""
    logger.info(f"Loading {config.name} dataset ({split} split)...")
    if config.subset:
        dataset = load_dataset(config.source, config.subset, split=split)
    else:
        dataset = load_dataset(config.source, split=split)
    logger.info(f"Loaded {len(dataset)} samples")
    return dataset


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
# Metrics (HotPotQA official evaluation)
# =============================================================================

def normalize_answer(s: str) -> str:
    """Normalize answer for evaluation (official HotPotQA normalization)."""
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def calculate_f1(prediction: str, ground_truth: str) -> tuple[float, float, float]:
    """Calculate F1, Precision, Recall (official HotPotQA implementation).

    Returns:
        Tuple of (f1, precision, recall)
    """
    normalized_prediction = normalize_answer(prediction)
    normalized_ground_truth = normalize_answer(ground_truth)

    ZERO_METRIC = (0.0, 0.0, 0.0)

    # Special handling for yes/no answers
    if normalized_prediction in ['yes', 'no', 'noanswer'] and normalized_prediction != normalized_ground_truth:
        return ZERO_METRIC
    if normalized_ground_truth in ['yes', 'no', 'noanswer'] and normalized_prediction != normalized_ground_truth:
        return ZERO_METRIC

    prediction_tokens = normalized_prediction.split()
    ground_truth_tokens = normalized_ground_truth.split()

    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())

    if num_same == 0:
        return ZERO_METRIC

    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1, precision, recall


def calculate_exact_match(prediction: str, ground_truth: str) -> bool:
    """Calculate exact match (official HotPotQA implementation)."""
    return normalize_answer(prediction) == normalize_answer(ground_truth)


# Stopwords for adherence calculation
STOPWORDS = {
    'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
    'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
    'should', 'may', 'might', 'can', 'to', 'of', 'in', 'for', 'on', 'with',
    'at', 'by', 'from', 'as', 'into', 'through', 'during', 'before', 'after',
    'above', 'below', 'between', 'under', 'again', 'further', 'then', 'once',
    'and', 'but', 'or', 'nor', 'so', 'yet', 'both', 'each', 'few', 'more',
    'most', 'other', 'some', 'such', 'no', 'not', 'only', 'same', 'than',
    'too', 'very', 'just', 'also', 'now', 'it', 'its', 'this', 'that',
    'these', 'those', 'i', 'you', 'he', 'she', 'they', 'we', 'what', 'which',
    'who', 'whom', 'how', 'when', 'where', 'why', 'all', 'any', 'if'
}


def calculate_context_relevance(retrieved_docs: List[str], gold_docs: List[str]) -> float:
    """Calculate how many retrieved docs are in the gold set."""
    if not retrieved_docs or not gold_docs:
        return 0.0
    retrieved_set = set(doc.strip().lower() for doc in retrieved_docs)
    gold_set = set(doc.strip().lower() for doc in gold_docs)
    overlap = len(retrieved_set & gold_set)
    return overlap / len(retrieved_set) if retrieved_set else 0.0


def calculate_context_utilization(answer: str, retrieved_docs: List[str]) -> float:
    """Calculate how much of the retrieved docs are used in the answer."""
    if not answer or not retrieved_docs:
        return 0.0
    answer_tokens = set(answer.lower().split())
    docs_used = sum(1 for doc in retrieved_docs if len(answer_tokens & set(doc.lower().split())) > 3)
    return docs_used / len(retrieved_docs)


def calculate_adherence(answer: str, retrieved_docs: List[str]) -> float:
    """Calculate if the answer is grounded in the retrieved docs."""
    if not answer or not retrieved_docs:
        return 0.0

    answer_tokens = set(answer.lower().split()) - STOPWORDS
    if not answer_tokens:
        return 1.0

    all_doc_tokens = set()
    for doc in retrieved_docs:
        all_doc_tokens.update(doc.lower().split())

    grounded_tokens = answer_tokens & all_doc_tokens
    return len(grounded_tokens) / len(answer_tokens)


def llm_judge_correctness(predicted: str, ground_truth: str, question: str, llm_client, model: str) -> dict:
    """Use an LLM to judge if the predicted answer is correct."""
    judge_prompt = f"""You are an expert evaluator judging the correctness of question-answering systems.

    Question: {question}
    Ground Truth Answer: {ground_truth}
    Predicted Answer: {predicted}

    Evaluate if the predicted answer is semantically equivalent to the ground truth answer.
    Respond in this exact format:
        CORRECT: [yes/no]
        CONFIDENCE: [0.0-1.0]
        REASONING: [brief explanation]"""

    try:
        response = llm_client.chat.completions.create(
            messages=[
                {"role": "system", "content": "You are a precise answer evaluator."},
                {"role": "user", "content": judge_prompt}
            ],
            model=model, temperature=0, max_tokens=200
        )
        content = response.choices[0].message.content.strip()

        is_correct, confidence, reasoning = False, 0.5, ""
        for line in content.split('\n'):
            line = line.strip()
            if line.startswith("CORRECT:"):
                is_correct = "yes" in line.lower()
            elif line.startswith("CONFIDENCE:"):
                try:
                    confidence = float(line.split(":")[-1].strip())
                except:
                    confidence = 1.0 if is_correct else 0.0
            elif line.startswith("REASONING:"):
                reasoning = line.split(":", 1)[-1].strip()

        return {
            "score": confidence if is_correct else (1.0 - confidence),
            "is_correct": is_correct,
            "confidence": confidence,
            "reasoning": reasoning
        }
    except Exception as e:
        logger.error(f"LLM judge error: {e}")
        return {"score": 0.0, "is_correct": False, "confidence": 0.0, "reasoning": str(e)}


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
    logger.info(f"{results['dataset'].upper()} BENCHMARK RESULTS")
    logger.info("="*60)

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



def save_results(results: Dict, output_dir: str = None) -> None:
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
):
    """Run RAG benchmark pipeline."""
    config = _make_ragbench_config(subset)
    retrieval_limit = retrieval_limit or Config.DEFAULT_RETRIEVAL_LIMIT
    logger.info(f"{config.name.upper()} Evaluation")
    logger.info("="*60)

    if use_hyde:
        rag = HyDERAGSystem()
        rag_type = "HyDE"
    else:
        rag = RAGSystem()
        rag_type = "standard"
    logger.info(f"Using {rag_type} RAG system")
    rag.setup()

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

    results = run_benchmark(eval_dataset, config, max_samples=max_samples)
    print_results(results)
    save_results(results)

    logger.info("\nBenchmark complete")
