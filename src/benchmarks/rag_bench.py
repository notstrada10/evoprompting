import argparse
import json
import logging
import os
import random
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

from datasets import load_dataset, concatenate_datasets

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
    is_multiple_choice: bool
    source: str  # HuggingFace dataset path
    subset: Optional[str] = None  # For datasets with subsets (e.g., ragbench)

    # Field extractors
    question_field: str = "question"
    answer_field: str = "response"  # ground truth answer
    documents_field: str = "documents"  # for knowledge base

    # Multiple choice specific
    distractors_fields: List[str] = field(default_factory=list)


RAGBENCH_CONFIG = DatasetConfig(
    name="ragbench",
    is_multiple_choice=False,
    source=Config.RAGBENCH_DATASET,
    subset=Config.RAGBENCH_SUBSET,
    question_field="question",
    answer_field="response",
    documents_field="documents",
)

SCIQ_CONFIG = DatasetConfig(
    name="sciq",
    is_multiple_choice=True,
    source=Config.SCIQ_DATASET,
    subset=None,
    question_field="question",
    answer_field="correct_answer",
    documents_field="support",
    distractors_fields=["distractor1", "distractor2", "distractor3"],
)

DATASET_CONFIGS = {
    "ragbench": RAGBENCH_CONFIG,
    "sciq": SCIQ_CONFIG,
}


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

    # Collect unique documents
    documents_added = set()
    for item in dataset:
        docs = item.get(config.documents_field, [])
        if isinstance(docs, str):
            # Single document (e.g., SciQ support field)
            if docs.strip():
                documents_added.add(docs.strip())
        else:
            # List of documents (e.g., RagBench)
            for doc in docs:
                doc_text = doc if isinstance(doc, str) else str(doc)
                if doc_text:
                    documents_added.add(doc_text)

    logger.info(f"Embedding {len(documents_added)} unique documents...")
    texts_with_metadata = [(doc, {"source": config.name}) for doc in documents_added]
    rag.vector_search.add_texts(texts_with_metadata)
    logger.info(f"Added {len(documents_added)} documents to knowledge base")


# =============================================================================
# Metrics
# =============================================================================

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


def calculate_f1(predicted: str, ground_truth: str) -> float:
    """Calculate F1 score based on token overlap."""
    pred_tokens = set(predicted.lower().split())
    truth_tokens = set(ground_truth.lower().split())

    if not truth_tokens:
        return 1.0 if not pred_tokens else 0.0

    common = pred_tokens & truth_tokens
    if not common:
        return 0.0

    precision = len(common) / len(pred_tokens) if pred_tokens else 0.0
    recall = len(common) / len(truth_tokens)

    if precision + recall == 0:
        return 0.0
    return 2 * (precision * recall) / (precision + recall)


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


def evaluate_multiple_choice(predicted: str, correct_answer: str, distractors: List[str]) -> dict:
    """Evaluate a multiple choice answer."""
    predicted_lower = predicted.lower().strip()
    correct_lower = correct_answer.lower().strip()

    exact_match = predicted_lower == correct_lower
    contains_correct = correct_lower in predicted_lower
    selected_distractor = any(d.lower().strip() in predicted_lower for d in distractors)
    is_correct = exact_match or (contains_correct and not selected_distractor)

    return {
        "exact_match": exact_match,
        "contains_correct": contains_correct,
        "selected_distractor": selected_distractor,
        "is_correct": is_correct,
    }


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

def run_benchmark(rag, dataset, config: DatasetConfig, max_samples: int = None,
                  retrieval_limit: int = None, use_llm_judge: bool = False,
                  use_hyde: bool = False) -> Dict:
    """Run RAG benchmark on any dataset."""
    retrieval_limit = retrieval_limit or Config.DEFAULT_RETRIEVAL_LIMIT
    logger.info(f"Running {config.name} benchmark...")

    # Initialize results structure
    results = {
        "dataset": config.name,
        "is_multiple_choice": config.is_multiple_choice,
        "total": 0,
        "adherence_scores": [],
        "predictions": [],
        "metadata": {
            "retrieval_limit": retrieval_limit,
            "max_samples": max_samples,
            "use_hyde": use_hyde,
            "embedding_model": Config.EMBEDDING_MODEL,
            "llm_model": Config.DEEPSEEK_MODEL if Config.LLM_PROVIDER == "deepseek" else Config.GROQ_MODEL,
            "chunk_size": Config.CHUNK_SIZE,
            "chunk_overlap": Config.CHUNK_OVERLAP,
        }
    }

    if config.is_multiple_choice:
        results.update({"correct": 0, "exact_match": 0, "contains_correct": 0, "selected_distractor": 0})
    else:
        results.update({"exact_match": 0, "contains": 0, "f1_scores": [], "relevance_scores": [], "utilization_scores": []})

    if use_llm_judge:
        results["llm_judge_scores"] = []
        results["llm_correct_count"] = 0

    samples_to_test = min(len(dataset), max_samples) if max_samples else len(dataset)

    for idx, item in enumerate(dataset):
        if max_samples and idx >= max_samples:
            break

        question = item.get(config.question_field, '')
        ground_truth = item.get(config.answer_field, '')
        if not question or not ground_truth:
            continue

        # Build query (with MC options if applicable)
        if config.is_multiple_choice:
            distractors = [item.get(f, '') for f in config.distractors_fields]
            all_options = [ground_truth] + distractors
            random.seed(idx)
            random.shuffle(all_options)
            options_text = "\n".join([f"{chr(65+i)}. {opt}" for i, opt in enumerate(all_options)])
            correct_letter = chr(65 + all_options.index(ground_truth))
            query = f"{question}\n\nOptions:\n{options_text}\n\nAnswer with ONLY the exact text of the correct option, without the letter."
        else:
            query = question
            distractors = []
            correct_letter = None
            gold_docs = item.get(config.documents_field, [])

        # Get RAG response
        result = rag.ask(query, limit=retrieval_limit)
        predicted = result['answer']
        retrieved_docs = result['sources']

        # Log first few examples
        if idx < 3:
            logger.info(f"\n--- Query {idx+1}: {question[:80]}...")
            logger.info(f"Ground truth: {ground_truth[:100]}...")
            logger.info(f"Generated: {predicted[:100]}...")

        # Calculate metrics
        adherence = calculate_adherence(predicted, retrieved_docs)

        if config.is_multiple_choice:
            mc_eval = evaluate_multiple_choice(predicted, ground_truth, distractors)
            results["correct"] += int(mc_eval["is_correct"])
            results["exact_match"] += int(mc_eval["exact_match"])
            results["contains_correct"] += int(mc_eval["contains_correct"])
            results["selected_distractor"] += int(mc_eval["selected_distractor"])

            prediction_entry = {
                "id": idx,
                "question": question,
                "predicted": predicted,
                "correct_answer": ground_truth,
                "correct_letter": correct_letter,
                "distractors": distractors,
                "is_correct": mc_eval["is_correct"],
                "exact_match": mc_eval["exact_match"],
                "contains_correct": mc_eval["contains_correct"],
                "selected_distractor": mc_eval["selected_distractor"],
                "adherence": adherence,
                "sources": retrieved_docs
            }
        else:
            exact = predicted.strip().lower() == ground_truth.strip().lower()
            contains = ground_truth.strip().lower() in predicted.strip().lower()
            f1 = calculate_f1(predicted, ground_truth)
            relevance = calculate_context_relevance(retrieved_docs, gold_docs)
            utilization = calculate_context_utilization(predicted, retrieved_docs)

            results["exact_match"] += int(exact)
            results["contains"] += int(contains)
            results["f1_scores"].append(f1)
            results["relevance_scores"].append(relevance)
            results["utilization_scores"].append(utilization)

            prediction_entry = {
                "id": item.get('id', idx),
                "query": question,
                "predicted": predicted,
                "ground_truth": ground_truth,
                "exact_match": exact,
                "contains": contains,
                "f1": f1,
                "relevance": relevance,
                "utilization": utilization,
                "adherence": adherence,
                "sources": retrieved_docs
            }

        # LLM judge (optional)
        if use_llm_judge:
            judge_result = llm_judge_correctness(predicted, ground_truth, question, rag.llm, rag.model)
            results["llm_judge_scores"].append(judge_result["score"])
            results["llm_correct_count"] += int(judge_result["is_correct"])
            prediction_entry["llm_judge"] = judge_result

        results["total"] += 1
        results["adherence_scores"].append(adherence)
        results["predictions"].append(prediction_entry)

        # Progress logging
        if (idx + 1) % 10 == 0:
            avg_adherence = sum(results["adherence_scores"]) / len(results["adherence_scores"])
            if config.is_multiple_choice:
                accuracy = results["correct"] / results["total"]
                progress_msg = f"Progress: {idx + 1}/{samples_to_test} | Accuracy: {accuracy:.1%} | Adherence: {avg_adherence:.3f}"
            else:
                avg_f1 = sum(results["f1_scores"]) / len(results["f1_scores"])
                progress_msg = f"Progress: {idx + 1}/{samples_to_test} | F1: {avg_f1:.3f} | Adherence: {avg_adherence:.3f}"
            if use_llm_judge:
                progress_msg += f" | LLM Judge: {results['llm_correct_count'] / results['total']:.1%}"
            logger.info(progress_msg)

    return results


# =============================================================================
# Results Output
# =============================================================================

def compute_summary(results: Dict) -> Dict:
    """Compute summary statistics from results."""
    total = results["total"]
    if total == 0:
        return {}

    avg_adherence = sum(results["adherence_scores"]) / len(results["adherence_scores"])

    if results["is_multiple_choice"]:
        accuracy = results["correct"] / total
        summary = {
            "dataset": results["dataset"],
            "accuracy": accuracy,
            "exact_match_rate": results["exact_match"] / total,
            "contains_correct_rate": results["contains_correct"] / total,
            "selected_distractor_rate": results["selected_distractor"] / total,
            "avg_adherence": avg_adherence,
            "fitness_score": accuracy,
        }
    else:
        avg_f1 = sum(results["f1_scores"]) / len(results["f1_scores"])
        summary = {
            "dataset": results["dataset"],
            "exact_match_rate": results["exact_match"] / total,
            "contains_rate": results["contains"] / total,
            "avg_f1": avg_f1,
            "avg_relevance": sum(results["relevance_scores"]) / len(results["relevance_scores"]),
            "avg_utilization": sum(results["utilization_scores"]) / len(results["utilization_scores"]),
            "avg_adherence": avg_adherence,
            "fitness_score": avg_f1,
        }

    if "llm_judge_scores" in results:
        summary["llm_judge_accuracy"] = results["llm_correct_count"] / total
        summary["fitness_score"] = summary["llm_judge_accuracy"]

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

    logger.info(f"\n{'='*60}")
    logger.info(f"  FITNESS SCORE: {summary['fitness_score']:.1%}")
    logger.info(f"{'='*60}")

    logger.info(f"\n  Total samples: {total}")

    if results["is_multiple_choice"]:
        logger.info(f"  Accuracy: {summary['accuracy']:.2%}")
        logger.info(f"  Exact Match: {summary['exact_match_rate']:.2%}")
        logger.info(f"  Contains Correct: {summary['contains_correct_rate']:.2%}")
        logger.info(f"  Selected Distractor: {summary['selected_distractor_rate']:.2%}")
    else:
        logger.info(f"  Exact Match: {summary['exact_match_rate']:.2%}")
        logger.info(f"  Contains Answer: {summary['contains_rate']:.2%}")
        logger.info(f"  Average F1 Score: {summary['avg_f1']:.3f}")
        logger.info(f"  Context Relevance: {summary['avg_relevance']:.3f}")
        logger.info(f"  Context Utilization: {summary['avg_utilization']:.3f}")

    logger.info(f"  Adherence: {summary['avg_adherence']:.3f}")

    if "llm_judge_accuracy" in summary:
        logger.info(f"  LLM Judge Accuracy: {summary['llm_judge_accuracy']:.2%}")

    # Sample predictions
    logger.info(f"\nSample Predictions:")
    for i, pred in enumerate(results["predictions"][:3]):
        if results["is_multiple_choice"]:
            status = "correct" if pred["is_correct"] else "wrong"
            logger.info(f"\n  Example {i+1} [{status}]:")
            logger.info(f"    Question: {pred['question'][:100]}...")
            logger.info(f"    Correct: {pred['correct_letter']}. {pred['correct_answer']}")
            logger.info(f"    Predicted: {pred['predicted'][:150]}...")
        else:
            logger.info(f"\n  Example {i+1}:")
            logger.info(f"    Query: {pred['query'][:100]}...")
            logger.info(f"    Predicted: {pred['predicted'][:150]}...")
            logger.info(f"    Ground Truth: {pred['ground_truth'][:150]}...")
            logger.info(f"    F1: {pred['f1']:.3f}")


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

def main():
    """Main entry point for RAG benchmark."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    parser = argparse.ArgumentParser(description="Run RAG benchmark")
    parser.add_argument('--force-reload', action='store_true',
                       help='Clear and reload database')
    parser.add_argument('--max-samples', type=int, default=50,
                       help='Maximum number of samples to test')
    parser.add_argument('--retrieval-limit', type=int, default=None,
                       help=f'Documents to retrieve per query (default: {Config.DEFAULT_RETRIEVAL_LIMIT})')
    parser.add_argument('--use-llm-judge', action='store_true',
                       help='Enable LLM-as-judge for evaluation')
    parser.add_argument('--eval-split', type=str, default='test',
                       choices=['validation', 'test'],
                       help='Evaluation split (default: test)')
    parser.add_argument('--use-hyde', action='store_true',
                       help='Use HyDE RAG instead of standard RAG')
    parser.add_argument('--dataset', type=str, default=Config.BENCHMARK_DATASET,
                       choices=['ragbench', 'sciq'],
                       help=f'Dataset to use (default: {Config.BENCHMARK_DATASET})')
    args = parser.parse_args()

    config = DATASET_CONFIGS[args.dataset]
    logger.info(f"{config.name.upper()} Evaluation")
    logger.info("="*60)

    # Initialize RAG system
    rag = HyDERAGSystem() if args.use_hyde else RAGSystem()
    logger.info(f"Using {'HyDE' if args.use_hyde else 'standard'} RAG system")
    rag.setup()

    try:
        # Load and prepare knowledge base
        if args.force_reload or rag.vector_search.count() == 0:
            logger.info("Loading all splits into knowledge base...")
            all_datasets = []
            for split in ["train", "validation", "test"]:
                split_data = load_benchmark_dataset(config, split=split)
                all_datasets.append(split_data)
                logger.info(f"  - {split}: {len(split_data)} samples")

            combined_dataset = concatenate_datasets(all_datasets)
            logger.info(f"Total samples: {len(combined_dataset)}")
            prepare_knowledge_base(rag, combined_dataset, config, force_reload=args.force_reload)
        else:
            logger.info(f"Knowledge base already loaded ({rag.vector_search.count()} documents)")

        # Run benchmark
        eval_dataset = load_benchmark_dataset(config, split=args.eval_split)
        logger.info(f"Evaluating on {args.eval_split} split ({len(eval_dataset)} samples)")

        results = run_benchmark(
            rag, eval_dataset, config,
            max_samples=args.max_samples,
            retrieval_limit=args.retrieval_limit,
            use_llm_judge=args.use_llm_judge,
            use_hyde=args.use_hyde
        )
        print_results(results)
        save_results(results)

    finally:
        rag.close()
        logger.info("\nBenchmark complete")


if __name__ == "__main__":
    main()
