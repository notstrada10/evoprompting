import argparse
import json
import logging
import os
from datetime import datetime
from typing import Dict, List

import numpy as np
from datasets import load_dataset

from ..config import Config
from ..core.rag import RAGSystem

logger = logging.getLogger(__name__)


def load_dataset_from_ragbench(dataset_name: str, split: str = "test"):
    """
    Load dataset from RagBench.

    Args:
        dataset_name: Name of the dataset to load.
        split: Dataset split to load.

    Returns:
        Loaded dataset.
    """
    logger.info(f"Loading {dataset_name} dataset from RagBench...")
    dataset = load_dataset(Config.RAGBENCH_DATASET, dataset_name, split=split)
    logger.info(f"Loaded {len(dataset)} samples")
    return dataset


def prepare_knowledge_base(rag: RAGSystem, dataset, force_reload: bool = False) -> None:
    """
    Extract and add all unique documents to RAG knowledge base.

    Args:
        rag: RAG system instance.
        dataset: Dataset to extract documents from.
        force_reload: Whether to clear and reload the database.
    """
    current_count = rag.vector_search.count()

    if current_count > 0 and not force_reload:
        logger.info(f"Knowledge base already contains {current_count} documents, skipping load")
        logger.info("Use --force-reload to clear and reload the database")
        return

    if force_reload and current_count > 0:
        logger.info(f"Clearing existing {current_count} documents...")
        rag.vector_search.delete_all()

    logger.info("Preparing knowledge base...")

    # Collect unique documents first
    documents_added = set()
    for item in dataset:
        docs = item.get('documents', [])
        for doc in docs:
            doc_text = doc if isinstance(doc, str) else str(doc)
            if doc_text:
                documents_added.add(doc_text)

    # Batch add all documents
    logger.info(f"Embedding {len(documents_added)} unique documents in batches...")
    texts_with_metadata = [(doc, {"source": "ragbench"}) for doc in documents_added]
    rag.vector_search.add_texts(texts_with_metadata)

    logger.info(f"Added {len(documents_added)} documents to knowledge base")


def calculate_f1(predicted: str, ground_truth: str) -> float:
    """
    Calculate F1 score based on token overlap.

    Args:
        predicted: Predicted answer.
        ground_truth: Ground truth answer.

    Returns:
        F1 score between 0 and 1.
    """
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
    """
    Calculate context relevance: how many retrieved docs are in the gold set.

    Args:
        retrieved_docs: Documents retrieved by the system.
        gold_docs: Ground truth relevant documents.

    Returns:
        Relevance score between 0 and 1.
    """
    if not retrieved_docs or not gold_docs:
        return 0.0

    # Normalize for comparison
    retrieved_set = set(doc.strip().lower() for doc in retrieved_docs)
    gold_set = set(doc.strip().lower() for doc in gold_docs)

    # Check overlap
    overlap = len(retrieved_set & gold_set)
    return overlap / len(retrieved_set) if retrieved_set else 0.0


def calculate_context_utilization(answer: str, retrieved_docs: List[str]) -> float:
    """
    Calculate context utilization: how much of the retrieved docs are used in the answer.

    Args:
        answer: Generated answer.
        retrieved_docs: Documents retrieved by the system.

    Returns:
        Utilization score between 0 and 1.
    """
    if not answer or not retrieved_docs:
        return 0.0

    answer_tokens = set(answer.lower().split())

    docs_used = 0
    for doc in retrieved_docs:
        doc_tokens = set(doc.lower().split())
        # If significant overlap exists, consider doc as used
        overlap = len(answer_tokens & doc_tokens)
        if overlap > 3:  # At least 3 common tokens
            docs_used += 1

    return docs_used / len(retrieved_docs)


def calculate_adherence(answer: str, retrieved_docs: List[str]) -> float:
    """
    Calculate adherence: is the answer grounded in the retrieved docs?
    Simple heuristic based on token overlap.

    Args:
        answer: Generated answer.
        retrieved_docs: Documents retrieved by the system.

    Returns:
        Adherence score between 0 and 1.
    """
    if not answer or not retrieved_docs:
        return 0.0

    answer_tokens = set(answer.lower().split())
    # Remove common stopwords
    stopwords = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been',
                 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
                 'would', 'could', 'should', 'may', 'might', 'can', 'to', 'of',
                 'in', 'for', 'on', 'with', 'at', 'by', 'from', 'as', 'into',
                 'through', 'during', 'before', 'after', 'above', 'below',
                 'between', 'under', 'again', 'further', 'then', 'once', 'and',
                 'but', 'or', 'nor', 'so', 'yet', 'both', 'each', 'few', 'more',
                 'most', 'other', 'some', 'such', 'no', 'not', 'only', 'same',
                 'than', 'too', 'very', 'just', 'also', 'now', 'it', 'its',
                 'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she',
                 'they', 'we', 'what', 'which', 'who', 'whom', 'how', 'when',
                 'where', 'why', 'all', 'any', 'if'}

    answer_tokens = answer_tokens - stopwords

    if not answer_tokens:
        return 1.0  # Empty answer after removing stopwords

    # Combine all doc tokens
    all_doc_tokens = set()
    for doc in retrieved_docs:
        all_doc_tokens.update(doc.lower().split())

    # Calculate how many answer tokens appear in docs
    grounded_tokens = answer_tokens & all_doc_tokens
    return len(grounded_tokens) / len(answer_tokens)


def llm_judge_correctness(predicted: str, ground_truth: str, question: str, llm_client, model: str) -> dict:
    """
    Use an LLM to judge if the predicted answer is correct.

    Args:
        predicted: Predicted answer.
        ground_truth: Ground truth answer.
        question: The original question.
        llm_client: LLM client (OpenAI-compatible).
        model: Model name to use.

    Returns:
        Dict with 'score' (0-1), 'reasoning', and 'is_correct' (bool).
    """
    judge_prompt = f"""You are an expert evaluator judging the correctness of question-answering systems.

Question: {question}

Ground Truth Answer: {ground_truth}

Predicted Answer: {predicted}

Evaluate if the predicted answer is semantically equivalent to the ground truth answer. Consider:
1. Do they convey the same core information?
2. Are factual details accurate (dates, names, numbers)?
3. Minor wording differences are acceptable if the meaning is preserved.

Respond in this exact format:
CORRECT: [yes/no]
CONFIDENCE: [0.0-1.0]
REASONING: [brief explanation]"""

    try:
        response = llm_client.chat.completions.create(
            messages=[
                {"role": "system", "content": "You are a precise answer evaluator. Always follow the requested format exactly."},
                {"role": "user", "content": judge_prompt}
            ],
            model=model,
            temperature=0,
            max_tokens=200
        )

        content = response.choices[0].message.content.strip()

        # Parse response
        lines = content.split('\n')
        is_correct = False
        confidence = 0.5
        reasoning = ""

        for line in lines:
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

        score = confidence if is_correct else (1.0 - confidence)

        return {
            "score": score,
            "is_correct": is_correct,
            "confidence": confidence,
            "reasoning": reasoning
        }

    except Exception as e:
        logger.error(f"LLM judge error: {e}")
        return {
            "score": 0.0,
            "is_correct": False,
            "confidence": 0.0,
            "reasoning": f"Error: {str(e)}"
        }


def run_benchmark(rag: RAGSystem, dataset, max_samples: int = None, retrieval_limit: int = None, use_llm_judge: bool = False) -> Dict:
    """
    Run RAG system on benchmark and evaluate.

    Args:
        rag: RAG system instance.
        dataset: Dataset to evaluate on.
        max_samples: Maximum number of samples to test.
        retrieval_limit: Number of documents to retrieve per query.

    Returns:
        Dictionary with evaluation results.
    """
    retrieval_limit = retrieval_limit or Config.DEFAULT_RETRIEVAL_LIMIT
    logger.info("Running benchmark...")

    results = {
        "total": 0,
        "exact_match": 0,
        "contains": 0,
        "f1_scores": [],
        "relevance_scores": [],
        "utilization_scores": [],
        "adherence_scores": [],
        "predictions": []
    }

    if use_llm_judge:
        results["llm_judge_scores"] = []
        results["llm_correct_count"] = 0
        logger.info("LLM-as-judge enabled for evaluation")

    samples_to_test = min(len(dataset), max_samples) if max_samples else len(dataset)

    for idx, item in enumerate(dataset):
        if max_samples and idx >= max_samples:
            break

        query = item.get('question', '')
        ground_truth = item.get('response', '')
        gold_docs = item.get('documents', [])

        if not query or not ground_truth:
            continue

        result = rag.ask(query, limit=retrieval_limit)
        predicted = result['answer']
        retrieved_docs = result['sources']

        if idx < 3:
            logger.info(f"\n--- Query {idx+1}: {query[:80]}...")
            logger.info(f"Ground truth: {ground_truth[:100]}...")
            logger.info(f"Retrieved docs:")
            for i, src in enumerate(retrieved_docs):
                logger.info(f"  {i+1}. {src[:100]}...")
            logger.info(f"Generated: {predicted[:100]}...")

        # Answer quality metrics
        exact = 1.0 if predicted.strip().lower() == ground_truth.strip().lower() else 0.0
        contains = 1.0 if ground_truth.strip().lower() in predicted.strip().lower() else 0.0
        f1 = calculate_f1(predicted, ground_truth)

        # RAGBench component metrics
        relevance = calculate_context_relevance(retrieved_docs, gold_docs)
        utilization = calculate_context_utilization(predicted, retrieved_docs)
        adherence = calculate_adherence(predicted, retrieved_docs)

        # LLM judge evaluation (optional)
        llm_judge_result = None
        if use_llm_judge:
            llm_judge_result = llm_judge_correctness(
                predicted, ground_truth, query,
                rag.llm, rag.model
            )
            results["llm_judge_scores"].append(llm_judge_result["score"])
            if llm_judge_result["is_correct"]:
                results["llm_correct_count"] += 1

        results["total"] += 1
        results["exact_match"] += exact
        results["contains"] += contains
        results["f1_scores"].append(f1)
        results["relevance_scores"].append(relevance)
        results["utilization_scores"].append(utilization)
        results["adherence_scores"].append(adherence)

        prediction_entry = {
            "id": item.get('id'),
            "query": query,
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

        if use_llm_judge and llm_judge_result:
            prediction_entry["llm_judge"] = {
                "score": llm_judge_result["score"],
                "is_correct": llm_judge_result["is_correct"],
                "confidence": llm_judge_result["confidence"],
                "reasoning": llm_judge_result["reasoning"]
            }

        results["predictions"].append(prediction_entry)

        if (idx + 1) % 10 == 0:
            avg_f1 = sum(results["f1_scores"]) / len(results["f1_scores"])
            avg_adherence = sum(results["adherence_scores"]) / len(results["adherence_scores"])
            progress_msg = f"Progress: {idx + 1}/{samples_to_test} | F1: {avg_f1:.3f} | Adherence: {avg_adherence:.3f}"
            if use_llm_judge:
                llm_accuracy = results["llm_correct_count"] / results["total"]
                progress_msg += f" | LLM Judge: {llm_accuracy:.1%}"
            logger.info(progress_msg)

    return results


def print_results(results: Dict) -> None:
    """
    Print benchmark results.

    Args:
        results: Results dictionary from run_benchmark.
    """
    logger.info("\n" + "="*60)
    logger.info("BENCHMARK RESULTS")
    logger.info("="*60)

    total = results["total"]
    if total == 0:
        logger.warning("No samples processed")
        return

    exact_match_rate = results["exact_match"] / total
    contains_rate = results["contains"] / total
    avg_f1 = sum(results["f1_scores"]) / len(results["f1_scores"])
    avg_relevance = sum(results["relevance_scores"]) / len(results["relevance_scores"])
    avg_utilization = sum(results["utilization_scores"]) / len(results["utilization_scores"])
    avg_adherence = sum(results["adherence_scores"]) / len(results["adherence_scores"])

    # Calculate fitness score
    if "llm_judge_scores" in results:
        fitness_score = results["llm_correct_count"] / total
        fitness_metric = "LLM Judge Accuracy"
    else:
        fitness_score = avg_f1
        fitness_metric = "F1 Score"

    # Display fitness score prominently
    logger.info(f"\n{'='*60}")
    logger.info(f"  🎯 FITNESS SCORE: {fitness_score:.1%} ({fitness_metric})")
    logger.info(f"{'='*60}")

    logger.info(f"\n=== Answer Quality Metrics ===")
    logger.info(f"  Total samples: {total}")
    logger.info(f"  Exact Match: {exact_match_rate:.2%}")
    logger.info(f"  Contains Answer: {contains_rate:.2%}")
    logger.info(f"  Average F1 Score: {avg_f1:.3f}")

    if "llm_judge_scores" in results:
        llm_accuracy = results["llm_correct_count"] / total
        avg_llm_score = sum(results["llm_judge_scores"]) / len(results["llm_judge_scores"])
        logger.info(f"  LLM Judge Accuracy: {llm_accuracy:.2%}")
        logger.info(f"  LLM Judge Avg Score: {avg_llm_score:.3f}")

    logger.info(f"\n=== RAG Component Metrics ===")
    logger.info(f"  Context Relevance: {avg_relevance:.3f}  (Are retrieved docs relevant?)")
    logger.info(f"  Context Utilization: {avg_utilization:.3f}  (Is context used in answer?)")
    logger.info(f"  Adherence (Groundedness): {avg_adherence:.3f}  (Is answer grounded in docs?)")

    logger.info(f"\nSample Predictions:")
    for i, pred in enumerate(results["predictions"][:3]):
        logger.info(f"\n  Example {i+1} (ID: {pred.get('id')}):")
        logger.info(f"    Query: {pred['query'][:100]}...")
        logger.info(f"    Predicted: {pred['predicted'][:150]}...")
        logger.info(f"    Ground Truth: {pred['ground_truth'][:150]}...")
        logger.info(f"    F1: {pred['f1']:.3f} | Relevance: {pred['relevance']:.3f} | Adherence: {pred['adherence']:.3f}")


def save_results(results: Dict, output_dir: str = None) -> None:
    """
    Save results to JSON file.

    Args:
        results: Results dictionary to save.
        output_dir: Directory to save results. Defaults to Config.RESULTS_DIR.
    """
    output_dir = output_dir or Config.RESULTS_DIR
    os.makedirs(output_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(output_dir, f"benchmark_{timestamp}.json")

    # Calculate summary stats
    total = results["total"]
    summary = {
        "exact_match_rate": results["exact_match"] / total if total else 0,
        "contains_rate": results["contains"] / total if total else 0,
        "avg_f1": sum(results["f1_scores"]) / len(results["f1_scores"]) if results["f1_scores"] else 0,
        "avg_relevance": sum(results["relevance_scores"]) / len(results["relevance_scores"]) if results["relevance_scores"] else 0,
        "avg_utilization": sum(results["utilization_scores"]) / len(results["utilization_scores"]) if results["utilization_scores"] else 0,
        "avg_adherence": sum(results["adherence_scores"]) / len(results["adherence_scores"]) if results["adherence_scores"] else 0,
    }

    # Add LLM judge metrics if available
    if "llm_judge_scores" in results:
        summary["llm_judge_accuracy"] = results["llm_correct_count"] / total if total else 0
        summary["llm_judge_avg_score"] = sum(results["llm_judge_scores"]) / len(results["llm_judge_scores"]) if results["llm_judge_scores"] else 0
        # Fitness function: LLM Judge Accuracy (primary optimization target)
        summary["fitness_score"] = summary["llm_judge_accuracy"]
    else:
        # Fallback if LLM judge not used: use F1 score
        summary["fitness_score"] = summary["avg_f1"]

    # Create output with summary at the top
    output = {
        "summary": summary,
        "total": results["total"],
        "exact_match": results["exact_match"],
        "contains": results["contains"],
        "f1_scores": results["f1_scores"],
        "relevance_scores": results["relevance_scores"],
        "utilization_scores": results["utilization_scores"],
        "adherence_scores": results["adherence_scores"],
        "predictions": results["predictions"]
    }

    # Add LLM judge data if available
    if "llm_judge_scores" in results:
        output["llm_judge_scores"] = results["llm_judge_scores"]
        output["llm_correct_count"] = results["llm_correct_count"]

    with open(filename, 'w') as f:
        json.dump(output, f, indent=2)

    logger.info(f"\nResults saved to: {filename}")


def main():
    """Main entry point for RAG benchmark."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    parser = argparse.ArgumentParser(description="Run RAG benchmark")
    parser.add_argument('--force-reload', action='store_true',
                       help='Clear and reload database with new chunking')
    parser.add_argument('--max-samples', type=int, default=50,
                       help='Maximum number of samples to test')
    parser.add_argument('--retrieval-limit', type=int, default=None,
                       help=f'Number of documents to retrieve per query (default: {Config.DEFAULT_RETRIEVAL_LIMIT})')
    parser.add_argument('--use-llm-judge', action='store_true',
                       help='Enable LLM-as-judge for answer evaluation')
    parser.add_argument('--eval-split', type=str, default='test',
                       choices=['validation', 'test'],
                       help='Which split to evaluate on (default: test). Use validation for optimization, test for final eval.')
    args = parser.parse_args()

    logger.info("RagBench Evaluation")
    logger.info("="*60)

    rag = RAGSystem()
    rag.setup()

    try:
        # Load train split for knowledge base (documents to retrieve from)
        train_dataset = load_dataset_from_ragbench(Config.RAGBENCH_SUBSET, split="train")
        prepare_knowledge_base(rag, train_dataset, force_reload=args.force_reload)

        # Load evaluation split (validation for optimization, test for final eval)
        eval_dataset = load_dataset_from_ragbench(Config.RAGBENCH_SUBSET, split=args.eval_split)
        logger.info(f"Evaluating on {args.eval_split} split ({len(eval_dataset)} samples)")

        # Run benchmark
        results = run_benchmark(rag, eval_dataset,
                              max_samples=args.max_samples,
                              retrieval_limit=args.retrieval_limit,
                              use_llm_judge=args.use_llm_judge)
        print_results(results)
        save_results(results)
    finally:
        rag.close()
        logger.info("\nBenchmark complete")


if __name__ == "__main__":
    main()
