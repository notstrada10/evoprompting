import argparse
import json
from datetime import datetime
from typing import Dict

from datasets import load_dataset

from rag import RAG


def load_dataset_from_ragbench(dataset_name: str, split: str = "test"):
    """Load dataset from RagBench"""
    print(f"Loading {dataset_name} dataset from RagBench...")
    dataset = load_dataset("rungalileo/ragbench", dataset_name, split=split)
    print(f"Loaded {len(dataset)} samples")
    return dataset


def prepare_knowledge_base(rag: RAG, dataset, force_reload: bool = False) -> None:
    """Extract and add all unique documents to RAG knowledge base"""
    current_count = rag.vector_search.count()

    if current_count > 0 and not force_reload:
        print(f"Knowledge base already contains {current_count} documents, skipping load")
        print("Use --force-reload to clear and reload the database")
        return

    if force_reload and current_count > 0:
        print(f"Clearing existing {current_count} documents...")
        rag.vector_search.delete_all()

    print("Preparing knowledge base...")

    documents_added = set()

    for item in dataset:
        docs = item.get('documents', [])

        for doc in docs:
            doc_text = doc if isinstance(doc, str) else str(doc)
            if doc_text and doc_text not in documents_added:
                rag.add_document(doc_text, {"source": "ragbench"})
                documents_added.add(doc_text)

    print(f"Added {len(documents_added)} documents to knowledge base")


def calculate_f1(predicted: str, ground_truth: str) -> float:
    """Calculate F1 score based on token overlap"""
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


def run_benchmark(rag: RAG, dataset, max_samples: int = None, retrieval_limit: int = 5) -> Dict:
    """Run RAG system on benchmark and evaluate"""
    print("Running benchmark...")

    results = {
        "total": 0,
        "exact_match": 0,
        "contains": 0,
        "f1_scores": [],
        "predictions": []
    }

    samples_to_test = min(len(dataset), max_samples) if max_samples else len(dataset)

    for idx, item in enumerate(dataset):
        if max_samples and idx >= max_samples:
            break

        query = item.get('question', '')
        ground_truth = item.get('response', '')

        if not query or not ground_truth:
            continue

        result = rag.ask(query, limit=retrieval_limit)
        predicted = result['answer']

        if idx < 3:
            print(f"\n--- Query {idx+1}: {query[:80]}...")
            print(f"Ground truth: {ground_truth[:100]}...")
            print(f"Retrieved docs:")
            for i, src in enumerate(result['sources']):
                print(f"  {i+1}. {src[:100]}...")
            print(f"Generated: {predicted[:100]}...")

        exact = 1.0 if predicted.strip().lower() == ground_truth.strip().lower() else 0.0
        contains = 1.0 if ground_truth.strip().lower() in predicted.strip().lower() else 0.0
        f1 = calculate_f1(predicted, ground_truth)

        results["total"] += 1
        results["exact_match"] += exact
        results["contains"] += contains
        results["f1_scores"].append(f1)

        results["predictions"].append({
            "id": item.get('id'),
            "query": query,
            "predicted": predicted,
            "ground_truth": ground_truth,
            "exact_match": exact,
            "contains": contains,
            "f1": f1,
            "sources": result['sources']
        })

        if (idx + 1) % 10 == 0:
            avg_f1 = sum(results["f1_scores"]) / len(results["f1_scores"])
            print(f"Progress: {idx + 1}/{samples_to_test} | Avg F1: {avg_f1:.3f}")

    return results


def print_results(results: Dict) -> None:
    """Print benchmark results"""
    print("\n" + "="*60)
    print("BENCHMARK RESULTS")
    print("="*60)

    total = results["total"]
    if total == 0:
        print("No samples processed")
        return

    exact_match_rate = results["exact_match"] / total
    contains_rate = results["contains"] / total
    avg_f1 = sum(results["f1_scores"]) / len(results["f1_scores"])

    print(f"\nOverall Metrics:")
    print(f"  Total samples: {total}")
    print(f"  Exact Match: {exact_match_rate:.2%}")
    print(f"  Contains Answer: {contains_rate:.2%}")
    print(f"  Average F1 Score: {avg_f1:.3f}")

    print(f"\nSample Predictions:")
    for i, pred in enumerate(results["predictions"][:3]):
        print(f"\n  Example {i+1} (ID: {pred.get('id')}):")
        print(f"    Query: {pred['query'][:100]}...")
        print(f"    Predicted: {pred['predicted'][:150]}...")
        print(f"    Ground Truth: {pred['ground_truth'][:150]}...")
        print(f"    F1 Score: {pred['f1']:.3f}")


def save_results(results: Dict, output_dir: str = "results") -> None:
    """Save results to JSON file"""
    import os
    os.makedirs(output_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(output_dir, f"benchmark_{timestamp}.json")

    with open(filename, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {filename}")


def main():
    parser = argparse.ArgumentParser(description="Run RAG benchmark")
    parser.add_argument('--force-reload', action='store_true',
                       help='Clear and reload database with new chunking')
    parser.add_argument('--max-samples', type=int, default=50,
                       help='Maximum number of samples to test')
    parser.add_argument('--retrieval-limit', type=int, default=5,
                       help='Number of documents to retrieve per query')
    args = parser.parse_args()

    print("RagBench Evaluation")
    print("="*60)

    rag = RAG()
    rag.setup()

    try:
        dataset = load_dataset_from_ragbench("covidqa")
        prepare_knowledge_base(rag, dataset, force_reload=args.force_reload)
        results = run_benchmark(rag, dataset,
                              max_samples=args.max_samples,
                              retrieval_limit=args.retrieval_limit)
        print_results(results)
        save_results(results)
    finally:
        rag.close()
        print("\nBenchmark complete")


if __name__ == "__main__":
    main()
