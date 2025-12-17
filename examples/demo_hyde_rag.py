import logging
import os
import sys

# Suppress tokenizer warnings
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core.hyde_rag import HyDERAGSystem

# Minimal logging
logging.basicConfig(level=logging.ERROR)
logging.getLogger('sentence_transformers').setLevel(logging.ERROR)


def main():
    print("=== HyDE RAG Demo ===")
    print("Using existing knowledge base from benchmark dataset")
    print("HyDE generates hypothetical answers for better retrieval\n")

    rag = HyDERAGSystem()

    doc_count = rag.vector_search.count()
    if doc_count == 0:
        print("Error: Knowledge base is empty!")
        print("Run benchmark first: python -m src.cli benchmark --force-reload")
        return

    print(f"Knowledge base ready with {doc_count} documents")
    print("Type 'quit' to exit\n")

    while True:
        question = input("Your question: ").strip()

        if question.lower() in ["quit", "exit", "q"]:
            print("Goodbye!")
            break

        if not question:
            continue

        result = rag.ask(question, limit=5)
        print(f"\nAnswer: {result['answer']}\n")


if __name__ == "__main__":
    main()
