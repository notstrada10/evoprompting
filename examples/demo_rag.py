import logging
import os
import sys

# Suppress warnings
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

sys.path.insert(0, '/Users/marcostrada/Desktop/evoprompting')

from src.core.rag import RAGSystem

logging.basicConfig(level=logging.ERROR)
logging.getLogger('sentence_transformers').setLevel(logging.ERROR)
logger = logging.getLogger(__name__)


def main():
    print("=== RAG Demo ===")
    print("Using existing knowledge base from benchmark dataset\n")

    rag = RAGSystem()

    doc_count = rag.vector_search.count()
    if doc_count == 0:
        print("Error: Knowledge base is empty!")
        print("Run: python -m src.cli benchmark --force-reload --max-samples 10")
        print("This will load the knowledge base first.\n")
        rag.close()
        return

    print(f"Knowledge base ready with {doc_count} documents")
    print("Type 'quit' to exit\n")

    while True:
        question = input("Your question: ").strip()

        if question.lower() in ["quit", "exit", "q"]:
            break

        if not question:
            continue

        result = rag.ask(question, limit=5)
        print(f"\nAnswer: {result['answer']}\n")

    rag.close()
    print("\nDone!")


if __name__ == "__main__":
    main()
