from vector_search import VectorSearch


def main():
    print("=== Vector Search Demo ===\n")

    # Inizializza
    vs = VectorSearch()

    # Setup database
    print("Setting up database...")
    vs.setup()

    # Aggiungi documenti
    print("\nAdding documents...")
    documents = [
        ("Fast language models are crucial for real-time applications", {"type": "tech"}),
        ("Embeddings capture semantic meaning of text", {"type": "ml"}),
        ("PostgreSQL is a powerful relational database", {"type": "database"}),
        ("Vector databases enable similarity search", {"type": "database"}),
        ("Machine learning models need good data", {"type": "ml"}),
        ("Python is great for data science", {"type": "programming"}),
    ]

    for text, metadata in documents:
        doc_id = vs.add_text(text, metadata)
        print(f"  ✅ Added document {doc_id}: {text[:40]}...")

    print(f"\nTotal documents: {vs.count()}")

    # Cerca
    print("\n" + "="*60)
    query = "How do databases work with vectors?"
    print(f"Query: '{query}'\n")

    results = vs.search(query, limit=3)

    print("Top 3 results:\n")
    for i, (doc_id, text, similarity, metadata) in enumerate(results, 1):
        print(f"{i}. [Score: {similarity:.4f}]")
        print(f"   Text: {text}")
        print(f"   Metadata: {metadata}")
        print()

    # Chiudi
    vs.close()
    print("Done! 🚀")

if __name__ == "__main__":
    main()
