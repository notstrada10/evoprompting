from rag import RAG


def main():
    print("=== RAG Demo ===\n")

    # Inizializza
    rag = RAG()
    rag.setup()

    # Aggiungi documenti alla knowledge base
    print("Adding documents to knowledge base...\n")

    documents = [
        ("Python was created by Guido van Rossum and first released in 1991.", {"topic": "python"}),
        ("PostgreSQL is an open-source relational database that supports JSON and vector data.", {"topic": "database"}),
        ("pgvector is a PostgreSQL extension for vector similarity search.", {"topic": "database"}),
        ("RAG stands for Retrieval Augmented Generation, a technique that combines search with LLMs.", {"topic": "ai"}),
        ("Embeddings are numerical representations of text that capture semantic meaning.", {"topic": "ai"}),
        ("Groq provides fast inference for large language models like Llama.", {"topic": "ai"}),
        ("Vector databases store embeddings and enable similarity search.", {"topic": "database"}),
        ("LangChain is a framework for building applications with LLMs.", {"topic": "ai"}),
        ("Transformers are the architecture behind modern language models like GPT and Llama.", {"topic": "ai"}),
        ("Cosine similarity measures the angle between two vectors to determine similarity.", {"topic": "math"}),
    ]

    rag.add_documents(documents)

    print(f"\n✅ Knowledge base ready with {rag.vector_search.count()} documents\n")

    # Fai alcune domande
    questions = [
        "What is RAG and how does it work?",
        "How can I store vectors in a database?",
        "Who created Python?",
    ]

    for question in questions:
        print("=" * 60)
        print(f"❓ Question: {question}\n")

        result = rag.ask(question, limit=3)

        print(f"💡 Answer: {result['answer']}\n")

        print("📚 Sources used:")
        for i, source in enumerate(result['sources'], 1):
            print(f"   {i}. {source}")
        print()

    # Modalità interattiva
    print("=" * 60)
    print("Interactive mode - type 'quit' to exit\n")

    while True:
        question = input("❓ Your question: ").strip()

        if question.lower() in ['quit', 'exit', 'q']:
            break

        if not question:
            continue

        result = rag.ask(question)
        print(f"\n💡 Answer: {result['answer']}\n")
        print("📚 Sources:")
        for i, source in enumerate(result['sources'], 1):
            print(f"   {i}. {source}")
        print()

    rag.close()
    print("\nDone! 🚀")

if __name__ == "__main__":
    main()
