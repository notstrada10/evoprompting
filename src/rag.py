import os

from dotenv import load_dotenv
from groq import Groq

from vector_search import VectorSearch

load_dotenv()

class RAG:
    def __init__(self, model: str = "llama-3.3-70b-versatile"):
        """
        Inizializza il sistema RAG

        Args:
            model: Modello Groq da usare per la generazione
        """
        self.llm = Groq(api_key=os.environ.get("GROQ_API_KEY"))
        self.model = model
        self.vector_search = VectorSearch()

    def setup(self):
        """Setup del database (esegui solo la prima volta)"""
        self.vector_search.setup()

    def add_document(self, text: str, metadata: dict = None) -> int:
        """Aggiungi un documento alla knowledge base"""
        return self.vector_search.add_text(text, metadata)

    def add_documents(self, documents: list[tuple[str, dict]]) -> list[int]:
        """Aggiungi più documenti alla knowledge base"""
        ids = []
        for text, metadata in documents:
            doc_id = self.add_document(text, metadata)
            if doc_id:
                ids.append(doc_id)
                print(f"  ✅ Added: {text[:50]}...")
        return ids

    def retrieve(self, query: str, limit: int = 3) -> list[str]:
        """
        Cerca i documenti più rilevanti per la query

        Args:
            query: La domanda dell'utente
            limit: Numero di documenti da recuperare

        Returns:
            Lista di testi rilevanti
        """
        results = self.vector_search.search(query, limit=limit)
        documents = [text for (id, text, similarity, metadata) in results]
        return documents

    def generate(self, query: str, context: list[str]) -> str:
        """
        Genera una risposta usando l'LLM

        Args:
            query: La domanda dell'utente
            context: Lista di documenti rilevanti

        Returns:
            Risposta generata
        """
        # Costruisci il prompt
        context_text = "\n\n".join([f"- {doc}" for doc in context])

        system_prompt = """You are a helpful assistant. Answer the user's question based ONLY on the provided context.
If the context doesn't contain enough information to answer, say "I don't have enough information to answer that."
Be concise and direct."""

        user_prompt = f"""Context:
{context_text}

Question: {query}

Answer:"""

        # Chiama l'LLM
        response = self.llm.chat.completions.create(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            model=self.model,
            temperature=0.1  # Bassa temperatura per risposte più precise
        )

        return response.choices[0].message.content

    def ask(self, query: str, limit: int = 3) -> dict:
        """
        Pipeline RAG completa: Retrieve + Generate

        Args:
            query: La domanda dell'utente
            limit: Numero di documenti da recuperare

        Returns:
            Dict con risposta e documenti usati
        """
        # 1. Retrieve
        documents = self.retrieve(query, limit=limit)

        # 2. Generate
        answer = self.generate(query, documents)

        return {
            "query": query,
            "answer": answer,
            "sources": documents
        }

    def close(self):
        """Chiudi le connessioni"""
        self.vector_search.close()
