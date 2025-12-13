import os

from dotenv import load_dotenv
from groq import Groq

from vector_search import VectorSearch

load_dotenv()


class RAG:
    def __init__(self, model: str = "llama-3.1-8b-instant"):
        """
        Inizializza il sistema RAG

        Args:
            model: Modello Groq da usare per la generazione
        """
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY not found in environment variables")

        self.llm = Groq(api_key=api_key)
        self.model = model
        self.vector_search = VectorSearch()

    def setup(self):
        """Setup del database"""
        self.vector_search.setup()

    def add_document(self, text: str, metadata: dict | None = None) -> int:
        """Aggiungi un documento alla knowledge base"""
        return self.vector_search.add_text(text, metadata or {})

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
        context_text = "\n\n".join([f"Document {i+1}:\n{doc}" for i, doc in enumerate(context)])

        system_prompt = """You are a helpful assistant that answers questions using ONLY the provided documents.
    You must base your answer strictly on the information in the documents.
    If multiple documents contain relevant information, synthesize them.
    Be concise and direct in your answer."""

        user_prompt = f"""Documents:
    {context_text}

    Question: {query}

    Provide a direct answer based on the documents above."""

        try:
            response = self.llm.chat.completions.create(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                model=self.model,
                temperature=0.1,
                max_tokens=512
            )

            if response and response.choices and len(response.choices) > 0:
                message = response.choices[0].message
                if message and hasattr(message, 'content') and message.content:
                    return message.content
                else:
                    return "Error: No content in response message"
            else:
                return "Error: No choices in response"

        except Exception as e:
            return f"Error generating response: {str(e)}"

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
        raw_documents = self.retrieve(query, limit=limit)

        # 2. Deduplicate similar chunks
        documents = []
        seen_prefixes = set()

        for doc in raw_documents:
            # Use first 100 chars as fingerprint
            prefix = doc[:100].strip()
            if prefix not in seen_prefixes:
                documents.append(doc)
                seen_prefixes.add(prefix)

        # 3. Generate
        answer = self.generate(query, documents)

        return {"query": query, "answer": answer, "sources": documents}

    def close(self):
        """Chiudi le connessioni"""
        self.vector_search.close()
