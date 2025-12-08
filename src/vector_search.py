from db import VectorDB
from embeddings import EmbeddingService


class VectorSearch:
    def __init__(self, model: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.embedding_service = EmbeddingService(model=model)
        self.db = VectorDB()
        self.db.connect()

    def setup(self):
        """Setup del database (esegui solo la prima volta)"""
        self.db.setup_database()

    def add_text(self, text: str, metadata: dict = None) -> int:
        """Aggiungi un testo al database"""
        embedding = self.embedding_service.get_embedding(text)
        if embedding:
            return self.db.insert_embedding(text, embedding, metadata)
        return None

    def add_texts(self, texts: list[tuple[str, dict]]) -> list[int]:
        """Aggiungi più testi al database"""
        ids = []
        for text, metadata in texts:
            doc_id = self.add_text(text, metadata)
            if doc_id:
                ids.append(doc_id)
        return ids

    def search(self, query: str, limit: int = 5):
        """Cerca i testi più simili alla query"""
        query_embedding = self.embedding_service.get_embedding(query)
        if not query_embedding:
            return []
        return self.db.search_similar(query_embedding, limit)

    def count(self):
        """Conta il numero di documenti"""
        return self.db.count()

    def delete_all(self):
        """Cancella tutti i documenti"""
        self.db.delete_all()

    def close(self):
        """Chiudi le connessioni"""
        self.db.close()
