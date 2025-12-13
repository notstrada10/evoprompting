from db import VectorDB
from embeddings import EmbeddingService


def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 100) -> list[str]:
    """
    Split text into overlapping chunks

    Args:
        text: Text to chunk
        chunk_size: Characters per chunk
        overlap: Characters overlap between chunks

    Returns:
        List of text chunks
    """
    if len(text) <= chunk_size:
        return [text]

    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]

        # Try to break at sentence boundary
        if end < len(text):
            last_period = chunk.rfind('.')
            last_newline = chunk.rfind('\n')
            break_point = max(last_period, last_newline)

            if break_point > chunk_size * 0.5:  # Only break if not too early
                chunk = chunk[:break_point + 1]
                end = start + break_point + 1

        chunks.append(chunk.strip())
        start = end - overlap  # Overlap for context

    return chunks


class VectorSearch:
    def __init__(self, model: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.embedding_service = EmbeddingService(model=model)
        self.db = VectorDB()
        self.db.connect()

    def setup(self):
        """Setup del database (esegui solo la prima volta)"""
        self.db.setup_database()

    def add_text(self, text: str, metadata: dict = None) -> list[int]:
        """
        Add text to database with chunking

        Returns:
            List of IDs for all chunks created
        """
        chunks = chunk_text(text, chunk_size=500, overlap=50)
        ids = []

        for i, chunk in enumerate(chunks):
            chunk_metadata = {
                **(metadata or {}),
                "chunk_index": i,
                "total_chunks": len(chunks),
                "original_text_length": len(text)
            }

            embedding = self.embedding_service.get_embedding(chunk)
            if embedding:
                doc_id = self.db.insert_embedding(chunk, embedding, chunk_metadata)
                ids.append(doc_id)

        return ids

    def add_texts(self, texts: list[tuple[str, dict]]) -> list[int]:
        """Aggiungi più testi al database"""
        all_ids = []
        for text, metadata in texts:
            chunk_ids = self.add_text(text, metadata)
            all_ids.extend(chunk_ids)
        return all_ids

    def search(self, query: str, limit: int = 5):
        """Cerca i testi più simili alla query"""
        query_embedding = self.embedding_service.get_embedding(query)
        if not query_embedding:
            return []
        return self.db.search_similar(query_embedding, limit)

    def count(self):
        """Conta il numero di documenti (chunks)"""
        return self.db.count()

    def delete_all(self):
        """Cancella tutti i documenti"""
        self.db.delete_all()

    def close(self):
        """Chiudi le connessioni"""
        self.db.close()
