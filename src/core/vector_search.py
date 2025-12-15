import logging

from ..config import Config
from .db import VectorDatabase
from .embeddings import EmbeddingService

logger = logging.getLogger(__name__)


def chunk_text(text: str, chunk_size: int = None, overlap: int = None) -> list[str]:
    """
    Split text into overlapping chunks.

    Args:
        text: Text to chunk.
        chunk_size: Characters per chunk. Defaults to Config.CHUNK_SIZE.
        overlap: Characters overlap between chunks. Defaults to Config.CHUNK_OVERLAP.

    Returns:
        List of text chunks.
    """
    chunk_size = chunk_size or Config.CHUNK_SIZE
    overlap = overlap or Config.CHUNK_OVERLAP

    text = text.strip()
    if not text:
        return []

    if len(text) <= chunk_size:
        return [text]

    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap

    return chunks


class VectorSearch:
    def __init__(self, model: str = None):
        """
        Initialize the vector search system.

        Args:
            model: Embedding model name. Defaults to Config.EMBEDDING_MODEL.
        """
        self.embedding_service = EmbeddingService(model=model)
        self.db = VectorDatabase()
        self.db.connect()

    def setup(self):
        """Setup the database (run only once)."""
        self.db.setup_database()

    def add_text(self, text: str, metadata: dict = None) -> list[int]:
        """
        Add text to database with chunking.

        Args:
            text: Text to add to the database.
            metadata: Optional metadata to attach to the text.

        Returns:
            List of IDs for all chunks created.
        """
        chunks = chunk_text(text)
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
        """
        Add multiple texts to the database.

        Args:
            texts: List of (text, metadata) tuples.

        Returns:
            List of all chunk IDs created.
        """
        all_ids = []
        for text, metadata in texts:
            chunk_ids = self.add_text(text, metadata)
            all_ids.extend(chunk_ids)
        return all_ids

    def search(self, query: str, limit: int = 5):
        """
        Search for texts most similar to the query.

        Args:
            query: Search query text.
            limit: Maximum number of results to return.

        Returns:
            List of search results.
        """
        query_embedding = self.embedding_service.get_embedding(query)
        if not query_embedding:
            return []
        return self.db.search_similar(query_embedding, limit)

    def count(self):
        """
        Count the number of documents (chunks) in the database.

        Returns:
            Total number of chunks.
        """
        return self.db.count()

    def delete_all(self):
        """Delete all documents from the database."""
        self.db.delete_all()

    def close(self):
        """Close database connections."""
        self.db.close()
