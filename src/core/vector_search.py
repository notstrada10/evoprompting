import logging

from ..config import Config
from .db import VectorDatabase
from .embeddings import EmbeddingService

logger = logging.getLogger(__name__)


def chunk_text(text: str, chunk_size: int | None = None, overlap: int | None = None) -> list[str]:
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
    def __init__(self, db: VectorDatabase | None = None, embedding_service: EmbeddingService | None = None, model: str | None = None):
        """
        Initialize the vector search system.

        Args:
            db: Optional VectorDatabase instance. If not provided, creates a new one.
            embedding_service: Optional EmbeddingService instance. If not provided, creates a new one.
            model: Embedding model name (only used if embedding_service is not provided). Defaults to Config.EMBEDDING_MODEL.
        """
        self.embedding_service = embedding_service or EmbeddingService(model=model)
        self.db = db or VectorDatabase()
        if db is None:
            self.db.connect()


    def setup(self):
        """Setup the database (run only once)."""
        self.db.setup_database()

    def add_text(self, text: str, metadata: dict | None = None) -> list[int]:
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

    def add_texts(self, texts: list[tuple[str, dict]], batch_size: int = 16) -> list[int]:
        """
        Add multiple texts to the database using batch processing.

        Args:
            texts: List of (text, metadata) tuples.
            batch_size: Number of texts to embed and insert in each batch. Defaults to 16.

        Returns:
            List of all chunk IDs created.
        """
        # First, chunk all texts and prepare metadata
        all_chunks = []
        all_metadata = []

        for text, metadata in texts:
            chunks = chunk_text(text)
            for i, chunk in enumerate(chunks):
                chunk_metadata = {
                    **(metadata or {}),
                    "chunk_index": i,
                    "total_chunks": len(chunks),
                    "original_text_length": len(text)
                }
                all_chunks.append(chunk)
                all_metadata.append(chunk_metadata)

        if not all_chunks:
            return []

        all_ids = []
        total_chunks = len(all_chunks)

        # Process in batches: embed and insert each batch
        for i in range(0, total_chunks, batch_size):
            batch_chunks = all_chunks[i:i + batch_size]
            batch_metadata = all_metadata[i:i + batch_size]

            # Embed this batch
            embeddings = self.embedding_service.get_embeddings_batch(batch_chunks, batch_size=batch_size)

            # Prepare items for batch insert (filter out failed embeddings)
            items = [
                (chunk, emb, meta)
                for chunk, emb, meta in zip(batch_chunks, embeddings, batch_metadata)
                if emb is not None
            ]

            if items:
                ids = self.db.insert_embeddings_batch(items)
                all_ids.extend(ids)

            logger.info(f"Processed {min(i + batch_size, total_chunks)}/{total_chunks} chunks")

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
        """Count the number of documents (chunks) in the DB."""
        return self.db.count()

    def delete_all(self):
        """Delete all documents from the database."""
        self.db.delete_all()

    def close(self):
        """Close database connections."""
        self.db.close()
