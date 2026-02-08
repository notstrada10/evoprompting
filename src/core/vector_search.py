import logging

from ..config import Config
from .db import VectorDatabase
from .embeddings import EmbeddingService

logger = logging.getLogger(__name__)


def chunk_text(text: str, chunk_size: int | None = None, overlap: int | None = None) -> list[str]:
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
        self.embedding_service = embedding_service or EmbeddingService(model=model)
        self.db = db or VectorDatabase()
        if db is None:
            self.db.connect()

    def setup(self):
        self.db.setup_database()

    def add_text(self, text: str, metadata: dict | None = None) -> list[int]:
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

        for i in range(0, total_chunks, batch_size):
            batch_chunks = all_chunks[i:i + batch_size]
            batch_metadata = all_metadata[i:i + batch_size]
            embeddings = self.embedding_service.get_embeddings_batch(batch_chunks, batch_size=batch_size)

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
        query_embedding = self.embedding_service.get_embedding(query)
        if not query_embedding:
            return []
        return self.db.search_similar(query_embedding, limit)

    def count(self):
        return self.db.count()

    def delete_all(self):
        self.db.delete_all()

    def close(self):
        self.db.close()
