import logging

from sentence_transformers import SentenceTransformer

from ..config import Config

logger = logging.getLogger(__name__)


class EmbeddingService:
    def __init__(self, model: str | None = None):
        self.model_name = model or Config.EMBEDDING_MODEL
        self.model = None

    def load_model(self):
        if self.model is None:
            logger.info(f"Loading local embedding model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)
        return self.model

    def get_embedding(self, text: str) -> list | None:
        try:
            result = self.load_model().encode(text, show_progress_bar=False)
            return result.tolist()
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            return None

    def get_embeddings_batch(self, texts: list[str], batch_size: int = 16) -> list[list[float] | None]:
        if not texts:
            return []

        results: list[list[float] | None] = [None] * len(texts)
        try:
            all_embeddings = self.load_model().encode(
                texts,
                batch_size=batch_size,
                show_progress_bar=len(texts) > 100
            )
            for i, emb in enumerate(all_embeddings):
                results[i] = emb.tolist()
        except Exception as e:
            logger.error(f"Error in batch embedding: {e}")
        return results
