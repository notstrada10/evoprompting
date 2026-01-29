import logging

from sentence_transformers import SentenceTransformer

from ..config import Config

logger = logging.getLogger(__name__)


class EmbeddingService:
    def __init__(self, model: str | None = None):
        """
        Initialize the embedding service.

        Args:
            model: Model name to use for embeddings. Defaults to Config.EMBEDDING_MODEL.
        """
        self.model_name = model or Config.EMBEDDING_MODEL
        self._local_model = None

    @property
    def local_model(self):
        """Lazy load the local model."""
        if self._local_model is None:
            logger.info(f"Loading local embedding model: {self.model_name}")
            self._local_model = SentenceTransformer(self.model_name)
        return self._local_model

    def get_embedding(self, text: str) -> list | None:
        """
        Generate embedding for a single text.

        Args:
            text: Input text to embed.

        Returns:
            List of floats representing the embedding, or None if error occurs.
        """
        try:
            result = self.local_model.encode(text, show_progress_bar=False)
            return result.tolist()
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            return None

    def get_embeddings_batch(self, texts: list[str], batch_size: int = 16) -> list[list[float] | None]:
        """
        Generate embeddings for multiple texts in batches.

        Args:
            texts: List of input texts to embed.
            batch_size: Number of texts to process in each batch. Defaults to 16.

        Returns:
            List of embeddings (or None for failed texts) in the same order as input.
        """
        if not texts:
            return []

        results: list[list[float] | None] = [None] * len(texts)

        try:
            all_embeddings = self.local_model.encode(
                texts,
                batch_size=batch_size,
                show_progress_bar=len(texts) > 100
            )
            for i, emb in enumerate(all_embeddings):
                results[i] = emb.tolist()
        except Exception as e:
            logger.error(f"Error in batch embedding: {e}")

        return results
