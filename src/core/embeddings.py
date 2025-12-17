import logging
import os

from dotenv import load_dotenv

from ..config import Config

load_dotenv()
logger = logging.getLogger(__name__)


class EmbeddingService:
    def __init__(self, model: str = None, use_local: bool = None):
        """
        Initialize the embedding service.

        Args:
            model: Model name to use for embeddings. Defaults to Config.EMBEDDING_MODEL.
            use_local: If True, use local sentence-transformers. If False, use HF API.
                      Defaults to Config.USE_LOCAL_EMBEDDINGS.
        """
        self.model_name = model or Config.EMBEDDING_MODEL
        self.use_local = use_local if use_local is not None else Config.USE_LOCAL_EMBEDDINGS
        self._local_model = None
        self._api_client = None

    @property
    def local_model(self):
        """Lazy load the local model."""
        if self._local_model is None:
            from sentence_transformers import SentenceTransformer
            logger.info(f"Loading local embedding model: {self.model_name}")
            self._local_model = SentenceTransformer(self.model_name)
        return self._local_model

    @property
    def api_client(self):
        """Lazy load the API client."""
        if self._api_client is None:
            from huggingface_hub import InferenceClient
            self._api_client = InferenceClient(api_key=os.environ.get("HF_TOKEN"))
        return self._api_client

    def get_embedding(self, text: str) -> list | None:
        """
        Generate embedding for a single text.

        Args:
            text: Input text to embed.

        Returns:
            List of floats representing the embedding, or None if error occurs.
        """
        try:
            if self.use_local:
                result = self.local_model.encode(text, show_progress_bar=False)
                return result.tolist()
            else:
                result = self.api_client.feature_extraction(text, model=self.model_name)
                if hasattr(result, 'tolist'):
                    return result.tolist()
                return list(result)
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            return None
