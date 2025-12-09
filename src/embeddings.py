import os

from dotenv import load_dotenv
from huggingface_hub import InferenceClient

load_dotenv()

class EmbeddingService:
    def __init__(self, model: str = "google/embeddinggemma-300m"):
        self.client = InferenceClient(api_key=os.environ.get("HF_TOKEN"))
        self.model = model

    def get_embedding(self, text: str) -> list:
        """Genera embedding per un singolo testo"""
        try:
            result = self.client.feature_extraction(text, model=self.model)
            # Converti in lista di float
            if hasattr(result, 'tolist'):
                return result.tolist()
            return list(result)
        except Exception as e:
            print(f"❌ Error generating embedding: {e}")
            return None

    def get_embeddings_batch(self, texts: list[str]) -> list:
        """Genera embeddings per più testi"""
        embeddings = []
        for text in texts:
            emb = self.get_embedding(text)
            if emb:
                embeddings.append(emb)
        return embeddings
