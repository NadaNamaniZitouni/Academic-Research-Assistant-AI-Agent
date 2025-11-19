from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List
import os


class EmbeddingService:
    def __init__(self, model_name: str = None):
        model_name = model_name or os.getenv(
            "EMBEDDING_MODEL",
            "sentence-transformers/all-MiniLM-L6-v2"
        )
        self.model = SentenceTransformer(model_name)
        self.dimension = self.model.get_sentence_embedding_dimension()

    def encode(self, texts: List[str], show_progress_bar: bool = True) -> np.ndarray:
        """Encode texts to embeddings"""
        return self.model.encode(
            texts,
            show_progress_bar=show_progress_bar,
            convert_to_numpy=True
        )

    def encode_query(self, query: str) -> np.ndarray:
        """Encode a single query"""
        return self.encode([query], show_progress_bar=False)[0]


# Global instance
_embedding_service = None


def get_embedding_service() -> EmbeddingService:
    """Get or create embedding service singleton"""
    global _embedding_service
    if _embedding_service is None:
        _embedding_service = EmbeddingService()
    return _embedding_service

