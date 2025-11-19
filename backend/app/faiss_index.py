import faiss
import numpy as np
from pathlib import Path
from typing import List, Tuple
from sklearn.preprocessing import normalize
import os
from .embeddings import get_embedding_service


class FAISSIndex:
    def __init__(self, dimension: int = 384, index_path: str = None):
        self.dimension = dimension
        self.index_path = index_path or os.getenv(
            "FAISS_INDEX_PATH",
            str(Path(__file__).parent.parent.parent / "data" / "indices" / "faiss_index.idx")
        )

        # Ensure directory exists
        Path(self.index_path).parent.mkdir(parents=True, exist_ok=True)

        # Initialize or load index
        if Path(self.index_path).exists():
            self.index = faiss.read_index(self.index_path)
        else:
            # Use Inner Product for cosine similarity (after normalization)
            self.index = faiss.IndexFlatIP(dimension)

    def add_vectors(self, vectors: np.ndarray):
        """Add vectors to index (normalize for cosine similarity)"""
        if len(vectors.shape) == 1:
            vectors = vectors.reshape(1, -1)

        # Normalize for cosine similarity
        vectors_normalized = normalize(vectors, norm='l2').astype('float32')
        self.index.add(vectors_normalized)

    def search(
        self,
        query_vector: np.ndarray,
        k: int = 5
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Search for similar vectors.

        Returns:
            Tuple of (distances, indices)
        """
        if self.index.ntotal == 0:
            raise ValueError("FAISS index is empty. No vectors have been added yet.")
        
        if len(query_vector.shape) == 1:
            query_vector = query_vector.reshape(1, -1)

        # Normalize query
        query_normalized = normalize(query_vector, norm='l2').astype('float32')

        # Adjust k if index has fewer vectors
        search_k = min(k, self.index.ntotal)
        
        distances, indices = self.index.search(query_normalized, search_k)
        return distances[0], indices[0]

    def save(self):
        """Save index to disk"""
        Path(self.index_path).parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, self.index_path)

    def get_total(self) -> int:
        """Get total number of vectors in index"""
        return self.index.ntotal

    def rebuild(self):
        """Rebuild index (useful for incremental updates)"""
        # For IndexFlatIP, no rebuild needed
        # For other index types, you might need to rebuild
        pass


# Global instance
_faiss_index = None


def get_faiss_index() -> FAISSIndex:
    """Get or create FAISS index singleton"""
    global _faiss_index
    if _faiss_index is None:
        embedding_service = get_embedding_service()
        _faiss_index = FAISSIndex(dimension=embedding_service.dimension)
    return _faiss_index

