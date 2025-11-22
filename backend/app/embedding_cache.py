"""
Embedding Cache Manager
Stores and retrieves chunk embeddings for fast reranking and diversity selection.
"""
import numpy as np
from pathlib import Path
import os
import pickle
from typing import Dict, Optional, Tuple
import json


class EmbeddingCache:
    """Manages cached embeddings for fast retrieval and reranking"""
    
    def __init__(self, cache_path: str = None):
        self.cache_path = cache_path or os.getenv(
            "EMBEDDING_CACHE_PATH",
            str(Path(__file__).parent.parent.parent / "data" / "indices" / "embeddings_cache.npy")
        )
        self.mapping_path = str(Path(self.cache_path).with_suffix('.json'))
        
        # In-memory cache: {chunk_id: embedding_vector}
        self._cache: Dict[int, np.ndarray] = {}
        self._dimension: Optional[int] = None
        
        # Load existing cache if available
        self._load_cache()
    
    def _load_cache(self):
        """Load embeddings cache from disk"""
        try:
            if Path(self.cache_path).exists():
                # Load numpy array
                embeddings_array = np.load(self.cache_path, mmap_mode='r')
                
                # Load chunk_id mapping
                if Path(self.mapping_path).exists():
                    with open(self.mapping_path, 'r') as f:
                        chunk_id_to_index = json.load(f)
                        index_to_chunk_id = {v: int(k) for k, v in chunk_id_to_index.items()}
                    
                    # Build in-memory cache
                    for idx, chunk_id in index_to_chunk_id.items():
                        self._cache[chunk_id] = embeddings_array[int(idx)]
                    
                    if len(embeddings_array) > 0:
                        self._dimension = embeddings_array.shape[1]
                    
                    print(f"Loaded {len(self._cache)} embeddings from cache")
        except Exception as e:
            print(f"Warning: Could not load embedding cache: {e}")
            self._cache = {}
    
    def add_embeddings(self, chunk_ids: list, embeddings: np.ndarray, replace_existing: bool = True):
        """
        Add embeddings to cache
        
        Args:
            chunk_ids: List of chunk IDs corresponding to embeddings
            embeddings: numpy array of shape (n_chunks, dimension)
            replace_existing: If True, replace existing embeddings for same chunk_id (default: True)
        """
        if len(chunk_ids) != len(embeddings):
            raise ValueError(f"chunk_ids length ({len(chunk_ids)}) != embeddings length ({len(embeddings)})")
        
        # Set dimension if not set
        if self._dimension is None:
            self._dimension = embeddings.shape[1]
        elif embeddings.shape[1] != self._dimension:
            raise ValueError(f"Embedding dimension mismatch: expected {self._dimension}, got {embeddings.shape[1]}")
        
        # Add to in-memory cache (replace if exists)
        added_count = 0
        replaced_count = 0
        for chunk_id, embedding in zip(chunk_ids, embeddings):
            chunk_id_int = int(chunk_id)
            if chunk_id_int in self._cache:
                if replace_existing:
                    self._cache[chunk_id_int] = embedding.astype('float32')
                    replaced_count += 1
            else:
                self._cache[chunk_id_int] = embedding.astype('float32')
                added_count += 1
        
        # Save to disk
        self._save_cache()
        
        if replaced_count > 0:
            print(f"Updated {replaced_count} existing embeddings in cache")
        if added_count > 0:
            print(f"Added {added_count} new embeddings to cache")
    
    def get_embedding(self, chunk_id: int) -> Optional[np.ndarray]:
        """Get embedding for a specific chunk_id"""
        return self._cache.get(int(chunk_id))
    
    def get_embeddings(self, chunk_ids: list) -> np.ndarray:
        """Get embeddings for multiple chunk_ids"""
        embeddings = []
        for chunk_id in chunk_ids:
            emb = self._cache.get(int(chunk_id))
            if emb is not None:
                embeddings.append(emb)
            else:
                # Return zero vector if not found (shouldn't happen)
                embeddings.append(np.zeros(self._dimension, dtype='float32'))
        return np.array(embeddings)
    
    def has_chunk(self, chunk_id: int) -> bool:
        """Check if chunk_id exists in cache"""
        return int(chunk_id) in self._cache
    
    def _save_cache(self):
        """Save cache to disk"""
        try:
            # Ensure directory exists
            Path(self.cache_path).parent.mkdir(parents=True, exist_ok=True)
            
            if not self._cache:
                return
            
            # Build arrays and mapping
            chunk_ids = sorted(self._cache.keys())
            embeddings_list = [self._cache[chunk_id] for chunk_id in chunk_ids]
            embeddings_array = np.array(embeddings_list, dtype='float32')
            
            # Save numpy array
            np.save(self.cache_path, embeddings_array)
            
            # Save chunk_id to index mapping
            chunk_id_to_index = {str(chunk_id): idx for idx, chunk_id in enumerate(chunk_ids)}
            with open(self.mapping_path, 'w') as f:
                json.dump(chunk_id_to_index, f)
            
            print(f"Saved {len(self._cache)} embeddings to cache")
        except Exception as e:
            print(f"Warning: Could not save embedding cache: {e}")
    
    def clear(self):
        """Clear the cache"""
        self._cache = {}
        if Path(self.cache_path).exists():
            Path(self.cache_path).unlink()
        if Path(self.mapping_path).exists():
            Path(self.mapping_path).unlink()
    
    def size(self) -> int:
        """Get number of cached embeddings"""
        return len(self._cache)
    
    @property
    def dimension(self) -> Optional[int]:
        """Get embedding dimension"""
        return self._dimension


# Global instance
_embedding_cache = None


def get_embedding_cache() -> EmbeddingCache:
    """Get or create embedding cache singleton"""
    global _embedding_cache
    if _embedding_cache is None:
        _embedding_cache = EmbeddingCache()
    return _embedding_cache

