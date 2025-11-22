from pathlib import Path
from sqlalchemy.orm import Session
from .faiss_index import get_faiss_index, FAISSIndex
from .embeddings import get_embedding_service
from .embedding_cache import get_embedding_cache
from .models import ChunkMetadata
import os


def rebuild_index_from_database(db: Session):
    """Rebuild FAISS index from all chunks in database"""
    faiss_idx = get_faiss_index()
    embedding_service = get_embedding_service()

    # Get all chunks
    chunks = db.query(ChunkMetadata).order_by(ChunkMetadata.chunk_id).all()

    if not chunks:
        return {"status": "no_chunks", "count": 0}

    # Generate embeddings
    texts = [chunk.text for chunk in chunks]
    embeddings = embedding_service.encode(texts, show_progress_bar=True)

    # Create new index
    index_path = os.getenv(
        "FAISS_INDEX_PATH",
        str(Path(__file__).parent.parent.parent / "data" / "indices" / "faiss_index.idx")
    )

    # Remove old index
    if Path(index_path).exists():
        Path(index_path).unlink()

    # Create new index
    new_index = FAISSIndex(dimension=embedding_service.dimension, index_path=index_path)
    new_index.add_vectors(embeddings)
    new_index.save()
    
    # Rebuild embedding cache (clear and rebuild from scratch)
    try:
        embedding_cache = get_embedding_cache()
        # Clear existing cache to avoid conflicts
        embedding_cache.clear()
        chunk_ids = [chunk.chunk_id for chunk in chunks]
        embedding_cache.add_embeddings(chunk_ids, embeddings, replace_existing=True)
        print(f"Rebuilt embedding cache with {len(chunks)} embeddings")
    except Exception as e:
        print(f"Warning: Could not rebuild embedding cache: {e}")

    return {
        "status": "success",
        "chunks_indexed": len(chunks),
        "index_path": index_path,
        "cache_size": get_embedding_cache().size()
    }

