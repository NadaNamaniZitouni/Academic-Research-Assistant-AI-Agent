from typing import List, Dict, Tuple
import numpy as np
from sqlalchemy.orm import Session

from .embeddings import get_embedding_service
from .faiss_index import get_faiss_index
from .models import ChunkMetadata


def retrieve_chunks(
    query: str,
    db: Session,
    k: int = 5
) -> List[Dict]:
    """
    Retrieve relevant chunks for a query.

    Args:
        query: Search query
        db: Database session
        k: Number of chunks to retrieve

    Returns:
        List of chunk dictionaries with metadata
    """
    # Check if index has any vectors
    faiss_idx = get_faiss_index()
    total_vectors = faiss_idx.get_total()
    
    if total_vectors == 0:
        raise ValueError(
            "No documents have been indexed yet. Please upload and process at least one PDF first."
        )
    
    # Encode query
    embedding_service = get_embedding_service()
    query_embedding = embedding_service.encode_query(query)

    # Search FAISS index
    # Adjust k if index has fewer vectors than requested
    search_k = min(k, total_vectors)
    distances, chunk_ids = faiss_idx.search(query_embedding, search_k)

    # Load chunk metadata from database
    chunks = []
    for chunk_id, distance in zip(chunk_ids, distances):
        # Skip invalid chunk IDs (FAISS returns -1 for empty results)
        if int(chunk_id) < 0:
            continue
            
        chunk_meta = db.query(ChunkMetadata).filter(
            ChunkMetadata.chunk_id == int(chunk_id)
        ).first()

        if chunk_meta:
            chunks.append({
                "chunk_id": chunk_meta.chunk_id,
                "doc_id": chunk_meta.doc_id,
                "doc_title": chunk_meta.doc_title,
                "text": chunk_meta.text,
                "page_start": chunk_meta.page_start,
                "page_end": chunk_meta.page_end,
                "source_path": chunk_meta.source_path,
                "similarity_score": float(distance)
            })

    if not chunks:
        raise ValueError(
            "No chunks found in database. The index may be out of sync. Try re-uploading your PDFs."
        )

    return chunks


def format_context_for_llm(chunks: List[Dict], max_chunk_length: int = 800) -> str:
    """
    Format retrieved chunks as context for LLM.
    Limits chunk text length to prevent prompt truncation while providing sufficient context.
    Default max_chunk_length=800 provides good balance between context and token limits.
    """
    context_parts = []
    for i, chunk in enumerate(chunks, 1):
        # Truncate chunk text if too long to prevent token limit issues
        chunk_text = chunk['text']
        if len(chunk_text) > max_chunk_length:
            chunk_text = chunk_text[:max_chunk_length] + "..."
        
        source = f"[{chunk['doc_title']}, p{chunk['page_start']}-{chunk['page_end']}]"
        context_parts.append(f"Source {i} {source}:\n{chunk_text}\n")
    return "\n".join(context_parts)

