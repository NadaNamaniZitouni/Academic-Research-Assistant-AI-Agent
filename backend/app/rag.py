from typing import List, Dict, Tuple
import numpy as np
from sqlalchemy.orm import Session
from sklearn.preprocessing import normalize

from .embeddings import get_embedding_service
from .faiss_index import get_faiss_index
from .embedding_cache import get_embedding_cache
from .models import ChunkMetadata


def rerank_chunks(query_embedding: np.ndarray, chunks: List[Dict], cache) -> List[Dict]:
    """
    Rerank chunks using cached embeddings for more accurate similarity scores.
    
    Args:
        query_embedding: Query embedding vector
        chunks: List of chunk dictionaries
        cache: EmbeddingCache instance
    
    Returns:
        Reranked chunks with updated similarity scores
    """
    if not chunks or cache.size() == 0:
        return chunks
    
    # Normalize query embedding
    query_embedding = normalize(query_embedding.reshape(1, -1), norm='l2')[0]
    
    # Get embeddings for all chunks
    chunk_ids = [c["chunk_id"] for c in chunks]
    chunk_embeddings = cache.get_embeddings(chunk_ids)
    
    # Normalize chunk embeddings
    chunk_embeddings = normalize(chunk_embeddings, norm='l2')
    
    # Compute cosine similarities
    similarities = np.dot(chunk_embeddings, query_embedding)
    
    # Update similarity scores
    for i, chunk in enumerate(chunks):
        chunk["similarity_score"] = float(similarities[i])
    
    # Sort by similarity (higher is better)
    chunks.sort(key=lambda x: x["similarity_score"], reverse=True)
    
    return chunks


def mmr_diversity_selection(
    query_embedding: np.ndarray,
    chunks: List[Dict],
    cache,
    lambda_param: float = 0.5,
    final_k: int = 12,
    ensure_doc_diversity: bool = True
) -> List[Dict]:
    """
    Select diverse chunks using Maximal Marginal Relevance (MMR).
    
    MMR balances relevance and diversity:
    - lambda_param = 1.0: Pure relevance (no diversity)
    - lambda_param = 0.0: Pure diversity (no relevance)
    - lambda_param = 0.5: Balanced (recommended)
    
    Args:
        query_embedding: Query embedding vector
        chunks: List of chunk dictionaries (should be pre-sorted by relevance)
        cache: EmbeddingCache instance
        lambda_param: Balance between relevance (1.0) and diversity (0.0)
        final_k: Number of chunks to select
        ensure_doc_diversity: If True, ensure chunks come from different documents when possible
    
    Returns:
        Selected diverse chunks
    """
    if len(chunks) <= final_k:
        return chunks
    
    if cache.size() == 0:
        # Fallback: just return top k
        return chunks[:final_k]
    
    # Normalize query embedding
    query_embedding = normalize(query_embedding.reshape(1, -1), norm='l2')[0]
    
    # Get embeddings for all chunks
    chunk_ids = [c["chunk_id"] for c in chunks]
    chunk_embeddings = cache.get_embeddings(chunk_ids)
    chunk_embeddings = normalize(chunk_embeddings, norm='l2')
    
    selected = []
    selected_indices = set()
    remaining_indices = set(range(len(chunks)))
    selected_doc_ids = set()  # Track documents already represented
    
    # Start with most relevant chunk
    selected.append(0)
    selected_indices.add(0)
    remaining_indices.remove(0)
    if ensure_doc_diversity:
        selected_doc_ids.add(chunks[0]["doc_id"])
    
    # Select remaining chunks using MMR with document diversity
    while len(selected) < final_k and remaining_indices:
        best_score = -float('inf')
        best_idx = None
        
        for idx in remaining_indices:
            # Relevance: similarity to query
            relevance = float(np.dot(chunk_embeddings[idx], query_embedding))
            
            # Diversity: max similarity to already selected chunks
            if selected:
                similarities_to_selected = [
                    float(np.dot(chunk_embeddings[idx], chunk_embeddings[sel_idx]))
                    for sel_idx in selected
                ]
                max_similarity = max(similarities_to_selected)
            else:
                max_similarity = 0.0
            
            # Document diversity bonus: prefer chunks from documents not yet represented
            doc_diversity_bonus = 0.0
            if ensure_doc_diversity:
                chunk_doc_id = chunks[idx]["doc_id"]
                if chunk_doc_id not in selected_doc_ids:
                    # Bonus for new documents (encourages diversity)
                    doc_diversity_bonus = 0.1 * relevance
            
            # MMR score: balance relevance and diversity + document diversity bonus
            mmr_score = lambda_param * relevance - (1 - lambda_param) * max_similarity + doc_diversity_bonus
            
            if mmr_score > best_score:
                best_score = mmr_score
                best_idx = idx
        
        if best_idx is not None:
            selected.append(best_idx)
            selected_indices.add(best_idx)
            remaining_indices.remove(best_idx)
            if ensure_doc_diversity:
                selected_doc_ids.add(chunks[best_idx]["doc_id"])
        else:
            break
    
    # Return selected chunks in order of selection
    return [chunks[i] for i in selected]


def retrieve_chunks(
    query: str,
    db: Session,
    k: int = 8,
    use_reranking: bool = True,
    use_diversity: bool = True,
    initial_k: int = 25,
    final_k: int = 12,
    doc_id: str = None
) -> List[Dict]:
    """
    Retrieve relevant chunks using hybrid approach: FAISS retrieval + reranking + diversity selection.

    Args:
        query: Search query
        db: Database session
        k: Final number of chunks to return (for backward compatibility)
        use_reranking: Whether to rerank using cached embeddings
        use_diversity: Whether to apply MMR diversity selection
        initial_k: Number of chunks to retrieve from FAISS (default: 25)
        final_k: Final number of chunks after reranking/diversity (default: 12)
        doc_id: Optional document ID to filter by (None = all documents, single doc_id = only that document)

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
    
    # Get embedding cache
    cache = get_embedding_cache()
    
    # Step 1: Initial retrieval from FAISS
    # If doc_id is specified, get all chunks for that document first, then search within them
    if doc_id is not None:
        # Get all chunks for this document
        doc_chunks = db.query(ChunkMetadata).filter(
            ChunkMetadata.doc_id == doc_id
        ).all()
        
        if not doc_chunks:
            raise ValueError(
                f"Document {doc_id} has no chunks. The document may not have been processed yet, or the index may be out of sync. Try re-uploading the PDF."
            )
        
        print(f"[retrieve_chunks] doc_id={doc_id}, found {len(doc_chunks)} chunks in database")
        
        # Get embeddings for all chunks in this document
        doc_chunk_ids = [chunk.chunk_id for chunk in doc_chunks]
        doc_chunk_embeddings = cache.get_embeddings(doc_chunk_ids)
        
        if len(doc_chunk_embeddings) == 0:
            # Fallback: if cache doesn't have embeddings, use FAISS but with larger search
            print(f"[retrieve_chunks] Cache missing embeddings, using FAISS with larger search_k")
            search_k = min(initial_k * 5, total_vectors)  # Get many more candidates
            distances, chunk_ids = faiss_idx.search(query_embedding, search_k)
            
            # Filter to only chunks from this document
            chunks = []
            for chunk_id, distance in zip(chunk_ids, distances):
                if int(chunk_id) < 0:
                    continue
                chunk_meta = db.query(ChunkMetadata).filter(
                    ChunkMetadata.chunk_id == int(chunk_id),
                    ChunkMetadata.doc_id == doc_id
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
        else:
            # Calculate similarity between query and each document chunk
            from sklearn.preprocessing import normalize
            query_normalized = normalize(query_embedding.reshape(1, -1), norm='l2')[0]
            doc_embeddings_normalized = normalize(doc_chunk_embeddings, norm='l2')
            similarities = np.dot(doc_embeddings_normalized, query_normalized)
            
            # Create chunks with similarity scores
            chunks = []
            for chunk, similarity in zip(doc_chunks, similarities):
                chunks.append({
                    "chunk_id": chunk.chunk_id,
                    "doc_id": chunk.doc_id,
                    "doc_title": chunk.doc_title,
                    "text": chunk.text,
                    "page_start": chunk.page_start,
                    "page_end": chunk.page_end,
                    "source_path": chunk.source_path,
                    "similarity_score": float(similarity)
                })
            
            # Sort by similarity (highest first)
            chunks.sort(key=lambda x: x["similarity_score"], reverse=True)
    else:
        # No doc_id filter - search all documents
        search_k = min(initial_k, total_vectors)
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
        if doc_id is not None:
            raise ValueError(
                f"No chunks found for document {doc_id}. The document may not have been processed yet, or the index may be out of sync. Try re-uploading the PDF."
            )
        else:
            raise ValueError(
                "No chunks found in database. The index may be out of sync. Try re-uploading your PDFs."
            )
    
    # Step 2: Rerank using cached embeddings (more accurate similarity)
    if use_reranking and cache.size() > 0:
        chunks = rerank_chunks(query_embedding, chunks, cache)
    
    # Step 3: Apply MMR diversity selection
    # Note: If doc_id is specified, all chunks are from same document, so doc diversity is not needed
    if use_diversity and len(chunks) > final_k and cache.size() > 0:
        chunks = mmr_diversity_selection(
            query_embedding, chunks, cache, 
            lambda_param=0.5, 
            final_k=final_k,
            ensure_doc_diversity=(doc_id is None)  # Only ensure doc diversity if searching across all docs
        )
    
    # Step 4: Return top k chunks (respect backward compatibility)
    return chunks[:k if not use_diversity else min(k, final_k)]


def format_context_for_llm(chunks: List[Dict], max_chunk_length: int = None) -> str:
    """
    Format retrieved chunks as context for LLM.
    
    Args:
        chunks: List of chunk dictionaries
        max_chunk_length: Optional maximum length per chunk (None = no limit, use full chunks)
    """
    context_parts = []
    for i, chunk in enumerate(chunks, 1):
        chunk_text = chunk['text']
        
        # Only truncate if max_chunk_length is specified
        if max_chunk_length and len(chunk_text) > max_chunk_length:
            chunk_text = chunk_text[:max_chunk_length] + "..."
        
        source = f"[{chunk['doc_title']}, p{chunk['page_start']}-{chunk['page_end']}]"
        context_parts.append(f"Source {i} {source}:\n{chunk_text}\n")
    return "\n".join(context_parts)

