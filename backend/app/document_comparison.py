"""
Multi-document comparison functionality
"""
from typing import List, Dict
from sqlalchemy.orm import Session
from .models import Document, ChunkMetadata
from .embeddings import get_embedding_service
from .embedding_cache import get_embedding_cache
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def compare_documents(
    doc_ids: List[str],
    db: Session,
    user_id: str
) -> Dict:
    """
    Compare multiple documents to find:
    - Similarities and differences
    - Contradictions
    - Complementary information
    - Common themes
    """
    # Verify all documents belong to user
    docs = db.query(Document).filter(
        Document.doc_id.in_(doc_ids),
        Document.user_id == user_id
    ).all()
    
    if len(docs) != len(doc_ids):
        raise ValueError("Some documents not found or don't belong to user")
    
    if len(docs) < 2:
        raise ValueError("Need at least 2 documents to compare")
    
    # Get chunks for each document
    doc_chunks = {}
    for doc in docs:
        chunks = db.query(ChunkMetadata).filter(
            ChunkMetadata.doc_id == doc.doc_id
        ).all()
        doc_chunks[doc.doc_id] = chunks
    
    # Get embeddings for comparison
    embedding_service = get_embedding_service()
    cache = get_embedding_cache()
    
    # Calculate document-level embeddings (average of chunk embeddings)
    doc_embeddings = {}
    for doc_id, chunks in doc_chunks.items():
        if not chunks:
            continue
        
        chunk_ids = [chunk.chunk_id for chunk in chunks]
        embeddings = cache.get_embeddings(chunk_ids)
        
        if len(embeddings) > 0:
            # Average embedding for document
            doc_embeddings[doc_id] = np.mean(embeddings, axis=0)
    
    # Calculate pairwise similarities
    similarities = {}
    doc_list = list(doc_embeddings.keys())
    for i, doc_id1 in enumerate(doc_list):
        for doc_id2 in doc_list[i+1:]:
            sim = cosine_similarity(
                doc_embeddings[doc_id1].reshape(1, -1),
                doc_embeddings[doc_id2].reshape(1, -1)
            )[0][0]
            similarities[f"{doc_id1}_{doc_id2}"] = float(sim)
    
    # Find common themes (top chunks across documents)
    all_chunks = []
    for doc_id, chunks in doc_chunks.items():
        for chunk in chunks[:10]:  # Top 10 chunks per doc
            all_chunks.append({
                'doc_id': doc_id,
                'doc_title': chunk.doc_title,
                'text': chunk.text[:200],  # First 200 chars
                'page': chunk.page_start
            })
    
    # Find contradictions (requires semantic analysis - simplified here)
    contradictions = []
    # This would require more sophisticated NLP - placeholder for now
    # In production, use LLM to analyze semantic contradictions
    
    return {
        "documents": [
            {
                "doc_id": doc.doc_id,
                "title": doc.title,
                "authors": doc.authors,
                "year": doc.year,
                "chunk_count": len(doc_chunks.get(doc.doc_id, []))
            }
            for doc in docs
        ],
        "similarities": similarities,
        "common_themes": all_chunks[:5],  # Top 5 chunks
        "contradictions": contradictions,
        "summary": f"Compared {len(docs)} documents. Found {len(similarities)} pairwise comparisons."
    }


def find_contradictions(
    doc_ids: List[str],
    db: Session,
    user_id: str
) -> List[Dict]:
    """
    Find contradictions between documents using semantic analysis
    """
    # This would use LLM to analyze semantic contradictions
    # For now, return empty list
    return []

