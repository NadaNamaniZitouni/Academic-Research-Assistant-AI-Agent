"""
Citation network visualization functionality
"""
from typing import List, Dict
from sqlalchemy.orm import Session
from .models import Document, ChunkMetadata
from .embeddings import get_embedding_service
from .embedding_cache import get_embedding_cache
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def build_citation_network(
    doc_ids: List[str],
    db: Session,
    user_id: str,
    similarity_threshold: float = 0.7
) -> Dict:
    """
    Build a citation network graph showing relationships between documents
    
    Returns nodes (documents) and edges (relationships) for visualization
    """
    # Get all user documents if doc_ids is empty
    if not doc_ids:
        docs = db.query(Document).filter(
            Document.user_id == user_id
        ).all()
        doc_ids = [doc.doc_id for doc in docs]
    else:
        # Verify documents belong to user
        docs = db.query(Document).filter(
            Document.doc_id.in_(doc_ids),
            Document.user_id == user_id
        ).all()
        
        if len(docs) != len(doc_ids):
            raise ValueError("Some documents not found or don't belong to user")
    
    if len(docs) < 2:
        return {
            "nodes": [],
            "edges": [],
            "message": "Need at least 2 documents to build a network"
        }
    
    # Get chunks for each document
    doc_chunks = {}
    for doc in docs:
        chunks = db.query(ChunkMetadata).filter(
            ChunkMetadata.doc_id == doc.doc_id
        ).all()
        doc_chunks[doc.doc_id] = chunks
    
    # Get embeddings
    cache = get_embedding_cache()
    
    # Calculate document-level embeddings
    doc_embeddings = {}
    for doc_id, chunks in doc_chunks.items():
        if not chunks:
            continue
        
        chunk_ids = [chunk.chunk_id for chunk in chunks]
        embeddings = cache.get_embeddings(chunk_ids)
        
        if len(embeddings) > 0:
            doc_embeddings[doc_id] = np.mean(embeddings, axis=0)
    
    # Build nodes
    nodes = []
    for doc in docs:
        if doc.doc_id in doc_embeddings:
            nodes.append({
                "id": doc.doc_id,
                "label": doc.title or doc.doc_id[:8],
                "title": doc.title,
                "authors": doc.authors,
                "year": doc.year,
                "chunk_count": len(doc_chunks.get(doc.doc_id, [])),
                "group": 1  # For visualization grouping
            })
    
    # Build edges (relationships)
    edges = []
    doc_list = list(doc_embeddings.keys())
    for i, doc_id1 in enumerate(doc_list):
        for doc_id2 in doc_list[i+1:]:
            sim = cosine_similarity(
                doc_embeddings[doc_id1].reshape(1, -1),
                doc_embeddings[doc_id2].reshape(1, -1)
            )[0][0]
            
            if sim >= similarity_threshold:
                edges.append({
                    "from": doc_id1,
                    "to": doc_id2,
                    "value": float(sim),
                    "label": f"{sim:.2f}",
                    "title": f"Similarity: {sim:.3f}"
                })
    
    return {
        "nodes": nodes,
        "edges": edges,
        "stats": {
            "total_documents": len(nodes),
            "total_relationships": len(edges),
            "average_similarity": float(np.mean([e["value"] for e in edges])) if edges else 0.0,
            "threshold": similarity_threshold
        }
    }

