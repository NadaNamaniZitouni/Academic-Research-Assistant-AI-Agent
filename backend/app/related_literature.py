from typing import List, Dict
from sqlalchemy.orm import Session
import numpy as np

from .embeddings import get_embedding_service
from .faiss_index import get_faiss_index
from .models import Document, ChunkMetadata
from .keywords import extract_keywords


def find_related_papers(
    query_text: str,
    answer_text: str,
    db: Session,
    top_k: int = 5
) -> List[Dict]:
    """
    Find related papers based on query and answer.

    Args:
        query_text: Original query
        answer_text: Generated answer
        db: Database session
        top_k: Number of related papers to return

    Returns:
        List of related paper dictionaries
    """
    # Combine query and answer for semantic search
    search_text = f"{query_text} {answer_text}"

    # Extract keywords
    keywords = extract_keywords(search_text, top_n=15)
    keyword_text = " ".join(keywords)

    # Use semantic search on document titles/abstracts
    # For simplicity, search chunks and aggregate by document
    embedding_service = get_embedding_service()
    query_embedding = embedding_service.encode_query(keyword_text)

    faiss_idx = get_faiss_index()
    distances, chunk_ids = faiss_idx.search(query_embedding, k=top_k * 3)  # Get more chunks

    # Aggregate by document
    doc_scores = {}
    for chunk_id, distance in zip(chunk_ids, distances):
        chunk = db.query(ChunkMetadata).filter(
            ChunkMetadata.chunk_id == int(chunk_id)
        ).first()

        if chunk:
            doc_id = chunk.doc_id
            if doc_id not in doc_scores:
                doc_scores[doc_id] = {
                    "doc_id": doc_id,
                    "score": float(distance),
                    "chunks": []
                }
            doc_scores[doc_id]["chunks"].append(chunk)
            # Update score (max similarity)
            doc_scores[doc_id]["score"] = max(
                doc_scores[doc_id]["score"],
                float(distance)
            )

    # Get document metadata
    related_papers = []
    for doc_id, doc_data in sorted(
        doc_scores.items(),
        key=lambda x: x[1]["score"],
        reverse=True
    )[:top_k]:
        doc = db.query(Document).filter(Document.doc_id == doc_id).first()
        if doc:
            related_papers.append({
                "doc_id": doc.doc_id,
                "title": doc.title,
                "authors": doc.authors,
                "year": doc.year,
                "doi": doc.doi,
                "relevance_score": doc_data["score"],
                "matching_chunks": len(doc_data["chunks"])
            })

    return related_papers

