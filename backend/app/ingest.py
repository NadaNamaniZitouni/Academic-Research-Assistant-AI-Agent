from pathlib import Path
from typing import List, Dict
import uuid
from datetime import datetime
import json
from sqlalchemy.orm import Session

from .pdf_extractor import extract_text_by_page, extract_text_from_pdf
from .chunking import chunk_text_by_pages
from .models import Document, ChunkMetadata
from .database import SessionLocal
from .embeddings import get_embedding_service
from .faiss_index import get_faiss_index
from .embedding_cache import get_embedding_cache


def generate_doc_id() -> str:
    """Generate unique document ID"""
    return str(uuid.uuid4())


def save_document_metadata(
    db: Session,
    doc_id: str,
    title: str,
    authors: str,
    year: int,
    doi: str,
    path: str,
    user_id: str = None
) -> Document:
    """Save document metadata to database"""
    doc = Document(
        doc_id=doc_id,
        user_id=user_id,
        title=title,
        authors=authors,
        year=year,
        doi=doi,
        path=path
    )
    db.add(doc)
    db.commit()
    db.refresh(doc)
    return doc


def ingest_pdf(
    pdf_path: Path,
    db: Session,
    start_chunk_id: int = 0,
    user_id: str = None
) -> Dict:
    """
    Ingest a PDF file: extract text, chunk, and prepare for embedding.

    Args:
        pdf_path: Path to PDF file
        db: Database session
        start_chunk_id: Starting chunk ID (for FAISS index)

    Returns:
        Dict with doc_id, chunks, and metadata
    """
    print(f"Step 1: Extracting text from PDF: {pdf_path}")
    try:
        # Extract text and metadata
        full_text, pdf_metadata = extract_text_from_pdf(pdf_path)
        pages = extract_text_by_page(pdf_path)
        print(f"Extracted {len(pages)} pages, {len(full_text)} characters")
    except Exception as e:
        print(f"Error extracting text from PDF: {e}")
        import traceback
        print(traceback.format_exc())
        raise ValueError(f"Failed to extract text from PDF: {str(e)}")

    # Generate document ID
    doc_id = generate_doc_id()
    print(f"Step 2: Generated document ID: {doc_id}")

    # Chunk text with page information
    print(f"Step 3: Chunking text...")
    try:
        chunked_data = chunk_text_by_pages(pages, chunk_size=800, overlap=200)
        print(f"Created {len(chunked_data)} chunks")
    except Exception as e:
        print(f"Error chunking text: {e}")
        import traceback
        print(traceback.format_exc())
        raise ValueError(f"Failed to chunk PDF text: {str(e)}")

    # Save document metadata
    # Clean title - remove template variables and normalize
    title = pdf_metadata.get("title") or pdf_path.stem
    if title:
        # Remove common template placeholders
        title = title.replace("{", "").replace("}", "").strip()
        if not title or len(title) < 3:
            title = pdf_path.stem
    
    doc = save_document_metadata(
        db=db,
        doc_id=doc_id,
        title=title,
        authors=pdf_metadata.get("authors") or "",
        year=pdf_metadata.get("year"),
        doi=pdf_metadata.get("doi") or "",
        path=str(pdf_path),
        user_id=user_id
    )

    # Prepare chunks for storage
    print(f"Step 4: Saving chunks to database...")
    chunks = []
    try:
        for idx, chunk_data in enumerate(chunked_data):
            chunk_id = start_chunk_id + idx
            chunk_metadata = ChunkMetadata(
                chunk_id=chunk_id,
                doc_id=doc_id,
                doc_title=doc.title,
                source_path=str(pdf_path),
                page_start=chunk_data["page_start"],
                page_end=chunk_data["page_end"],
                text=chunk_data["text"]
            )
            db.add(chunk_metadata)
            chunks.append({
                "chunk_id": chunk_id,
                "text": chunk_data["text"],
                "page_start": chunk_data["page_start"],
                "page_end": chunk_data["page_end"]
            })

        db.commit()
        print(f"Saved {len(chunks)} chunks to database")
    except Exception as e:
        db.rollback()
        print(f"Error saving chunks to database: {e}")
        import traceback
        print(traceback.format_exc())
        raise ValueError(f"Failed to save chunks to database: {str(e)}")

    return {
        "doc_id": doc_id,
        "chunks": chunks,
        "metadata": pdf_metadata,
        "num_chunks": len(chunks)
    }


def ingest_pdf_with_embeddings(
    pdf_path: Path,
    db: Session,
    start_chunk_id: int = None,
    user_id: str = None
) -> Dict:
    """
    Full ingestion pipeline: extract, chunk, embed, and index.
    
    Steps:
    1. Get FAISS index and determine starting chunk ID
    2. Ingest PDF (extract, chunk, save metadata)
    3. Generate embeddings for all chunks
    4. Add embeddings to FAISS index
    5. Save FAISS index to disk
    """
    print(f"Starting full ingestion pipeline for: {pdf_path}")
    
    # Step 1: Get current chunk count for ID assignment
    try:
        faiss_idx = get_faiss_index()
        start_chunk_id = start_chunk_id or faiss_idx.get_total()
        print(f"Starting chunk ID: {start_chunk_id}, Current index size: {faiss_idx.get_total()}")
    except Exception as e:
        print(f"Error getting FAISS index: {e}")
        import traceback
        print(traceback.format_exc())
        raise RuntimeError(f"Failed to initialize FAISS index: {str(e)}")

    # Step 2: Ingest PDF (extract, chunk, save metadata)
    result = ingest_pdf(pdf_path, db, start_chunk_id, user_id=user_id)

    # Step 3: Generate embeddings
    print(f"Step 5: Generating embeddings for {len(result['chunks'])} chunks...")
    try:
        embedding_service = get_embedding_service()
        chunk_texts = [chunk["text"] for chunk in result["chunks"]]
        print(f"Encoding {len(chunk_texts)} text chunks...")
        embeddings = embedding_service.encode(chunk_texts, show_progress_bar=True)
        print(f"Generated {len(embeddings)} embeddings of dimension {len(embeddings[0])}")
    except Exception as e:
        print(f"Error generating embeddings: {e}")
        import traceback
        print(traceback.format_exc())
        raise RuntimeError(f"Failed to generate embeddings: {str(e)}")

    # Step 4: Add to FAISS index
    print(f"Step 6: Adding embeddings to FAISS index...")
    try:
        faiss_idx.add_vectors(embeddings)
        print(f"Added {len(embeddings)} vectors to index. New index size: {faiss_idx.get_total()}")
    except Exception as e:
        print(f"Error adding vectors to FAISS index: {e}")
        import traceback
        print(traceback.format_exc())
        raise RuntimeError(f"Failed to add vectors to FAISS index: {str(e)}")
    
    # Step 4b: Add to embedding cache (replace existing if same chunk_id)
    print(f"Step 6b: Adding embeddings to cache...")
    try:
        embedding_cache = get_embedding_cache()
        chunk_ids = [chunk["chunk_id"] for chunk in result["chunks"]]
        embedding_cache.add_embeddings(chunk_ids, embeddings, replace_existing=True)
        print(f"Added/updated {len(embeddings)} embeddings to cache. Cache size: {embedding_cache.size()}")
    except Exception as e:
        print(f"Warning: Error adding embeddings to cache: {e}")
        import traceback
        print(traceback.format_exc())
        # Don't fail the whole ingestion if cache fails

    # Step 5: Save FAISS index
    print(f"Step 7: Saving FAISS index to disk...")
    try:
        faiss_idx.save()
        print(f"FAISS index saved successfully")
    except Exception as e:
        print(f"Error saving FAISS index: {e}")
        import traceback
        print(traceback.format_exc())
        raise RuntimeError(f"Failed to save FAISS index: {str(e)}")

    result["embeddings_added"] = len(embeddings)
    print(f"Ingestion pipeline completed successfully!")
    return result

