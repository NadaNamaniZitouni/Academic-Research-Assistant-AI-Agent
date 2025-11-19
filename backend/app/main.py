from fastapi import FastAPI, UploadFile, File, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session
from pathlib import Path
import os
import uuid
from typing import Optional

from .database import init_db, get_db
from .ingest import ingest_pdf_with_embeddings
from .agents import full_rag_pipeline
from .models import ChunkMetadata, Document
from .index_manager import rebuild_index_from_database
from .middleware import setup_rate_limiting
from .utils import validate_pdf, sanitize_filename
from .schemas import QueryRequest

# Initialize database
init_db()

app = FastAPI(title="Academic Research Assistant API")

# CORS middleware - allow all origins for development
cors_origins = os.getenv("CORS_ORIGINS", "http://localhost:5173,http://localhost:3000").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for debugging
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)

# Setup rate limiting
app = setup_rate_limiting(app)

# Ensure upload directory exists
UPLOAD_DIR = Path(os.getenv("UPLOAD_DIR", "data/uploads"))
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


@app.on_event("startup")
async def startup_event():
    """Initialize on startup"""
    print("Starting Academic Research Assistant API...")
    # Ensure database is initialized
    init_db()
    print("Database initialized.")
    # Optionally rebuild index
    # db = next(get_db())
    # rebuild_index_from_database(db)


@app.get("/")
async def root():
    return {"message": "Academic Research Assistant API"}


@app.post("/upload")
async def upload_pdf(
    file: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    """Upload and ingest a PDF file"""
    import sys
    from datetime import datetime
    
    print(f"\n{'='*60}")
    print(f"[UPLOAD] Request received at {datetime.now()}")
    print(f"[UPLOAD] Filename: {file.filename}")
    print(f"[UPLOAD] Content-Type: {getattr(file, 'content_type', 'unknown')}")
    print(f"[UPLOAD] Size: {getattr(file, 'size', 'unknown')}")
    sys.stdout.flush()
    
    if not file.filename or not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="File must be a PDF")
    
    # Save file
    file_id = str(uuid.uuid4())
    file_path = UPLOAD_DIR / f"{file_id}.pdf"
    
    print(f"[UPLOAD] Saving file to: {file_path}")
    sys.stdout.flush()
    
    try:
        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)
        print(f"[UPLOAD] File saved successfully. Size: {len(content)} bytes")
        sys.stdout.flush()
    except Exception as e:
        print(f"[UPLOAD] Error saving file: {e}")
        import traceback
        print(traceback.format_exc())
        sys.stdout.flush()
        raise HTTPException(status_code=500, detail=f"Error saving file: {str(e)}")
    
    try:
        print(f"[UPLOAD] Starting PDF ingestion...")
        sys.stdout.flush()
        # Ingest PDF
        result = ingest_pdf_with_embeddings(file_path, db)
        print(f"[UPLOAD] Ingestion completed. Doc ID: {result['doc_id']}, Chunks: {result['num_chunks']}")
        print(f"{'='*60}\n")
        sys.stdout.flush()
        return {
            "doc_id": result["doc_id"],
            "status": "completed",
            "num_chunks": result["num_chunks"]
        }
    except Exception as e:
        print(f"[UPLOAD] Error ingesting PDF: {e}")
        import traceback
        print(traceback.format_exc())
        sys.stdout.flush()
        # Clean up file on error
        if file_path.exists():
            file_path.unlink()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/ingest/status/{doc_id}")
async def get_ingest_status(doc_id: str, db: Session = Depends(get_db)):
    """Get ingestion status for a document"""
    doc = db.query(Document).filter(Document.doc_id == doc_id).first()
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")

    # Check if chunks exist
    chunks = db.query(ChunkMetadata).filter(ChunkMetadata.doc_id == doc_id).all()

    if chunks:
        return {"status": "completed", "num_chunks": len(chunks)}
    else:
        return {"status": "processing"}


@app.post("/query")
async def query_endpoint(
    query: QueryRequest,
    db: Session = Depends(get_db)
):
    """Query the RAG system"""
    query_text = query.query
    k = query.k

    if not query_text or not query_text.strip():
        raise HTTPException(status_code=400, detail="Query text is required")

    try:
        result = full_rag_pipeline(query_text, db, k=k)
        return result
    except ValueError as e:
        # User-friendly error messages for common issues
        error_msg = str(e)
        print(f"ValueError in query endpoint: {error_msg}")
        raise HTTPException(status_code=400, detail=error_msg)
    except Exception as e:
        # Log the full error for debugging
        import traceback
        error_details = traceback.format_exc()
        error_msg = str(e)
        print(f"Error in query endpoint: {error_details}")
        # Return detailed error for debugging (in production, you might want to hide this)
        raise HTTPException(
            status_code=500, 
            detail=f"Internal server error: {error_msg}. Check server logs for details."
        )


@app.get("/doc/{doc_id}/chunks/{chunk_id}")
async def get_chunk(
    doc_id: str,
    chunk_id: int,
    db: Session = Depends(get_db)
):
    """Get a specific chunk by ID"""
    chunk = db.query(ChunkMetadata).filter(
        ChunkMetadata.chunk_id == chunk_id,
        ChunkMetadata.doc_id == doc_id
    ).first()

    if not chunk:
        raise HTTPException(status_code=404, detail="Chunk not found")

    return {
        "chunk_id": chunk.chunk_id,
        "doc_id": chunk.doc_id,
        "doc_title": chunk.doc_title,
        "text": chunk.text,
        "page_start": chunk.page_start,
        "page_end": chunk.page_end,
        "source_path": chunk.source_path
    }


@app.get("/search-metadata")
async def search_metadata(
    q: str,
    db: Session = Depends(get_db)
):
    """Search document metadata"""
    docs = db.query(Document).filter(
        Document.title.contains(q) | Document.authors.contains(q)
    ).limit(10).all()

    return [{
        "doc_id": doc.doc_id,
        "title": doc.title,
        "authors": doc.authors,
        "year": doc.year,
        "doi": doc.doi
    } for doc in docs]


@app.get("/debug/index-status")
async def get_index_status(db: Session = Depends(get_db)):
    """Debug endpoint to check index and database status"""
    from .faiss_index import get_faiss_index
    from .models import ChunkMetadata, Document
    
    try:
        faiss_idx = get_faiss_index()
        total_vectors = faiss_idx.get_total()
        
        # Count chunks in database
        chunk_count = db.query(ChunkMetadata).count()
        doc_count = db.query(Document).count()
        
        # Get sample chunk IDs
        sample_chunks = db.query(ChunkMetadata.chunk_id).limit(5).all()
        chunk_ids = [c[0] for c in sample_chunks]
        
        return {
            "faiss_index_vectors": total_vectors,
            "database_chunks": chunk_count,
            "database_documents": doc_count,
            "sample_chunk_ids": chunk_ids,
            "index_synced": total_vectors == chunk_count,
            "status": "ok" if total_vectors > 0 else "empty"
        }
    except Exception as e:
        import traceback
        return {
            "error": str(e),
            "traceback": traceback.format_exc(),
            "status": "error"
        }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

