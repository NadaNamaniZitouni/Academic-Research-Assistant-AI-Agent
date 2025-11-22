from fastapi import FastAPI, UploadFile, File, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from sqlalchemy.orm import Session
from pathlib import Path
import os
import uuid
from typing import Optional, List, Dict
from datetime import datetime
import time

from .database import init_db, get_db
from .ingest import ingest_pdf_with_embeddings
from .agents import full_rag_pipeline
from .models import ChunkMetadata, Document, User, UserUsage, QueryHistory
from .index_manager import rebuild_index_from_database
from .middleware import setup_rate_limiting
from .utils import validate_pdf, sanitize_filename
from .schemas import (
    QueryRequest, 
    ExportBibTeXRequest, 
    ExportQueryRequest,
    DocumentComparisonRequest,
    CitationNetworkRequest
)
from .auth_routes import router as auth_router
from .auth import get_current_active_user, check_tier_limit
from .analytics import get_query_analytics, get_user_statistics
from .export import generate_bibtex_entry, format_answer_for_export, generate_markdown_export
from .document_comparison import compare_documents
from .citation_network import build_citation_network

# Initialize database
init_db()

app = FastAPI(
    title="Academic Research Assistant API",
    description="An intelligent research assistant with RAG capabilities",
    version="1.0.0"
)

# Include auth router
app.include_router(auth_router)

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
    # Run migrations
    try:
        from .migrations import migrate_database
        migrate_database()
    except Exception as e:
        print(f"Warning: Migration error (may be expected on first run): {e}")


@app.get("/")
async def root():
    return {"message": "Academic Research Assistant API"}


@app.post("/upload")
async def upload_pdf(
    file: UploadFile = File(...),
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Upload and ingest a PDF file (requires authentication)"""
    import sys
    
    # Check document limit
    try:
        if not check_tier_limit(current_user, "documents", db):
            doc_count = db.query(Document).filter(Document.user_id == current_user.user_id).count()
            tier_limits = {
                "free": 5, "starter": 50, "pro": 200, "team": 1000
            }
            limit = tier_limits.get(current_user.tier, 5)
            raise HTTPException(
                status_code=403,
                detail=f"Document limit reached ({doc_count}/{limit}). Upgrade your plan to upload more documents."
            )
    except Exception as e:
        # If user_id column doesn't exist, allow upload (migration will fix it)
        if "no such column" in str(e).lower() or "user_id" in str(e).lower():
            print(f"Warning: user_id column issue, allowing upload: {e}")
        else:
            raise
    
    print(f"\n{'='*60}")
    print(f"[UPLOAD] Request received at {datetime.now()}")
    print(f"[UPLOAD] User: {current_user.username} ({current_user.user_id})")
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
        # Ingest PDF (will be updated to include user_id)
        result = ingest_pdf_with_embeddings(file_path, db, user_id=current_user.user_id)
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
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Query the RAG system (requires authentication)"""
    query_text = query.query
    k = query.k
    doc_id = query.doc_id  # Get doc_id from request (None = search all documents)

    if not query_text or not query_text.strip():
        raise HTTPException(status_code=400, detail="Query text is required")

    # Check query limit
    if not check_tier_limit(current_user, "queries", db):
        current_month = datetime.utcnow().strftime("%Y-%m")
        usage = db.query(UserUsage).filter(
            UserUsage.user_id == current_user.user_id,
            UserUsage.month == current_month
        ).first()
        queries_used = usage.queries_count if usage else 0
        tier_limits = {
            "free": 20, "starter": 500, "pro": -1, "team": -1
        }
        limit = tier_limits.get(current_user.tier, 20)
        raise HTTPException(
            status_code=403,
            detail=f"Monthly query limit reached ({queries_used}/{limit}). Upgrade your plan for more queries."
        )

    start_time = time.time()
    
    try:
        result = full_rag_pipeline(query_text, db, k=k, doc_id=doc_id)
        
        # Track query in history
        response_time_ms = int((time.time() - start_time) * 1000)
        query_history = QueryHistory(
            user_id=current_user.user_id,
            query_text=query_text,
            doc_id=doc_id,
            answer=result.get("answer", "")[:500],  # Store first 500 chars
            response_time_ms=response_time_ms
        )
        db.add(query_history)
        
        # Update monthly usage
        current_month = datetime.utcnow().strftime("%Y-%m")
        usage = db.query(UserUsage).filter(
            UserUsage.user_id == current_user.user_id,
            UserUsage.month == current_month
        ).first()
        
        if usage:
            usage.queries_count += 1
            usage.updated_at = datetime.utcnow()
        else:
            usage = UserUsage(
                user_id=current_user.user_id,
                month=current_month,
                queries_count=1
            )
            db.add(usage)
        
        db.commit()
        
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


@app.get("/documents")
async def get_user_documents(
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Get all documents for the current user"""
    docs = db.query(Document).filter(Document.user_id == current_user.user_id).all()
    
    return [
        {
            "doc_id": doc.doc_id,
            "title": doc.title,
            "authors": doc.authors,
            "year": doc.year,
            "doi": doc.doi,
            "created_at": doc.created_at.isoformat() if doc.created_at else None
        }
        for doc in docs
    ]


@app.get("/analytics/queries")
async def get_analytics(
    days: int = 30,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Get query analytics for the current user"""
    try:
        analytics = get_query_analytics(db, current_user.user_id, days=days)
        return analytics
    except Exception as e:
        import traceback
        print(f"Error getting analytics: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Error getting analytics: {str(e)}")


@app.get("/analytics/stats")
async def get_stats(
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Get overall user statistics"""
    try:
        stats = get_user_statistics(db, current_user.user_id)
        return stats
    except Exception as e:
        import traceback
        print(f"Error getting stats: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Error getting stats: {str(e)}")


@app.post("/export/bibtex")
async def export_bibtex(
    request: ExportBibTeXRequest,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Export documents as BibTeX"""
    if not check_tier_limit(current_user, "export", db):
        raise HTTPException(
            status_code=403,
            detail="Export feature is not available in your current tier. Upgrade to access this feature."
        )
    
    try:
        docs = db.query(Document).filter(
            Document.doc_id.in_(request.doc_ids),
            Document.user_id == current_user.user_id
        ).all()
        
        if not docs:
            raise HTTPException(status_code=404, detail="No documents found")
        
        bibtex_entries = []
        for doc in docs:
            doc_dict = {
                "doc_id": doc.doc_id,
                "title": doc.title or "Untitled",
                "authors": doc.authors or "",
                "year": doc.year,
                "doi": doc.doi or ""
            }
            bibtex_entries.append(generate_bibtex_entry(doc_dict))
        
        bibtex_content = "\n".join(bibtex_entries)
        
        return Response(
            content=bibtex_content,
            media_type="text/plain",
            headers={
                "Content-Disposition": f'attachment; filename="citations_{datetime.now().strftime("%Y%m%d")}.bib"'
            }
        )
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        print(f"Error exporting BibTeX: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Error exporting BibTeX: {str(e)}")


@app.post("/export/markdown")
async def export_markdown(
    request: ExportQueryRequest,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Export query result as Markdown"""
    if not check_tier_limit(current_user, "export", db):
        raise HTTPException(
            status_code=403,
            detail="Export feature is not available in your current tier. Upgrade to access this feature."
        )
    
    try:
        answer = request.query_result.get("answer", "")
        sources = request.query_result.get("sources", [])
        
        markdown_content = generate_markdown_export(answer, sources, request.question)
        
        return Response(
            content=markdown_content,
            media_type="text/markdown",
            headers={
                "Content-Disposition": f'attachment; filename="query_result_{datetime.now().strftime("%Y%m%d_%H%M%S")}.md"'
            }
        )
    except Exception as e:
        import traceback
        print(f"Error exporting Markdown: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Error exporting Markdown: {str(e)}")


@app.post("/export/text")
async def export_text(
    request: ExportQueryRequest,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Export query result as plain text"""
    if not check_tier_limit(current_user, "export", db):
        raise HTTPException(
            status_code=403,
            detail="Export feature is not available in your current tier. Upgrade to access this feature."
        )
    
    try:
        answer = request.query_result.get("answer", "")
        sources = request.query_result.get("sources", [])
        
        text_content = format_answer_for_export(answer, sources)
        text_content = f"Question: {request.question}\n\n{text_content}"
        
        return Response(
            content=text_content,
            media_type="text/plain",
            headers={
                "Content-Disposition": f'attachment; filename="query_result_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt"'
            }
        )
    except Exception as e:
        import traceback
        print(f"Error exporting text: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Error exporting text: {str(e)}")


@app.post("/compare/documents")
async def compare_documents_endpoint(
    request: DocumentComparisonRequest,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Compare multiple documents to find similarities and differences"""
    try:
        result = compare_documents(request.doc_ids, db, current_user.user_id)
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        import traceback
        print(f"Error comparing documents: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Error comparing documents: {str(e)}")


@app.post("/citation-network")
async def get_citation_network(
    request: CitationNetworkRequest,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Build citation network graph for documents"""
    try:
        result = build_citation_network(
            request.doc_ids,
            db,
            current_user.user_id,
            similarity_threshold=request.similarity_threshold
        )
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        import traceback
        print(f"Error building citation network: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Error building citation network: {str(e)}")


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
