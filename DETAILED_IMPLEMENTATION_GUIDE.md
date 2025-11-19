# Academic Research Assistant AI Agent - Detailed Implementation Guide

## Table of Contents
1. [Project Overview](#project-overview)
2. [Phase 0: Project Setup](#phase-0-project-setup)
3. [Phase 1: PDF Ingestion Pipeline](#phase-1-pdf-ingestion-pipeline)
4. [Phase 2: Embeddings & FAISS Index](#phase-2-embeddings--faiss-index)
5. [Phase 3: RAG Retrieval & LLM Answering](#phase-3-rag-retrieval--llm-answering)
6. [Phase 4: Multi-step Reasoning & Related Literature](#phase-4-multi-step-reasoning--related-literature)
7. [Phase 5: Frontend Development](#phase-5-frontend-development)
8. [Phase 6: Testing & Deployment](#phase-6-testing--deployment)

---

## Project Overview

### Objective
Build an intelligent Research Assistant that:
- Ingests academic PDFs with metadata extraction
- Indexes content using FAISS vector database
- Performs RAG (Retrieval-Augmented Generation) queries
- Produces citation-aware summaries and answers
- Suggests related literature and identifies research gaps
- Provides a React frontend for interaction

### Technology Stack
- **Backend**: Python 3.10+, FastAPI, Uvicorn
- **Vector DB**: FAISS (faiss-cpu)
- **Embeddings**: SentenceTransformers (all-MiniLM-L6-v2)
- **LLM**: Local OSS model (Mistral/Llama via transformers) or cloud API
- **Orchestration**: LangChain
- **PDF Processing**: PyPDF2 / pdfminer.six
- **Frontend**: React (Vite or CRA) + axios
- **Storage**: SQLite / JSONL for metadata
- **Deployment**: Docker

---

## Phase 0: Project Setup

### Step 0.1: Initialize Repository Structure

```bash
# Create project directory
mkdir academic-research-assistant
cd academic-research-assistant

# Create directory structure
mkdir -p backend/app
mkdir -p frontend
mkdir -p data
mkdir -p notebooks
mkdir -p docs
mkdir -p tests
mkdir -p data/uploads
mkdir -p data/indices
mkdir -p data/metadata
```

### Step 0.2: Initialize Python Backend

```bash
cd backend

# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# Create requirements.txt
```

**backend/requirements.txt:**
```txt
fastapi==0.104.1
uvicorn[standard]==0.24.0
python-multipart==0.0.6
pydantic==2.5.0
pydantic-settings==2.1.0

# PDF Processing
pypdf==3.17.0
pdfminer.six==20221105

# Embeddings & Vector DB
sentence-transformers==2.2.2
faiss-cpu==1.7.4
numpy==1.24.3
scikit-learn==1.3.2

# LLM & Orchestration
langchain==0.1.0
langchain-community==0.0.10
transformers==4.36.0
torch==2.1.0
accelerate==0.25.0

# Database
sqlalchemy==2.0.23
aiosqlite==0.19.0

# Utilities
python-dotenv==1.0.0
tiktoken==0.5.1
python-dateutil==2.8.2

# Testing
pytest==7.4.3
pytest-asyncio==0.21.1
httpx==0.25.2
```

```bash
# Install dependencies
pip install -r requirements.txt
```

### Step 0.3: Initialize Frontend

```bash
cd ../frontend

# Using Vite (recommended)
npm create vite@latest react-app -- --template react
cd react-app
npm install

# Install additional dependencies
npm install axios
npm install react-router-dom
```

### Step 0.4: Create Configuration Files

**backend/.env.example:**
```env
# LLM Configuration
LLM_PROVIDER=local  # or "openai", "ollama"
LLM_MODEL_NAME=mistral-7b-instruct-v0.2  # or "gpt-3.5-turbo" for OpenAI
OLLAMA_BASE_URL=http://localhost:11434  # if using Ollama
OPENAI_API_KEY=  # if using OpenAI

# Embedding Model
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2

# Paths
DATA_DIR=../data
UPLOAD_DIR=../data/uploads
INDEX_DIR=../data/indices
METADATA_DIR=../data/metadata

# FAISS Index
FAISS_INDEX_PATH=../data/indices/faiss_index.idx
METADATA_STORE_PATH=../data/metadata/chunks.jsonl
DB_PATH=../data/research_assistant.db

# API Settings
API_HOST=0.0.0.0
API_PORT=8000
CORS_ORIGINS=http://localhost:5173,http://localhost:3000
```

**backend/.gitignore:**
```
venv/
__pycache__/
*.pyc
*.pyo
*.pyd
.Python
*.so
*.egg
*.egg-info
dist/
build/
.env
*.db
*.idx
*.jsonl
data/
```

**frontend/.gitignore:**
```
node_modules/
dist/
.env
.env.local
```

### Step 0.5: Setup Linting & Formatting

```bash
cd backend
pip install black flake8 pre-commit

# Create .pre-commit-config.yaml
```

**backend/.pre-commit-config.yaml:**
```yaml
repos:
  - repo: https://github.com/psf/black
    rev: 23.11.0
    hooks:
      - id: black
  - repo: https://github.com/pycqa/flake8
    rev: 6.1.0
    hooks:
      - id: flake8
```

```bash
pre-commit install
```

### Step 0.6: Create Initial README

**README.md:**
```markdown
# Academic Research Assistant AI Agent

An intelligent research assistant that ingests academic PDFs, performs RAG queries, and suggests related literature.

## Setup

### Backend
```bash
cd backend
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
# Edit .env with your configuration
```

### Frontend
```bash
cd frontend/react-app
npm install
```

## Running

### Backend
```bash
cd backend
uvicorn app.main:app --reload
```

### Frontend
```bash
cd frontend/react-app
npm run dev
```

## License
MIT
```

---

## Phase 1: PDF Ingestion Pipeline

### Step 1.1: Create Database Models

**backend/app/models.py:**
```python
from sqlalchemy import Column, Integer, String, DateTime, Text
from sqlalchemy.ext.declarative import declarative_base
from datetime import datetime

Base = declarative_base()

class Document(Base):
    __tablename__ = "documents"
    
    doc_id = Column(String, primary_key=True)
    title = Column(String)
    authors = Column(Text)
    year = Column(Integer)
    doi = Column(String)
    path = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)

class ChunkMetadata(Base):
    __tablename__ = "chunk_metadata"
    
    chunk_id = Column(Integer, primary_key=True, autoincrement=True)
    doc_id = Column(String, index=True)
    doc_title = Column(String)
    source_path = Column(String)
    page_start = Column(Integer)
    page_end = Column(Integer)
    text = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)
```

### Step 1.2: Create Database Connection

**backend/app/database.py:**
```python
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from .models import Base
import os
from pathlib import Path

# Get database path from environment or use default
DB_PATH = os.getenv("DB_PATH", str(Path(__file__).parent.parent.parent / "data" / "research_assistant.db"))

# Ensure data directory exists
Path(DB_PATH).parent.mkdir(parents=True, exist_ok=True)

engine = create_engine(f"sqlite:///{DB_PATH}", connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def init_db():
    """Initialize database tables"""
    Base.metadata.create_all(bind=engine)

def get_db():
    """Dependency for FastAPI"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
```

### Step 1.3: Implement PDF Text Extraction

**backend/app/pdf_extractor.py:**
```python
from pathlib import Path
from typing import Optional, Tuple
import pypdf
from pdfminer.high_level import extract_text as pdfminer_extract
import re

def extract_text_from_pdf(pdf_path: Path, method: str = "pypdf") -> Tuple[str, dict]:
    """
    Extract text from PDF and return text with metadata.
    
    Args:
        pdf_path: Path to PDF file
        method: "pypdf" or "pdfminer"
    
    Returns:
        Tuple of (text, metadata_dict)
    """
    metadata = {
        "title": None,
        "authors": None,
        "year": None,
        "doi": None,
        "num_pages": 0
    }
    
    try:
        if method == "pypdf":
            with open(pdf_path, "rb") as file:
                pdf_reader = pypdf.PdfReader(file)
                metadata["num_pages"] = len(pdf_reader.pages)
                
                # Extract metadata
                if pdf_reader.metadata:
                    metadata["title"] = pdf_reader.metadata.get("/Title", "")
                    metadata["authors"] = pdf_reader.metadata.get("/Author", "")
                    if "/CreationDate" in pdf_reader.metadata:
                        date_str = pdf_reader.metadata["/CreationDate"]
                        year_match = re.search(r"(\d{4})", date_str)
                        if year_match:
                            metadata["year"] = int(year_match.group(1))
                
                # Extract text page by page
                text_pages = []
                for page_num, page in enumerate(pdf_reader.pages, 1):
                    page_text = page.extract_text()
                    text_pages.append((page_num, page_text))
                
                full_text = "\n\n".join([text for _, text in text_pages])
        
        else:  # pdfminer
            full_text = pdfminer_extract(str(pdf_path))
            # Try to get page count with pypdf
            try:
                with open(pdf_path, "rb") as file:
                    pdf_reader = pypdf.PdfReader(file)
                    metadata["num_pages"] = len(pdf_reader.pages)
            except:
                metadata["num_pages"] = len(full_text.split("\f"))
    
    except Exception as e:
        raise ValueError(f"Failed to extract text from PDF: {str(e)}")
    
    # Clean text
    full_text = clean_text(full_text)
    
    # Try to extract DOI from text
    doi_match = re.search(r"doi[:\s]+([0-9.]+/[^\s]+)", full_text, re.IGNORECASE)
    if doi_match:
        metadata["doi"] = doi_match.group(1)
    
    return full_text, metadata

def clean_text(text: str) -> str:
    """Clean extracted text"""
    # Remove excessive whitespace
    text = re.sub(r"\s+", " ", text)
    # Remove page breaks
    text = text.replace("\f", "\n")
    # Remove excessive newlines
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

def extract_text_by_page(pdf_path: Path) -> list[Tuple[int, str]]:
    """Extract text page by page for better chunking with page numbers"""
    pages = []
    try:
        with open(pdf_path, "rb") as file:
            pdf_reader = pypdf.PdfReader(file)
            for page_num, page in enumerate(pdf_reader.pages, 1):
                text = page.extract_text()
                text = clean_text(text)
                pages.append((page_num, text))
    except Exception as e:
        raise ValueError(f"Failed to extract pages: {str(e)}")
    return pages
```

### Step 1.4: Implement Token-Aware Chunking

**backend/app/chunking.py:**
```python
from typing import List, Tuple
import tiktoken
from pathlib import Path

def get_tokenizer(model: str = "gpt-3.5-turbo"):
    """Get tiktoken tokenizer"""
    try:
        return tiktoken.encoding_for_model(model)
    except:
        return tiktoken.get_encoding("cl100k_base")

def chunk_text(
    text: str,
    chunk_size: int = 800,
    overlap: int = 200,
    tokenizer_model: str = "gpt-3.5-turbo"
) -> List[str]:
    """
    Chunk text by tokens with overlap.
    
    Args:
        text: Text to chunk
        chunk_size: Target chunk size in tokens
        overlap: Overlap size in tokens
        tokenizer_model: Model name for tokenizer
    
    Returns:
        List of text chunks
    """
    tokenizer = get_tokenizer(tokenizer_model)
    tokens = tokenizer.encode(text)
    
    chunks = []
    i = 0
    while i < len(tokens):
        chunk_tokens = tokens[i:i + chunk_size]
        chunk_text = tokenizer.decode(chunk_tokens)
        chunks.append(chunk_text)
        
        if i + chunk_size >= len(tokens):
            break
        i += chunk_size - overlap
    
    return chunks

def chunk_text_by_pages(
    pages: List[Tuple[int, str]],
    chunk_size: int = 800,
    overlap: int = 200
) -> List[dict]:
    """
    Chunk text with page information preserved.
    
    Args:
        pages: List of (page_num, text) tuples
        chunk_size: Target chunk size in tokens
        overlap: Overlap size in tokens
    
    Returns:
        List of dicts with 'text', 'page_start', 'page_end'
    """
    # Combine all pages
    full_text = "\n\n".join([text for _, text in pages])
    chunks = chunk_text(full_text, chunk_size, overlap)
    
    # Map chunks back to pages (simplified - could be more sophisticated)
    chunked_pages = []
    current_page = 1
    
    for chunk in chunks:
        # Estimate which pages this chunk spans
        # This is approximate - for exact mapping, track character positions
        chunk_start_page = current_page
        # Estimate end page based on text length
        estimated_pages = max(1, len(chunk) // 2000)  # Rough estimate
        chunk_end_page = min(current_page + estimated_pages - 1, len(pages))
        
        chunked_pages.append({
            "text": chunk,
            "page_start": chunk_start_page,
            "page_end": chunk_end_page
        })
        
        current_page = max(1, chunk_end_page - (overlap // 100))  # Rough estimate
    
    return chunked_pages
```

### Step 1.5: Create Ingestion Module

**backend/app/ingest.py:**
```python
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
    path: str
) -> Document:
    """Save document metadata to database"""
    doc = Document(
        doc_id=doc_id,
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
    start_chunk_id: int = 0
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
    # Extract text and metadata
    full_text, pdf_metadata = extract_text_from_pdf(pdf_path)
    pages = extract_text_by_page(pdf_path)
    
    # Generate document ID
    doc_id = generate_doc_id()
    
    # Chunk text with page information
    chunked_data = chunk_text_by_pages(pages, chunk_size=800, overlap=200)
    
    # Save document metadata
    doc = save_document_metadata(
        db=db,
        doc_id=doc_id,
        title=pdf_metadata.get("title") or pdf_path.stem,
        authors=pdf_metadata.get("authors") or "",
        year=pdf_metadata.get("year"),
        doi=pdf_metadata.get("doi") or "",
        path=str(pdf_path)
    )
    
    # Prepare chunks for storage
    chunks = []
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
    
    return {
        "doc_id": doc_id,
        "chunks": chunks,
        "metadata": pdf_metadata,
        "num_chunks": len(chunks)
    }
```

### Step 1.6: Create Unit Tests for Ingestion

**tests/test_ingest.py:**
```python
import pytest
from pathlib import Path
from backend.app.pdf_extractor import extract_text_from_pdf, clean_text
from backend.app.chunking import chunk_text
from backend.app.ingest import ingest_pdf
from backend.app.database import init_db, SessionLocal

@pytest.fixture
def db_session():
    """Create test database session"""
    init_db()
    db = SessionLocal()
    yield db
    db.close()

def test_clean_text():
    """Test text cleaning"""
    text = "Hello   world\n\n\nTest"
    cleaned = clean_text(text)
    assert "\n\n\n" not in cleaned
    assert "  " not in cleaned

def test_chunk_text():
    """Test text chunking"""
    text = " ".join(["word"] * 2000)  # Create long text
    chunks = chunk_text(text, chunk_size=100, overlap=20)
    assert len(chunks) > 1
    assert all(len(chunk.split()) > 0 for chunk in chunks)

def test_extract_text_from_pdf():
    """Test PDF extraction (requires sample PDF)"""
    # This test requires a sample PDF file
    # pdf_path = Path("tests/sample.pdf")
    # if pdf_path.exists():
    #     text, metadata = extract_text_from_pdf(pdf_path)
    #     assert len(text) > 0
    #     assert "num_pages" in metadata
    pass
```

Run tests:
```bash
cd backend
pytest tests/test_ingest.py -v
```

---

## Phase 2: Embeddings & FAISS Index

### Step 2.1: Create Embeddings Module

**backend/app/embeddings.py:**
```python
from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List
import os

class EmbeddingService:
    def __init__(self, model_name: str = None):
        model_name = model_name or os.getenv(
            "EMBEDDING_MODEL",
            "sentence-transformers/all-MiniLM-L6-v2"
        )
        self.model = SentenceTransformer(model_name)
        self.dimension = self.model.get_sentence_embedding_dimension()
    
    def encode(self, texts: List[str], show_progress_bar: bool = True) -> np.ndarray:
        """Encode texts to embeddings"""
        return self.model.encode(
            texts,
            show_progress_bar=show_progress_bar,
            convert_to_numpy=True
        )
    
    def encode_query(self, query: str) -> np.ndarray:
        """Encode a single query"""
        return self.encode([query], show_progress_bar=False)[0]

# Global instance
_embedding_service = None

def get_embedding_service() -> EmbeddingService:
    """Get or create embedding service singleton"""
    global _embedding_service
    if _embedding_service is None:
        _embedding_service = EmbeddingService()
    return _embedding_service
```

### Step 2.2: Create FAISS Index Manager

**backend/app/faiss_index.py:**
```python
import faiss
import numpy as np
from pathlib import Path
from typing import List, Tuple
from sklearn.preprocessing import normalize
import os

class FAISSIndex:
    def __init__(self, dimension: int = 384, index_path: str = None):
        self.dimension = dimension
        self.index_path = index_path or os.getenv(
            "FAISS_INDEX_PATH",
            str(Path(__file__).parent.parent.parent / "data" / "indices" / "faiss_index.idx")
        )
        
        # Ensure directory exists
        Path(self.index_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Initialize or load index
        if Path(self.index_path).exists():
            self.index = faiss.read_index(self.index_path)
        else:
            # Use Inner Product for cosine similarity (after normalization)
            self.index = faiss.IndexFlatIP(dimension)
    
    def add_vectors(self, vectors: np.ndarray):
        """Add vectors to index (normalize for cosine similarity)"""
        if len(vectors.shape) == 1:
            vectors = vectors.reshape(1, -1)
        
        # Normalize for cosine similarity
        vectors_normalized = normalize(vectors, norm='l2').astype('float32')
        self.index.add(vectors_normalized)
    
    def search(
        self,
        query_vector: np.ndarray,
        k: int = 5
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Search for similar vectors.
        
        Returns:
            Tuple of (distances, indices)
        """
        if len(query_vector.shape) == 1:
            query_vector = query_vector.reshape(1, -1)
        
        # Normalize query
        query_normalized = normalize(query_vector, norm='l2').astype('float32')
        
        distances, indices = self.index.search(query_normalized, k)
        return distances[0], indices[0]
    
    def save(self):
        """Save index to disk"""
        Path(self.index_path).parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, self.index_path)
    
    def get_total(self) -> int:
        """Get total number of vectors in index"""
        return self.index.ntotal
    
    def rebuild(self):
        """Rebuild index (useful for incremental updates)"""
        # For IndexFlatIP, no rebuild needed
        # For other index types, you might need to rebuild
        pass

# Global instance
_faiss_index = None

def get_faiss_index() -> FAISSIndex:
    """Get or create FAISS index singleton"""
    global _faiss_index
    if _faiss_index is None:
        embedding_service = get_embedding_service()
        _faiss_index = FAISSIndex(dimension=embedding_service.dimension)
    return _faiss_index
```

### Step 2.3: Update Ingestion to Include Embeddings

**backend/app/ingest.py** (add to existing file):
```python
from .embeddings import get_embedding_service
from .faiss_index import get_faiss_index

# Add to ingest_pdf function after chunking:

def ingest_pdf_with_embeddings(
    pdf_path: Path,
    db: Session,
    start_chunk_id: int = None
) -> Dict:
    """
    Full ingestion pipeline: extract, chunk, embed, and index.
    """
    # Get current chunk count for ID assignment
    faiss_idx = get_faiss_index()
    start_chunk_id = start_chunk_id or faiss_idx.get_total()
    
    # Ingest PDF (extract, chunk, save metadata)
    result = ingest_pdf(pdf_path, db, start_chunk_id)
    
    # Generate embeddings
    embedding_service = get_embedding_service()
    chunk_texts = [chunk["text"] for chunk in result["chunks"]]
    embeddings = embedding_service.encode(chunk_texts, show_progress_bar=True)
    
    # Add to FAISS index
    faiss_idx.add_vectors(embeddings)
    faiss_idx.save()
    
    result["embeddings_added"] = len(embeddings)
    return result
```

### Step 2.4: Create Index Management Utilities

**backend/app/index_manager.py:**
```python
from pathlib import Path
from sqlalchemy.orm import Session
from .faiss_index import get_faiss_index
from .embeddings import get_embedding_service
from .models import ChunkMetadata

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
    from .faiss_index import FAISSIndex
    import os
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
    
    return {
        "status": "success",
        "chunks_indexed": len(chunks),
        "index_path": index_path
    }
```

---

## Phase 3: RAG Retrieval & LLM Answering

### Step 3.1: Create RAG Retrieval Module

**backend/app/rag.py:**
```python
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
    # Encode query
    embedding_service = get_embedding_service()
    query_embedding = embedding_service.encode_query(query)
    
    # Search FAISS index
    faiss_idx = get_faiss_index()
    distances, chunk_ids = faiss_idx.search(query_embedding, k)
    
    # Load chunk metadata from database
    chunks = []
    for chunk_id, distance in zip(chunk_ids, distances):
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
    
    return chunks

def format_context_for_llm(chunks: List[Dict]) -> str:
    """Format retrieved chunks as context for LLM"""
    context_parts = []
    for i, chunk in enumerate(chunks, 1):
        source = f"[{chunk['doc_title']}, p{chunk['page_start']}-{chunk['page_end']}]"
        context_parts.append(f"Source {i} {source}:\n{chunk['text']}\n")
    return "\n".join(context_parts)
```

### Step 3.2: Create LLM Wrapper

**backend/app/llm_wrapper.py:**
```python
from typing import Optional
import os
from langchain.llms.base import LLM
from langchain.callbacks.manager import CallbackManagerForLLMRun
from pydantic import Field

# Option 1: Local model with transformers
try:
    from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

# Option 2: OpenAI
try:
    from langchain.llms import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

# Option 3: Ollama
try:
    from langchain.llms import Ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False

class LocalLLM(LLM):
    """Local LLM using transformers"""
    model_name: str = Field(default="mistralai/Mistral-7B-Instruct-v0.2")
    pipeline: Optional[object] = None
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if TRANSFORMERS_AVAILABLE:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16,
                device_map="auto"
            )
            self.pipeline = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                max_new_tokens=512,
                temperature=0.7
            )
    
    @property
    def _llm_type(self) -> str:
        return "local_transformers"
    
    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs
    ) -> str:
        if not self.pipeline:
            return "LLM not initialized"
        
        result = self.pipeline(prompt, return_full_text=False)
        return result[0]["generated_text"]

def get_llm():
    """Get LLM instance based on configuration"""
    provider = os.getenv("LLM_PROVIDER", "openai").lower()
    
    if provider == "openai" and OPENAI_AVAILABLE:
        api_key = os.getenv("OPENAI_API_KEY")
        if api_key:
            return OpenAI(temperature=0.7, model_name="gpt-3.5-turbo")
    
    elif provider == "ollama" and OLLAMA_AVAILABLE:
        base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        model_name = os.getenv("LLM_MODEL_NAME", "mistral")
        return Ollama(base_url=base_url, model=model_name)
    
    elif provider == "local" and TRANSFORMERS_AVAILABLE:
        model_name = os.getenv("LLM_MODEL_NAME", "mistralai/Mistral-7B-Instruct-v0.2")
        return LocalLLM(model_name=model_name)
    
    else:
        # Fallback: use a simple mock for testing
        class MockLLM(LLM):
            def _call(self, prompt: str, **kwargs) -> str:
                return "This is a mock response. Please configure a real LLM."
            @property
            def _llm_type(self) -> str:
                return "mock"
        return MockLLM()
```

### Step 3.3: Create RAG Chain

**backend/app/agents.py:**
```python
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from typing import List, Dict

from .llm_wrapper import get_llm
from .rag import format_context_for_llm

# Initialize LLM
_llm = None

def get_llm_instance():
    """Get or create LLM instance"""
    global _llm
    if _llm is None:
        _llm = get_llm()
    return _llm

def answer_with_rag(
    chunks: List[Dict],
    question: str
) -> str:
    """
    Generate answer using RAG with retrieved chunks.
    
    Args:
        chunks: List of retrieved chunk dictionaries
        question: User question
    
    Returns:
        Generated answer with citations
    """
    # Format context
    context = format_context_for_llm(chunks)
    
    # Create prompt
    prompt_template = PromptTemplate(
        input_variables=["context", "question"],
        template="""You are a research assistant. Use the following context from academic papers to answer the question. Include citations in the format [Source Title, pX-Y] where appropriate.

Context:
{context}

Question: {question}

Answer concisely and accurately, citing sources where you use information from them. If the context doesn't contain enough information to answer the question, say so.
"""
    )
    
    # Create chain
    llm = get_llm_instance()
    chain = LLMChain(llm=llm, prompt=prompt_template)
    
    # Generate answer
    result = chain.run(context=context, question=question)
    return result.strip()
```

---

## Phase 4: Multi-step Reasoning & Related Literature

### Step 4.1: Create Keyword Extraction

**backend/app/keywords.py:**
```python
from typing import List
import re
from collections import Counter

def extract_keywords(text: str, top_n: int = 10) -> List[str]:
    """Extract keywords from text using simple frequency analysis"""
    # Remove common stop words
    stop_words = {
        "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
        "of", "with", "by", "from", "as", "is", "was", "are", "were", "be",
        "been", "being", "have", "has", "had", "do", "does", "did", "will",
        "would", "could", "should", "may", "might", "must", "can", "this",
        "that", "these", "those", "it", "its", "they", "them", "their"
    }
    
    # Extract words (alphanumeric, at least 3 chars)
    words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
    
    # Filter stop words and count
    keywords = [w for w in words if w not in stop_words]
    counter = Counter(keywords)
    
    # Return top N
    return [word for word, _ in counter.most_common(top_n)]

def extract_keyphrases(text: str, n: int = 3) -> List[str]:
    """Extract n-gram keyphrases"""
    words = text.lower().split()
    keyphrases = []
    
    for i in range(len(words) - n + 1):
        phrase = " ".join(words[i:i+n])
        keyphrases.append(phrase)
    
    # Return most common
    counter = Counter(keyphrases)
    return [phrase for phrase, _ in counter.most_common(10)]
```

### Step 4.2: Create Related Literature Suggestion

**backend/app/related_literature.py:**
```python
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
```

### Step 4.3: Create Research Gap Analysis

**backend/app/gap_analysis.py:**
```python
from typing import List, Dict
from .llm_wrapper import get_llm_instance
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

def identify_research_gaps(
    answer: str,
    related_papers: List[Dict],
    question: str
) -> List[Dict]:
    """
    Identify research gaps based on answer and related literature.
    
    Args:
        answer: Generated answer
        related_papers: List of related paper dictionaries
        question: Original question
    
    Returns:
        List of research gap dictionaries
    """
    # Format related papers
    papers_text = "\n".join([
        f"- {p['title']} ({p.get('year', 'N/A')}): {p.get('authors', 'N/A')}"
        for p in related_papers[:5]  # Top 5
    ])
    
    # Create prompt
    prompt_template = PromptTemplate(
        input_variables=["answer", "related_papers", "question"],
        template="""You are a research analyst. Based on the following answer to a research question and the related literature, identify 3-5 research gaps and suggest potential experimental approaches.

Question: {question}

Answer: {answer}

Related Literature:
{related_papers}

Identify research gaps (areas not fully addressed) and suggest:
1. What questions remain unanswered?
2. What methodological approaches could address these gaps?
3. What experiments or studies would be valuable?

Format your response as a numbered list of gaps, each with:
- Gap description
- Suggested approach/experiment

Be specific and actionable.
"""
    )
    
    # Generate gaps
    llm = get_llm_instance()
    chain = LLMChain(llm=llm, prompt=prompt_template)
    
    gaps_text = chain.run(
        answer=answer,
        related_papers=papers_text,
        question=question
    )
    
    # Parse gaps (simple parsing - could be improved)
    gaps = []
    lines = gaps_text.split("\n")
    current_gap = None
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        # Check if it's a numbered item
        if line and line[0].isdigit():
            if current_gap:
                gaps.append(current_gap)
            current_gap = {"description": line, "suggestions": []}
        elif current_gap and ("approach" in line.lower() or "experiment" in line.lower()):
            current_gap["suggestions"].append(line)
        elif current_gap:
            current_gap["description"] += " " + line
    
    if current_gap:
        gaps.append(current_gap)
    
    return gaps[:5]  # Return top 5
```

### Step 4.4: Integrate Everything in Agents Module

**backend/app/agents.py** (add to existing):
```python
from .related_literature import find_related_papers
from .gap_analysis import identify_research_gaps

def full_rag_pipeline(
    question: str,
    db: Session,
    k: int = 5
) -> Dict:
    """
    Complete RAG pipeline: retrieve, answer, find related papers, identify gaps.
    
    Returns:
        Dict with answer, sources, related_papers, gaps
    """
    from .rag import retrieve_chunks
    
    # Step 1: Retrieve relevant chunks
    chunks = retrieve_chunks(question, db, k=k)
    
    # Step 2: Generate answer
    answer = answer_with_rag(chunks, question)
    
    # Step 3: Find related papers
    related_papers = find_related_papers(question, answer, db, top_k=5)
    
    # Step 4: Identify research gaps
    gaps = identify_research_gaps(answer, related_papers, question)
    
    # Format sources
    sources = [
        {
            "chunk_id": c["chunk_id"],
            "doc_id": c["doc_id"],
            "doc_title": c["doc_title"],
            "page_range": f"{c['page_start']}-{c['page_end']}",
            "snippet": c["text"][:200] + "..." if len(c["text"]) > 200 else c["text"],
            "similarity_score": c["similarity_score"]
        }
        for c in chunks
    ]
    
    return {
        "answer": answer,
        "sources": sources,
        "related_papers": related_papers,
        "gaps": gaps
    }
```

---

## Phase 5: Frontend Development

### Step 5.1: Setup React App Structure

```bash
cd frontend/react-app
mkdir -p src/components
mkdir -p src/pages
mkdir -p src/services
mkdir -p src/utils
```

### Step 5.2: Create API Service

**frontend/react-app/src/services/api.js:**
```javascript
import axios from 'axios';

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

export const uploadPDF = async (file) => {
  const formData = new FormData();
  formData.append('file', file);
  
  const response = await api.post('/upload', formData, {
    headers: {
      'Content-Type': 'multipart/form-data',
    },
  });
  
  return response.data;
};

export const checkIngestStatus = async (docId) => {
  const response = await api.get(`/ingest/status/${docId}`);
  return response.data;
};

export const query = async (queryText, k = 5) => {
  const response = await api.post('/query', {
    query: queryText,
    k: k,
  });
  return response.data;
};

export const getChunk = async (docId, chunkId) => {
  const response = await api.get(`/doc/${docId}/chunks/${chunkId}`);
  return response.data;
};

export const searchMetadata = async (query) => {
  const response = await api.get('/search-metadata', {
    params: { q: query },
  });
  return response.data;
};

export default api;
```

### Step 5.3: Create Upload Component

**frontend/react-app/src/components/UploadPDF.jsx:**
```javascript
import { useState } from 'react';
import { uploadPDF, checkIngestStatus } from '../services/api';

const UploadPDF = () => {
  const [file, setFile] = useState(null);
  const [uploading, setUploading] = useState(false);
  const [status, setStatus] = useState(null);
  const [docId, setDocId] = useState(null);

  const handleFileChange = (e) => {
    setFile(e.target.files[0]);
    setStatus(null);
  };

  const handleUpload = async () => {
    if (!file) {
      alert('Please select a file');
      return;
    }

    setUploading(true);
    setStatus('Uploading...');

    try {
      const result = await uploadPDF(file);
      setDocId(result.doc_id);
      setStatus('Processing...');

      // Poll for status
      const pollStatus = async () => {
        const statusResult = await checkIngestStatus(result.doc_id);
        setStatus(statusResult.status);

        if (statusResult.status === 'completed') {
          setUploading(false);
          alert('PDF processed successfully!');
        } else if (statusResult.status === 'failed') {
          setUploading(false);
          alert('Processing failed: ' + statusResult.error);
        } else {
          setTimeout(pollStatus, 2000);
        }
      };

      setTimeout(pollStatus, 1000);
    } catch (error) {
      setUploading(false);
      setStatus('Error: ' + error.message);
      alert('Upload failed: ' + error.message);
    }
  };

  return (
    <div className="upload-container">
      <h2>Upload PDF</h2>
      <div className="upload-form">
        <input
          type="file"
          accept=".pdf"
          onChange={handleFileChange}
          disabled={uploading}
        />
        <button onClick={handleUpload} disabled={uploading || !file}>
          {uploading ? 'Uploading...' : 'Upload PDF'}
        </button>
      </div>
      {status && (
        <div className="status">
          <p>Status: {status}</p>
        </div>
      )}
    </div>
  );
};

export default UploadPDF;
```

### Step 5.4: Create Query Component

**frontend/react-app/src/components/QueryInterface.jsx:**
```javascript
import { useState } from 'react';
import { query, getChunk } from '../services/api';

const QueryInterface = () => {
  const [queryText, setQueryText] = useState('');
  const [loading, setLoading] = useState(false);
  const [results, setResults] = useState(null);
  const [expandedChunks, setExpandedChunks] = useState(new Set());

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!queryText.trim()) return;

    setLoading(true);
    try {
      const response = await query(queryText);
      setResults(response);
    } catch (error) {
      alert('Query failed: ' + error.message);
    } finally {
      setLoading(false);
    }
  };

  const toggleChunk = async (docId, chunkId) => {
    const key = `${docId}-${chunkId}`;
    if (expandedChunks.has(key)) {
      const newSet = new Set(expandedChunks);
      newSet.delete(key);
      setExpandedChunks(newSet);
    } else {
      try {
        const chunkData = await getChunk(docId, chunkId);
        // Store full chunk data (you might want to use state for this)
        const newSet = new Set(expandedChunks);
        newSet.add(key);
        setExpandedChunks(newSet);
      } catch (error) {
        alert('Failed to load chunk: ' + error.message);
      }
    }
  };

  return (
    <div className="query-container">
      <h2>Ask a Question</h2>
      <form onSubmit={handleSubmit}>
        <textarea
          value={queryText}
          onChange={(e) => setQueryText(e.target.value)}
          placeholder="Enter your research question..."
          rows={4}
          style={{ width: '100%', padding: '10px' }}
        />
        <button type="submit" disabled={loading || !queryText.trim()}>
          {loading ? 'Searching...' : 'Search'}
        </button>
      </form>

      {results && (
        <div className="results">
          <div className="answer-section">
            <h3>Answer</h3>
            <div className="answer-text">{results.answer}</div>
          </div>

          <div className="sources-section">
            <h3>Sources</h3>
            {results.sources.map((source, idx) => (
              <div key={idx} className="source-card">
                <h4>
                  {source.doc_title} (p{source.page_range})
                  <span className="score">
                    Score: {source.similarity_score.toFixed(3)}
                  </span>
                </h4>
                <p className="snippet">{source.snippet}</p>
                <button
                  onClick={() => toggleChunk(source.doc_id, source.chunk_id)}
                >
                  {expandedChunks.has(`${source.doc_id}-${source.chunk_id}`)
                    ? 'Show Less'
                    : 'Show More'}
                </button>
              </div>
            ))}
          </div>

          {results.related_papers && results.related_papers.length > 0 && (
            <div className="related-papers-section">
              <h3>Related Papers</h3>
              {results.related_papers.map((paper, idx) => (
                <div key={idx} className="paper-card">
                  <h4>{paper.title}</h4>
                  <p>
                    {paper.authors} ({paper.year}) - Relevance:{' '}
                    {paper.relevance_score.toFixed(3)}
                  </p>
                  {paper.doi && <p>DOI: {paper.doi}</p>}
                </div>
              ))}
            </div>
          )}

          {results.gaps && results.gaps.length > 0 && (
            <div className="gaps-section">
              <h3>Research Gaps</h3>
              {results.gaps.map((gap, idx) => (
                <div key={idx} className="gap-card">
                  <h4>Gap {idx + 1}</h4>
                  <p>{gap.description}</p>
                  {gap.suggestions && gap.suggestions.length > 0 && (
                    <ul>
                      {gap.suggestions.map((suggestion, sIdx) => (
                        <li key={sIdx}>{suggestion}</li>
                      ))}
                    </ul>
                  )}
                </div>
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  );
};

export default QueryInterface;
```

### Step 5.5: Create Main App Component

**frontend/react-app/src/App.jsx:**
```javascript
import { useState } from 'react';
import UploadPDF from './components/UploadPDF';
import QueryInterface from './components/QueryInterface';
import './App.css';

function App() {
  const [activeTab, setActiveTab] = useState('query');

  return (
    <div className="App">
      <header>
        <h1>Academic Research Assistant</h1>
        <nav>
          <button
            onClick={() => setActiveTab('upload')}
            className={activeTab === 'upload' ? 'active' : ''}
          >
            Upload PDF
          </button>
          <button
            onClick={() => setActiveTab('query')}
            className={activeTab === 'query' ? 'active' : ''}
          >
            Query
          </button>
        </nav>
      </header>

      <main>
        {activeTab === 'upload' && <UploadPDF />}
        {activeTab === 'query' && <QueryInterface />}
      </main>
    </div>
  );
}

export default App;
```

### Step 5.6: Add Basic Styling

**frontend/react-app/src/App.css:**
```css
* {
  margin: 0;
  padding: 0;
  box-sizing: border-box;
}

body {
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', 'Oxygen',
    'Ubuntu', 'Cantarell', 'Fira Sans', 'Droid Sans', 'Helvetica Neue',
    sans-serif;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
}

.App {
  max-width: 1200px;
  margin: 0 auto;
  padding: 20px;
}

header {
  margin-bottom: 30px;
  border-bottom: 2px solid #eee;
  padding-bottom: 20px;
}

header h1 {
  margin-bottom: 20px;
}

nav {
  display: flex;
  gap: 10px;
}

nav button {
  padding: 10px 20px;
  border: 1px solid #ddd;
  background: white;
  cursor: pointer;
  border-radius: 4px;
}

nav button.active {
  background: #007bff;
  color: white;
  border-color: #007bff;
}

.upload-container,
.query-container {
  background: white;
  padding: 30px;
  border-radius: 8px;
  box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
}

.upload-form {
  display: flex;
  gap: 10px;
  margin-bottom: 20px;
}

.upload-form input[type="file"] {
  flex: 1;
  padding: 10px;
}

.upload-form button {
  padding: 10px 20px;
  background: #007bff;
  color: white;
  border: none;
  border-radius: 4px;
  cursor: pointer;
}

.upload-form button:disabled {
  background: #ccc;
  cursor: not-allowed;
}

.query-container form {
  margin-bottom: 30px;
}

.query-container textarea {
  margin-bottom: 10px;
}

.query-container button {
  padding: 10px 20px;
  background: #28a745;
  color: white;
  border: none;
  border-radius: 4px;
  cursor: pointer;
}

.results {
  margin-top: 30px;
}

.answer-section {
  background: #f8f9fa;
  padding: 20px;
  border-radius: 8px;
  margin-bottom: 30px;
}

.answer-text {
  line-height: 1.6;
  white-space: pre-wrap;
}

.sources-section,
.related-papers-section,
.gaps-section {
  margin-bottom: 30px;
}

.source-card,
.paper-card,
.gap-card {
  background: white;
  border: 1px solid #ddd;
  padding: 15px;
  margin-bottom: 15px;
  border-radius: 4px;
}

.source-card h4,
.paper-card h4 {
  margin-bottom: 10px;
  color: #333;
}

.score {
  float: right;
  font-size: 0.9em;
  color: #666;
}

.snippet {
  color: #666;
  margin: 10px 0;
}

button {
  margin-top: 10px;
  padding: 5px 15px;
  background: #17a2b8;
  color: white;
  border: none;
  border-radius: 4px;
  cursor: pointer;
}
```

---

## Phase 6: Testing & Deployment

### Step 6.1: Create FastAPI Main Application

**backend/app/main.py:**
```python
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

# Initialize database
init_db()

app = FastAPI(title="Academic Research Assistant API")

# CORS middleware
cors_origins = os.getenv("CORS_ORIGINS", "http://localhost:5173,http://localhost:3000").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Ensure upload directory exists
UPLOAD_DIR = Path(os.getenv("UPLOAD_DIR", "data/uploads"))
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

@app.on_event("startup")
async def startup_event():
    """Initialize on startup"""
    print("Starting Academic Research Assistant API...")
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
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="File must be a PDF")
    
    # Save file
    file_id = str(uuid.uuid4())
    file_path = UPLOAD_DIR / f"{file_id}.pdf"
    
    with open(file_path, "wb") as f:
        content = await file.read()
        f.write(content)
    
    try:
        # Ingest PDF
        result = ingest_pdf_with_embeddings(file_path, db)
        return {
            "doc_id": result["doc_id"],
            "status": "ingesting",
            "num_chunks": result["num_chunks"]
        }
    except Exception as e:
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
    query: dict,
    db: Session = Depends(get_db)
):
    """Query the RAG system"""
    query_text = query.get("query", "")
    k = query.get("k", 5)
    
    if not query_text:
        raise HTTPException(status_code=400, detail="Query text is required")
    
    try:
        result = full_rag_pipeline(query_text, db, k=k)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

### Step 6.2: Create Pydantic Models for Request/Response

**backend/app/schemas.py:**
```python
from pydantic import BaseModel
from typing import List, Optional, Dict, Any

class QueryRequest(BaseModel):
    query: str
    k: int = 5

class SourceResponse(BaseModel):
    chunk_id: int
    doc_id: str
    doc_title: str
    page_range: str
    snippet: str
    similarity_score: float

class RelatedPaperResponse(BaseModel):
    doc_id: str
    title: str
    authors: Optional[str]
    year: Optional[int]
    doi: Optional[str]
    relevance_score: float
    matching_chunks: int

class GapResponse(BaseModel):
    description: str
    suggestions: List[str]

class QueryResponse(BaseModel):
    answer: str
    sources: List[SourceResponse]
    related_papers: List[RelatedPaperResponse]
    gaps: List[GapResponse]

class UploadResponse(BaseModel):
    doc_id: str
    status: str
    num_chunks: int
```

### Step 6.3: Create Comprehensive Tests

**tests/test_rag.py:**
```python
import pytest
from backend.app.rag import retrieve_chunks, format_context_for_llm
from backend.app.database import SessionLocal, init_db
from backend.app.models import ChunkMetadata, Document

@pytest.fixture
def db():
    init_db()
    db = SessionLocal()
    yield db
    db.close()

def test_format_context_for_llm():
    """Test context formatting"""
    chunks = [
        {
            "doc_title": "Test Paper",
            "page_start": 1,
            "page_end": 2,
            "text": "This is test text."
        }
    ]
    context = format_context_for_llm(chunks)
    assert "Test Paper" in context
    assert "p1-2" in context
    assert "This is test text" in context

def test_retrieve_chunks(db):
    """Test chunk retrieval (requires populated database)"""
    # This test requires a populated index
    # query = "test query"
    # chunks = retrieve_chunks(query, db, k=3)
    # assert isinstance(chunks, list)
    pass
```

**tests/test_integration.py:**
```python
import pytest
from fastapi.testclient import TestClient
from backend.app.main import app
from backend.app.database import init_db, SessionLocal
from pathlib import Path
import tempfile

client = TestClient(app)

@pytest.fixture
def test_db():
    """Create test database"""
    init_db()
    db = SessionLocal()
    yield db
    db.close()

def test_root_endpoint():
    """Test root endpoint"""
    response = client.get("/")
    assert response.status_code == 200
    assert "message" in response.json()

def test_upload_pdf_invalid():
    """Test upload with invalid file"""
    response = client.post("/upload", files={"file": ("test.txt", b"not a pdf")})
    assert response.status_code == 400

def test_query_empty():
    """Test query with empty text"""
    response = client.post("/query", json={"query": "", "k": 5})
    assert response.status_code == 400
```

### Step 6.4: Create Docker Configuration

**Dockerfile:**
```dockerfile
FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY backend/requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY backend/ /app/

# Create data directories
RUN mkdir -p /app/data/uploads /app/data/indices /app/data/metadata

# Expose port
EXPOSE 8000

# Run application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

**docker-compose.yml:**
```yaml
version: '3.8'

services:
  backend:
    build: .
    ports:
      - "8000:8000"
    volumes:
      - ./data:/app/data
      - ./backend:/app
    environment:
      - LLM_PROVIDER=${LLM_PROVIDER:-openai}
      - OPENAI_API_KEY=${OPENAI_API_KEY:-}
      - EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
      - DB_PATH=/app/data/research_assistant.db
      - FAISS_INDEX_PATH=/app/data/indices/faiss_index.idx
      - CORS_ORIGINS=http://localhost:5173,http://localhost:3000
    restart: unless-stopped

  frontend:
    build:
      context: ./frontend/react-app
      dockerfile: Dockerfile
    ports:
      - "5173:5173"
    volumes:
      - ./frontend/react-app:/app
      - /app/node_modules
    environment:
      - VITE_API_URL=http://localhost:8000
    depends_on:
      - backend
```

**frontend/react-app/Dockerfile:**
```dockerfile
FROM node:18-alpine

WORKDIR /app

COPY package*.json ./
RUN npm install

COPY . .

EXPOSE 5173

CMD ["npm", "run", "dev", "--", "--host"]
```

### Step 6.5: Create Demo Notebook

**notebooks/demo.ipynb:**
```python
# This would be a Jupyter notebook - showing key cells:

# Cell 1: Setup
import sys
sys.path.append('../backend')
from app.database import init_db, SessionLocal
from app.ingest import ingest_pdf_with_embeddings
from app.rag import retrieve_chunks
from app.agents import full_rag_pipeline
from pathlib import Path

# Initialize
init_db()
db = SessionLocal()

# Cell 2: Ingest sample PDFs
pdf_paths = [
    Path("../data/sample_papers/paper1.pdf"),
    Path("../data/sample_papers/paper2.pdf"),
    Path("../data/sample_papers/paper3.pdf")
]

for pdf_path in pdf_paths:
    if pdf_path.exists():
        print(f"Ingesting {pdf_path.name}...")
        result = ingest_pdf_with_embeddings(pdf_path, db)
        print(f"  - Created {result['num_chunks']} chunks")
        print(f"  - Doc ID: {result['doc_id']}")

# Cell 3: Test retrieval
query = "What is the main contribution of this research?"
chunks = retrieve_chunks(query, db, k=5)
print(f"Retrieved {len(chunks)} chunks:")
for i, chunk in enumerate(chunks, 1):
    print(f"\n{i}. {chunk['doc_title']} (p{chunk['page_start']}-{chunk['page_end']})")
    print(f"   Score: {chunk['similarity_score']:.3f}")
    print(f"   Snippet: {chunk['text'][:200]}...")

# Cell 4: Full RAG pipeline
result = full_rag_pipeline(query, db, k=5)
print("Answer:")
print(result['answer'])
print("\nSources:")
for source in result['sources']:
    print(f"  - {source['doc_title']} (p{source['page_range']})")
print("\nRelated Papers:")
for paper in result['related_papers']:
    print(f"  - {paper['title']} ({paper.get('year', 'N/A')})")
print("\nResearch Gaps:")
for gap in result['gaps']:
    print(f"  - {gap['description']}")

db.close()
```

### Step 6.6: Create Environment Setup Script

**backend/setup.sh:**
```bash
#!/bin/bash

# Setup script for backend

echo "Setting up Academic Research Assistant Backend..."

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Create .env file if it doesn't exist
if [ ! -f .env ]; then
    cp .env.example .env
    echo "Created .env file. Please edit it with your configuration."
fi

# Create data directories
mkdir -p ../data/uploads
mkdir -p ../data/indices
mkdir -p ../data/metadata

# Initialize database
python -c "from app.database import init_db; init_db()"

echo "Setup complete!"
echo "To start the server, run: uvicorn app.main:app --reload"
```

**backend/setup.bat** (Windows):
```batch
@echo off
echo Setting up Academic Research Assistant Backend...

python -m venv venv
call venv\Scripts\activate

pip install --upgrade pip
pip install -r requirements.txt

if not exist .env (
    copy .env.example .env
    echo Created .env file. Please edit it with your configuration.
)

mkdir ..\data\uploads 2>nul
mkdir ..\data\indices 2>nul
mkdir ..\data\metadata 2>nul

python -c "from app.database import init_db; init_db()"

echo Setup complete!
echo To start the server, run: uvicorn app.main:app --reload
```

### Step 6.7: Update README with Complete Instructions

**README.md** (complete version):
```markdown
# Academic Research Assistant AI Agent

An intelligent research assistant that ingests academic PDFs, performs RAG (Retrieval-Augmented Generation) queries, and suggests related literature and research gaps.

## Features

- 📄 **PDF Ingestion**: Extract text and metadata from academic PDFs
- 🔍 **Semantic Search**: FAISS-based vector search for relevant content
- 💬 **RAG Queries**: Generate citation-aware answers using LLMs
- 📚 **Related Literature**: Suggest related papers based on queries
- 🔬 **Research Gaps**: Identify research gaps and suggest approaches
- 🎨 **Modern UI**: React-based frontend for easy interaction

## Architecture

- **Backend**: FastAPI with Python 3.10+
- **Vector DB**: FAISS for semantic search
- **Embeddings**: SentenceTransformers (all-MiniLM-L6-v2)
- **LLM**: Configurable (OpenAI, Ollama, or local transformers)
- **Frontend**: React with Vite
- **Database**: SQLite for metadata storage

## Quick Start

### Prerequisites

- Python 3.10+
- Node.js 18+
- (Optional) Docker and Docker Compose

### Backend Setup

```bash
cd backend

# Run setup script
# Linux/Mac:
bash setup.sh

# Windows:
setup.bat

# Or manually:
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
# Edit .env with your configuration

# Initialize database
python -c "from app.database import init_db; init_db()"

# Start server
uvicorn app.main:app --reload
```

### Frontend Setup

```bash
cd frontend/react-app
npm install

# Create .env file
echo "VITE_API_URL=http://localhost:8000" > .env

# Start dev server
npm run dev
```

### Using Docker

```bash
# Build and start all services
docker-compose up --build

# Backend will be at http://localhost:8000
# Frontend will be at http://localhost:5173
```

## Configuration

### Backend (.env)

```env
# LLM Configuration
LLM_PROVIDER=openai  # or "ollama", "local"
LLM_MODEL_NAME=gpt-3.5-turbo
OPENAI_API_KEY=your_key_here

# Embedding Model
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2

# Paths
DATA_DIR=../data
UPLOAD_DIR=../data/uploads
INDEX_DIR=../data/indices
METADATA_DIR=../data/metadata

# API Settings
API_HOST=0.0.0.0
API_PORT=8000
CORS_ORIGINS=http://localhost:5173,http://localhost:3000
```

### Frontend (.env)

```env
VITE_API_URL=http://localhost:8000
```

## Usage

### 1. Upload PDFs

- Navigate to the Upload tab in the frontend
- Select a PDF file
- Wait for processing to complete

### 2. Query the System

- Navigate to the Query tab
- Enter your research question
- View the answer with citations, related papers, and research gaps

### 3. Explore Sources

- Click "Show More" on any source to see the full chunk
- Review similarity scores to assess relevance

## API Endpoints

- `POST /upload` - Upload and ingest a PDF
- `GET /ingest/status/{doc_id}` - Check ingestion status
- `POST /query` - Query the RAG system
- `GET /doc/{doc_id}/chunks/{chunk_id}` - Get a specific chunk
- `GET /search-metadata?q=...` - Search document metadata

## Testing

```bash
# Run backend tests
cd backend
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=app --cov-report=html
```

## Demo Notebook

See `notebooks/demo.ipynb` for an end-to-end demonstration with sample PDFs.

## Development

### Code Formatting

```bash
# Format code
black backend/app

# Lint code
flake8 backend/app
```

### Project Structure

```
academic-research-assistant/
├── backend/
│   ├── app/
│   │   ├── main.py          # FastAPI application
│   │   ├── ingest.py        # PDF ingestion
│   │   ├── rag.py           # RAG retrieval
│   │   ├── embeddings.py    # Embedding service
│   │   ├── faiss_index.py   # FAISS index manager
│   │   ├── agents.py        # LLM agents
│   │   ├── models.py        # Database models
│   │   └── ...
│   ├── requirements.txt
│   └── .env
├── frontend/
│   └── react-app/
│       ├── src/
│       │   ├── components/
│       │   ├── services/
│       │   └── ...
│       └── package.json
├── data/
│   ├── uploads/             # Uploaded PDFs
│   ├── indices/             # FAISS indices
│   └── metadata/            # Metadata stores
├── notebooks/               # Demo notebooks
└── tests/                   # Test files
```

## Troubleshooting

### Common Issues

1. **FAISS index not found**: Run ingestion on at least one PDF first
2. **LLM errors**: Check your LLM_PROVIDER and API keys in .env
3. **CORS errors**: Ensure CORS_ORIGINS includes your frontend URL
4. **Memory issues**: Use smaller models or reduce chunk sizes

### Performance Tips

- Use GPU for embeddings if available
- Consider using Pinecone for production instead of FAISS
- Adjust chunk_size and overlap based on your documents
- Use smaller LLM models for faster inference

## License

MIT

## Contributing

Contributions welcome! Please open an issue or submit a pull request.

## Acknowledgments

- Built with FastAPI, LangChain, and React
- Uses SentenceTransformers for embeddings
- FAISS for vector search
```

### Step 6.8: Add Additional Utilities

**backend/app/utils.py:**
```python
from typing import List, Dict
import re
from pathlib import Path

def validate_pdf(file_path: Path) -> bool:
    """Validate that file is a PDF"""
    if not file_path.exists():
        return False
    if not file_path.suffix.lower() == '.pdf':
        return False
    # Could add more validation (check PDF header)
    return True

def sanitize_filename(filename: str) -> str:
    """Sanitize filename for storage"""
    # Remove path components
    filename = Path(filename).name
    # Remove dangerous characters
    filename = re.sub(r'[^\w\-_\.]', '_', filename)
    return filename

def format_citation(doc_title: str, page_start: int, page_end: int) -> str:
    """Format citation string"""
    if page_start == page_end:
        return f"[{doc_title}, p{page_start}]"
    return f"[{doc_title}, p{page_start}-{page_end}]"

def truncate_text(text: str, max_length: int = 200, suffix: str = "...") -> str:
    """Truncate text to max length"""
    if len(text) <= max_length:
        return text
    return text[:max_length - len(suffix)] + suffix
```

### Step 6.9: Add Rate Limiting and Security

**backend/app/middleware.py:**
```python
from fastapi import Request, HTTPException
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

limiter = Limiter(key_func=get_remote_address)

def setup_rate_limiting(app):
    """Setup rate limiting middleware"""
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
    return app
```

Update **backend/requirements.txt** to include:
```txt
slowapi==0.1.9
```

Update **backend/app/main.py** to use rate limiting:
```python
from .middleware import setup_rate_limiting

# After creating app
app = setup_rate_limiting(app)

# Add rate limits to endpoints
@app.post("/query")
@limiter.limit("10/minute")
async def query_endpoint(...):
    ...
```

### Step 6.10: Final Checklist and Next Steps

## Final Implementation Checklist

- [ ] **Phase 0**: Project setup complete
  - [ ] Directory structure created
  - [ ] Virtual environment configured
  - [ ] Dependencies installed
  - [ ] Linting/formatting setup

- [ ] **Phase 1**: PDF Ingestion
  - [ ] PDF extraction working
  - [ ] Chunking implemented
  - [ ] Metadata extraction working
  - [ ] Database models created
  - [ ] Unit tests passing

- [ ] **Phase 2**: Embeddings & FAISS
  - [ ] Embedding service working
  - [ ] FAISS index creation
  - [ ] Index persistence
  - [ ] Incremental updates working

- [ ] **Phase 3**: RAG & LLM
  - [ ] Retrieval function working
  - [ ] LLM integration configured
  - [ ] RAG chain implemented
  - [ ] Citation formatting working

- [ ] **Phase 4**: Multi-step Reasoning
  - [ ] Keyword extraction
  - [ ] Related literature search
  - [ ] Research gap analysis
  - [ ] Full pipeline integration

- [ ] **Phase 5**: Frontend
  - [ ] Upload component
  - [ ] Query interface
  - [ ] Results display
  - [ ] Source exploration
  - [ ] Styling complete

- [ ] **Phase 6**: Testing & Deployment
  - [ ] Unit tests written
  - [ ] Integration tests passing
  - [ ] Docker configuration
  - [ ] Documentation complete
  - [ ] Demo notebook working

## Next Steps After Implementation

1. **Performance Optimization**
   - Profile embedding generation
   - Optimize FAISS index type (IndexIVFFlat for large datasets)
   - Add caching for frequent queries
   - Implement async processing for ingestion

2. **Enhanced Features**
   - Add support for multiple file formats (DOCX, TXT)
   - Implement document summarization
   - Add export functionality (PDF reports, citations)
   - Implement user authentication and document sharing

3. **Production Readiness**
   - Set up monitoring (Prometheus, Grafana)
   - Add logging (structured logging with Python logging)
   - Implement backup strategies for database and indices
   - Set up CI/CD pipeline
   - Add health check endpoints

4. **Advanced RAG Techniques**
   - Implement re-ranking (using cross-encoders)
   - Add query expansion
   - Implement multi-query retrieval
   - Add conversation memory for follow-up questions

5. **Evaluation & Metrics**
   - Create evaluation dataset
   - Measure retrieval precision/recall
   - Evaluate answer quality (BLEU, ROUGE)
   - User feedback collection

## Troubleshooting Guide

### Issue: Embeddings are slow
**Solution**: 
- Use GPU if available: `torch.cuda.is_available()`
- Consider using smaller embedding models
- Batch process embeddings

### Issue: FAISS index too large
**Solution**:
- Use IndexIVFFlat for approximate search
- Implement index sharding
- Consider using Pinecone for cloud deployment

### Issue: LLM responses are poor
**Solution**:
- Tune prompt templates
- Increase context window (k parameter)
- Use better LLM models
- Add few-shot examples to prompts

### Issue: Memory errors during ingestion
**Solution**:
- Process PDFs one at a time
- Reduce chunk_size
- Use streaming for large files
- Increase system memory or use cloud instance

## Resources

- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [LangChain Documentation](https://python.langchain.com/)
- [FAISS Documentation](https://github.com/facebookresearch/faiss)
- [SentenceTransformers Documentation](https://www.sbert.net/)
- [React Documentation](https://react.dev/)

## Support

For issues and questions:
1. Check the troubleshooting section
2. Review the demo notebook
3. Check existing GitHub issues
4. Create a new issue with detailed error logs

---

**Congratulations!** You now have a complete implementation guide for building an Academic Research Assistant AI Agent. Follow each phase step-by-step, test thoroughly, and iterate based on your specific needs. 