# Academic Research Assistant - Complete Project Flow Documentation

## Overview
This document provides a comprehensive, n8n-style workflow diagram of the entire project, showing how every file interacts and the complete data flow from user upload to query response.

---

## 🏗️ Architecture Overview

```
┌─────────────────┐
│   Frontend      │  React + Vite (Port 5173)
│   (React App)   │
└────────┬────────┘
         │ HTTP/REST API
         ▼
┌─────────────────┐
│   Backend       │  FastAPI (Port 8010)
│   (Python)      │
└────────┬────────┘
         │
    ┌────┴────┐
    ▼         ▼
┌────────┐ ┌────────┐
│ Ollama │ │ FAISS  │
│ (LLM)  │ │ (Index)│
└────────┘ └────────┘
```

---

## 📋 Table of Contents

1. [Frontend Flow](#frontend-flow)
2. [Backend Upload Flow](#backend-upload-flow)
3. [Backend Query Flow](#backend-query-flow)
4. [File-by-File Breakdown](#file-by-file-breakdown)
5. [Data Flow Diagrams](#data-flow-diagrams)
6. [Key Integration Points](#key-integration-points)

---

## 🎨 Frontend Flow

### Component: `main.jsx`
**Location:** `frontend/react-app/src/main.jsx`
**Purpose:** React application entry point
**Flow:**
```
main.jsx
  └─> Renders <App /> component
  └─> Imports index.css (global styles)
```

### Component: `App.jsx`
**Location:** `frontend/react-app/src/App.jsx`
**Purpose:** Main application container, manages upload/query state
**Flow:**
```
App.jsx
  ├─> State: hasUploaded (tracks if document uploaded)
  ├─> Renders Header (title + subtitle)
  ├─> Renders UploadPDF component
  │   └─> onUploadComplete() callback → sets hasUploaded = true
  └─> Renders QueryInterface component
      └─> Shows info box when hasUploaded = true
```

**Key Functions:**
- `handleUploadComplete()`: Called when upload succeeds, updates state

### Component: `UploadPDF.jsx`
**Location:** `frontend/react-app/src/components/UploadPDF.jsx`
**Purpose:** Handles PDF file upload UI and logic
**Flow:**
```
User selects file
  └─> handleFileSelect()
      └─> Validates file (PDF, size, not empty)
      └─> Sets file state
      
User clicks "Upload"
  └─> handleUpload()
      └─> Calls api.uploadPDF(file, onProgress)
          └─> api.js: uploadPDF()
              └─> POST /upload (multipart/form-data)
                  └─> Backend: main.py /upload endpoint
      
Upload Success
  └─> onUploadComplete() callback
      └─> Updates App.jsx state
      └─> Shows success message
```

**Key Functions:**
- `handleFileSelect()`: Validates and stores selected file
- `handleUpload()`: Initiates upload via API
- `onUploadComplete()`: Notifies parent component

### Component: `QueryInterface.jsx`
**Location:** `frontend/react-app/src/components/QueryInterface.jsx`
**Purpose:** Handles query submission and result display
**Flow:**
```
User enters query
  └─> handleSubmit()
      └─> Calls api.query(queryText)
          └─> api.js: query()
              └─> POST /query (JSON)
                  └─> Backend: main.py /query endpoint
      
Response received
  └─> Displays:
      ├─> Answer (from LLM)
      ├─> Sources (chunks with citations)
      ├─> Related Papers
      └─> Research Gaps
```

**Key Functions:**
- `handleSubmit()`: Submits query to backend
- `toggleChunk()`: Expands/collapses chunk details
- `fetchChunkDetails()`: Gets full chunk text for display

### Service: `api.js`
**Location:** `frontend/react-app/src/services/api.js`
**Purpose:** Centralized API client for backend communication
**Flow:**
```
api.js (Axios instance)
  ├─> Base URL: http://localhost:8010
  ├─> Timeout: 15 minutes
  ├─> Interceptors:
  │   ├─> Request: Handles FormData (removes Content-Type)
  │   └─> Response: Logs responses
  │
  ├─> uploadPDF(file, onProgress)
  │   └─> POST /upload
  │       └─> FormData with file
  │       └─> onUploadProgress callback
  │
  └─> query(queryText)
      └─> POST /query
          └─> JSON: { query: string, k: number }
```

**Key Functions:**
- `uploadPDF()`: Handles file upload with progress tracking
- `query()`: Sends query request
- `getChunk()`: Fetches individual chunk details

---

## 📤 Backend Upload Flow

### Entry Point: `main.py` - `/upload` Endpoint
**Location:** `backend/app/main.py`
**Flow:**
```
POST /upload
  ├─> Receives: UploadFile (multipart/form-data)
  ├─> Validates: PDF file extension
  ├─> Generates: file_id (UUID)
  ├─> Saves file: UPLOAD_DIR / {file_id}.pdf
  └─> Calls: ingest_pdf_with_embeddings(file_path, db)
      └─> ingest.py: ingest_pdf_with_embeddings()
```

**Key Functions:**
- `upload_pdf()`: Main upload handler
- Returns: `{ doc_id, status, num_chunks }`

### Step 1: PDF Ingestion - `ingest.py`
**Location:** `backend/app/ingest.py`
**Flow:**
```
ingest_pdf_with_embeddings(pdf_path, db)
  │
  ├─> Step 1: Get FAISS Index
  │   └─> faiss_index.py: get_faiss_index()
  │       └─> Returns: FAISSIndex instance
  │       └─> start_chunk_id = faiss_idx.get_total()
  │
  ├─> Step 2: Extract & Chunk PDF
  │   └─> ingest_pdf(pdf_path, db, start_chunk_id)
  │       │
  │       ├─> pdf_extractor.py: extract_text_from_pdf()
  │       │   └─> Extracts: full_text, metadata (title, authors, year, DOI)
  │       │
  │       ├─> pdf_extractor.py: extract_text_by_page()
  │       │   └─> Returns: List of {page_num, text}
  │       │
  │       ├─> chunking.py: chunk_text_by_pages()
  │       │   └─> Chunks text: size=800, overlap=200
  │       │   └─> Returns: List of {text, page_start, page_end}
  │       │
  │       ├─> models.py: save_document_metadata()
  │       │   └─> Creates: Document record in database
  │       │   └─> Returns: doc_id (UUID)
  │       │
  │       └─> models.py: ChunkMetadata
  │           └─> Saves chunks to database
  │           └─> Each chunk: chunk_id, doc_id, text, pages, etc.
  │
  ├─> Step 3: Generate Embeddings
  │   └─> embeddings.py: get_embedding_service()
  │       └─> Returns: EmbeddingService instance
  │       └─> embedding_service.encode(chunk_texts)
  │           └─> Uses: sentence-transformers/all-MiniLM-L6-v2
  │           └─> Returns: numpy array (n_chunks × 384)
  │
  ├─> Step 4: Add to FAISS Index
  │   └─> faiss_index.py: faiss_idx.add_vectors(embeddings)
  │       └─> Normalizes vectors (L2)
  │       └─> Adds to FAISS IndexFlatIP
  │       └─> faiss_idx.save() → saves to disk
  │
  └─> Step 5: Add to Embedding Cache
      └─> embedding_cache.py: get_embedding_cache()
          └─> Returns: EmbeddingCache instance
          └─> cache.add_embeddings(chunk_ids, embeddings)
              └─> Stores in memory: {chunk_id: embedding}
              └─> Saves to disk: embeddings_cache.npy + mapping.json
```

**Key Functions:**
- `ingest_pdf_with_embeddings()`: Main ingestion orchestrator
- `ingest_pdf()`: Extracts, chunks, and saves metadata
- Returns: `{ doc_id, chunks, num_chunks, embeddings_added }`

### Supporting Files in Upload Flow:

#### `pdf_extractor.py`
**Purpose:** PDF text and metadata extraction
**Key Functions:**
- `extract_text_from_pdf()`: Full text + metadata
- `extract_text_by_page()`: Page-by-page text

#### `chunking.py`
**Purpose:** Text chunking with overlap
**Key Functions:**
- `chunk_text_by_pages()`: Creates chunks with page info
- `chunk_text()`: Generic text chunking

#### `models.py`
**Purpose:** Database models (SQLAlchemy)
**Key Classes:**
- `Document`: Document metadata table
- `ChunkMetadata`: Chunk storage table

#### `database.py`
**Purpose:** Database connection and session management
**Key Functions:**
- `init_db()`: Creates tables
- `get_db()`: FastAPI dependency for DB sessions

#### `embeddings.py`
**Purpose:** Embedding generation service
**Key Classes:**
- `EmbeddingService`: Wraps SentenceTransformer
- `get_embedding_service()`: Singleton instance

#### `faiss_index.py`
**Purpose:** FAISS vector index management
**Key Classes:**
- `FAISSIndex`: Manages FAISS index operations
- `get_faiss_index()`: Singleton instance

#### `embedding_cache.py`
**Purpose:** Embedding cache for fast reranking
**Key Classes:**
- `EmbeddingCache`: In-memory + disk cache
- `get_embedding_cache()`: Singleton instance

---

## 🔍 Backend Query Flow

### Entry Point: `main.py` - `/query` Endpoint
**Location:** `backend/app/main.py`
**Flow:**
```
POST /query
  ├─> Receives: QueryRequest { query: str, k: int, doc_id: Optional[str] }
  ├─> Validates: query text not empty
  └─> Calls: full_rag_pipeline(query_text, db, k=k, doc_id=doc_id)
      └─> agents.py: full_rag_pipeline()
```

**Key Functions:**
- `query_endpoint()`: Main query handler
- Returns: `{ answer, sources, related_papers, gaps }`

### Step 1: Retrieve Chunks - `rag.py`
**Location:** `backend/app/rag.py`
**Flow:**
```
retrieve_chunks(query, db, k, doc_id=None)
  │
  ├─> Step 1.1: Encode Query
  │   └─> embeddings.py: embedding_service.encode_query(query)
  │       └─> Returns: query_embedding (384-dim vector)
  │
  ├─> Step 1.2: FAISS Search
  │   └─> faiss_index.py: faiss_idx.search(query_embedding, initial_k=25)
  │       └─> Returns: (distances, chunk_ids)
  │
  ├─> Step 1.3: Load Chunk Metadata
  │   └─> database.py: db.query(ChunkMetadata)
  │       └─> Filters by: chunk_id in results
  │       └─> Filters by: doc_id (if specified) ⚠️ KEY FILTER
  │       └─> Returns: List of chunk dicts
  │
  ├─> Step 1.4: Rerank Chunks
  │   └─> rerank_chunks(query_embedding, chunks, cache)
  │       └─> embedding_cache.py: cache.get_embeddings(chunk_ids)
  │       └─> Computes: cosine similarity (query × chunks)
  │       └─> Sorts: by similarity score
  │
  └─> Step 1.5: MMR Diversity Selection
      └─> mmr_diversity_selection(query_embedding, chunks, cache)
          └─> Balances: relevance vs diversity
          └─> Ensures: doc diversity (if doc_id=None)
          └─> Returns: Top final_k chunks (default: 12)
```

**Key Functions:**
- `retrieve_chunks()`: Main retrieval orchestrator
- `rerank_chunks()`: Reranks using cached embeddings
- `mmr_diversity_selection()`: Selects diverse chunks
- `format_context_for_llm()`: Formats chunks for LLM prompt

### Step 2: Generate Answer - `agents.py`
**Location:** `backend/app/agents.py`
**Flow:**
```
answer_with_rag(chunks, question)
  │
  ├─> Format Context
  │   └─> rag.py: format_context_for_llm(chunks, max_chunk_length=None)
  │       └─> Creates: "Source 1 [Title, pX-Y]:\n{text}\n..."
  │
  ├─> Create Prompt
  │   └─> PromptTemplate with:
  │       ├─> Context: Formatted chunks
  │       └─> Question: User query
  │
  └─> Call LLM
      └─> llm_wrapper.py: get_llm_instance()
          └─> Returns: LLM instance (Ollama/OpenAI/etc.)
          └─> chain.run(context=context, question=question)
              └─> LLM generates answer with citations
```

**Key Functions:**
- `answer_with_rag()`: Generates answer using RAG
- `get_llm_instance()`: Gets LLM singleton

### Step 3: Find Related Papers - `related_literature.py`
**Location:** `backend/app/related_literature.py`
**Flow:**
```
find_related_papers(query_text, answer_text, db)
  ├─> Combines: query + answer text
  ├─> keywords.py: extract_keywords() → top 15 keywords
  ├─> embeddings.py: encode_query(keywords)
  ├─> faiss_index.py: search() → top_k * 3 chunks
  ├─> Aggregates: by doc_id
  └─> Returns: Top 5 related papers
```

### Step 4: Identify Gaps - `gap_analysis.py`
**Location:** `backend/app/gap_analysis.py`
**Flow:**
```
identify_research_gaps(answer, related_papers, question)
  ├─> Formats: related papers text
  ├─> Creates: PromptTemplate
  ├─> Calls: LLM with gap analysis prompt
  └─> Parses: Gap descriptions and suggestions
```

### Complete Query Pipeline - `agents.py`
**Location:** `backend/app/agents.py`
**Flow:**
```
full_rag_pipeline(question, db, k, doc_id)
  ├─> Step 1: Retrieve Chunks
  │   └─> rag.py: retrieve_chunks(question, db, k, doc_id)
  │
  ├─> Step 2: Generate Answer
  │   └─> answer_with_rag(chunks, question)
  │
  ├─> Step 3: Find Related Papers
  │   └─> related_literature.py: find_related_papers()
  │
  ├─> Step 4: Identify Gaps
  │   └─> gap_analysis.py: identify_research_gaps()
  │
  └─> Returns: {
        answer: str,
        sources: List[SourceResponse],
        related_papers: List[RelatedPaperResponse],
        gaps: List[GapResponse]
      }
```

---

## 📁 File-by-File Breakdown

### Frontend Files

| File | Purpose | Key Exports/Functions |
|------|---------|----------------------|
| `main.jsx` | React entry point | Renders `<App />` |
| `App.jsx` | Main container | Manages upload/query state |
| `App.css` | App styles | Component styling |
| `index.css` | Global styles | Base styles, dark theme |
| `UploadPDF.jsx` | Upload component | File selection, upload logic |
| `QueryInterface.jsx` | Query component | Query submission, results display |
| `api.js` | API client | Axios instance, API functions |
| `vite.config.js` | Vite config | Dev server, proxy, HMR |

### Backend Files

| File | Purpose | Key Functions/Classes |
|------|---------|----------------------|
| `main.py` | FastAPI app | `/upload`, `/query` endpoints |
| `models.py` | Database models | `Document`, `ChunkMetadata` |
| `database.py` | DB connection | `init_db()`, `get_db()` |
| `schemas.py` | Pydantic schemas | `QueryRequest`, `QueryResponse` |
| `ingest.py` | PDF ingestion | `ingest_pdf_with_embeddings()` |
| `pdf_extractor.py` | PDF extraction | `extract_text_from_pdf()` |
| `chunking.py` | Text chunking | `chunk_text_by_pages()` |
| `embeddings.py` | Embedding service | `EmbeddingService`, `get_embedding_service()` |
| `faiss_index.py` | FAISS index | `FAISSIndex`, `get_faiss_index()` |
| `embedding_cache.py` | Embedding cache | `EmbeddingCache`, `get_embedding_cache()` |
| `rag.py` | RAG retrieval | `retrieve_chunks()`, `rerank_chunks()`, `mmr_diversity_selection()` |
| `agents.py` | RAG pipeline | `full_rag_pipeline()`, `answer_with_rag()` |
| `llm_wrapper.py` | LLM management | `get_llm()`, LLM provider setup |
| `related_literature.py` | Related papers | `find_related_papers()` |
| `gap_analysis.py` | Gap analysis | `identify_research_gaps()` |
| `keywords.py` | Keyword extraction | `extract_keywords()` |
| `index_manager.py` | Index management | `rebuild_index_from_database()` |
| `middleware.py` | Middleware | `setup_rate_limiting()` |
| `utils.py` | Utilities | Helper functions |

---

## 🔄 Data Flow Diagrams

### Upload Flow (Complete)
```
User (Browser)
  │
  ▼
UploadPDF.jsx
  │ handleUpload()
  ▼
api.js: uploadPDF()
  │ POST /upload (FormData)
  ▼
main.py: /upload endpoint
  │ Validates PDF
  │ Saves file to disk
  ▼
ingest.py: ingest_pdf_with_embeddings()
  │
  ├─> ingest_pdf()
  │   ├─> pdf_extractor.py: extract_text_from_pdf()
  │   ├─> pdf_extractor.py: extract_text_by_page()
  │   ├─> chunking.py: chunk_text_by_pages()
  │   ├─> models.py: save_document_metadata()
  │   └─> models.py: ChunkMetadata (save chunks)
  │
  ├─> embeddings.py: encode()
  │   └─> SentenceTransformer model
  │
  ├─> faiss_index.py: add_vectors()
  │   └─> FAISS IndexFlatIP
  │
  └─> embedding_cache.py: add_embeddings()
      └─> Memory + Disk cache
  │
  ▼
Response: { doc_id, status, num_chunks }
  │
  ▼
UploadPDF.jsx: onUploadComplete()
  │
  ▼
App.jsx: hasUploaded = true
```

### Query Flow (Complete)
```
User (Browser)
  │
  ▼
QueryInterface.jsx
  │ handleSubmit()
  ▼
api.js: query()
  │ POST /query (JSON)
  ▼
main.py: /query endpoint
  │ Validates request
  ▼
agents.py: full_rag_pipeline()
  │
  ├─> Step 1: Retrieve Chunks
  │   └─> rag.py: retrieve_chunks()
  │       ├─> embeddings.py: encode_query()
  │       ├─> faiss_index.py: search()
  │       ├─> database.py: query(ChunkMetadata)
  │       │   └─> ⚠️ FILTER: doc_id (if specified)
  │       ├─> rag.py: rerank_chunks()
  │       │   └─> embedding_cache.py: get_embeddings()
  │       └─> rag.py: mmr_diversity_selection()
  │
  ├─> Step 2: Generate Answer
  │   └─> agents.py: answer_with_rag()
  │       ├─> rag.py: format_context_for_llm()
  │       └─> llm_wrapper.py: get_llm_instance()
  │           └─> LLM (Ollama/OpenAI)
  │
  ├─> Step 3: Related Papers
  │   └─> related_literature.py: find_related_papers()
  │
  └─> Step 4: Research Gaps
      └─> gap_analysis.py: identify_research_gaps()
  │
  ▼
Response: { answer, sources, related_papers, gaps }
  │
  ▼
QueryInterface.jsx: Display results
```

---

## 🔗 Key Integration Points

### 1. Document ID Filtering (⚠️ CRITICAL)
**Location:** `backend/app/rag.py` - `retrieve_chunks()`
**Issue:** When `doc_id` is provided, chunks are filtered:
```python
if doc_id is not None and chunk_meta.doc_id != doc_id:
    continue  # Skip chunks from other documents
```

**Problem:** This filter happens AFTER FAISS search, so:
- FAISS returns top 25 chunks (may include other documents)
- Filter removes chunks from other documents
- If top results are from wrong document, query fails

**Solution Needed:** Filter at FAISS level OR increase initial_k when doc_id specified

### 2. Embedding Cache Sync
**Location:** `backend/app/embedding_cache.py`
**Flow:**
- New document uploaded → embeddings added to cache
- Cache stores: `{chunk_id: embedding}`
- Query uses cache for reranking

**Potential Issue:** If cache not synced, reranking uses wrong embeddings

### 3. Chunk ID Assignment
**Location:** `backend/app/ingest.py`
**Flow:**
```python
start_chunk_id = faiss_idx.get_total()  # Sequential IDs
chunk_id = start_chunk_id + idx
```

**Issue:** Chunk IDs are sequential across all documents, not per-document

### 4. FAISS Index
**Location:** `backend/app/faiss_index.py`
**Flow:**
- All document embeddings in single FAISS index
- Search returns top-k by similarity (across all docs)
- No built-in document filtering

**Issue:** FAISS doesn't know about doc_id, only chunk_id

---

## 🐛 Potential Issues & Debugging Points

### Issue: Query Returns Wrong Document

**Check Points:**
1. **FAISS Search** (`rag.py:193`)
   - Does FAISS return chunks from correct document?
   - Check: `chunk_ids` from FAISS search

2. **Document Filter** (`rag.py:208`)
   - Is `doc_id` being passed correctly?
   - Check: `doc_id` parameter in `retrieve_chunks()`

3. **Chunk Metadata** (`rag.py:202`)
   - Are chunks loaded with correct `doc_id`?
   - Check: `chunk_meta.doc_id` matches expected

4. **Cache Embeddings** (`rag.py:229`)
   - Are cached embeddings for correct chunks?
   - Check: `cache.get_embeddings(chunk_ids)`

5. **MMR Selection** (`rag.py:234`)
   - Does MMR preserve document filter?
   - Check: Selected chunks still have correct `doc_id`

### Recommended Fix:
Increase `initial_k` when `doc_id` is specified to ensure enough chunks from target document:
```python
if doc_id is not None:
    initial_k = min(initial_k * 3, total_vectors)  # Get more candidates
```

---

## 📊 Data Storage

### Database (SQLite)
**Location:** `data/research_assistant.db`
**Tables:**
- `documents`: Document metadata
- `chunk_metadata`: Chunk text and metadata

### FAISS Index
**Location:** `data/indices/faiss_index.idx`
**Content:** Vector embeddings (384-dim) for all chunks

### Embedding Cache
**Location:** `data/indices/embeddings_cache.npy` + `embeddings_cache.json`
**Content:** Embeddings + chunk_id mapping

### Uploaded Files
**Location:** `data/uploads/{file_id}.pdf`
**Content:** Original PDF files

---

## 🔧 Configuration

### Environment Variables
**Location:** `.env` (root) or `docker-compose.yml`
**Key Variables:**
- `LLM_PROVIDER`: ollama/openai/sambanova/gemini
- `LLM_MODEL_NAME`: mistral/gpt-3.5-turbo/etc.
- `OLLAMA_BASE_URL`: http://ollama:11434
- `EMBEDDING_MODEL`: sentence-transformers/all-MiniLM-L6-v2
- `DB_PATH`: Database file path
- `FAISS_INDEX_PATH`: FAISS index file path
- `UPLOAD_DIR`: Upload directory path

---

## 🚀 Docker Services

### Services (docker-compose.yml)
1. **ollama**: LLM service (port 11434)
2. **backend**: FastAPI app (port 8010)
3. **frontend**: React app (port 5173)

### Network
- All services on `research-network`
- Backend connects to Ollama via service name

---

## 📝 Summary

This document provides a complete workflow view of the Academic Research Assistant project. The key insight is that **document filtering happens AFTER FAISS search**, which may cause queries to return chunks from wrong documents if the top FAISS results are from other documents.

**Recommended Fix:** Increase `initial_k` when `doc_id` is specified, or implement document-aware FAISS filtering.

