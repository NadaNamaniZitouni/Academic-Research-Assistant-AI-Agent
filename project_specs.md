Academic Research Assistant AI Agent (Large Project)
1.1 Purpose & Objectives

Purpose: Build an intelligent Research Assistant that ingests academic PDFs, indexes content, performs retrieval-augmented generation (RAG), produces citation-aware summaries, extracts key insights, and suggests related literature and research gaps.

Primary objectives

Reliable PDF ingestion and canonical chunking with metadata (title, authors, page ranges, DOI if available).

Local vector index (FAISS) for semantic retrieval; embeddings via open-source sentence-transformers.

RAG pipeline that returns concise, citation-tagged answers with supporting evidence (source + page).

Multi-step reasoning agent that proposes related literature and identifies gaps / research questions.

Simple React frontend for uploads, queries, and source exploration.

Self-contained, reproducible, and deployable as a FastAPI service.

Success metrics

Retrieval precision: relevant chunk in top-5 for 90% of evaluation queries (on demo set).

Citation accuracy: returned source identifiers match stored metadata.

Latency: average query latency < 1.5s for local CPU inference (depends on model).

Demonstration: working demo on 3 scholarly PDFs with QA, summaries, and related literature suggestions.

1.2 Target users

Graduate students & researchers who want quick literature summarization and suggestions.

Professors reviewing papers.

Non-expert users who need understandable summaries with citations.

1.3 High-level architecture

Frontend (React): Upload PDFs, ask questions, view answers and sources.

Backend (FastAPI): ingestion endpoints, query endpoints, agent orchestration endpoints.

Embeddings service (local): sentence-transformers embedding model.

Vector DB: FAISS (local) or Pinecone option (cloud).

LLM: OSS model (Llama/Mistral via transformers / local runtime) or cloud fallback.

Orchestration: LangChain chains (RAG chain, multi-step chain).

Persistence: index files (faiss.idx), metadata store (JSONL or SQLite), file storage.

1.4 Technology stack

Python 3.10+

FastAPI + Uvicorn

LangChain

FAISS (faiss-cpu)

SentenceTransformers (all-MiniLM-L6-v2 or similar)

Transformers + Hugging Face / local inference runtime (e.g., transformers, accelerate, or llama.cpp wrapper)

PyPDF2 / pdfminer.six for extraction

React (CRA or Vite) + axios

SQLite / JSONL for metadata store

Docker for deployment

1.5 Data model & schemas

Metadata record (JSON)

{
  "chunk_id": "<int>",
  "doc_id": "<str>",
  "doc_title": "<str>",
  "source_path": "<str>",
  "page_start": <int>,
  "page_end": <int>,
  "text": "<chunk_text>",
  "created_at": "<iso8601>"
}


Documents table (SQLite)

doc_id (PK), title, authors, year, doi, path

1.6 API contracts (endpoints)

POST /upload — multipart form upload a PDF → returns {"doc_id": "...", "status":"ingesting"}.

GET /ingest/status/{doc_id} — ingestion status.

POST /query — body { "query": "...", "k": 5 } → returns {"answer": "...", "sources": [{chunk_id, doc_id, page_range, snippet}], "related_papers": [...], "gaps": [...]}.

GET /doc/{doc_id}/chunks/{chunk_id} — returns chunk text and metadata.

GET /search-metadata?q=... — fuzzy metadata search for related literature.

1.7 Step-by-step implementation plan
Phase 0 — Project setup (1 day)

Initialize repo, venv, linting (black/flake8), pre-commit hooks.

Create directories: backend/app, frontend, data, notebooks, docs.

Add basic README skeleton and license.

Phase 1 — Ingest pipeline (2–3 days)

Implement PDF text extraction utils (PyPDF2/pypdf).

Implement canonicalization: clean text, remove headers/footers if obvious.

Implement token-aware chunking function (chunk size ~800–1,000 words tokens, overlap 200 tokens).

Save chunks to local store (JSONL) with metadata (doc_id, page ranges).

Unit tests: extraction on 3 public PDFs, verify chunk counts and metadata.

Phase 2 — Embeddings & FAISS index (2 days)

Integrate SentenceTransformers embedder.

Create FAISS index creation flow: embeddings -> index.add -> write_index.

Persist metadata and map FAISS ids -> chunk metadata.

Implement index rebuild and incremental add.

Phase 3 — RAG retrieval & LLM answering (3–4 days)

Implement retrieve(query, k) to return chunk IDs and similarity scores.

Build prompt template for citation-aware answers (include chunk context + source tags).

Integrate LLM wrapper:

Option A (local OSS): create a transformers pipeline using a small LLM (Mistral/ggml-backed).

Option B (cloud): OpenAI (if key).

Implement answer_with_rag(context_chunks, question) using LangChain LLMChain.

Add tests and tune prompt: ensure answer conciseness and include citations.

Phase 4 — Multi-step reasoning & related literature (2 days)

Implement keyword extractor from query/answer (keyphrases or nouns).

Use metadata index (title, abstract) to run semantic search for related papers using embeddings.

Implement a prompt to ask LLM for research gaps given the context and related paper abstracts.

Return gaps and related_papers in the API.

Phase 5 — Frontend (3 days)

Build React app:

Upload page (shows ingestion status)

Query page (input, results with citations, expand chunk text)

Related literature & gap suggestions UI

Integrate file preview and source snippets.

Phase 6 — Polishing, testing & demo (2–3 days)

Add unit tests and integration tests.

Add a demo notebook showing end-to-end flow.

Create Dockerfile for backend and simple docker-compose for local dev.

1.8 Testing & validation

Unit tests for chunking, embedding dimensions, retrieval.

End-to-end test: upload public PDF → query for a known fact → assert source returned contains the sentence.

Load test: simulate parallel queries with locust or wrk.

1.9 Security, ethics & legal notes

Respect copyright: use only CC-BY or user-provided PDFs.

Never expose raw uploaded files publicly.

Rate-limit queries to prevent model overload.

Sanitize inputs to avoid prompt injection vectors.

1.10 Deliverables / end result

FastAPI backend with /upload and /query endpoints.

FAISS index persisted under data/.

React frontend demonstrating upload + QA + source exploration.

README with setup and demo instructions and small demo dataset.

Notebook proving efficacy on 3 papers.