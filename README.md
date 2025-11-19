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

### Docker Setup (Recommended)

The easiest way to run everything together:

```bash
# Windows
docker-start.bat

# Linux/Mac
chmod +x docker-start.sh
./docker-start.sh

# Or manually:
docker-compose up --build
```

This will start:
- **Backend** at http://localhost:8001
- **Frontend** at http://localhost:5173

**Note**: Create a `.env` file in the root directory with your configuration:
```env
LLM_PROVIDER=openai
OPENAI_API_KEY=your_key_here
```

To stop all services:
```bash
docker-compose down
```

To view logs:
```bash
docker-compose logs -f
```

### Manual Setup

### Backend Setup

```bash
cd backend

# Run setup script (Windows)
setup.bat

# Or manually:
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac
pip install -r requirements.txt
copy env.example .env
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
echo "VITE_API_URL=http://localhost:8001" > .env

# Start dev server
npm run dev
```

## Configuration

### Backend (.env)

Copy `backend/env.example` to `backend/.env` and configure:

```env
# LLM Configuration
LLM_PROVIDER=openai  # or "ollama", "local"
LLM_MODEL_NAME=gpt-3.5-turbo
OPENAI_API_KEY=your_key_here

# Embedding Model
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2

# API Settings
CORS_ORIGINS=http://localhost:5173,http://localhost:3000
```

### Frontend (.env)

```env
VITE_API_URL=http://localhost:8001
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

**Note**: Backend runs on port 8001 (not 8000) to avoid conflicts.

## Project Structure

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
│   └── metadata/             # Metadata stores
└── README.md
```

## Development

### Code Formatting

```bash
# Format code
black backend/app

# Lint code
flake8 backend/app
```

## Docker Commands

### Development Mode
```bash
# Start all services
docker-compose up

# Start in background
docker-compose up -d

# Rebuild and start
docker-compose up --build

# View logs
docker-compose logs -f

# Stop services
docker-compose down

# Stop and remove volumes
docker-compose down -v
```

### Production Mode
```bash
docker-compose -f docker-compose.prod.yml up --build -d
```

## Troubleshooting

### Common Issues

1. **FAISS index not found**: Run ingestion on at least one PDF first
2. **LLM errors**: Check your LLM_PROVIDER and API keys in .env
3. **CORS errors**: Ensure CORS_ORIGINS includes your frontend URL
4. **Memory issues**: Use smaller models or reduce chunk sizes
5. **Docker port conflicts**: Change ports in docker-compose.yml if 8001 or 5173 are in use
6. **Container won't start**: Check logs with `docker-compose logs backend` or `docker-compose logs frontend`

### Performance Tips

- Use GPU for embeddings if available
- Consider using Pinecone for production instead of FAISS
- Adjust chunk_size and overlap based on your documents
- Use smaller LLM models for faster inference

## License

MIT

## Acknowledgments

- Built with FastAPI, LangChain, and React
- Uses SentenceTransformers for embeddings
- FAISS for vector search
