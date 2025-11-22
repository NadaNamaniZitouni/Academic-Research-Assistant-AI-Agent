# Academic Research Assistant AI Agent

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-green.svg)](https://fastapi.tiangolo.com/)
[![React](https://img.shields.io/badge/React-18+-61dafb.svg)](https://reactjs.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED.svg)](https://www.docker.com/)

An intelligent research assistant that ingests academic PDFs, performs RAG (Retrieval-Augmented Generation) queries, and suggests related literature and research gaps. Built with modern technologies for production-ready deployment.

## ✨ Features

### Core Functionality
- 📄 **PDF Ingestion**: Extract text and metadata from academic PDFs with automatic chunking
- 🔍 **Semantic Search**: FAISS-based vector search with hybrid retrieval (reranking + MMR diversity)
- 💬 **RAG Queries**: Generate citation-aware answers using multiple LLM providers
- 📚 **Related Literature**: Suggest related papers based on semantic similarity
- 🔬 **Research Gaps**: Identify research gaps and suggest methodological approaches

### Advanced Features
- 🔐 **User Authentication**: JWT-based authentication with tier-based access control
- 📊 **Analytics Dashboard**: Track queries, response times, and document usage
- 📤 **Export Capabilities**: Export answers as Markdown, Text, or BibTeX citations
- 🎯 **Multi-LLM Support**: OpenAI, Ollama (Mistral), SambaNova, Gemini
- 🚀 **Production Ready**: Docker containerization, rate limiting, error handling
- 🎨 **Modern UI**: Responsive React frontend with dark theme

## 🏗️ Architecture

### Tech Stack
- **Backend**: FastAPI with Python 3.10+
- **Vector DB**: FAISS for semantic search with embedding cache
- **Embeddings**: SentenceTransformers (all-MiniLM-L6-v2)
- **LLM**: Configurable providers (OpenAI, Ollama/Mistral, SambaNova, Gemini)
- **Frontend**: React 18+ with Vite
- **Database**: SQLite for metadata storage
- **Authentication**: JWT with bcrypt password hashing
- **Containerization**: Docker & Docker Compose

### System Architecture
```
┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│   React     │────▶│   FastAPI     │────▶│   FAISS     │
│  Frontend   │     │   Backend     │     │  Vector DB  │
└─────────────┘     └──────────────┘     └─────────────┘
                            │
                            ├──▶ SQLite (Metadata)
                            ├──▶ Embedding Cache
                            └──▶ LLM Providers
```

### Key Components
- **RAG Pipeline**: Hybrid retrieval with FAISS → Reranking → MMR diversity selection
- **Embedding Cache**: In-memory and disk-persisted cache for fast reranking
- **User Management**: Tier-based access control (Free, Starter, Pro, Team)
- **Analytics**: Query tracking, usage statistics, performance metrics

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
- **Backend** at http://localhost:8010
- **Frontend** at http://localhost:5173
- **Ollama** (if using Ollama provider) at http://localhost:11434

**Note**: Create a `.env` file in the root directory with your configuration:
```env
# LLM Configuration
LLM_PROVIDER=ollama  # or "openai", "sambanova", "gemini"
LLM_MODEL_NAME=mistral  # Model name for Ollama
OLLAMA_BASE_URL=http://ollama:11434

# For OpenAI
# OPENAI_API_KEY=your_key_here

# For SambaNova
# SAMBANOVA_API_KEY=your_key_here

# For Gemini
# GEMINI_API_KEY=your_key_here

# JWT Secret (change in production!)
JWT_SECRET_KEY=your-secret-key-change-in-production
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

## 📡 API Endpoints

### Authentication
- `POST /auth/register` - Register a new user
- `POST /auth/login` - Login and get JWT token
- `POST /auth/login-json` - Login using JSON (alternative)
- `GET /auth/me` - Get current user information
- `GET /auth/usage` - Get usage statistics

### Document Management
- `POST /upload` - Upload and ingest a PDF (requires auth)
- `GET /ingest/status/{doc_id}` - Check ingestion status
- `GET /doc/{doc_id}/chunks/{chunk_id}` - Get a specific chunk
- `GET /search-metadata?q=...` - Search document metadata

### Query & RAG
- `POST /query` - Query the RAG system (requires auth)
  ```json
  {
    "query": "Your research question",
    "k": 12,
    "doc_id": "optional-document-id"
  }
  ```

### Analytics
- `GET /analytics/queries?days=30` - Get query analytics
- `GET /analytics/stats` - Get user statistics

### Export
- `POST /export/bibtex` - Export documents as BibTeX (requires paid tier)
- `POST /export/markdown` - Export query result as Markdown (requires paid tier)
- `POST /export/text` - Export query result as plain text (requires paid tier)

### API Documentation
- **Swagger UI**: http://localhost:8010/docs
- **ReDoc**: http://localhost:8010/redoc

**Note**: Backend runs on port 8010. Most endpoints require authentication.

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

## 🧪 Testing

```bash
# Run backend tests
cd backend
pytest

# Run with coverage
pytest --cov=app --cov-report=html
```

## 📊 Usage Statistics

The system tracks:
- Total queries per user
- Monthly query usage
- Document count per user
- Response time metrics
- Most asked questions
- Document query frequency

Access analytics via the dashboard in the frontend or `/analytics/queries` endpoint.

## 🔒 Security Features

- JWT-based authentication
- Password hashing with bcrypt
- Rate limiting on API endpoints
- Tier-based access control
- User-specific document isolation
- Input validation and sanitization

## 🚀 Deployment

### Production Considerations
1. Set strong `JWT_SECRET_KEY` in environment variables
2. Use PostgreSQL instead of SQLite for production
3. Configure proper CORS origins
4. Set up SSL/TLS certificates
5. Use environment-specific configuration
6. Enable logging and monitoring
7. Set up backup strategies for database and indices

### Docker Production
```bash
docker-compose -f docker-compose.prod.yml up --build -d
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📝 License

MIT License - see LICENSE file for details

## 🙏 Acknowledgments

- Built with [FastAPI](https://fastapi.tiangolo.com/), [LangChain](https://www.langchain.com/), and [React](https://reactjs.org/)
- Uses [SentenceTransformers](https://www.sbert.net/) for embeddings
- [FAISS](https://github.com/facebookresearch/faiss) for vector search
- [Ollama](https://ollama.ai/) for local LLM inference

## 📧 Contact

For questions or support, please open an issue on GitHub.

---

**Made with ❤️ for researchers and academics**
