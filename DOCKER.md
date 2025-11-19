# Docker Setup Guide

This guide explains how to run the Academic Research Assistant using Docker Compose.

## Quick Start

### 1. Create Environment File

Create a `.env` file in the root directory:

```env
LLM_PROVIDER=openai
OPENAI_API_KEY=your_openai_api_key_here
LLM_MODEL_NAME=gpt-3.5-turbo
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
CORS_ORIGINS=http://localhost:5173,http://localhost:3000
```

### 2. Start Services

**Windows:**
```bash
docker-start.bat
```

**Linux/Mac:**
```bash
chmod +x docker-start.sh
./docker-start.sh
```

**Or manually:**
```bash
docker-compose up --build
```

### 3. Access the Application

- **Backend API**: http://localhost:8001
- **Frontend UI**: http://localhost:5173
- **API Docs**: http://localhost:8001/docs

## Services

### Backend Service
- **Container**: `academic-research-backend`
- **Port**: 8001 (mapped from container port 8000)
- **Health Check**: Enabled
- **Auto-restart**: Yes

### Frontend Service
- **Container**: `academic-research-frontend`
- **Port**: 5173
- **Depends on**: Backend
- **Auto-restart**: Yes

## Docker Compose Commands

### Start Services
```bash
# Start in foreground (see logs)
docker-compose up

# Start in background
docker-compose up -d

# Rebuild and start
docker-compose up --build
```

### Stop Services
```bash
# Stop services
docker-compose down

# Stop and remove volumes (deletes data!)
docker-compose down -v
```

### View Logs
```bash
# All services
docker-compose logs -f

# Specific service
docker-compose logs -f backend
docker-compose logs -f frontend
```

### Execute Commands
```bash
# Run command in backend container
docker-compose exec backend python -c "from app.database import init_db; init_db()"

# Access backend shell
docker-compose exec backend bash

# Access frontend shell
docker-compose exec frontend sh
```

## Production Deployment

For production, use the production compose file:

```bash
docker-compose -f docker-compose.prod.yml up --build -d
```

This uses:
- Nginx for frontend (port 80)
- Optimized builds
- Resource limits
- Production-ready configuration

## Data Persistence

Data is stored in the `./data` directory which is mounted as a volume:
- `./data/uploads/` - Uploaded PDFs
- `./data/indices/` - FAISS indices
- `./data/metadata/` - Metadata files
- `./data/research_assistant.db` - SQLite database

**Important**: The `./data` directory is mounted as a volume, so data persists between container restarts.

## Troubleshooting

### Port Already in Use
If ports 8001 or 5173 are already in use, modify `docker-compose.yml`:
```yaml
ports:
  - "8002:8000"  # Change host port (backend)
  - "5174:5173"  # Change host port (frontend)
```

### Container Won't Start
1. Check logs: `docker-compose logs backend`
2. Verify .env file exists and has correct values
3. Ensure Docker has enough resources (memory/CPU)

### Database Issues
Initialize database manually:
```bash
docker-compose exec backend python -c "from app.database import init_db; init_db()"
```

### Rebuild Everything
```bash
docker-compose down -v
docker-compose build --no-cache
docker-compose up
```

### View Container Status
```bash
docker-compose ps
```

### Clean Up
```bash
# Remove containers and volumes
docker-compose down -v

# Remove images
docker-compose down --rmi all
```

## Network

All services are connected via a bridge network (`research-network`), allowing them to communicate using service names:
- Frontend can access backend at `http://backend:8000` (internal container port)
- Services can communicate internally

## Environment Variables

All environment variables can be set in:
1. `.env` file in root directory (recommended)
2. `docker-compose.yml` directly
3. System environment variables

Variables are passed to containers and can be accessed by the application.

## Resource Requirements

**Minimum:**
- 2 CPU cores
- 4GB RAM
- 10GB disk space

**Recommended:**
- 4 CPU cores
- 8GB RAM
- 20GB disk space

Adjust resources in `docker-compose.prod.yml` if needed.

