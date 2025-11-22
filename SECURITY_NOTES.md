# Security & Privacy Notes

## Files Protected by .gitignore

This document outlines what sensitive data is excluded from version control.

### 🔒 Sensitive Data (Never Committed)

1. **Environment Variables & Secrets**
   - `.env` files (API keys, JWT secrets, database credentials)
   - `*.key`, `*.pem`, `*.cert` files
   - `secrets/` and `credentials/` directories

2. **User Data**
   - All uploaded PDFs (`data/uploads/*`)
   - User documents and files
   - Database files (`*.db`, `*.sqlite`, `*.sqlite3`)
   - Database WAL/SHM files

3. **Embeddings & Vector Data**
   - FAISS indices (`*.idx`, `data/indices/*`)
   - Embedding cache files (`*.npy`, `*.pkl`, `*.pickle`)
   - Embedding metadata JSON files

4. **Logs & Cache**
   - All log files (`*.log`)
   - Cache directories
   - Temporary files

5. **Build Artifacts**
   - Compiled Python files (`__pycache__/`, `*.pyc`)
   - Frontend build outputs (`dist/`, `build/`)
   - Node modules

6. **Internal Documentation**
   - Implementation guides
   - Monetization strategy documents
   - Internal project specs

### ✅ Safe to Commit

- Source code (`.py`, `.jsx`, `.js` files)
- Configuration templates (`.env.example`)
- Public documentation (`README.md`, `LICENSE`)
- Docker configuration files
- CI/CD workflows
- Test files
- Package manifests (`package.json`, `requirements.txt`)

## Best Practices

1. **Never commit:**
   - Real API keys or secrets
   - User-uploaded content
   - Database files with real data
   - Embedding caches

2. **Always use:**
   - `.env.example` for configuration templates
   - Environment variables for secrets
   - `.gitkeep` files to preserve directory structure

3. **Before pushing:**
   - Run `git status` to review what will be committed
   - Check for any `.env` files
   - Verify no user data is included
   - Ensure no API keys are hardcoded

## Data Directory Structure

```
data/
├── .gitkeep                    # ✅ Committed (empty file)
├── research_assistant.db       # ❌ Ignored (user data)
├── indices/
│   ├── .gitkeep               # ✅ Committed (empty file)
│   ├── faiss_index.idx        # ❌ Ignored (vector data)
│   └── embeddings_cache.*     # ❌ Ignored (embeddings)
└── uploads/
    ├── .gitkeep               # ✅ Committed (empty file)
    └── *.pdf                  # ❌ Ignored (user documents)
```

