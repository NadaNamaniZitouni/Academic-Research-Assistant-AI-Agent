FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY backend/requirements.txt .

# Install Python dependencies with increased timeout
# First install PyTorch CPU-only to avoid downloading large CUDA packages
RUN pip install --upgrade pip setuptools wheel && \
    pip install --no-cache-dir --default-timeout=1000 \
        --index-url https://download.pytorch.org/whl/cpu \
        torch==2.1.0+cpu torchvision==0.16.0+cpu torchaudio==2.1.0+cpu && \
    pip install --no-cache-dir --default-timeout=1000 -r requirements.txt

# Copy application code
COPY backend/ /app/

# Create data directories
RUN mkdir -p /app/data/uploads /app/data/indices /app/data/metadata

# Expose port
EXPOSE 8000

# Run application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

