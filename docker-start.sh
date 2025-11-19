#!/bin/bash

# Docker startup script

echo "Starting Academic Research Assistant with Docker Compose..."

# Check if .env file exists
if [ ! -f .env ]; then
    echo "Creating .env file from .env.example..."
    cp .env.example .env
    echo "Please edit .env file with your configuration before starting!"
    exit 1
fi

# Create data directories if they don't exist
mkdir -p data/uploads data/indices data/metadata

# Start services
docker-compose up --build -d

echo "Services are starting..."
echo "Backend will be available at: http://localhost:8001"
echo "Frontend will be available at: http://localhost:5173"
echo ""
echo "To view logs: docker-compose logs -f"
echo "To stop services: docker-compose down"

