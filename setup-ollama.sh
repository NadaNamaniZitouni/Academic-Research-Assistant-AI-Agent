#!/bin/bash

echo "========================================"
echo "Setting up Ollama and pulling Mistral model"
echo "========================================"
echo ""

echo "Waiting for Ollama to be ready..."
sleep 10

echo "Pulling Mistral model (this may take a few minutes)..."
docker exec academic-research-ollama ollama pull mistral

echo ""
echo "========================================"
echo "Done! Mistral model should now be available."
echo "========================================"
echo ""
echo "You can verify by running:"
echo "docker exec academic-research-ollama ollama list"

