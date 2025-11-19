@echo off
echo ========================================
echo Setting up Ollama and pulling Mistral model
echo ========================================
echo.

echo Waiting for Ollama to be ready...
timeout /t 10 /nobreak >nul

echo Pulling Mistral model (this may take a few minutes)...
docker exec academic-research-ollama ollama pull mistral

echo.
echo ========================================
echo Done! Mistral model should now be available.
echo ========================================
echo.
echo You can verify by running:
echo docker exec academic-research-ollama ollama list

