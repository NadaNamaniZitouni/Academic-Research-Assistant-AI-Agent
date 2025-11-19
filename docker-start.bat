@echo off
echo Starting Academic Research Assistant with Docker Compose...

REM Check if .env file exists
if not exist .env (
    echo Creating .env file from .env.example...
    copy .env.example .env
    echo Please edit .env file with your configuration before starting!
    pause
    exit /b 1
)

REM Create data directories if they don't exist
if not exist data\uploads mkdir data\uploads
if not exist data\indices mkdir data\indices
if not exist data\metadata mkdir data\metadata

REM Start services
docker-compose up --build -d

echo.
echo Services are starting...
echo Backend will be available at: http://localhost:8001
echo Frontend will be available at: http://localhost:5173
echo.
echo To view logs: docker-compose logs -f
echo To stop services: docker-compose down
pause

