@echo off
echo Setting up Academic Research Assistant Backend...

python -m venv venv
call venv\Scripts\activate

pip install --upgrade pip
pip install -r requirements.txt

if not exist .env (
    copy env.example .env
    echo Created .env file. Please edit it with your configuration.
)

mkdir ..\data\uploads 2>nul
mkdir ..\data\indices 2>nul
mkdir ..\data\metadata 2>nul

python -c "from app.database import init_db; init_db()"

echo Setup complete!
echo To start the server, run: uvicorn app.main:app --reload

