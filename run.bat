@echo off
title SignMate
echo SignMate
echo.

cd /d "%~dp0"

if not exist "venv\Scripts\python.exe" (
    echo [ERROR] Virtual environment not found at .\venv
    echo Please run setup_env.bat to install dependencies.
    echo.
    pause
    exit /b 1
)

echo Starting at http://127.0.0.1:8000 ...
echo Opening browser...
echo Press Ctrl+C in this console to stop the server.
echo.

start "" "http://127.0.0.1:8000"
"venv\Scripts\python.exe" -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
pause
