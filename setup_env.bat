@echo off
title SignMate - Environment Setup
echo ========================================================
echo              SIGNMATE - ENVIRONMENT SETUP
echo ========================================================
echo.

cd /d "%~dp0"

echo [1/3] Creating virtual environment (Python venv)...
python -m venv venv
if %errorlevel% neq 0 (
    echo [ERROR] Python not found on PATH. Please install Python 3.10+ and add it to PATH.
    pause
    exit /b 1
)

echo [2/3] Upgrading pip...
call venv\Scripts\activate.bat
python -m pip install --upgrade pip

echo [3/3] Installing dependencies from requirements.txt...
pip install -r requirements.txt

echo.
echo ========================================================
echo Setup complete! You can now run the app with run.bat.
echo ========================================================
pause
