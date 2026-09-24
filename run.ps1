# Start SignMate (FastAPI)
$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ScriptDir

Write-Host "SignMate" -ForegroundColor Cyan

$PythonExe = Join-Path $ScriptDir "venv\Scripts\python.exe"

if (-not (Test-Path $PythonExe)) {
    Write-Host "`n[ERROR] Virtual environment not found at $PythonExe" -ForegroundColor Red
    Write-Host "Please run setup_env.bat to install dependencies.`n" -ForegroundColor Yellow
    exit 1
}

Write-Host "`nStarting server..." -ForegroundColor Green
Write-Host "URL: http://127.0.0.1:8000" -ForegroundColor Yellow
Write-Host "Press Ctrl+C to stop.`n" -ForegroundColor Gray

Start-Process "http://127.0.0.1:8000"
& $PythonExe -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
