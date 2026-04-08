@echo off
REM ============================================================
REM  CleanOps OpenEnv — Windows Environment Setup Script
REM  Run this ONCE to create the virtual environment and install
REM  all dependencies.
REM
REM  Usage:
REM    Double-click setup_env.bat
REM    OR run in Command Prompt / PowerShell:
REM       setup_env.bat
REM ============================================================

title CleanOps OpenEnv - Environment Setup

echo.
echo ============================================================
echo   CleanOps OpenEnv - Environment Setup
echo ============================================================
echo.

REM ── Step 1: Check Python ─────────────────────────────────────
echo [1/5] Checking Python version...
python --version >nul 2>&1
if errorlevel 1 (
    echo  ERROR: Python not found. Please install Python 3.11+
    echo  Download: https://www.python.org/downloads/
    pause
    exit /b 1
)
python --version
echo  Python found OK
echo.

REM ── Step 2: Create virtual environment ───────────────────────
echo [2/5] Creating virtual environment (.venv)...
if exist .venv (
    echo  .venv already exists — skipping creation
) else (
    python -m venv .venv
    if errorlevel 1 (
        echo  ERROR: Failed to create virtual environment
        pause
        exit /b 1
    )
    echo  .venv created OK
)
echo.

REM ── Step 3: Activate virtual environment ─────────────────────
echo [3/5] Activating virtual environment...
call .venv\Scripts\activate.bat
if errorlevel 1 (
    echo  ERROR: Failed to activate .venv
    pause
    exit /b 1
)
echo  Activated OK
echo.

REM ── Step 4: Upgrade pip ──────────────────────────────────────
echo [4/5] Upgrading pip...
python -m pip install --upgrade pip --quiet
echo  pip upgraded OK
echo.

REM ── Step 5: Install requirements ─────────────────────────────
echo [5/5] Installing requirements...
pip install -r requirements.txt
if errorlevel 1 (
    echo  ERROR: Failed to install requirements
    pause
    exit /b 1
)
echo.
echo  All requirements installed OK
echo.

REM ── Done ─────────────────────────────────────────────────────
echo ============================================================
echo   Setup Complete!
echo ============================================================
echo.
echo  Next steps:
echo.
echo   1. START THE SERVER:
echo      run_server.bat
echo      OR: .venv\Scripts\activate ^& uvicorn server.app:app --host 0.0.0.0 --port 7860 --reload
echo.
echo   2. RUN BASELINE (no API key needed):
echo      run_baseline.bat
echo      OR: .venv\Scripts\activate ^& python baseline.py
echo.
echo   3. RUN INFERENCE SCRIPT:
echo      run_inference.bat
echo.
echo   4. OPEN API DOCS in browser:
echo      http://localhost:7860/docs
echo.
echo ============================================================
echo.
pause
