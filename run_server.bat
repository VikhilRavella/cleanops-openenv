@echo off
REM ============================================================
REM  CleanOps OpenEnv — Start Server
REM  Run AFTER setup_env.bat
REM ============================================================
title CleanOps OpenEnv - Server

echo.
echo ============================================================
echo   CleanOps OpenEnv - Starting Server
echo ============================================================
echo.
echo  Server URL:  http://localhost:7860
echo  API Docs:    http://localhost:7860/docs
echo  Health:      http://localhost:7860/health
echo.
echo  Press Ctrl+C to stop
echo ============================================================
echo.

call .venv\Scripts\activate.bat
python -m uvicorn server.app:app --host 0.0.0.0 --port 7860 --reload
