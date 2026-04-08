@echo off
REM ============================================================
REM  CleanOps OpenEnv — Run Baseline Agent
REM  No API key needed — uses heuristic agent
REM  Run AFTER setup_env.bat
REM ============================================================
title CleanOps OpenEnv - Baseline

echo.
echo ============================================================
echo   CleanOps OpenEnv - Baseline Evaluation
echo   (No API key needed)
echo ============================================================
echo.

call .venv\Scripts\activate.bat
python baseline.py
echo.
pause
