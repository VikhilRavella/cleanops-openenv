@echo off
REM ============================================================
REM  CleanOps OpenEnv — Run Inference Script (Hackathon)
REM  This is the SUBMISSION script the validator will run.
REM
REM  Modes:
REM    Heuristic (no key):  just run this bat file
REM    LLM mode:            set HF_TOKEN, MODEL_NAME, API_BASE_URL below
REM ============================================================
title CleanOps OpenEnv - Inference

echo.
echo ============================================================
echo   CleanOps OpenEnv - Inference Script
echo ============================================================
echo.

REM ── OPTIONAL: Set these for LLM mode ─────────────────────────
REM  Uncomment and fill in your values:
REM
REM set API_BASE_URL=https://api-inference.huggingface.co/v1
REM set MODEL_NAME=Qwen/Qwen2.5-72B-Instruct
REM set HF_TOKEN=hf_YOUR_TOKEN_HERE

REM ── Show current mode ─────────────────────────────────────────
if defined HF_TOKEN (
    echo  Mode: LLM Agent
    echo  Model: %MODEL_NAME%
    echo  Endpoint: %API_BASE_URL%
) else (
    echo  Mode: Heuristic fallback  [no HF_TOKEN set]
)
echo.
echo ============================================================
echo.

call .venv\Scripts\activate.bat
python inference.py
echo.
pause
