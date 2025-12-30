@echo off
REM Convenience script to run RAG evaluation with Python 3.11 venv
REM Uses default HuggingFace cache (C:\Users\jakob\.cache\huggingface)

echo ============================================================
echo RAG Evaluation - Baseline
echo Cache: Default HuggingFace location
echo ============================================================
echo.

venv311\Scripts\python.exe experiments\eval_rag_baseline.py %*
