@echo off
rem Pulls every model the system uses (about 33 GB in total). Ollama must be installed and running.
rem The 14B escalation model (last line) is optional; delete that line to skip it.
ollama pull qwen2.5:3b
ollama pull qwen2.5-coder:7b
ollama pull llama3.2:3b
ollama pull llama3.1:8b
ollama pull deepseek-r1:8b
ollama pull gemma2:9b
ollama pull qwen2.5-coder:14b
echo.
echo Done. Run: .venv\Scripts\python.exe setup_check.py
pause
