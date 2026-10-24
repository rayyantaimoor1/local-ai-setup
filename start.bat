@echo off
rem One-click launcher. Uses the project's own virtual environment (.venv).
cd /d "%~dp0"

if not exist ".venv\Scripts\activate.bat" (
    echo Virtual environment not found. Run setup.bat first.
    pause
    exit /b 1
)
call ".venv\Scripts\activate.bat"

rem Start Ollama if it is installed but not running.
curl -s -m 2 http://127.0.0.1:11434/api/tags >nul 2>&1
if errorlevel 1 (
    if exist "%LOCALAPPDATA%\Programs\Ollama\ollama.exe" (
        echo Starting Ollama...
        start "" /min "%LOCALAPPDATA%\Programs\Ollama\ollama.exe" serve
        timeout /t 4 /nobreak >nul
    ) else (
        echo Ollama is not installed. See README.md.
    )
)

rem Open the browser a few seconds after the server starts, not before.
start "" /b cmd /c "timeout /t 4 /nobreak >nul & start http://127.0.0.1:8420"

uvicorn main:app --host 127.0.0.1 --port 8420
pause
