@echo off
rem One-time setup: create .venv and install dependencies.
cd /d "%~dp0"

if not exist ".venv\Scripts\activate.bat" (
    python -m venv .venv
    if errorlevel 1 (
        echo Could not create the virtual environment. Is Python installed?
        pause
        exit /b 1
    )
)
call ".venv\Scripts\activate.bat"
python -m pip install --upgrade pip
pip install -r requirements.txt
if errorlevel 1 (
    echo Install failed. Check your connection and run setup.bat again.
    pause
    exit /b 1
)
echo.
python setup_check.py
pause
