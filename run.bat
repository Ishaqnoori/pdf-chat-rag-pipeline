@echo off
setlocal
title Ask your PDF
REM Always run from this file's folder (the project path has spaces)
cd /d "%~dp0"

REM --- 1. Check Python ---
where python >nul 2>nul
if errorlevel 1 (
    echo Python was not found. Install Python 3.12 or 3.13 from https://www.python.org/downloads/
    echo and tick "Add python.exe to PATH" during setup, then run this file again.
    pause
    exit /b 1
)
for /f "delims=" %%v in ('python --version 2^>^&1') do echo Using %%v

REM --- 2. Rebuild the venv if it points to a Python that no longer exists ---
if exist "venv\Scripts\python.exe" (
    "venv\Scripts\python.exe" -c "import sys" >nul 2>nul
    if errorlevel 1 (
        echo The existing virtual environment is broken ^(its Python was moved or uninstalled^). Rebuilding it...
        rmdir /s /q "venv"
    )
)

REM --- 3. Create the virtual environment if needed ---
if not exist "venv\Scripts\python.exe" (
    echo Creating virtual environment...
    python -m venv venv
    if errorlevel 1 (
        echo Could not create the virtual environment.
        pause
        exit /b 1
    )
)

REM --- 4. Install packages only when requirements.txt has changed ---
fc /b "requirements.txt" "venv\.installed_requirements.txt" >nul 2>nul
if errorlevel 1 (
    echo Installing packages. The first time takes a few minutes...
    "venv\Scripts\python.exe" -m pip install --upgrade pip --quiet
    "venv\Scripts\python.exe" -m pip install -r requirements.txt
    if errorlevel 1 (
        echo Package installation failed. Scroll up to see the error.
        pause
        exit /b 1
    )
    copy /y "requirements.txt" "venv\.installed_requirements.txt" >nul
)

REM --- 5. Make sure API keys exist ---
if not exist ".env" if not exist ".streamlit\secrets.toml" (
    copy ".env.example" ".env" >nul
    echo.
    echo No API keys found. A .env file has been created and will open in Notepad.
    echo Paste your GOOGLE_API_KEY and GROQ_API_KEY, save, close Notepad, then press any key.
    notepad ".env"
    pause
)

REM --- 6. Start the app (opens in your browser) ---
echo.
echo Starting the app at http://localhost:8501  -  close this window to stop it.
"venv\Scripts\python.exe" -m streamlit run app.py
pause
