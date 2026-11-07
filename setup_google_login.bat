@echo off
setlocal
title Set up Continue with Google
cd /d "%~dp0"
if not exist "venv\Scripts\python.exe" (
    echo Run run.bat once first so the environment exists.
    pause
    exit /b 1
)
"venv\Scripts\python.exe" setup_google_login.py
pause
