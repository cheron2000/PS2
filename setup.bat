@echo off
REM setup.bat — Windows entry point. Checks Python exists, then hands off to
REM the real cross-platform logic in setup.py. See SETUP_GUIDE.md for details.

where python >nul 2>nul
if %errorlevel% neq 0 (
    echo Python 3.9+ not found.
    echo   Download it from https://www.python.org/downloads/
    echo   IMPORTANT: tick "Add Python to PATH" during install, then reopen this terminal.
    exit /b 1
)

if not exist "%~dp0setup.py" (
    curl.exe -sO https://raw.githubusercontent.com/cheron2000/PS2/main/setup.py
    python setup.py %*
    exit /b %errorlevel%
)

python "%~dp0setup.py" %*
