@echo off
title Orient-Anything Server (Port 8004)

echo ========================================
echo   Orient-Anything Server - Port 8004
echo ========================================
echo.

python --version > nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python not found. Please install Python 3.8+
    pause
    exit /b 1
)

echo [INFO] Installing dependencies...
pip install fastapi "uvicorn[standard]" python-multipart --quiet
pip install -r requirements.txt --quiet

echo [INFO] Dependencies OK
echo.
echo [INFO] Server starting at http://localhost:8004
echo [INFO] Open test.html in your browser to test
echo [INFO] Press Ctrl+C to stop
echo.

python server.py

pause
