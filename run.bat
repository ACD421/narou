@echo off
cd /d "%~dp0"
echo ============================================
echo   Narou - AI Job Search Assistant
echo   Starting...
echo ============================================
echo.

:: Check for venv
if not exist .venv\Scripts\activate.bat (
    echo ERROR: Virtual environment not found.
    echo Run install.bat first.
    pause
    exit /b 1
)

call .venv\Scripts\activate.bat

echo Opening Narou in your default browser...
echo.
echo If the browser does not open automatically,
echo go to: http://localhost:8501
echo.
echo Press Ctrl+C in this window to stop the app.
echo ============================================
echo.

streamlit run app.py
