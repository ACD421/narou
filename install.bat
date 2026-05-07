@echo off
cd /d "%~dp0"
setlocal EnableDelayedExpansion
echo ============================================
echo   Narou - AI Job Search Assistant
echo   First-time setup (Windows)
echo ============================================
echo.

:: Check for Python
where python >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Python is not installed or not in PATH.
    echo.
    echo Download Python 3.10 or later from https://www.python.org/downloads/
    echo Make sure to check "Add Python to PATH" during installation.
    pause
    exit /b 1
)

:: Verify Python 3.10+
for /f "tokens=2 delims= " %%a in ('python --version 2^>^&1') do set PYVER=%%a
for /f "tokens=1,2 delims=." %%a in ("!PYVER!") do (
    set PYMAJOR=%%a
    set PYMINOR=%%b
)
if !PYMAJOR! LSS 3 goto :oldpy
if !PYMAJOR! EQU 3 if !PYMINOR! LSS 10 goto :oldpy
echo Python !PYVER! detected.
goto :pyok
:oldpy
echo ERROR: Python !PYVER! is too old. Need Python 3.10 or later.
echo Download from https://www.python.org/downloads/
pause
exit /b 1
:pyok

echo [1/4] Creating virtual environment...
if exist .venv (
    echo    (removing stale .venv)
    rmdir /s /q .venv
)
python -m venv .venv
if %errorlevel% neq 0 (
    echo ERROR: Could not create virtual environment.
    pause
    exit /b 1
)

echo [2/4] Upgrading pip and clearing stale index cache...
call .venv\Scripts\activate.bat
python -m pip install --upgrade pip >nul 2>&1
python -m pip cache purge >nul 2>&1

echo [3/4] Installing dependencies (bypassing any stale pip cache)...
python -m pip install --no-cache-dir -r requirements.txt
if %errorlevel% neq 0 (
    echo.
    echo ERROR: Package install failed.
    echo If you see "No matching distribution", your corporate proxy or mirror
    echo may be blocking PyPI. Try from a non-corporate network, or configure
    echo pip to use https://pypi.org/simple explicitly.
    pause
    exit /b 1
)

echo [4/4] Creating data directories...
if not exist data\cache mkdir data\cache
if not exist data\labeled mkdir data\labeled
if not exist data\samples mkdir data\samples

echo.
echo ============================================
echo   Setup complete!
echo.
echo   To start the app, double-click run.bat
echo ============================================
pause
