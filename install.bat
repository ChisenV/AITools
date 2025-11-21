@echo off
echo Installing AITools package...
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo Error: Python is not installed or not in PATH
    echo Please install Python 3.8 or higher and try again
    pause
    exit /b 1
)

REM Check Python version
for /f "tokens=2" %%i in ('python --version 2^>^&1') do set PYTHON_VERSION=%%i
echo Python version: %PYTHON_VERSION%

REM Install dependencies
echo.
echo Installing dependencies from requirements.txt...
pip install -r requirements.txt
if errorlevel 1 (
    echo Error: Failed to install dependencies
    pause
    exit /b 1
)

REM Install the package
echo.
echo Installing AITools package...
pip install -e .
if errorlevel 1 (
    echo Error: Failed to install AITools package
    pause
    exit /b 1
)

echo.
echo ========================================
echo AITools installation completed successfully!
echo ========================================
echo.
echo You can now import AITools in your Python scripts:
echo   import AITools
echo.
pause
