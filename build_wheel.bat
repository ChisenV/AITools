@echo off
echo Building AITools wheel package...
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo Error: Python is not installed or not in PATH
    echo Please install Python 3.8 or higher and try again
    pause
    exit /b 1
)

REM Install build tools if not present
echo Installing build tools...
pip install build wheel

REM Clean previous builds
if exist "build" rmdir /s /q build
if exist "dist" rmdir /s /q dist
if exist "AITools.egg-info" rmdir /s /q AITools.egg-info

REM Build the wheel
echo.
echo Building wheel package...
python -m build --wheel
if errorlevel 1 (
    echo Error: Failed to build wheel package
    pause
    exit /b 1
)

REM Show the built wheel file
echo.
echo ========================================
echo Wheel package built successfully!
echo ========================================
echo.
dir dist\*.whl
echo.
echo You can install the wheel package using:
echo   pip install dist\AITools-*.whl
echo.
@REM pause
