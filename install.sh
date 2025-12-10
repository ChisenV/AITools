#!/bin/bash

echo "Installing AITools package..."
echo

# Check if Python is installed
if ! command -v python &> /dev/null; then
    echo "Error: Python is not installed or not in PATH"
    echo "Please install Python 3.8 or higher and try again"
    exit 1
fi

# Check Python version
PYTHON_VERSION=$(python --version 2>&1 | cut -d' ' -f2)
echo "Python version: $PYTHON_VERSION"

# Install dependencies
echo
echo "Installing dependencies from requirements.txt..."
pip install -r requirements.txt
if [ $? -ne 0 ]; then
    echo "Error: Failed to install dependencies"
    exit 1
fi

# Install the package
echo
echo "Installing AITools package..."
pip install -e .
if [ $? -ne 0 ]; then
    echo "Error: Failed to install AITools package"
    exit 1
fi

echo
echo "========================================"
echo "AITools installation completed successfully!"
echo "========================================"
echo
echo "You can now import AITools in your Python scripts:"
echo "  import AITools"
