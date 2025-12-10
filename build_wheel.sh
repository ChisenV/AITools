#!/bin/bash

echo "Building AITools wheel package..."
echo

# Check if Python is installed
if ! command -v python &> /dev/null; then
    echo "Error: Python is not installed or not in PATH"
    echo "Please install Python 3.8 or higher and try again"
    exit 1
fi

# Install build tools if not present
echo "Installing build tools..."
pip install build wheel

# Clean previous builds
echo "Cleaning previous builds..."
rm -rf build/ dist/ AITools.egg-info/

# Build the wheel
echo
echo "Building wheel package..."
python -m build --wheel
if [ $? -ne 0 ]; then
    echo "Error: Failed to build wheel package"
    exit 1
fi

# Show the built wheel file
echo
echo "========================================"
echo "Wheel package built successfully!"
echo "========================================"
echo
ls -la dist/*.whl
echo
echo "You can install the wheel package using:"
echo "  pip install dist/AITools-*.whl"
