#!/bin/bash

# Check if Python3 is installed
if ! command -v python3 &> /dev/null; then
    echo "Python3 is not installed. Please install Python3 first."
    exit 1
fi

# Remove existing venv if it exists
if [ -d "venv" ]; then
    echo "Removing existing virtual environment..."
    rm -rf venv
fi

# Create virtual environment
echo "Creating new virtual environment..."
python3 -m venv venv

# Get the absolute path to the virtual environment
VENV_PATH=$(pwd)/venv

# Activate virtual environment
echo "Activating virtual environment..."
source "$VENV_PATH/bin/activate"

# Verify we're in the virtual environment
if [ -z "$VIRTUAL_ENV" ]; then
    echo "Failed to activate virtual environment"
    exit 1
fi

# Install pip directly in the virtual environment
echo "Installing pip..."
curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
"$VENV_PATH/bin/python3" get-pip.py
rm get-pip.py

# Upgrade pip using the virtual environment's pip
echo "Upgrading pip..."
"$VENV_PATH/bin/pip3" install --upgrade pip

# Install requirements using the virtual environment's pip
echo "Installing requirements..."
"$VENV_PATH/bin/pip3" install -r requirements.txt

# Create necessary directories if they don't exist
echo "Creating project directories..."
mkdir -p code tests data docs

echo "Setup completed successfully!"
echo ""
echo "==================================================="
echo "To activate the virtual environment, please run:"
echo "source \"$VENV_PATH/bin/activate\""
echo ""
echo "You'll know it's activated when you see (venv) at"
echo "the beginning of your command prompt"
echo "===================================================" 