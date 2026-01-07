#!/bin/bash
set -e

# Install system dependencies
apt-get update
apt-get install -y \
    tesseract-ocr \
    tesseract-ocr-eng \
    ffmpeg \
    libsm6 \
    libxext6 \
    python3-dev \
    python3-pip \
    build-essential

# Install Python packages
pip install --upgrade pip
pip install -r requirements.txt

# Run the application
exec python app.py
