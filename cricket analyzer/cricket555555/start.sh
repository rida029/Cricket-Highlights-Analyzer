#!/bin/bash
# Install system dependencies
apt-get update && apt-get install -y tesseract-ocr

# Run the application
python app.py
chmod +x start.sh
