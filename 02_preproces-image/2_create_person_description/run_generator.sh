#!/bin/bash

# Image Description Generator using Amazon Bedrock Claude 3.7
# This script runs the Python generator to create detailed image descriptions

set -e

echo "=== Image Description Generator ==="
echo "Using Amazon Bedrock Claude 3.7 Sonnet"
echo ""

# Check if AWS credentials are configured
if ! aws sts get-caller-identity > /dev/null 2>&1; then
    echo "Error: AWS credentials not configured. Please run 'aws configure' first."
    exit 1
fi

# Check if required files exist
METADATA_PATH="/home/ubuntu/musinsaigo/test_data/metadata.jsonl"
if [ ! -f "$METADATA_PATH" ]; then
    echo "Error: Metadata file not found at $METADATA_PATH"
    exit 1
fi

# Install dependencies if needed
echo "Installing dependencies..."
pip install -r requirements.txt

# Run the generator
echo "Starting description generation..."
python generate_descriptions.py

echo ""
echo "=== Generation Complete ==="
echo "Check the log file 'description_generation.log' for details"
echo "Updated metadata saved to: /home/ubuntu/musinsaigo/test_data/metadata_updated.jsonl" 