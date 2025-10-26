#!/bin/bash
# Kaggle API Setup Script
# This script helps you configure Kaggle API credentials

set -e  # Exit on error

echo "=========================================="
echo "Kaggle API Setup"
echo "=========================================="
echo ""

# Check if kaggle is installed
if ! command -v kaggle &> /dev/null; then
    echo "❌ Kaggle CLI not found. Installing..."
    pip install kaggle
    echo "✅ Kaggle installed successfully!"
    echo ""
fi

# Create .kaggle directory
KAGGLE_DIR="$HOME/.kaggle"
if [ ! -d "$KAGGLE_DIR" ]; then
    echo "📁 Creating Kaggle config directory: $KAGGLE_DIR"
    mkdir -p "$KAGGLE_DIR"
fi

# Check if kaggle.json exists
KAGGLE_JSON="$KAGGLE_DIR/kaggle.json"
if [ -f "$KAGGLE_JSON" ]; then
    echo "✅ kaggle.json already exists at: $KAGGLE_JSON"
    echo ""
else
    echo "📥 kaggle.json not found. Please follow these steps:"
    echo ""
    echo "1. Go to: https://www.kaggle.com/settings"
    echo "2. Scroll to 'API' section"
    echo "3. Click 'Create New API Token'"
    echo "4. This will download kaggle.json"
    echo ""
    echo "5. Move it to the correct location:"
    echo "   mv ~/Downloads/kaggle.json $KAGGLE_DIR/"
    echo ""
    echo "Then run this script again!"
    exit 1
fi

# Set correct permissions
chmod 600 "$KAGGLE_JSON"
echo "🔒 Set secure permissions (600) on kaggle.json"

# Test API connection
echo ""
echo "🧪 Testing Kaggle API connection..."
if kaggle datasets list --max-size 1 &> /dev/null; then
    echo "✅ Kaggle API is working correctly!"
    echo ""
    echo "=========================================="
    echo "Setup Complete! 🎉"
    echo "=========================================="
    echo ""
    echo "You can now download datasets with:"
    echo "  kaggle datasets download -d DATASET_ID"
    echo ""
else
    echo "❌ Kaggle API test failed. Please check your credentials."
    echo ""
    echo "Make sure kaggle.json contains valid credentials:"
    echo "  cat $KAGGLE_JSON"
    exit 1
fi
