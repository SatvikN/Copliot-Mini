#!/bin/bash

# Enable Custom AI mode for CopilotMini
echo "🚀 Enabling Custom AI mode..."

# Set environment variable
export USE_CUSTOM_AI=true

# Verify the trained model exists
if [ -d "training/checkpoints/quick_test" ]; then
    echo "✅ Trained model found at: training/checkpoints/quick_test"
    echo 📊 Model files:
    ls -la training/checkpoints/quick_test/
else
    echo "❌ No trained model found. Please run training first:"
    echo "   python training/scripts/quick_train.py"
    exit 1
fi

echo 🎯 Custom AI mode enabled!"
echo "📝 To start the backend with custom AI:"
echo python backend/app.py
echo 📝 To start with custom AI in a new terminal:"
echo   source enable_custom_ai.sh && python backend/app.py" 