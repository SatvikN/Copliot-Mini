#!/bin/bash

# 🎯 Enable Custom AI for CopilotMini
# This script enables the use of fine-tuned custom models

set -e

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color

print_header() {
    echo -e "${BLUE}=== $1 ===${NC}"
}

print_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

echo "🎯 CopilotMini Custom AI Setup"
echo "==============================="
echo

# Check if training has been done
print_header "Checking for trained models"

CHECKPOINTS_DIR="training/checkpoints"
MODELS_FOUND=false

if [ -d "$CHECKPOINTS_DIR" ]; then
    for model_type in "codeparrot" "codet5" "codegen"; do
        if [ -d "$CHECKPOINTS_DIR/$model_type" ] && [ "$(ls -A $CHECKPOINTS_DIR/$model_type 2>/dev/null)" ]; then
            print_info "✅ Found trained $model_type model"
            MODELS_FOUND=true
        else
            print_warning "⚠️ No trained $model_type model found"
        fi
    done
else
    print_warning "⚠️ No checkpoints directory found"
fi

if [ "$MODELS_FOUND" = false ]; then
    print_error "❌ No trained models found!"
    echo
    echo "You need to train models first. Run:"
    echo "  python training/scripts/train_models.py --models all"
    echo
    echo "Or for a quick test:"
    echo "  python training/scripts/test_training.py"
    exit 1
fi

# Check dependencies
print_header "Checking dependencies"

python -c "import torch, transformers" 2>/dev/null
if [ $? -eq 0 ]; then
    print_info "✅ PyTorch and Transformers are installed"
else
    print_error "❌ Missing dependencies"
    echo "Install training dependencies:"
    echo "  pip install -r requirements.txt"
    exit 1
fi

# Set environment variable
print_header "Enabling Custom AI"

export USE_CUSTOM_AI=true

# Add to shell configuration files
for shell_config in ~/.bashrc ~/.zshrc; do
    if [ -f "$shell_config" ]; then
        if ! grep -q "USE_CUSTOM_AI" "$shell_config"; then
            echo "export USE_CUSTOM_AI=true" >> "$shell_config"
            print_info "✅ Added USE_CUSTOM_AI=true to $shell_config"
        else
            print_info "ℹ️ USE_CUSTOM_AI already set in $shell_config"
        fi
    fi
done

# Test the custom inference engine
print_header "Testing Custom AI Engine"

python -c "
import asyncio
import sys
sys.path.append('.')
from backend.models.custom_inference import CustomInferenceEngine

async def test():
    engine = CustomInferenceEngine()
    success = await engine.initialize()
    if success:
        info = engine.get_model_info()
        print(f'✅ Custom AI initialized successfully!')
        print(f'📊 Models available: {info[\"total_models\"]}')
        print(f'🎯 Default model: {info[\"default_model\"]}')
        for model in info['models']:
            print(f'  • {model[\"model_type\"]}: {model[\"architecture\"]} (loss: {model[\"performance\"][\"eval_loss\"]:.4f})')
        await engine.cleanup()
        return True
    else:
        print('❌ Failed to initialize custom AI')
        return False

success = asyncio.run(test())
exit(0 if success else 1)
"

if [ $? -eq 0 ]; then
    print_info "✅ Custom AI engine test successful!"
else
    print_error "❌ Custom AI engine test failed"
    exit 1
fi

# Instructions
print_header "Setup Complete!"

echo "🎉 Custom AI is now enabled!"
echo
echo "Your CopilotMini will now use your fine-tuned models by default."
echo
echo "📋 What's enabled:"
echo "  • Custom fine-tuned models will be loaded automatically"
echo "  • Best performing model will be used as default"
echo "  • All model types (CodeParrot, CodeT5, CodeGen) supported"
echo
echo "🚀 Start the backend:"
echo "  cd backend && python app.py"
echo
echo "🔍 Check engine status:"
echo "  curl http://localhost:8000/api/v1/engines/status"
echo
echo "⚙️ To switch back to other engines:"
echo "  export USE_CUSTOM_AI=false USE_REAL_AI=true  # Use real AI"
echo "  export USE_CUSTOM_AI=false USE_REAL_AI=false # Use mock engine"
echo

print_info "✨ Happy coding with your custom AI models!" 