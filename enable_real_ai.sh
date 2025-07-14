#!/bin/bash

# 🤖 Enable Real AI for CopilotMini
# This script helps you set up real AI models instead of the mock engine

set -e

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
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

echo "🤖 CopilotMini Real AI Setup"
echo "=============================="
echo
echo "Choose your preferred AI backend:"
echo "1) OpenAI GPT-4 (requires API key, best quality)"
echo "2) Ollama Local Models (free, runs locally)"
echo "3) HuggingFace Models (free, runs locally)"
echo "4) Show current status"
echo

read -p "Enter your choice (1-4): " choice

case $choice in
    1)
        print_header "Setting up OpenAI GPT-4"
        
        # Install OpenAI dependency
        print_info "Installing OpenAI package..."
        pip install openai>=1.3.0
        
        # Get API key
        echo
        print_warning "You need an OpenAI API key to use GPT-4"
        print_info "Get your API key from: https://platform.openai.com/api-keys"
        echo
        read -p "Enter your OpenAI API key (or press Enter to set it later): " api_key
        
        if [ ! -z "$api_key" ]; then
            export OPENAI_API_KEY="$api_key"
            echo "export OPENAI_API_KEY=\"$api_key\"" >> ~/.bashrc
            echo "export OPENAI_API_KEY=\"$api_key\"" >> ~/.zshrc
            print_info "✅ API key saved to shell configuration"
        else
            print_warning "Remember to set OPENAI_API_KEY environment variable later"
            print_info "Run: export OPENAI_API_KEY='your-key-here'"
        fi
        
        # Enable real AI
        export USE_REAL_AI=true
        print_info "✅ Real AI enabled!"
        ;;
        
    2)
        print_header "Setting up Ollama Local Models"
        
        # Check if Ollama is installed
        if ! command -v ollama &> /dev/null; then
            print_warning "Ollama is not installed"
            print_info "Install Ollama from: https://ollama.ai"
            
            # Try to install Ollama on macOS
            if [[ "$OSTYPE" == "darwin"* ]]; then
                if command -v brew &> /dev/null; then
                    print_info "Installing Ollama with Homebrew..."
                    brew install ollama
                else
                    print_warning "Please install Ollama manually from: https://ollama.ai"
                    exit 1
                fi
            else
                print_warning "Please install Ollama manually from: https://ollama.ai"
                exit 1
            fi
        fi
        
        # Install aiohttp for Ollama communication
        print_info "Installing aiohttp..."
        pip install aiohttp>=3.8.0
        
        # Download a coding model
        print_info "Starting Ollama service..."
        ollama serve &
        sleep 5
        
        print_info "Downloading CodeLlama model (this may take a few minutes)..."
        ollama pull codellama:7b
        
        # Enable real AI
        export USE_REAL_AI=true
        print_info "✅ Ollama setup complete!"
        ;;
        
    3)
        print_header "Setting up HuggingFace Local Models"
        
        print_warning "HuggingFace models require significant disk space and RAM"
        read -p "Continue? (y/N): " confirm
        
        if [[ $confirm =~ ^[Yy]$ ]]; then
            print_info "Installing HuggingFace dependencies..."
            pip install transformers>=4.30.0 torch>=2.0.0
            
            # Enable real AI
            export USE_REAL_AI=true
            print_info "✅ HuggingFace setup complete!"
        else
            print_info "Setup cancelled"
            exit 0
        fi
        ;;
        
    4)
        print_header "Current AI Status"
        
        # Check environment variables
        if [ "$USE_REAL_AI" = "true" ]; then
            print_info "✅ Real AI is ENABLED"
        else
            print_warning "❌ Real AI is DISABLED (using mock)"
            print_info "Set USE_REAL_AI=true to enable"
        fi
        
        # Check OpenAI
        if [ ! -z "$OPENAI_API_KEY" ]; then
            print_info "✅ OpenAI API key is set"
        else
            print_warning "❌ OpenAI API key not found"
        fi
        
        # Check Ollama
        if command -v ollama &> /dev/null; then
            print_info "✅ Ollama is installed"
            
            # Check if Ollama is running
            if curl -s http://localhost:11434/api/tags &> /dev/null; then
                print_info "✅ Ollama is running"
                ollama list
            else
                print_warning "❌ Ollama is not running (run: ollama serve)"
            fi
        else
            print_warning "❌ Ollama not installed"
        fi
        
        # Check Python packages
        echo
        print_info "Checking Python packages..."
        python -c "import openai; print('✅ OpenAI package installed')" 2>/dev/null || echo "❌ OpenAI package not installed"
        python -c "import transformers; print('✅ HuggingFace transformers installed')" 2>/dev/null || echo "❌ HuggingFace transformers not installed"
        python -c "import aiohttp; print('✅ aiohttp installed')" 2>/dev/null || echo "❌ aiohttp not installed"
        
        exit 0
        ;;
        
    *)
        print_warning "Invalid choice"
        exit 1
        ;;
esac

echo
print_header "🎉 Setup Complete!"
echo
print_info "Real AI is now enabled for CopilotMini!"
print_info "Restart your backend server to use the new AI engine:"
echo
print_info "  export USE_REAL_AI=true"
print_info "  cd backend && python app.py"
echo
print_warning "Note: Real AI models may be slower than the mock engine"
print_info "Check logs to see which AI backend is being used"
echo 