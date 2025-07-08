#!/bin/bash

# 🚀 CopilotMini Deployment Script
# This script automates the setup and deployment of CopilotMini

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_header() {
    echo -e "${BLUE}=== $1 ===${NC}"
}

# Check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Check prerequisites
check_prerequisites() {
    print_header "Checking Prerequisites"
    
    # Check Python
    if command_exists python3; then
        PYTHON_VERSION=$(python3 --version | cut -d' ' -f2)
        print_status "Python found: $PYTHON_VERSION"
    else
        print_error "Python 3 is required but not installed"
        exit 1
    fi
    
    # Check Node.js
    if command_exists node; then
        NODE_VERSION=$(node --version)
        print_status "Node.js found: $NODE_VERSION"
    else
        print_error "Node.js is required but not installed"
        exit 1
    fi
    
    # Check npm
    if command_exists npm; then
        NPM_VERSION=$(npm --version)
        print_status "npm found: $NPM_VERSION"
    else
        print_error "npm is required but not installed"
        exit 1
    fi
    
    # Check if VSCode is installed
    if command_exists code; then
        print_status "VSCode CLI found"
    else
        print_warning "VSCode CLI not found - you'll need to install the extension manually"
    fi
}

# Install backend dependencies
setup_backend() {
    print_header "Setting up Backend"
    
    # Create virtual environment if it doesn't exist
    if [ ! -d "venv" ]; then
        print_status "Creating Python virtual environment..."
        python3 -m venv venv
    fi
    
    # Activate virtual environment
    print_status "Activating virtual environment..."
    source venv/bin/activate
    
    # Install dependencies
    print_status "Installing Python dependencies..."
    pip install --upgrade pip
    pip install -r requirements-minimal.txt
    
    print_status "Backend setup complete!"
}

# Setup VSCode extension
setup_extension() {
    print_header "Setting up VSCode Extension"
    
    cd frontend/vscode-extension
    
    print_status "Installing Node.js dependencies..."
    npm install
    
    print_status "Compiling TypeScript..."
    npm run compile
    
    print_status "Packaging extension..."
    npm run package
    
    if command_exists code; then
        print_status "Installing extension in VSCode..."
        code --install-extension copilot-mini-0.1.0.vsix --force
        print_status "Extension installed successfully!"
    else
        print_warning "Please install the extension manually:"
        print_warning "1. Open VSCode"
        print_warning "2. Go to Extensions (Cmd+Shift+X)"
        print_warning "3. Click '...' -> Install from VSIX"
        print_warning "4. Select frontend/vscode-extension/copilot-mini-0.1.0.vsix"
    fi
    
    cd ../..
}

# Start backend server
start_backend() {
    print_header "Starting Backend Server"
    
    # Check if port 8000 is in use
    if lsof -Pi :8000 -sTCP:LISTEN -t >/dev/null; then
        print_warning "Port 8000 is already in use. Stopping existing processes..."
        kill -9 $(lsof -t -i:8000) || true
        sleep 2
    fi
    
    # Activate virtual environment
    source venv/bin/activate
    
    print_status "Starting backend server on http://localhost:8000"
    cd backend
    
    # Start server in background
    nohup python app.py > ../logs/server.log 2>&1 &
    SERVER_PID=$!
    echo $SERVER_PID > ../server.pid
    
    cd ..
    
    # Wait for server to start
    print_status "Waiting for server to start..."
    for i in {1..30}; do
        if curl -s http://localhost:8000/health >/dev/null 2>&1; then
            print_status "✅ Backend server is running! (PID: $SERVER_PID)"
            break
        fi
        if [ $i -eq 30 ]; then
            print_error "Server failed to start within 30 seconds"
            exit 1
        fi
        sleep 1
    done
}

# Test the installation
test_installation() {
    print_header "Testing Installation"
    
    # Test API endpoints
    print_status "Testing API health endpoint..."
    if curl -s http://localhost:8000/health | grep -q "healthy"; then
        print_status "✅ Health endpoint working"
    else
        print_error "❌ Health endpoint failed"
        exit 1
    fi
    
    print_status "Testing completion endpoint..."
    if curl -s -X POST http://localhost:8000/api/v1/complete \
        -H "Content-Type: application/json" \
        -d '{"code":"def hello", "language":"python"}' | grep -q "suggestions"; then
        print_status "✅ Completion endpoint working"
    else
        print_error "❌ Completion endpoint failed"
        exit 1
    fi
    
    print_status "Testing WebSocket connection..."
    if command_exists wscat; then
        timeout 5s wscat -c ws://localhost:8000/ws -x '{"type":"ping"}' >/dev/null 2>&1 || true
        print_status "✅ WebSocket endpoint accessible"
    else
        print_warning "wscat not found - install with: npm install -g wscat"
    fi
}

# Docker deployment
deploy_docker() {
    print_header "Docker Deployment"
    
    if ! command_exists docker; then
        print_error "Docker is not installed"
        exit 1
    fi
    
    print_status "Building Docker image..."
    docker build -t copilot-mini .
    
    print_status "Starting Docker container..."
    docker run -d --name copilot-mini-backend -p 8000:8000 copilot-mini
    
    print_status "✅ Docker deployment complete!"
    print_status "Container is running on http://localhost:8000"
}

# Show final instructions
show_final_instructions() {
    print_header "🎉 Installation Complete!"
    
    echo
    print_status "CopilotMini is now ready to use!"
    echo
    print_status "Backend API: http://localhost:8000"
    print_status "API Documentation: http://localhost:8000/docs"
    print_status "Health Check: http://localhost:8000/health"
    echo
    print_status "VSCode Extension Commands:"
    print_status "  • Cmd+Shift+C - Open Chat Panel"
    print_status "  • Cmd+Shift+E - Explain Selected Code"
    print_status "  • Right-click on errors for AI fixes"
    echo
    print_status "To stop the backend server:"
    print_status "  kill \$(cat server.pid)"
    echo
    print_status "To view server logs:"
    print_status "  tail -f logs/server.log"
    echo
    print_status "For troubleshooting, see INSTALLATION.md"
    echo
}

# Cleanup function
cleanup() {
    if [ -f server.pid ]; then
        print_status "Stopping backend server..."
        kill $(cat server.pid) 2>/dev/null || true
        rm server.pid
    fi
}

# Main deployment logic
main() {
    # Handle script termination
    trap cleanup EXIT
    
    echo "🚀 CopilotMini Deployment Script"
    echo "==============================="
    echo
    
    # Parse command line arguments
    DEPLOY_MODE="local"
    SKIP_TESTS=false
    
    while [[ $# -gt 0 ]]; do
        case $1 in
            --docker)
                DEPLOY_MODE="docker"
                shift
                ;;
            --skip-tests)
                SKIP_TESTS=true
                shift
                ;;
            --help|-h)
                echo "Usage: $0 [OPTIONS]"
                echo "Options:"
                echo "  --docker      Deploy using Docker"
                echo "  --skip-tests  Skip installation tests"
                echo "  --help        Show this help message"
                exit 0
                ;;
            *)
                print_error "Unknown option: $1"
                exit 1
                ;;
        esac
    done
    
    # Create logs directory
    mkdir -p logs
    
    # Run deployment steps
    check_prerequisites
    
    if [ "$DEPLOY_MODE" = "docker" ]; then
        deploy_docker
    else
        setup_backend
        setup_extension
        start_backend
        
        if [ "$SKIP_TESTS" = false ]; then
            test_installation
        fi
    fi
    
    show_final_instructions
}

# Run main function with all arguments
main "$@" 