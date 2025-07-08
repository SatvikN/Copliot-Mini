# 🚀 CopilotMini Installation Guide

Complete guide for installing and deploying CopilotMini in different environments.

## 📋 Table of Contents

1. [Quick Start](#quick-start)
2. [Development Setup](#development-setup)
3. [VSCode Extension Installation](#vscode-extension-installation)
4. [Backend Deployment](#backend-deployment)
5. [Docker Deployment](#docker-deployment)
6. [Production Deployment](#production-deployment)
7. [Troubleshooting](#troubleshooting)

---

## 🏃‍♂️ Quick Start

### Prerequisites
- **Python 3.11+**
- **Node.js 16+**
- **VSCode**
- **Git**

### 1-Minute Setup
```bash
# Clone the repository
git clone https://github.com/SatvikN/Copliot-Mini.git
cd CopilotMini

# Install backend dependencies
pip install -r requirements-minimal.txt

# Start the backend server
cd backend && python app.py

# In a new terminal, install the VSCode extension
cd frontend/vscode-extension
npm install
npm run compile
```

Then in VSCode:
1. Open the `frontend/vscode-extension` folder
2. Press `F5` to launch Extension Development Host
3. Start coding with AI assistance! 🎉

---

## 🛠 Development Setup

### Backend Setup

1. **Create Virtual Environment** (Recommended)
```bash
python -m venv copilot-mini-env
source copilot-mini-env/bin/activate  # On Windows: copilot-mini-env\Scripts\activate
```

2. **Install Dependencies**
```bash
# Minimal setup (recommended for development)
pip install -r requirements-minimal.txt

# Full ML stack (for production with real models)
pip install -r requirements.txt
```

3. **Start Development Server**
```bash
cd backend
python app.py
```

The API will be available at `http://localhost:8000`

### VSCode Extension Setup

1. **Install Dependencies**
```bash
cd frontend/vscode-extension
npm install
```

2. **Development Mode**
```bash
# Compile TypeScript
npm run compile

# Watch mode (auto-recompile on changes)
npm run watch
```

3. **Launch Extension**
- Open `frontend/vscode-extension` in VSCode
- Press `F5` to open Extension Development Host
- Or package and install: `npm run package` then install the `.vsix` file

---

## 📦 VSCode Extension Installation

### Method 1: Development Host (Recommended for Testing)
1. Open VSCode
2. Open folder: `frontend/vscode-extension`
3. Press `F5` → Extension Development Host opens
4. Your extension is active in the new window!

### Method 2: Package Installation
```bash
cd frontend/vscode-extension
npm run package
code --install-extension copilot-mini-0.1.0.vsix
```

### Method 3: Manual Installation
1. Build the extension: `npm run package`
2. Open VSCode → Extensions (`Cmd+Shift+X`)
3. Click `...` → Install from VSIX
4. Select `copilot-mini-0.1.0.vsix`

### Extension Features
- **Inline Code Completion**: Type code and see AI suggestions
- **Chat Panel**: `Cmd+Shift+C` to open AI chat
- **Error Fixes**: Hover over errors for AI-powered fixes
- **Code Explanation**: Select code → `Cmd+Shift+E`
- **Status Bar**: Shows connection status

---

## 🖥 Backend Deployment

### Local Development
```bash
cd backend
python app.py
```

### Production with Gunicorn
```bash
pip install gunicorn
gunicorn -w 4 -k uvicorn.workers.UvicornWorker backend.app:app --bind 0.0.0.0:8000
```

### Environment Variables
```bash
export API_HOST=0.0.0.0
export API_PORT=8000
export LOG_LEVEL=info
export ENVIRONMENT=production
```

### API Endpoints
- **Health Check**: `GET /health`
- **Code Completion**: `POST /api/v1/complete`
- **Chat**: `POST /api/v1/chat`
- **WebSocket**: `ws://localhost:8000/ws`
- **API Docs**: `http://localhost:8000/docs`

---

## 🐳 Docker Deployment

### Single Container
```bash
# Build the image
docker build -t copilot-mini .

# Run the container
docker run -p 8000:8000 copilot-mini
```

### Docker Compose (Recommended)
```bash
# Start all services
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

### Docker Health Check
```bash
# Check container health
docker ps
docker exec copilot-mini-backend curl -f http://localhost:8000/health
```

---

## 🌐 Production Deployment

### AWS Deployment
```bash
# Build for AWS Lambda
docker build -t copilot-mini-lambda -f Dockerfile.lambda .

# Deploy with AWS CLI
aws ecr get-login-password | docker login --username AWS --password-stdin <account>.dkr.ecr.<region>.amazonaws.com
docker tag copilot-mini-lambda <account>.dkr.ecr.<region>.amazonaws.com/copilot-mini:latest
docker push <account>.dkr.ecr.<region>.amazonaws.com/copilot-mini:latest
```

### Google Cloud Run
```bash
# Build and deploy
gcloud builds submit --tag gcr.io/PROJECT-ID/copilot-mini
gcloud run deploy --image gcr.io/PROJECT-ID/copilot-mini --platform managed
```

### Kubernetes
```yaml
# k8s-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: copilot-mini
spec:
  replicas: 3
  selector:
    matchLabels:
      app: copilot-mini
  template:
    metadata:
      labels:
        app: copilot-mini
    spec:
      containers:
      - name: copilot-mini
        image: copilot-mini:latest
        ports:
        - containerPort: 8000
        env:
        - name: API_HOST
          value: "0.0.0.0"
        - name: API_PORT
          value: "8000"
```

### HTTPS/SSL Setup
1. **With Nginx** (included in docker-compose)
2. **With Cloudflare** (recommended for simplicity)
3. **With Let's Encrypt**:
```bash
certbot --nginx -d your-domain.com
```

---

## 🛡 Security Configuration

### Production Security Checklist
- [ ] Use HTTPS in production
- [ ] Set up API rate limiting
- [ ] Configure CORS properly
- [ ] Use environment variables for secrets
- [ ] Enable container security scanning
- [ ] Set up monitoring and logging

### Environment Variables
```bash
# Required
API_HOST=0.0.0.0
API_PORT=8000

# Optional
LOG_LEVEL=info
ENVIRONMENT=production
CORS_ORIGINS=https://your-domain.com
MAX_REQUESTS_PER_MINUTE=60
```

---

## 🔧 Troubleshooting

### Common Issues

#### Backend Won't Start
```bash
# Check port availability
lsof -ti:8000

# Kill processes on port 8000
kill -9 $(lsof -ti:8000)

# Check dependencies
pip list | grep fastapi
```

#### Extension Not Connecting
1. **Check Backend Status**: Visit `http://localhost:8000/health`
2. **Verify WebSocket URL**: Should be `ws://localhost:8000/ws`
3. **Reload Extension**: Press `Cmd+R` in Extension Development Host
4. **Check VSCode Output**: View → Output → CopilotMini

#### No Code Suggestions
1. **Check Status Bar**: Should show "✓ CopilotMini" (green)
2. **Verify Language Support**: Python, JS, TS, Java, C++, Go, Rust, PHP, Ruby
3. **Check Configuration**: `Cmd+,` → Search "CopilotMini"

#### Docker Issues
```bash
# Check container logs
docker-compose logs copilot-mini-api

# Restart services
docker-compose restart

# Rebuild with no cache
docker-compose build --no-cache
```

### Debug Mode
```bash
# Backend debug mode
export LOG_LEVEL=debug
python backend/app.py

# Extension debug mode
# In VSCode: View → Output → Extension Host
```

### Performance Tuning
```bash
# Increase worker processes
gunicorn -w 8 -k uvicorn.workers.UvicornWorker backend.app:app

# Optimize Docker
docker run --memory=512m --cpus=2 copilot-mini
```

---

## 📊 Monitoring & Metrics

### Health Monitoring
```bash
# Backend health
curl http://localhost:8000/health

# WebSocket connection test
wscat -c ws://localhost:8000/ws

# Container resource usage
docker stats copilot-mini-backend
```

### Logs
```bash
# Application logs
tail -f logs/copilot_mini.log

# Docker logs
docker-compose logs -f --tail=100
```

---

## 🎯 Next Steps

After installation:

1. **Test the Extension**: Try code completion in Python/JavaScript files
2. **Explore Chat Mode**: Press `Cmd+Shift+C` and ask coding questions
3. **Configure Settings**: Customize in VSCode preferences
4. **Integrate Real AI Models**: Replace mock engine with actual models
5. **Set up Analytics**: Track usage and improve suggestions

### Upgrading to Real AI Models

Replace the mock inference engine with:
- **Local Models**: Ollama, LM Studio
- **Cloud APIs**: OpenAI, Anthropic, Google
- **Self-hosted**: Deploy CodeLlama, StarCoder

See `backend/models/inference.py` for implementation details.

---

## 🆘 Support

- **Issues**: [GitHub Issues](https://github.com/SatvikN/Copliot-Mini/issues)
- **Discussions**: [GitHub Discussions](https://github.com/SatvikN/Copliot-Mini/discussions)
- **Documentation**: See `README.md` for features and architecture

---

**Happy Coding with CopilotMini! 🤖✨** 