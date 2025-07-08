# CopilotMini - GitHub Copilot Clone

A comprehensive GitHub Copilot clone featuring a fine-tuned code generation model, RAG-enhanced completions, and VSCode integration.

## 🎯 Features

- **Fine-tuned Code Generation**: Custom models based on CodeParrot/CodeGen/CodeT5
- **RAG-Enhanced Completions**: Retrieval of local code, docs, APIs, and Stack Overflow
- **VSCode Integration**: Native extension with real-time suggestions
- **Chat Mode**: Interactive code assistance and explanations
- **Error Autofix**: Automatic detection and fixing of coding errors
- **Project-Aware**: Context-aware completions based on your codebase

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   VSCode Ext    │◄──►│  Backend API    │◄──►│  Fine-tuned LLM │
│  (TypeScript)   │    │ (Flask/FastAPI) │    │ (CodeParrot/T5) │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                              │
                              ▼
                       ┌─────────────────┐
                       │   RAG System    │
                       │ (FAISS/ChromaDB)│
                       └─────────────────┘
```

## 📁 Project Structure

```
CopilotMini/
├── backend/               # Backend API server
│   ├── api/              # API routes and endpoints
│   ├── models/           # Model loading and inference
│   ├── rag/              # RAG pipeline and retrieval
│   └── utils/            # Utility functions
├── frontend/             # Frontend components
│   └── vscode-extension/ # VSCode extension
├── training/             # Model training scripts
│   ├── scripts/          # Training and fine-tuning scripts
│   ├── configs/          # Model and training configurations
│   └── checkpoints/      # Saved model checkpoints
├── data/                 # Data processing
│   ├── raw/              # Raw datasets
│   └── processed/        # Processed and tokenized data
├── docs/                 # Documentation
└── tests/                # Test suites
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- Node.js 16+
- VSCode
- CUDA-capable GPU (recommended for training)

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd CopilotMini
   ```

2. **Set up backend**
   ```bash
   cd backend
   pip install -r requirements.txt
   ```

3. **Set up VSCode extension**
   ```bash
   cd frontend/vscode-extension
   npm install
   ```

4. **Install the extension**
   ```bash
   code --install-extension frontend/vscode-extension
   ```

### Usage

1. **Start the backend server**
   ```bash
   cd backend
   python app.py
   ```

2. **Open VSCode** and start coding - the extension will automatically provide suggestions!

## 🔧 Development Phases

- [x] **Phase 1**: Dataset and Preprocessing
- [x] **Phase 2**: Fine-tune Code Model  
- [x] **Phase 3**: Inference Server (LLM + RAG)
- [x] **Phase 4**: VSCode Extension
- [ ] **Phase 5**: Chat Mode (Optional)
- [ ] **Phase 6**: Error Fix Mode (Optional)
- [ ] **Phase 7**: Deployment and Testing

## 🛠️ Tech Stack

| Component | Technology |
|-----------|------------|
| Code Model | CodeParrot / CodeT5 / CodeGen |
| RAG Retriever | FAISS or ChromaDB |
| Embeddings | codeBERT, intfloat/e5-code |
| Server | Flask/FastAPI with WebSocket |
| Extension | VSCode with TypeScript |
| Hosting | Render / EC2 / GCP |

## 📊 Model Performance

*Performance metrics will be added after training completion*

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📝 License

MIT License - see LICENSE file for details

## 🔗 Links

- [Documentation](./docs/)
- [API Reference](./docs/api.md)
- [Training Guide](./docs/training.md) 