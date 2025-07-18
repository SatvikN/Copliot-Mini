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
   cd ..
   ```

3. **Set up VSCode extension**
   ```bash
   cd frontend/vscode-extension
   npm install
   cd ../../..
   ```

### Usage

1. **Build the RAG index** (first time or after changing code/docs)
   ```bash
   python -m backend.rag.index_code_and_docs
   ```
   This will index your codebase and documentation for retrieval-augmented generation (RAG).

2. **Start the backend server**
   - To use your custom fine-tuned model:
     ```bash
     export USE_CUSTOM_AI=true
     python backend/app.py
     # or
     export USE_CUSTOM_AI=true
     uvicorn backend.app:app --reload
     ```
   - The backend will load your model and the RAG index. The first load may take several minutes for large models.

3. **Run the VSCode extension**
   - Open the project in VSCode.
   - In VSCode, open the `frontend/vscode-extension/` folder and press `F5` to launch the extension in a new Extension Development Host window.
   - The extension will connect to the backend and provide completions and chat.

4. **(Optional) Test the API directly**
   ```bash
   curl -X POST "http://localhost:8000/api/v1/complete" \
     -H "Content-Type: application/json" \
     -d '{"code": "def fibonacci(n):", "language": "python"}'
   ```

## 🧠 RAG & Custom Model Usage

- **RAG Indexing**: The backend uses a persistent FAISS index to retrieve relevant code and documentation chunks for every request. To rebuild the index (e.g., after changing your codebase or docs):
  ```bash
  python -m backend.rag.index_code_and_docs
  ```
  The index is saved in `backend/rag/index_data/` and loaded automatically by the backend.

- **Custom Model**: To use your own fine-tuned model, ensure your model files are in `training/checkpoints/codegen/` and set:
  ```bash
  export USE_CUSTOM_AI=true
  ```
  The backend will load your model and use it for completions and chat. If not set, the backend will use a mock engine or external AI if available.

- **Troubleshooting**:
  - If the backend hangs on startup, it may be loading a large model. Wait several minutes.
  - If you see `External AI API not available`, it means OpenAI or other external APIs are not configured, but your custom model will still work.
  - If the extension cannot connect, ensure the backend is running and the WebSocket URL matches the backend port (default: 8000).
  - To test model loading, try loading your model in a Python shell:
    ```python
    from transformers import AutoModelForCausalLM, AutoTokenizer
    model = AutoModelForCausalLM.from_pretrained("training/checkpoints/codegen")
    tokenizer = AutoTokenizer.from_pretrained("training/checkpoints/codegen")
    ```

- **Updating the RAG index**: Re-run the indexer script any time you change your codebase or documentation to keep completions project-aware.

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