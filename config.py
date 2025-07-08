"""
Configuration file for CopilotMini project
"""

import os
from pathlib import Path

# Project root directory
PROJECT_ROOT = Path(__file__).parent

# Data configuration
DATA_CONFIG = {
    "raw_data_dir": PROJECT_ROOT / "data" / "raw",
    "processed_data_dir": PROJECT_ROOT / "data" / "processed",
    "supported_languages": ["python", "javascript", "typescript", "java", "cpp", "c", "go", "rust"],
    "max_file_size": 1024 * 1024,  # 1MB max file size
    "min_lines": 5,  # Minimum lines per code snippet
    "max_lines": 512,  # Maximum lines per code snippet
}

# Model configuration
MODEL_CONFIG = {
    "base_models": {
        "codeparrot": "codeparrot/codeparrot-small",
        "codet5": "Salesforce/codet5-base",
        "codegen": "Salesforce/codegen-350M-mono",
    },
    "max_length": 512,
    "max_new_tokens": 128,
    "temperature": 0.7,
    "top_p": 0.9,
    "top_k": 50,
    "repetition_penalty": 1.1,
}

# Training configuration
TRAINING_CONFIG = {
    "batch_size": 8,
    "learning_rate": 5e-5,
    "num_epochs": 3,
    "warmup_steps": 500,
    "logging_steps": 100,
    "save_steps": 1000,
    "eval_steps": 500,
    "gradient_accumulation_steps": 4,
    "max_grad_norm": 1.0,
    "weight_decay": 0.01,
    "adam_epsilon": 1e-8,
    "checkpoints_dir": PROJECT_ROOT / "training" / "checkpoints",
}

# RAG configuration
RAG_CONFIG = {
    "embedding_models": {
        "codebert": "microsoft/codebert-base",
        "e5_code": "intfloat/e5-base-v2",
        "minilm": "sentence-transformers/all-MiniLM-L6-v2",
    },
    "vector_db": "faiss",  # or "chromadb"
    "chunk_size": 256,
    "chunk_overlap": 50,
    "top_k_retrieval": 5,
    "similarity_threshold": 0.7,
    "index_dir": PROJECT_ROOT / "backend" / "rag" / "indexes",
}

# API configuration
API_CONFIG = {
    "host": "0.0.0.0",
    "port": 8000,
    "websocket_path": "/ws",
    "max_connections": 100,
    "timeout": 30,
    "cors_origins": ["*"],
    "api_prefix": "/api/v1",
}

# VSCode extension configuration
EXTENSION_CONFIG = {
    "suggestion_delay": 300,  # milliseconds
    "max_suggestions": 3,
    "debounce_time": 150,  # milliseconds
    "context_lines": 10,  # lines of context to send
    "supported_file_types": [
        "python", "javascript", "typescript", "java", 
        "cpp", "c", "go", "rust", "php", "ruby"
    ],
}

# Database configuration
DB_CONFIG = {
    "redis_url": os.getenv("REDIS_URL", "redis://localhost:6379"),
    "database_url": os.getenv("DATABASE_URL", "sqlite:///copilot_mini.db"),
    "cache_ttl": 3600,  # 1 hour
}

# Logging configuration
LOGGING_CONFIG = {
    "level": "INFO",
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "log_file": PROJECT_ROOT / "logs" / "copilot_mini.log",
}

# Environment-specific overrides
if os.getenv("ENVIRONMENT") == "production":
    API_CONFIG["host"] = "0.0.0.0"
    LOGGING_CONFIG["level"] = "WARNING"
elif os.getenv("ENVIRONMENT") == "development":
    API_CONFIG["port"] = 8001
    LOGGING_CONFIG["level"] = "DEBUG"

# Create necessary directories
for config in [DATA_CONFIG, TRAINING_CONFIG, RAG_CONFIG, LOGGING_CONFIG]:
    for key, value in config.items():
        if key.endswith("_dir") or key.endswith("_file"):
            if isinstance(value, Path):
                value.parent.mkdir(parents=True, exist_ok=True) 