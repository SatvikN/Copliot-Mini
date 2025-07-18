"""
Main FastAPI application for CopilotMini backend.
Handles code completion requests and WebSocket connections.
"""

import os
import asyncio
from typing import List, Dict, Any, Optional
from datetime import datetime
import json

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import uvicorn
from loguru import logger

# Add project root to path
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from config import API_CONFIG, LOGGING_CONFIG
from backend.models.inference import MockInferenceEngine
from backend.utils.connection_manager import ConnectionManager
from backend.rag.retriever import RAGRetriever

# Import real AI engine
try:
    from backend.models.real_inference import RealInferenceEngine
    REAL_AI_AVAILABLE = True
except ImportError as e:
    REAL_AI_AVAILABLE = False
    RealInferenceEngine = None
    logger.info(f"External AI API not available: {e}")

# Import custom AI engine
try:
    from backend.models.custom_inference import CustomInferenceEngine
    CUSTOM_AI_AVAILABLE = True
except ImportError as e:
    CUSTOM_AI_AVAILABLE = False
    CustomInferenceEngine = None
    logger.info(f"Custom AI not available: {e}")

# Initialize RAG retriever (adjust path if needed)
rag_retriever = RAGRetriever()

# Configure logging
logger.add(
    LOGGING_CONFIG["log_file"], 
    level=LOGGING_CONFIG["level"],
    format=LOGGING_CONFIG["format"],
    rotation="10 MB"
)

# Initialize FastAPI app
app = FastAPI(
    title="CopilotMini API",
    description="AI-powered code completion backend with RAG enhancement",
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=API_CONFIG["cors_origins"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize components
connection_manager = ConnectionManager()

# Choose inference engine based on environment variables
USE_REAL_AI = os.getenv("USE_REAL_AI", "false").lower() == "true"
USE_CUSTOM_AI = os.getenv("USE_CUSTOM_AI", "false").lower() == "true"

# Priority: Custom AI > Real AI > Mock
if USE_CUSTOM_AI and CUSTOM_AI_AVAILABLE:
    inference_engine = CustomInferenceEngine()
    logger.info("🎯 Using Custom fine-tuned models")
elif USE_REAL_AI and REAL_AI_AVAILABLE:
    inference_engine = RealInferenceEngine()
    logger.info("🤖 Using Real AI inference engine")
else:
    inference_engine = MockInferenceEngine()
    if USE_CUSTOM_AI and not CUSTOM_AI_AVAILABLE:
        logger.warning("⚠️ Custom AI requested but no trained models found, falling back to mock")
    elif USE_REAL_AI and not REAL_AI_AVAILABLE:
        logger.warning("⚠️ Real AI requested but not available, using mock engine")
    else:
        logger.info("🔧 Using Mock inference engine")

# Pydantic models for request/response
class CodeCompletionRequest(BaseModel):
    code: str
    language: str
    cursor_position: Optional[int] = None
    max_suggestions: Optional[int] = 3
    context_lines: Optional[int] = 10

class CodeCompletionResponse(BaseModel):
    suggestions: List[str]
    confidence_scores: List[float]
    processing_time_ms: float
    model_used: str

class ChatRequest(BaseModel):
    message: str
    code_context: Optional[str] = None
    language: Optional[str] = None

class ChatResponse(BaseModel):
    response: str
    code_suggestions: Optional[List[str]] = None
    processing_time_ms: float

class HealthResponse(BaseModel):
    status: str
    timestamp: str
    version: str
    active_connections: int
    model_status: str

# API Routes

@app.get("/", response_model=dict)
async def root():
    """Root endpoint with basic API information."""
    return {
        "name": "CopilotMini API",
        "version": "0.1.0",
        "status": "running",
        "docs": "/docs",
        "websocket": "/ws"
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now().isoformat(),
        version="0.1.0",
        active_connections=len(connection_manager.active_connections),
        model_status=inference_engine.get_status()
    )

@app.post("/api/v1/complete", response_model=CodeCompletionResponse)
async def complete_code(request: CodeCompletionRequest):
    """Generate code completions for the given input."""
    try:
        start_time = asyncio.get_event_loop().time()
        # RAG: Retrieve relevant context
        rag_results = rag_retriever.retrieve(request.code, top_k=3)
        rag_context = '\n\n'.join([r['text'] for r in rag_results])
        # Prepend RAG context to code
        code_with_context = f"# Relevant context from project:\n{rag_context}\n\n{request.code}"
        # Generate completions using inference engine
        result = await inference_engine.generate_completions(
            code=code_with_context,
            language=request.language,
            max_suggestions=request.max_suggestions or 3,
            cursor_position=request.cursor_position
        )
        end_time = asyncio.get_event_loop().time()
        processing_time = (end_time - start_time) * 1000  # Convert to milliseconds
        return CodeCompletionResponse(
            suggestions=result["suggestions"],
            confidence_scores=result["confidence_scores"],
            processing_time_ms=processing_time,
            model_used=result["model_used"]
        )
    except Exception as e:
        logger.error(f"Error in code completion: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/chat", response_model=ChatResponse)
async def chat_with_ai(request: ChatRequest):
    """Chat with AI about code."""
    try:
        start_time = asyncio.get_event_loop().time()
        # RAG: Retrieve relevant context
        rag_results = rag_retriever.retrieve(request.message, top_k=3)
        rag_context = '\n\n'.join([r['text'] for r in rag_results])
        # Prepend RAG context to code_context if present, else just use RAG
        if request.code_context:
            code_context_with_rag = f"# Relevant context from project:\n{rag_context}\n\n{request.code_context}"
        else:
            code_context_with_rag = f"# Relevant context from project:\n{rag_context}"
        # Process chat request
        result = await inference_engine.process_chat(
            message=request.message,
            code_context=code_context_with_rag,
            language=request.language
        )
        end_time = asyncio.get_event_loop().time()
        processing_time = (end_time - start_time) * 1000
        return ChatResponse(
            response=result["response"],
            code_suggestions=result.get("code_suggestions"),
            processing_time_ms=processing_time
        )
    except Exception as e:
        logger.error(f"Error in chat processing: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/stats")
async def get_stats():
    """Get API usage statistics."""
    return {
        "active_connections": len(connection_manager.active_connections),
        "total_requests": 0,  # TODO: Implement request counting
        "model_info": inference_engine.get_model_info(),
        "uptime": "0 minutes"  # TODO: Implement uptime tracking
    }

@app.get("/api/v1/engines/status")
async def get_engines_status():
    """Get status of all available inference engines."""
    status = {
        "current_engine": type(inference_engine).__name__,
        "available_engines": {},
        "environment": {
            "USE_REAL_AI": USE_REAL_AI,
            "USE_CUSTOM_AI": USE_CUSTOM_AI
        }
    }
    
    # Check mock engine (always available)
    status["available_engines"]["mock"] = {
        "available": True,
        "status": "ready",
        "description": "Mock inference engine for development"
    }
    
    # Check real AI engine
    if REAL_AI_AVAILABLE:
        real_engine = RealInferenceEngine()
        status["available_engines"]["real_ai"] = {
            "available": True,
            "status": real_engine.get_status(),
            "description": "Real AI models (OpenAI, Ollama, HuggingFace)"
        }
    else:
        status["available_engines"]["real_ai"] = {
            "available": False,
            "status": "not_available",
            "description": "Real AI dependencies not installed"
        }
    
    # Check custom AI engine
    if CUSTOM_AI_AVAILABLE:
        custom_engine = CustomInferenceEngine()
        try:
            custom_initialized = await custom_engine.initialize()
            if custom_initialized:
                model_info = custom_engine.get_model_info()
                status["available_engines"]["custom_ai"] = {
                    "available": True,
                    "status": "ready",
                    "description": f"Custom fine-tuned models ({model_info['total_models']} models available)",
                    "models": model_info.get("models", [])
                }
                await custom_engine.cleanup()
            else:
                status["available_engines"]["custom_ai"] = {
                    "available": False,
                    "status": "no_models",
                    "description": "No trained custom models found"
                }
        except Exception as e:
            status["available_engines"]["custom_ai"] = {
                "available": False,
                "status": "error",
                "description": f"Custom AI initialization failed: {str(e)}"
            }
    else:
        status["available_engines"]["custom_ai"] = {
            "available": False,
            "status": "not_available",
            "description": "Custom AI engine not available"
        }
    
    return status

# WebSocket endpoint for real-time communication
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time code completion."""
    await connection_manager.connect(websocket)
    logger.info(f"WebSocket connection established. Active connections: {len(connection_manager.active_connections)}")
    
    try:
        while True:
            # Receive message from client
            data = await websocket.receive_text()
            message = json.loads(data)
            
            # Process different message types
            if message.get("type") == "completion_request":
                await handle_completion_request(websocket, message)
            elif message.get("type") == "chat_request":
                await handle_chat_request(websocket, message)
            elif message.get("type") == "ping":
                await websocket.send_text(json.dumps({"type": "pong", "timestamp": datetime.now().isoformat()}))
            else:
                await websocket.send_text(json.dumps({
                    "type": "error",
                    "message": f"Unknown message type: {message.get('type')}"
                }))
                
    except WebSocketDisconnect:
        connection_manager.disconnect(websocket)
        logger.info(f"WebSocket connection closed. Active connections: {len(connection_manager.active_connections)}")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        connection_manager.disconnect(websocket)

async def handle_completion_request(websocket: WebSocket, message: Dict[str, Any]):
    """Handle code completion request via WebSocket."""
    try:
        # Extract request data
        code = message.get("code", "")
        language = message.get("language", "python")
        max_suggestions = message.get("max_suggestions", 3)
        cursor_position = message.get("cursor_position")
        
        # Generate completions
        result = await inference_engine.generate_completions(
            code=code,
            language=language,
            max_suggestions=max_suggestions,
            cursor_position=cursor_position
        )
        
        # Send response
        response = {
            "type": "completion_response",
            "request_id": message.get("request_id"),
            "suggestions": result["suggestions"],
            "confidence_scores": result["confidence_scores"],
            "model_used": result["model_used"]
        }
        
        await websocket.send_text(json.dumps(response))
        
    except Exception as e:
        logger.error(f"Error handling completion request: {e}")
        error_response = {
            "type": "error",
            "request_id": message.get("request_id"),
            "message": str(e)
        }
        await websocket.send_text(json.dumps(error_response))

async def handle_chat_request(websocket: WebSocket, message: Dict[str, Any]):
    """Handle chat request via WebSocket."""
    try:
        # Extract request data
        user_message = message.get("message", "")
        code_context = message.get("code_context")
        language = message.get("language")
        
        # Process chat
        result = await inference_engine.process_chat(
            message=user_message,
            code_context=code_context,
            language=language
        )
        
        # Send response
        response = {
            "type": "chat_response",
            "request_id": message.get("request_id"),
            "response": result["response"],
            "code_suggestions": result.get("code_suggestions")
        }
        
        await websocket.send_text(json.dumps(response))
        
    except Exception as e:
        logger.error(f"Error handling chat request: {e}")
        error_response = {
            "type": "error",
            "request_id": message.get("request_id"),
            "message": str(e)
        }
        await websocket.send_text(json.dumps(error_response))

# Startup and shutdown events
@app.on_event("startup")
async def startup_event():
    """Initialize services on startup."""
    logger.info("Starting CopilotMini API server...")
    
    # Initialize inference engine
    await inference_engine.initialize()
    logger.info("Inference engine initialized")
    
    logger.info(f"Server started on {API_CONFIG['host']}:{API_CONFIG['port']}")

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    logger.info("Shutting down CopilotMini API server...")
    
    # Cleanup inference engine
    await inference_engine.cleanup()
    
    # Close all WebSocket connections
    await connection_manager.disconnect_all()
    
    logger.info("Server shutdown complete")

if __name__ == "__main__":
    uvicorn.run(
        "backend.app:app",
        host=API_CONFIG["host"],
        port=API_CONFIG["port"],
        reload=True,
        log_level="info"
    ) 