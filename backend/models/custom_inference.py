"""
Custom inference engine for fine-tuned CopilotMini models.
Supports loading and using trained CodeParrot, CodeT5, and CodeGen models.
"""

import os
import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
import asyncio
import time

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    T5ForConditionalGeneration,
    RobertaTokenizer,
    GPT2Tokenizer
)
from loguru import logger

class CustomInferenceEngine:
    """Custom inference engine for the fine-tuned CodeGen model."""
    
    def __init__(self):
        self.models = {}
        self.tokenizers = {}
        self.model_configs = {}
        self.default_model = None
        self.is_initialized = False
        
        # Model type mappings
        self.model_types = {
            "codegen": {
                "model_class": AutoModelForCausalLM,
                "tokenizer_class": AutoTokenizer,
                "type": "causal_lm"
            }
        }
        
    async def initialize(self) -> bool:
        """Initialize the custom inference engine by loading the CodeGen model."""
        logger.info("🤖 Initializing custom inference engine...")
        
        try:
            # Discover the trained CodeGen model
            model_info = self._discover_trained_model()
            
            if not model_info:
                logger.warning("⚠️ No trained CodeGen model found, cannot initialize custom inference")
                return False
            
            # Load the model
            if await self._load_model(model_info):
                self.default_model = "codegen"
                self.is_initialized = True
                logger.info("✅ Custom inference engine initialized with CodeGen model")
                logger.info(f"Default model: {self.default_model}")
                return True
            else:
                logger.error("❌ Failed to load the custom CodeGen model")
                return False
                
        except Exception as e:
            logger.error(f"Failed to initialize custom inference engine: {e}")
            return False
    
    def _discover_trained_model(self) -> Optional[Dict[str, Any]]:
        """Discover the available trained CodeGen model."""
        checkpoints_dir = Path("training/checkpoints")
        model_type = "codegen"
        
        model_path = checkpoints_dir / model_type
        
        if model_path.exists() and any(model_path.iterdir()):
            logger.info(f"Found trained model: {model_type} at {model_path}")
            return {
                "model_type": model_type,
                "path": str(model_path)
            }
        return None
    
    async def _load_model(self, model_info: Dict[str, Any]) -> bool:
        model_type = model_info["model_type"]
        model_path = model_info["path"]

        logger.info(f"Loading {model_type} model from {model_path}")

        try:
            model_config = self.model_types[model_type]
            tokenizer_class = model_config["tokenizer_class"]
            tokenizer = tokenizer_class.from_pretrained(model_path)

            model_class = model_config["model_class"]
            print("About to load model...")
            model = model_class.from_pretrained(
                model_path,
                torch_dtype=torch.float32,
                device_map=None
            )
            print("Model loaded!")

            self.models[model_type] = model
            self.tokenizers[model_type] = tokenizer
            self.model_configs[model_type] = model_config

            logger.info(f"✅ Successfully loaded {model_type} model")
            return True

        except Exception as e:
            logger.error(f"Failed to load {model_type} model: {e}")
            return False
    
    async def generate_completions(
        self,
        code: str,
        language: str,
        max_suggestions: int = 3,
        cursor_position: Optional[int] = None,
        model_type: Optional[str] = None
    ) -> Dict[str, Any]:
        """Generate code completions using the custom CodeGen model."""
        
        if not self.is_initialized:
            raise RuntimeError("Custom inference engine not initialized")
        
        selected_model = "codegen"
        
        start_time = time.time()
        
        try:
            model = self.models[selected_model]
            tokenizer = self.tokenizers[selected_model]
            
            print(f"[DEBUG] Prompt sent to model:\n{code}\n---")
            suggestions = await self._generate_causal_completions(
                model, tokenizer, code, max_suggestions
            )
            print(f"[DEBUG] Model suggestions: {suggestions}")
            
            processing_time = (time.time() - start_time) * 1000  # Convert to ms
            
            # Calculate confidence scores
            confidence_scores = [0.9 - (i * 0.1) for i in range(len(suggestions))]
            
            return {
                "suggestions": suggestions,
                "confidence_scores": confidence_scores,
                "model_used": f"custom:{selected_model}",
                "language": language,
                "processing_time": processing_time
            }
            
        except Exception as e:
            logger.error(f"Error generating completions with {selected_model}: {e}")
            raise
    
    async def _generate_causal_completions(
        self,
        model,
        tokenizer,
        code: str,
        max_suggestions: int
    ) -> List[str]:
        """Generate completions using causal language models (CodeGen)."""
        
        model.eval()
        
        # Tokenize input
        inputs = tokenizer(code, return_tensors="pt")
        
        # Handle device placement for Apple Silicon
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}
            model = model.cuda()
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            inputs = {k: v.to('mps') for k, v in inputs.items()}
            model = model.to('mps')
        
        suggestions = []
        
        try:
            with torch.no_grad():
                # Try sampling first with safer parameters
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=30,
                    num_return_sequences=max_suggestions,
                    do_sample=True,
                    temperature=0.1,  # Lower temperature for stability
                    top_p=0.9,        # Lower top_p for stability
                    top_k=50,         # Add top_k for additional stability
                    pad_token_id=tokenizer.eos_token_id,
                    repetition_penalty=1.1,  # Prevent repetition
                    no_repeat_ngram_size=2   # Prevent repetition
                )
                
                # Decode and clean suggestions
                suggestions = [
                    tokenizer.decode(output, skip_special_tokens=True)
                    for output in outputs
                ]
                
                # Remove the original code from the suggestion
                suggestions = [s[len(code):].strip() for s in suggestions]
            print(f"[DEBUG] Raw model outputs: {suggestions}")
        except Exception as e:
            logger.warning(f"Sampling generation failed: {e}, falling back to greedy decoding")
            
            # Fallback to greedy decoding (more stable)
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=30,
                    num_return_sequences=1,
                    do_sample=False,  # Greedy decoding
                    pad_token_id=tokenizer.eos_token_id
                )
                
                # Decode and create variations
                base_suggestion = tokenizer.decode(outputs[0], skip_special_tokens=True)
                base_suggestion = base_suggestion[len(code):].strip()
                
                # Create simple variations for multiple suggestions
                suggestions = [base_suggestion]
                if len(base_suggestion) > 10:
                    # Add truncated versions
                    suggestions.append(base_suggestion[:len(base_suggestion)//2])
                    suggestions.append(base_suggestion[:len(base_suggestion)//3])
                else:
                    # Add simple variations
                    suggestions.extend([base_suggestion + "()", base_suggestion + ":"])
            print(f"[DEBUG] Greedy model outputs: {suggestions}")
        
        # Ensure we have the right number of suggestions
        while len(suggestions) < max_suggestions:
            suggestions.append("")  # Add empty suggestions if needed
        
        return suggestions[:max_suggestions]

    async def process_chat(
        self,
        message: str,
        code_context: Optional[str] = None,
        language: Optional[str] = None,
        model_type: Optional[str] = None
    ) -> Dict[str, Any]:
        """Process a chat message using the fine-tuned CodeGen model."""
        
        if not self.is_initialized:
            raise RuntimeError("Custom inference engine not initialized")
            
        selected_model = "codegen"
        
        start_time = time.time()
        
        try:
            model = self.models[selected_model]
            tokenizer = self.tokenizers[selected_model]
            
            response_text = await self._generate_chat_causal(
                model, tokenizer, message, code_context
            )
            
            processing_time = (time.time() - start_time) * 1000  # Convert to ms
            
            return {
                "response": response_text,
                "model_used": f"custom:{selected_model}",
                "processing_time": processing_time
            }
            
        except Exception as e:
            logger.error(f"Error processing chat with {selected_model}: {e}")
            raise

    async def _generate_chat_causal(
        self,
        model,
        tokenizer,
        message: str,
        code_context: Optional[str]
    ) -> str:
        """Generate a chat response using causal language models (CodeGen)."""
        
        prompt = f"User: {message}\n"
        if code_context:
            prompt += f"Code Context:\n```\n{code_context}\n```\n"
        prompt += "Assistant:"
        
        model.eval()
        
        inputs = tokenizer(prompt, return_tensors="pt")
        
        # Handle device placement for Apple Silicon
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}
            model = model.cuda()
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            inputs = {k: v.to('mps') for k, v in inputs.items()}
            model = model.to('mps')
            
        try:
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=100,  # Reduced for stability
                    num_return_sequences=1,
                    do_sample=True,
                    temperature=0.3,  # Lower temperature for stability
                    top_p=0.9,        # Lower top_p for stability
                    top_k=50,         # Add top_k for additional stability
                    pad_token_id=tokenizer.eos_token_id,
                    repetition_penalty=1.1,  # Prevent repetition
                    no_repeat_ngram_size=2   # Prevent repetition
                )
                
                response = tokenizer.decode(outputs[0], skip_special_tokens=True)
                
                # Clean up response
                response = response[len(prompt):].strip()
                
                return response
                
        except Exception as e:
            logger.warning(f"Chat generation failed: {e}, falling back to simple response")
            return f"I'm a fine-tuned code model. For '{message}', I'd recommend checking the documentation or trying a different approach."
            
    def get_status(self) -> str:
        """Get the status of the custom inference engine."""
        if self.is_initialized:
            return f"✅ Initialized and ready. Default model: {self.default_model}"
        return "❌ Not initialized."
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded models."""
        if not self.is_initialized:
            return {}
        
        return {
            "loaded_models": list(self.models.keys()),
            "default_model": self.default_model,
            "model_configs": {
                name: {"type": cfg["type"]} for name, cfg in self.model_configs.items()
            }
        }
    
    def get_available_models(self) -> List[str]:
        """Get a list of available model types."""
        return list(self.models.keys())
        
    def set_default_model(self, model_type: str) -> bool:
        """Set the default model for inference."""
        if model_type in self.models:
            self.default_model = model_type
            logger.info(f"Default model set to: {model_type}")
            return True
        logger.error(f"Cannot set default model, {model_type} not loaded.")
        return False
        
    async def cleanup(self):
        """Clean up resources."""
        self.models.clear()
        self.tokenizers.clear()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("Custom inference engine cleaned up.")

# Global instance
custom_inference_engine = CustomInferenceEngine() 