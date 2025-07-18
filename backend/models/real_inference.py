import os
import openai
from typing import List, Dict, Any, Optional
import asyncio
import time
from loguru import logger

class RealInferenceEngine:
    """Real AI inference engine using OpenAI GPT-4 or local models"""
    
    def __init__(self):
        self.client = None
        self.model_type = "openai"  # or "ollama", "huggingface"
        self.model_name = "gpt-4"
        self.is_initialized = False
        
    async def initialize(self) -> bool:
        """Initialize the real AI inference engine"""
        try:
            # Try OpenAI first
            if await self._try_openai():
                self.model_type = "openai"
                logger.info("✅ OpenAI GPT-4 initialized successfully")
                self.is_initialized = True
                return True
            
            # Fallback to Ollama
            if await self._try_ollama():
                self.model_type = "ollama"
                logger.info("✅ Ollama local model initialized successfully")
                self.is_initialized = True
                return True
            
            # Fallback to HuggingFace
            if await self._try_huggingface():
                self.model_type = "huggingface"
                logger.info("✅ HuggingFace model initialized successfully")
                self.is_initialized = True
                return True
            
            logger.warning("⚠️ No real AI models available, falling back to mock")
            return False
            
        except Exception as e:
            logger.error(f"Failed to initialize real inference engine: {e}")
            return False
    
    async def _try_openai(self) -> bool:
        """Try to initialize OpenAI"""
        try:
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                logger.info("No OPENAI_API_KEY found, skipping OpenAI")
                return False
            
            import openai
            self.client = openai.AsyncOpenAI(api_key=api_key)
            
            # Test the connection
            response = await self.client.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": "test"}],
                max_tokens=5
            )
            return True
            
        except ImportError:
            logger.info("OpenAI package not installed: pip install openai")
            return False
        except Exception as e:
            logger.info(f"OpenAI initialization failed: {e}")
            return False
    
    async def _try_ollama(self) -> bool:
        """Try to initialize Ollama local models"""
        try:
            import aiohttp
            
            # Check if Ollama is running
            async with aiohttp.ClientSession() as session:
                async with session.get("http://localhost:11434/api/tags") as response:
                    if response.status == 200:
                        models = await response.json()
                        available_models = [m["name"] for m in models.get("models", [])]
                        
                        # Prefer code-specific models
                        preferred_models = ["codellama:7b", "starcoder:3b", "codellama:13b", "llama2:7b"]
                        for model in preferred_models:
                            if model in available_models:
                                self.model_name = model
                                logger.info(f"Using Ollama model: {model}")
                                return True
                        
                        if available_models:
                            self.model_name = available_models[0]
                            logger.info(f"Using Ollama model: {self.model_name}")
                            return True
            
            return False
            
        except ImportError:
            logger.info("aiohttp package needed for Ollama: pip install aiohttp")
            return False
        except Exception as e:
            logger.info(f"Ollama not available: {e}")
            return False
    
    async def _try_huggingface(self) -> bool:
        """Try to initialize HuggingFace models"""
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            import torch
            
            # Use a small, fast model for MVP
            model_name = "microsoft/DialoGPT-small"  # Fast model for testing
            
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForCausalLM.from_pretrained(model_name)
            self.model_name = model_name
            
            # Add padding token if missing
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            logger.info(f"Loaded HuggingFace model: {model_name}")
            return True
            
        except ImportError:
            logger.info("HuggingFace transformers not installed: pip install transformers torch")
            return False
        except Exception as e:
            logger.info(f"HuggingFace model loading failed: {e}")
            return False
    
    async def generate_completion(
        self,
        code: str,
        language: str,
        max_length: int = 100,
        num_suggestions: int = 3
    ) -> List[Dict[str, Any]]:
        """Generate code completions using real AI models"""
        
        if not self.is_initialized:
            return await self._fallback_mock_completion(code, language, num_suggestions)
        
        try:
            start_time = time.time()
            
            if self.model_type == "openai":
                suggestions = await self._openai_completion(code, language, num_suggestions)
            elif self.model_type == "ollama":
                suggestions = await self._ollama_completion(code, language, num_suggestions)
            elif self.model_type == "huggingface":
                suggestions = await self._huggingface_completion(code, language, num_suggestions)
            else:
                suggestions = await self._fallback_mock_completion(code, language, num_suggestions)
            
            processing_time = time.time() - start_time
            
            # Add metadata to suggestions
            for suggestion in suggestions:
                suggestion["processing_time_ms"] = round(processing_time * 1000, 2)
                suggestion["model_used"] = f"{self.model_type}:{self.model_name}"
            
            return suggestions[:num_suggestions]
            
        except Exception as e:
            logger.error(f"Real inference failed: {e}")
            return await self._fallback_mock_completion(code, language, num_suggestions)
    
    async def _openai_completion(self, code: str, language: str, num_suggestions: int) -> List[Dict[str, Any]]:
        """Generate completions using OpenAI GPT-4. The 'code' argument may already include RAG context prepended."""
        
        # Create a focused prompt for code completion
        prompt = f"""You are an expert {language} programmer. Complete the following code with {num_suggestions} different suggestions.
Only return the completion code without explanations.

Code to complete:
```{language}
{code}
```

Complete this code with the most likely continuation:"""

        response = await self.client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a code completion AI. Only return code completions, no explanations."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=150,
            temperature=0.3,
            n=num_suggestions
        )
        
        suggestions = []
        for i, choice in enumerate(response.choices):
            completion = choice.message.content.strip()
            # Clean up the completion
            if "```" in completion:
                completion = completion.split("```")[1].strip()
                if completion.startswith(language):
                    completion = completion[len(language):].strip()
            
            suggestions.append({
                "text": completion,
                "confidence": 0.85 + (i * 0.05),  # Decreasing confidence
                "source": "gpt-4"
            })
        
        return suggestions
    
    async def _ollama_completion(self, code: str, language: str, num_suggestions: int) -> List[Dict[str, Any]]:
        """Generate completions using Ollama local models. The 'code' argument may already include RAG context prepended."""
        import aiohttp
        
        prompt = f"Complete this {language} code:\n{code}"
        
        suggestions = []
        async with aiohttp.ClientSession() as session:
            for i in range(num_suggestions):
                payload = {
                    "model": self.model_name,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.3 + (i * 0.2),  # Vary temperature for diversity
                        "top_p": 0.9,
                        "max_tokens": 100
                    }
                }
                
                async with session.post("http://localhost:11434/api/generate", json=payload) as response:
                    if response.status == 200:
                        result = await response.json()
                        completion = result.get("response", "").strip()
                        
                        suggestions.append({
                            "text": completion,
                            "confidence": 0.80 - (i * 0.1),
                            "source": self.model_name
                        })
        
        return suggestions
    
    async def _huggingface_completion(self, code: str, language: str, num_suggestions: int) -> List[Dict[str, Any]]:
        """Generate completions using HuggingFace models. The 'code' argument may already include RAG context prepended."""
        import torch
        
        # Prepare input
        prompt = f"# {language} code completion\n{code}"
        inputs = self.tokenizer.encode(prompt, return_tensors="pt")
        
        suggestions = []
        for i in range(num_suggestions):
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs,
                    max_length=inputs.shape[1] + 50,
                    temperature=0.7 + (i * 0.2),
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            # Decode completion
            completion = self.tokenizer.decode(outputs[0][inputs.shape[1]:], skip_special_tokens=True)
            
            suggestions.append({
                "text": completion.strip(),
                "confidence": 0.75 - (i * 0.1),
                "source": self.model_name
            })
        
        return suggestions
    
    async def _fallback_mock_completion(self, code: str, language: str, num_suggestions: int) -> List[Dict[str, Any]]:
        """Fallback to mock completions if real AI fails"""
        from .inference import MockInferenceEngine
        
        mock_engine = MockInferenceEngine()
        await mock_engine.initialize()
        return await mock_engine.generate_completion(code, language, num_suggestions=num_suggestions)
    
    async def generate_chat_response(self, message: str, context: Optional[str] = None) -> Dict[str, Any]:
        """Generate chat responses using real AI models"""
        
        if not self.is_initialized:
            return await self._fallback_mock_chat(message, context)
        
        try:
            start_time = time.time()
            
            if self.model_type == "openai":
                response = await self._openai_chat(message, context)
            elif self.model_type == "ollama":
                response = await self._ollama_chat(message, context)
            elif self.model_type == "huggingface":
                response = await self._huggingface_chat(message, context)
            else:
                response = await self._fallback_mock_chat(message, context)
            
            processing_time = time.time() - start_time
            response["processing_time_ms"] = round(processing_time * 1000, 2)
            response["model_used"] = f"{self.model_type}:{self.model_name}"
            
            return response
            
        except Exception as e:
            logger.error(f"Real chat failed: {e}")
            return await self._fallback_mock_chat(message, context)
    
    async def _openai_chat(self, message: str, context: Optional[str] = None) -> Dict[str, Any]:
        """Generate chat response using OpenAI"""
        
        system_prompt = """You are CopilotMini, an expert programming assistant. Help users with:
- Code completion and suggestions
- Debugging and error fixing
- Code explanation and documentation
- Best practices and optimization
- Architecture and design patterns

Be concise, practical, and provide code examples when helpful."""

        messages = [{"role": "system", "content": system_prompt}]
        
        if context:
            messages.append({"role": "system", "content": f"Code context:\n```\n{context}\n```"})
        
        messages.append({"role": "user", "content": message})
        
        response = await self.client.chat.completions.create(
            model="gpt-4",
            messages=messages,
            max_tokens=500,
            temperature=0.7
        )
        
        return {
            "response": response.choices[0].message.content,
            "confidence": 0.90,
            "source": "gpt-4"
        }
    
    async def _ollama_chat(self, message: str, context: Optional[str] = None) -> Dict[str, Any]:
        """Generate chat response using Ollama"""
        import aiohttp
        
        prompt = f"You are a helpful programming assistant.\n\nUser: {message}\nAssistant:"
        if context:
            prompt = f"Code context:\n{context}\n\n{prompt}"
        
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.7, "max_tokens": 300}
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post("http://localhost:11434/api/generate", json=payload) as response:
                if response.status == 200:
                    result = await response.json()
                    return {
                        "response": result.get("response", "Sorry, I couldn't generate a response."),
                        "confidence": 0.80,
                        "source": self.model_name
                    }
        
        return {"response": "Ollama service unavailable", "confidence": 0.0, "source": "error"}
    
    async def _huggingface_chat(self, message: str, context: Optional[str] = None) -> Dict[str, Any]:
        """Generate chat response using HuggingFace models"""
        # For now, return a simple response since DialoGPT is not ideal for code
        return {
            "response": f"I'm a basic HuggingFace model. For the question '{message}', I'd recommend checking the documentation or using a more advanced model.",
            "confidence": 0.60,
            "source": self.model_name
        }
    
    async def _fallback_mock_chat(self, message: str, context: Optional[str] = None) -> Dict[str, Any]:
        """Fallback to mock chat if real AI fails"""
        from .inference import MockInferenceEngine
        
        mock_engine = MockInferenceEngine()
        await mock_engine.initialize()
        return await mock_engine.generate_chat_response(message, context)

    def get_status(self) -> str:
        """Get the current status of the inference engine."""
        if self.is_initialized:
            return "ready"
        else:
            return "initializing"
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model."""
        return {
            "model_name": self.model_name,
            "model_type": self.model_type,
            "status": self.get_status(),
            "capabilities": ["code_completion", "chat", "explanation"],
            "supported_languages": ["python", "javascript", "java", "typescript", "go", "rust", "c", "cpp", "php", "ruby"]
        }
    
    async def cleanup(self):
        """Cleanup the inference engine."""
        logger.info("Cleaning up real inference engine...")
        self.is_initialized = False
    
    async def generate_completions(
        self,
        code: str,
        language: str,
        max_suggestions: int = 3,
        cursor_position: Optional[int] = None
    ) -> Dict[str, Any]:
        """Generate code completions for the given input (compatible with mock engine)."""
        
        # Call the generate_completion method and adapt the response
        suggestions_data = await self.generate_completion(code, language, num_suggestions=max_suggestions)
        
        # Extract just the text suggestions and confidence scores
        suggestions = [s["text"] for s in suggestions_data]
        confidence_scores = [s["confidence"] for s in suggestions_data]
        
        return {
            "suggestions": suggestions,
            "confidence_scores": confidence_scores,
            "model_used": f"{self.model_type}:{self.model_name}",
            "language": language,
            "processing_time": suggestions_data[0].get("processing_time_ms", 100) if suggestions_data else 100
        }
    
    async def process_chat(
        self,
        message: str,
        code_context: Optional[str] = None,
        language: Optional[str] = None
    ) -> Dict[str, Any]:
        """Process a chat message and generate a response (compatible with mock engine)."""
        
        # Call the generate_chat_response method and adapt the response
        response_data = await self.generate_chat_response(message, code_context)
        
        return {
            "response": response_data["response"],
            "code_suggestions": None,  # Could be enhanced later
            "model_used": f"{self.model_type}:{self.model_name}",
            "processing_time": response_data.get("processing_time_ms", 100)
        }

# Global instance
real_inference_engine = RealInferenceEngine() 