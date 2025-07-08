"""
Mock inference engine for testing CopilotMini API.
This simulates real model inference until we implement the actual models.
"""

import asyncio
import random
import time
from typing import List, Dict, Any, Optional
from loguru import logger

class MockInferenceEngine:
    """Mock inference engine that simulates code completion and chat responses."""
    
    def __init__(self):
        self.status = "initializing"
        self.model_name = "mock-codeparrot-v1"
        self.initialized = False
        
        # Mock completions database
        self.mock_completions = {
            "python": [
                "print('Hello, World!')",
                "def __init__(self):",
                "if __name__ == '__main__':",
                "return result",
                "except Exception as e:",
                "for i in range(len(items)):",
                "with open('file.txt', 'r') as f:",
                "import numpy as np",
                "class MyClass:",
                "async def process_data():"
            ],
            "javascript": [
                "console.log('Hello, World!');",
                "function() {",
                "const result = ",
                "return new Promise(",
                "} catch (error) {",
                "for (let i = 0; i < array.length; i++) {",
                "addEventListener('click', () => {",
                "import React from 'react';",
                "export default class",
                "async function fetchData()"
            ],
            "java": [
                "System.out.println(\"Hello, World!\");",
                "public class MyClass {",
                "public static void main(String[] args) {",
                "private static final",
                "} catch (Exception e) {",
                "for (int i = 0; i < array.length; i++) {",
                "new ArrayList<>()",
                "import java.util.*;",
                "public void method() {",
                "@Override"
            ],
            "default": [
                "// TODO: Implement this",
                "/* Comment */",
                "function main() {",
                "return 0;",
                "end",
                "class Example",
                "if condition:",
                "else:",
                "while loop:",
                "break;"
            ]
        }
        
        # Mock chat responses
        self.chat_responses = [
            "This code looks good! Here's what it does:",
            "I can help you with that. Consider this approach:",
            "There's a potential issue here. You might want to:",
            "Great question! The best practice would be to:",
            "I see what you're trying to do. A more efficient way would be:",
            "This pattern is commonly used for:",
            "You could optimize this by:",
            "Here's how you can fix this error:",
            "Consider using this alternative approach:",
            "This code follows good practices, but you could also:"
        ]
    
    async def initialize(self):
        """Initialize the mock inference engine."""
        logger.info("Initializing mock inference engine...")
        
        # Simulate initialization time
        await asyncio.sleep(0.5)
        
        self.status = "ready"
        self.initialized = True
        
        logger.info("Mock inference engine initialized successfully")
    
    async def cleanup(self):
        """Cleanup the inference engine."""
        logger.info("Cleaning up mock inference engine...")
        self.status = "shutdown"
        self.initialized = False
    
    def get_status(self) -> str:
        """Get the current status of the inference engine."""
        return self.status
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model."""
        return {
            "model_name": self.model_name,
            "model_type": "mock",
            "status": self.status,
            "capabilities": ["code_completion", "chat", "explanation"],
            "supported_languages": ["python", "javascript", "java", "typescript", "go", "rust"]
        }
    
    async def generate_completions(
        self,
        code: str,
        language: str,
        max_suggestions: int = 3,
        cursor_position: Optional[int] = None
    ) -> Dict[str, Any]:
        """Generate code completions for the given input."""
        if not self.initialized:
            raise RuntimeError("Inference engine not initialized")
        
        # Simulate processing time
        await asyncio.sleep(random.uniform(0.1, 0.3))
        
        # Get language-specific completions or default
        available_completions = self.mock_completions.get(language.lower(), self.mock_completions["default"])
        
        # Generate suggestions based on context
        suggestions = self._generate_contextual_suggestions(code, available_completions, max_suggestions)
        
        # Generate mock confidence scores
        confidence_scores = [random.uniform(0.7, 0.95) for _ in suggestions]
        confidence_scores.sort(reverse=True)  # Higher confidence first
        
        return {
            "suggestions": suggestions,
            "confidence_scores": confidence_scores,
            "model_used": self.model_name,
            "language": language,
            "processing_time": random.uniform(50, 200)  # Mock processing time in ms
        }
    
    def _generate_contextual_suggestions(self, code: str, available_completions: List[str], max_suggestions: int) -> List[str]:
        """Generate contextual suggestions based on the input code."""
        suggestions = []
        
        # Analyze the code context for better suggestions
        code_lower = code.lower().strip()
        last_line = code.split('\n')[-1] if code else ""
        
        # Context-aware suggestion logic
        if "def " in last_line or "function " in last_line:
            # Suggest function body starters
            function_suggestions = [comp for comp in available_completions if any(keyword in comp for keyword in ["return", "if", "for", "try"])]
            suggestions.extend(random.sample(function_suggestions, min(len(function_suggestions), max_suggestions)))
        
        elif "class " in last_line:
            # Suggest class body starters
            class_suggestions = [comp for comp in available_completions if any(keyword in comp for keyword in ["def", "__init__", "self"])]
            suggestions.extend(random.sample(class_suggestions, min(len(class_suggestions), max_suggestions)))
        
        elif "import " in last_line or "from " in last_line:
            # Suggest more imports
            import_suggestions = [comp for comp in available_completions if "import" in comp]
            suggestions.extend(random.sample(import_suggestions, min(len(import_suggestions), max_suggestions)))
        
        elif "print" in last_line or "console.log" in last_line:
            # Suggest similar output statements
            output_suggestions = [comp for comp in available_completions if any(keyword in comp for keyword in ["print", "console.log", "System.out"])]
            suggestions.extend(random.sample(output_suggestions, min(len(output_suggestions), max_suggestions)))
        
        else:
            # Default random suggestions
            suggestions.extend(random.sample(available_completions, min(len(available_completions), max_suggestions)))
        
        # Remove duplicates and limit to max_suggestions
        suggestions = list(dict.fromkeys(suggestions))[:max_suggestions]
        
        # If we don't have enough suggestions, fill with random ones
        while len(suggestions) < max_suggestions and len(suggestions) < len(available_completions):
            remaining = [comp for comp in available_completions if comp not in suggestions]
            if remaining:
                suggestions.append(random.choice(remaining))
        
        return suggestions
    
    async def process_chat(
        self,
        message: str,
        code_context: Optional[str] = None,
        language: Optional[str] = None
    ) -> Dict[str, Any]:
        """Process a chat message and generate a response."""
        if not self.initialized:
            raise RuntimeError("Inference engine not initialized")
        
        # Simulate processing time
        await asyncio.sleep(random.uniform(0.2, 0.5))
        
        # Generate contextual response
        response = self._generate_chat_response(message, code_context, language)
        
        # Sometimes include code suggestions
        code_suggestions = None
        if random.random() < 0.3 and language:  # 30% chance to include code suggestions
            completions = await self.generate_completions("", language, max_suggestions=2)
            code_suggestions = completions["suggestions"]
        
        return {
            "response": response,
            "code_suggestions": code_suggestions,
            "model_used": self.model_name,
            "context_used": bool(code_context),
            "language": language
        }
    
    def _generate_chat_response(self, message: str, code_context: Optional[str], language: Optional[str]) -> str:
        """Generate a contextual chat response."""
        message_lower = message.lower()
        
        # Context-aware responses
        if any(keyword in message_lower for keyword in ["error", "bug", "fix", "wrong"]):
            responses = [
                "I can help you debug this issue. Let me analyze the code...",
                "This looks like a common error. Here's what might be causing it:",
                "I see the problem! The issue is likely in this section:",
                "Let's fix this step by step. First, check if:",
            ]
        elif any(keyword in message_lower for keyword in ["explain", "what", "how", "why"]):
            responses = [
                "Let me explain how this works:",
                "This code does the following:",
                "The purpose of this function is to:",
                "Here's a breakdown of what's happening:",
            ]
        elif any(keyword in message_lower for keyword in ["optimize", "improve", "better", "performance"]):
            responses = [
                "Here are some ways to optimize this code:",
                "You can improve performance by:",
                "Consider these optimizations:",
                "A more efficient approach would be:",
            ]
        elif any(keyword in message_lower for keyword in ["refactor", "clean", "structure"]):
            responses = [
                "Here's how you could refactor this code:",
                "To improve code structure, consider:",
                "A cleaner approach would be:",
                "You could reorganize this by:",
            ]
        else:
            responses = self.chat_responses
        
        base_response = random.choice(responses)
        
        # Add context-specific details
        if code_context:
            base_response += f" Looking at your {language or 'code'}, I notice that..."
        elif language:
            base_response += f" In {language}, the best practice is..."
        
        return base_response
    
    async def explain_code(self, code: str, language: str) -> Dict[str, Any]:
        """Explain what a piece of code does."""
        if not self.initialized:
            raise RuntimeError("Inference engine not initialized")
        
        await asyncio.sleep(random.uniform(0.1, 0.3))
        
        # Generate explanation based on code patterns
        explanation = self._generate_code_explanation(code, language)
        
        return {
            "explanation": explanation,
            "language": language,
            "model_used": self.model_name,
            "confidence": random.uniform(0.8, 0.95)
        }
    
    def _generate_code_explanation(self, code: str, language: str) -> str:
        """Generate an explanation of the code."""
        explanations = [
            f"This {language} code defines a function that processes data.",
            f"This is a {language} class implementation with multiple methods.",
            f"This {language} code demonstrates error handling and input validation.",
            f"This is a {language} utility function for data manipulation.",
            f"This {language} code implements a common design pattern.",
            f"This is a {language} function that performs calculations and returns results.",
            f"This {language} code handles file operations and data processing.",
            f"This is a {language} implementation of an algorithm or data structure."
        ]
        
        return random.choice(explanations) 