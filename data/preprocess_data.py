"""
Data preprocessing script for CopilotMini.
Tokenizes and prepares downloaded code datasets for model training.
"""

import os
import json
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import re
from collections import defaultdict
import random

import pandas as pd
from tqdm import tqdm
from loguru import logger
from transformers import (
    AutoTokenizer, 
    GPT2Tokenizer, 
    CodeGenTokenizer,
    T5Tokenizer
)
from datasets import Dataset, DatasetDict
import ast
import tree_sitter
from tree_sitter import Language, Parser
import pygments
from pygments.lexers import get_lexer_by_name
from pygments.util import ClassNotFound

# Add project root to path
import sys
sys.path.append(str(Path(__file__).parent.parent))
from config import DATA_CONFIG, MODEL_CONFIG

class CodePreprocessor:
    """Preprocess code datasets for training."""
    
    def __init__(self, 
                 raw_data_dir: Path = None, 
                 processed_data_dir: Path = None,
                 model_name: str = "codeparrot"):
        self.raw_data_dir = raw_data_dir or DATA_CONFIG["raw_data_dir"]
        self.processed_data_dir = processed_data_dir or DATA_CONFIG["processed_data_dir"]
        self.processed_data_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize tokenizer based on model choice
        self.tokenizer = self._get_tokenizer(model_name)
        self.model_name = model_name
        
        # Language file extensions
        self.language_extensions = {
            "python": [".py"],
            "javascript": [".js", ".jsx"],
            "typescript": [".ts", ".tsx"],
            "java": [".java"],
            "cpp": [".cpp", ".cxx", ".cc", ".hpp"],
            "c": [".c", ".h"],
            "go": [".go"],
            "rust": [".rs"],
            "php": [".php"],
            "ruby": [".rb"]
        }
        
        # Code quality patterns
        self.quality_patterns = {
            "has_function": r'(def |function |class |public |private |protected )',
            "has_docstring": r'(""".*?"""|\'\'\'.*?\'\'\'|//.*|/\*.*?\*/)',
            "has_imports": r'(import |from |#include |require\(|use )',
            "minimal_complexity": r'[{}();]',  # Basic syntax elements
        }
    
    def _get_tokenizer(self, model_name: str):
        """Initialize appropriate tokenizer based on model."""
        tokenizers = {
            "codeparrot": "codeparrot/codeparrot-small",
            "codet5": "Salesforce/codet5-base", 
            "codegen": "Salesforce/codegen-350M-mono",
            "gpt2": "gpt2"
        }
        
        try:
            tokenizer_path = tokenizers.get(model_name, tokenizers["codeparrot"])
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
            
            # Add special tokens for code
            special_tokens = {
                "pad_token": "<|pad|>",
                "eos_token": "<|endoftext|>",
                "bos_token": "<|startoftext|>",
                "unk_token": "<|unk|>"
            }
            
            # Add tokens that don't exist
            for token_type, token in special_tokens.items():
                if getattr(tokenizer, token_type) is None:
                    setattr(tokenizer, token_type, token)
            
            # Add code-specific tokens
            code_tokens = ["<|code|>", "<|comment|>", "<|function|>", "<|class|>"]
            tokenizer.add_tokens(code_tokens)
            
            logger.info(f"Initialized {model_name} tokenizer with vocab size: {len(tokenizer)}")
            return tokenizer
            
        except Exception as e:
            logger.error(f"Error loading tokenizer: {e}")
            # Fallback to GPT2
            return GPT2Tokenizer.from_pretrained("gpt2")
    
    def detect_language(self, code: str, filename: str = "") -> str:
        """Detect programming language from code and filename."""
        # Try filename extension first
        if filename:
            ext = Path(filename).suffix.lower()
            for lang, extensions in self.language_extensions.items():
                if ext in extensions:
                    return lang
        
        # Use pygments for language detection
        try:
            lexer = get_lexer_by_name("text")  # Default
            
            # Simple heuristics
            if "def " in code and "import " in code:
                return "python"
            elif "function " in code and ("var " in code or "const " in code):
                return "javascript"
            elif "public class " in code and "static void main" in code:
                return "java"
            elif "#include" in code and "int main" in code:
                return "c"
            
            return "unknown"
            
        except Exception:
            return "unknown"
    
    def assess_code_quality(self, code: str) -> Dict[str, Any]:
        """Assess code quality using various metrics."""
        quality_score = 0
        metrics = {}
        
        # Length checks
        lines = code.split('\n')
        line_count = len(lines)
        char_count = len(code)
        
        metrics["line_count"] = line_count
        metrics["char_count"] = char_count
        metrics["avg_line_length"] = char_count / max(line_count, 1)
        
        # Quality patterns
        for pattern_name, pattern in self.quality_patterns.items():
            if re.search(pattern, code, re.MULTILINE | re.DOTALL):
                quality_score += 1
                metrics[pattern_name] = True
            else:
                metrics[pattern_name] = False
        
        # Syntax validity (basic check)
        try:
            # For Python, try parsing
            if "def " in code or "class " in code:
                ast.parse(code)
                quality_score += 2
                metrics["valid_syntax"] = True
        except:
            metrics["valid_syntax"] = False
        
        # Calculate final score
        max_score = len(self.quality_patterns) + 2  # +2 for syntax
        metrics["quality_score"] = quality_score / max_score
        
        return metrics
    
    def filter_code(self, code: str, language: str) -> bool:
        """Filter code based on quality and content criteria."""
        # Basic length filter
        lines = code.split('\n')
        if (len(lines) < DATA_CONFIG["min_lines"] or 
            len(lines) > DATA_CONFIG["max_lines"]):
            return False
        
        # Size filter
        if len(code) > DATA_CONFIG["max_file_size"]:
            return False
        
        # Quality filter
        quality = self.assess_code_quality(code)
        if quality["quality_score"] < 0.3:  # Minimum quality threshold
            return False
        
        # Language-specific filters
        if language == "python":
            # Must have some Python-specific syntax
            if not any(keyword in code for keyword in ["def ", "class ", "import ", "if __name__"]):
                return False
        
        elif language == "javascript":
            # Must have some JS-specific syntax
            if not any(keyword in code for keyword in ["function", "const ", "var ", "let "]):
                return False
        
        return True
    
    def extract_code_segments(self, code: str, language: str) -> List[Dict[str, Any]]:
        """Extract meaningful code segments (functions, classes, etc.)."""
        segments = []
        
        if language == "python":
            segments.extend(self._extract_python_segments(code))
        elif language in ["javascript", "typescript"]:
            segments.extend(self._extract_js_segments(code))
        else:
            # Fallback: split by functions/classes
            segments.extend(self._extract_generic_segments(code))
        
        return segments
    
    def _extract_python_segments(self, code: str) -> List[Dict[str, Any]]:
        """Extract Python functions and classes."""
        segments = []
        
        try:
            tree = ast.parse(code)
            
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    # Extract function
                    start_line = node.lineno - 1
                    end_line = node.end_lineno if hasattr(node, 'end_lineno') else start_line + 10
                    
                    lines = code.split('\n')
                    func_code = '\n'.join(lines[start_line:end_line])
                    
                    segments.append({
                        "type": "function",
                        "name": node.name,
                        "code": func_code,
                        "start_line": start_line,
                        "end_line": end_line
                    })
                
                elif isinstance(node, ast.ClassDef):
                    # Extract class
                    start_line = node.lineno - 1
                    end_line = node.end_lineno if hasattr(node, 'end_lineno') else start_line + 20
                    
                    lines = code.split('\n')
                    class_code = '\n'.join(lines[start_line:end_line])
                    
                    segments.append({
                        "type": "class", 
                        "name": node.name,
                        "code": class_code,
                        "start_line": start_line,
                        "end_line": end_line
                    })
                    
        except Exception as e:
            logger.debug(f"Error parsing Python code: {e}")
            # Fallback to whole code
            segments.append({
                "type": "file",
                "name": "main",
                "code": code,
                "start_line": 0,
                "end_line": len(code.split('\n'))
            })
        
        return segments
    
    def _extract_js_segments(self, code: str) -> List[Dict[str, Any]]:
        """Extract JavaScript functions and classes."""
        segments = []
        
        # Simple regex-based extraction (could be improved with proper parser)
        function_pattern = r'(?:function\s+(\w+)|const\s+(\w+)\s*=\s*(?:function|\([^)]*\)\s*=>))'
        class_pattern = r'class\s+(\w+)'
        
        lines = code.split('\n')
        
        for i, line in enumerate(lines):
            # Function detection
            func_match = re.search(function_pattern, line)
            if func_match:
                name = func_match.group(1) or func_match.group(2)
                # Extract surrounding context
                start = max(0, i)
                end = min(len(lines), i + 20)  # Approximate function length
                
                func_code = '\n'.join(lines[start:end])
                segments.append({
                    "type": "function",
                    "name": name,
                    "code": func_code,
                    "start_line": start,
                    "end_line": end
                })
            
            # Class detection
            class_match = re.search(class_pattern, line)
            if class_match:
                name = class_match.group(1)
                start = max(0, i)
                end = min(len(lines), i + 30)  # Approximate class length
                
                class_code = '\n'.join(lines[start:end])
                segments.append({
                    "type": "class",
                    "name": name,
                    "code": class_code,
                    "start_line": start,
                    "end_line": end
                })
        
        if not segments:
            # No segments found, use whole file
            segments.append({
                "type": "file",
                "name": "main",
                "code": code,
                "start_line": 0,
                "end_line": len(lines)
            })
        
        return segments
    
    def _extract_generic_segments(self, code: str) -> List[Dict[str, Any]]:
        """Generic segment extraction for unsupported languages."""
        # Split by common patterns
        lines = code.split('\n')
        
        # Look for function-like patterns
        function_indicators = [
            r'^\s*(def|function|public|private|protected|static)\s+',
            r'^\s*(class|struct|interface)\s+',
        ]
        
        segments = []
        current_segment = []
        current_start = 0
        
        for i, line in enumerate(lines):
            is_new_segment = any(re.match(pattern, line) for pattern in function_indicators)
            
            if is_new_segment and current_segment:
                # Save previous segment
                segment_code = '\n'.join(current_segment)
                segments.append({
                    "type": "segment",
                    "name": f"segment_{len(segments)}",
                    "code": segment_code,
                    "start_line": current_start,
                    "end_line": i
                })
                current_segment = [line]
                current_start = i
            else:
                current_segment.append(line)
        
        # Add final segment
        if current_segment:
            segment_code = '\n'.join(current_segment)
            segments.append({
                "type": "segment",
                "name": f"segment_{len(segments)}",
                "code": segment_code,
                "start_line": current_start,
                "end_line": len(lines)
            })
        
        return segments
    
    def tokenize_code(self, code: str, max_length: int = None) -> Dict[str, Any]:
        """Tokenize code using the configured tokenizer."""
        max_length = max_length or MODEL_CONFIG["max_length"]
        
        # Add special tokens
        formatted_code = f"<|code|>{code}<|endoftext|>"
        
        # Tokenize
        encoding = self.tokenizer(
            formatted_code,
            max_length=max_length,
            truncation=True,
            padding=False,
            return_tensors=None
        )
        
        return {
            "input_ids": encoding["input_ids"],
            "attention_mask": encoding["attention_mask"],
            "token_count": len(encoding["input_ids"]),
            "truncated": len(self.tokenizer.encode(formatted_code)) > max_length
        }
    
    def process_jsonl_file(self, input_file: Path, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Process a single JSONL data file."""
        logger.info(f"Processing file: {input_file}...")
        
        all_examples = []
        
        with open(input_file, 'r', encoding='utf-8') as f:
            for i, line in tqdm(enumerate(f), desc=f"Parsing {input_file.name}"):
                if limit and i >= limit:
                    logger.info(f"Reached limit of {limit} lines for {input_file.name}.")
                    break
                try:
                    row = json.loads(line)
                    code = row.get("content") or row.get("code")
                    
                    if not code or not isinstance(code, str):
                        continue
                        
                    language = self.detect_language(code, filename=row.get("path", ""))
                    
                    if self.filter_code(code, language):
                        tokenized_code = self.tokenize_code(code)
                        all_examples.append({
                            "text": code,
                            "language": language,
                            **tokenized_code
                        })
                except json.JSONDecodeError:
                    logger.warning(f"Skipping malformed JSON line in {input_file}")
                except Exception as e:
                    logger.warning(f"Error processing line: {e}")
                    
        return all_examples
        
    def create_training_splits(self, examples: List[Dict[str, Any]], 
                             train_ratio: float = 0.8,
                             val_ratio: float = 0.1) -> DatasetDict:
        
        # Shuffle data
        random.shuffle(examples)
        
        # Split into train, validation, test
        train_end = int(len(examples) * train_ratio)
        val_end = train_end + int(len(examples) * val_ratio)
        
        train_examples = examples[:train_end]
        val_examples = examples[train_end:val_end]
        test_examples = examples[val_end:]
        
        logger.info(f"Train samples: {len(train_examples)}, Validation: {len(val_examples)}, Test: {len(test_examples)}")
        
        # Create Hugging Face Dataset objects
        train_dataset = Dataset.from_pandas(pd.DataFrame(train_examples))
        val_dataset = Dataset.from_pandas(pd.DataFrame(val_examples))
        test_dataset = Dataset.from_pandas(pd.DataFrame(test_examples))
        
        return DatasetDict({
            "train": train_dataset,
            "validation": val_dataset,
            "test": test_dataset
        })
        
    def process_all_data(self, limit: Optional[int] = None) -> DatasetDict:
        """Process all raw data files."""
        logger.info("Starting data processing...")
        
        all_examples = []
        files_to_process = list(self.raw_data_dir.rglob("*.jsonl"))
        
        if not files_to_process:
            logger.warning(f"No .jsonl files found in {self.raw_data_dir}")
            return None
            
        for file_path in files_to_process:
            examples = self.process_jsonl_file(file_path, limit)
            all_examples.extend(examples)
            if limit and len(all_examples) >= limit:
                all_examples = all_examples[:limit]
                logger.info(f"Total processing limit of {limit} reached.")
                break

        if not all_examples:
            logger.error("No valid examples could be processed.")
            return None
            
        # Create train/val/test splits
        dataset_dict = self.create_training_splits(all_examples)
        
        # Save to disk
        output_path = self.processed_data_dir / f"processed_dataset_{self.model_name}"
        try:
            dataset_dict.save_to_disk(output_path)
            logger.info(f"✅ Processed data saved to: {output_path}")
        except Exception as e:
            logger.error(f"Failed to save dataset: {e}")
        
        return dataset_dict


def main():
    """Main function to run preprocessing."""
    parser = argparse.ArgumentParser(description="CopilotMini Data Preprocessing")
    parser.add_argument(
        "--model", 
        type=str, 
        default="codegen",
        choices=["codeparrot", "codet5", "codegen"],
        help="Model type to preprocess data for."
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit the number of lines to process for a quick test."
    )
    args = parser.parse_args()
    
    logger.info(f"Starting preprocessing for model: {args.model}")

    # Setup directories
    raw_data_dir = Path(DATA_CONFIG["raw_data_dir"])
    processed_data_dir = Path(DATA_CONFIG["processed_data_dir"])
    
    # Check if raw data exists. If not, stop and warn the user.
    if not any(raw_data_dir.iterdir()):
        logger.error(f"Raw data not found in {raw_data_dir}.")
        logger.error("Please run the `download_dataset.py` script first to download the data.")
        return

    preprocessor = CodePreprocessor(
        raw_data_dir=raw_data_dir,
        processed_data_dir=processed_data_dir,
        model_name=args.model
    )
    
    dataset = preprocessor.process_all_data(limit=args.limit)
    
    if dataset:
        logger.info("✅ Data preprocessing complete.")
        logger.info(f"Sample record: {dataset['train'][0]}")
    else:
        logger.error("❌ Data preprocessing failed.")

if __name__ == "__main__":
    main() 