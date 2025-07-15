"""
Dataset download and preparation script for CopilotMini.
Downloads code datasets from various sources and prepares them for training.
"""

import os
import json
import argparse
from pathlib import Path
from typing import List, Dict, Any
import requests
from tqdm import tqdm
import zipfile
import tarfile
from loguru import logger

# Optional imports - only needed for full dataset downloads  
try:
    from datasets import load_dataset, Dataset
    DATASETS_AVAILABLE = True
except ImportError:
    DATASETS_AVAILABLE = False
    logger.warning("datasets library not available - only sample dataset creation will work")

# Add project root to path
import sys
sys.path.append(str(Path(__file__).parent.parent))
from config import DATA_CONFIG

class DatasetDownloader:
    """Download and prepare code datasets for training."""
    
    def __init__(self, raw_data_dir: Path = None):
        self.raw_data_dir = raw_data_dir or DATA_CONFIG["raw_data_dir"]
        self.raw_data_dir.mkdir(parents=True, exist_ok=True)
        
        # Dataset configurations
        self.datasets = {
            "code_search_net": {
                "name": "code_search_net",
                "languages": ["python", "javascript", "java", "go", "php", "ruby"],
                "splits": ["train", "validation", "test"]
            },
            "github_code": {
                "name": "codeparrot/github-code",
                "languages": DATA_CONFIG["supported_languages"],
                "splits": ["train"]
            },
            "the_stack": {
                "name": "bigcode/the-stack-dedup",
                "languages": ["python", "javascript", "typescript", "java"],
                "splits": ["train"]
            }
        }
    
    def download_code_search_net(self) -> Dict[str, Any]:
        """Download CodeSearchNet dataset."""
        if not DATASETS_AVAILABLE:
            return {"name": "code_search_net", "files": [], "error": "datasets library not available"}
            
        logger.info("Downloading CodeSearchNet dataset...")
        
        dataset_info = {"name": "code_search_net", "files": []}
        
        try:
            for lang in self.datasets["code_search_net"]["languages"]:
                logger.info(f"Downloading CodeSearchNet for {lang}...")
                
                dataset = load_dataset(
                    "code_search_net", 
                    lang,
                    cache_dir=str(self.raw_data_dir / "cache")
                )
                
                # Save each split
                for split in ["train", "validation", "test"]:
                    if split in dataset:
                        output_file = self.raw_data_dir / f"code_search_net_{lang}_{split}.jsonl"
                        
                        with open(output_file, 'w', encoding='utf-8') as f:
                            for example in tqdm(dataset[split], desc=f"Saving {lang} {split}"):
                                json.dump(example, f)
                                f.write('\n')
                        
                        dataset_info["files"].append(str(output_file))
                        logger.info(f"Saved {len(dataset[split])} examples to {output_file}")
            
            return dataset_info
            
        except Exception as e:
            logger.error(f"Error downloading CodeSearchNet: {e}")
            return {"name": "code_search_net", "files": [], "error": str(e)}
    
    def download_github_code(self, max_samples: int = 10000) -> Dict[str, Any]:
        """Download GitHub code dataset."""
        if not DATASETS_AVAILABLE:
            return {"name": "github_code", "files": [], "error": "datasets library not available"}
            
        logger.info("Downloading GitHub code dataset...")
        
        dataset_info = {"name": "github_code", "files": []}
        
        try:
            for lang in ["python", "javascript", "java"]:  # Start with main languages
                logger.info(f"Downloading GitHub code for {lang}...")
                
                # Load streaming dataset to handle large size
                dataset = load_dataset(
                    "codeparrot/github-code",
                    languages=[lang],
                    streaming=True,
                    split="train"
                )
                
                output_file = self.raw_data_dir / f"github_code_{lang}.jsonl"
                
                with open(output_file, 'w', encoding='utf-8') as f:
                    count = 0
                    for example in tqdm(dataset, desc=f"Downloading {lang} code"):
                        if count >= max_samples:
                            break
                        
                        # Filter by file size and content quality
                        code = example.get('code', '')
                        if (len(code) > 100 and 
                            len(code) < DATA_CONFIG["max_file_size"] and
                            len(code.split('\n')) >= DATA_CONFIG["min_lines"]):
                            
                            json.dump(example, f)
                            f.write('\n')
                            count += 1
                
                dataset_info["files"].append(str(output_file))
                logger.info(f"Saved {count} examples to {output_file}")
            
            return dataset_info
            
        except Exception as e:
            logger.error(f"Error downloading GitHub code: {e}")
            return {"name": "github_code", "files": [], "error": str(e)}
    
    def download_stack_dataset(self, max_samples_per_lang: int = 5000) -> Dict[str, Any]:
        """Download The Stack dataset."""
        if not DATASETS_AVAILABLE:
            return {"name": "the_stack", "files": [], "error": "datasets library not available"}
            
        logger.info("Downloading The Stack dataset...")
        
        dataset_info = {"name": "the_stack", "files": []}
        
        try:
            for lang in ["python", "javascript", "typescript", "java"]:
                logger.info(f"Downloading The Stack for {lang}...")
                
                # Use specific language subset
                dataset = load_dataset(
                    "bigcode/the-stack-dedup",
                    data_dir=f"data/{lang}",
                    streaming=True,
                    split="train"
                )
                
                output_file = self.raw_data_dir / f"the_stack_{lang}.jsonl"
                
                with open(output_file, 'w', encoding='utf-8') as f:
                    count = 0
                    for example in tqdm(dataset, desc=f"Downloading {lang} from Stack"):
                        if count >= max_samples_per_lang:
                            break
                        
                        content = example.get('content', '')
                        if (len(content) > 100 and 
                            len(content) < DATA_CONFIG["max_file_size"] and
                            len(content.split('\n')) >= DATA_CONFIG["min_lines"]):
                            
                            json.dump(example, f)
                            f.write('\n')
                            count += 1
                
                dataset_info["files"].append(str(output_file))
                logger.info(f"Saved {count} examples to {output_file}")
            
            return dataset_info
            
        except Exception as e:
            logger.error(f"Error downloading The Stack: {e}")
            return {"name": "the_stack", "files": [], "error": str(e)}
    
    def create_sample_dataset(self) -> Dict[str, Any]:
        """Create a small sample dataset for quick testing."""
        logger.info("Creating sample dataset...")
        
        sample_code = {
            "python": [
                '''def fibonacci(n):
    """Calculate the nth Fibonacci number."""
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)

def main():
    for i in range(10):
        print(f"F({i}) = {fibonacci(i)}")

if __name__ == "__main__":
    main()''',
                '''class Calculator:
    """A simple calculator class."""
    
    def add(self, a, b):
        return a + b
    
    def subtract(self, a, b):
        return a - b
    
    def multiply(self, a, b):
        return a * b
    
    def divide(self, a, b):
        if b == 0:
            raise ValueError("Cannot divide by zero")
        return a / b''',
            ],
            "javascript": [
                '''function factorial(n) {
    // Calculate factorial of n
    if (n <= 1) return 1;
    return n * factorial(n - 1);
}

const numbers = [1, 2, 3, 4, 5];
const factorials = numbers.map(factorial);
console.log(factorials);''',
                '''class TodoApp {
    constructor() {
        this.todos = [];
        this.idCounter = 0;
    }
    
    addTodo(text) {
        const todo = {
            id: ++this.idCounter,
            text: text,
            completed: false
        };
        this.todos.push(todo);
        return todo;
    }
    
    toggleTodo(id) {
        const todo = this.todos.find(t => t.id === id);
        if (todo) {
            todo.completed = !todo.completed;
        }
    }
}''',
            ]
        }
        
        dataset_info = {"name": "sample_dataset", "files": []}
        
        for lang, code_samples in sample_code.items():
            output_file = self.raw_data_dir / f"sample_{lang}.jsonl"
            
            with open(output_file, 'w', encoding='utf-8') as f:
                for i, code in enumerate(code_samples):
                    example = {
                        "id": f"sample_{lang}_{i}",
                        "code": code,
                        "language": lang,
                        "license": "MIT",
                        "path": f"sample_{i}.{lang}",
                    }
                    json.dump(example, f)
                    f.write('\n')
            
            dataset_info["files"].append(str(output_file))
            logger.info(f"Created sample dataset: {output_file}")
        
        return dataset_info
    
    def download_all(self, include_large: bool = False) -> List[Dict[str, Any]]:
        """Download all available datasets."""
        logger.info("Starting dataset download process...")
        
        results = []
        
        # Always create sample dataset for testing
        results.append(self.create_sample_dataset())
        
        # Download CodeSearchNet (smaller dataset)
        results.append(self.download_code_search_net())
        
        if include_large:
            # Download larger datasets
            results.append(self.download_github_code(max_samples=10000))
            results.append(self.download_stack_dataset(max_samples_per_lang=5000))
        
        # Save download summary
        summary_file = self.raw_data_dir / "download_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Download complete. Summary saved to {summary_file}")
        return results

def main():
    """Main entry point for the download script."""
    
    # --- Configuration Variables ---
    # Change these variables to control which dataset is downloaded.
    # Options: "code_search_net", "github_code", "the_stack", "sample"
    DATASET_TO_DOWNLOAD = "the_stack"
    
    # Set the maximum number of code samples to download.
    MAX_SAMPLES = 5
    # -----------------------------

    # Setup logging
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    logger.add(log_dir / "download.log", rotation="10 MB")

    # Initialize downloader
    downloader = DatasetDownloader()

    if not DATASETS_AVAILABLE and DATASET_TO_DOWNLOAD != "sample":
        logger.error("The 'datasets' library is required to download full datasets.")
        logger.error("Please install it with: pip install datasets")
        logger.info("Falling back to creating a sample dataset.")
        downloader.create_sample_dataset()
        return

    # Execute download based on the configured dataset
    logger.info(f"Starting download for dataset: '{DATASET_TO_DOWNLOAD}' with a limit of {MAX_SAMPLES} samples.")
    
    if DATASET_TO_DOWNLOAD == "sample":
        downloader.create_sample_dataset()
    elif DATASET_TO_DOWNLOAD == "code_search_net":
        downloader.download_code_search_net()
    elif DATASET_TO_DOWNLOAD == "github_code":
        downloader.download_github_code(max_samples=MAX_SAMPLES)
    elif DATASET_TO_DOWNLOAD == "the_stack":
        downloader.download_stack_dataset(max_samples_per_lang=MAX_SAMPLES)
    else:
        logger.error(f"Unknown dataset specified: {DATASET_TO_DOWNLOAD}")


if __name__ == "__main__":
    main() 