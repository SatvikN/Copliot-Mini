"""
Fine-tuning script for CodeGen model on code completion tasks.
CodeGen is a GPT-style causal language model optimized for code generation.
"""

import os
import sys
import argparse
import json
from pathlib import Path
from typing import Dict, Any, Optional

import torch
import torch.nn as nn
import numpy as np
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    GPT2Tokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    EarlyStoppingCallback
)
from datasets import load_from_disk, Dataset
from loguru import logger

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))
from config import MODEL_CONFIG, TRAINING_CONFIG

class CodeGenTrainer:
    """Fine-tune CodeGen model for code completion."""
    
    def __init__(self, 
                 model_name: str = "Salesforce/codegen-350M-mono",
                 dataset_path: str = None,
                 output_dir: str = None,
                 resume_from_checkpoint: str = None):
        
        self.model_name = model_name
        self.dataset_path = dataset_path or "data/processed/processed_dataset_codegen"
        self.output_dir = output_dir or "training/checkpoints/codegen"
        self.resume_from_checkpoint = resume_from_checkpoint
        
        # Create output directory
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.tokenizer = None
        self.model = None
        self.dataset = None
        
    def setup_model_and_tokenizer(self):
        """Initialize CodeGen model and tokenizer."""
        logger.info(f"Loading CodeGen model: {self.model_name}")
        
        # Load tokenizer - CodeGen uses GPT2 tokenizer
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        except:
            # Fallback to GPT2 tokenizer
            self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        
        # Add special tokens if not present
        special_tokens = {
            "pad_token": "<|pad|>",
            "eos_token": "<|endoftext|>",
            "bos_token": "<|startoftext|>",
            "unk_token": "<|unk|>"
        }
        
        for token_type, token in special_tokens.items():
            if getattr(self.tokenizer, token_type) is None:
                setattr(self.tokenizer, token_type, token)
                if token_type == "pad_token":
                    # Set pad_token_id as well
                    self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        
        # Add code-specific tokens
        code_tokens = [
            "<|code|>", "<|comment|>", "<|function|>", "<|class|>", "<|variable|>",
            "<|import|>", "<|return|>", "<|if|>", "<|for|>", "<|while|>"
        ]
        num_added = self.tokenizer.add_tokens(code_tokens)
        
        # Load model
        device_map = None
        if torch.cuda.is_available():
            device_map = "auto"
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            # For MPS (Apple Silicon), load on CPU first, then move to MPS
            device_map = None
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map=device_map,
            trust_remote_code=True  # Required for some CodeGen models
        )
        
        # Move to MPS if available
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            self.model = self.model.to('mps')
        
        # Resize token embeddings if we added tokens
        if num_added > 0:
            self.model.resize_token_embeddings(len(self.tokenizer))
        
        logger.info(f"CodeGen model loaded with {sum(p.numel() for p in self.model.parameters())} parameters")
        
    def load_dataset(self):
        """Load and prepare the training dataset."""
        logger.info(f"Loading dataset from: {self.dataset_path}")
        
        try:
            self.dataset = load_from_disk(self.dataset_path)
            logger.info(f"Dataset loaded: {self.dataset}")
            
            # Check if dataset needs preprocessing
            if "text" in self.dataset["train"].column_names:
                logger.info("Dataset contains text fields, preprocessing for training...")
                self.dataset = self._preprocess_dataset(self.dataset)
            
            # Ensure all splits have required columns
            required_columns = ["input_ids", "attention_mask"]
            for split_name, split_data in self.dataset.items():
                missing_cols = [col for col in required_columns if col not in split_data.column_names]
                if missing_cols:
                    logger.error(f"Missing columns in {split_name}: {missing_cols}")
                    raise ValueError(f"Dataset missing required columns: {missing_cols}")
            
        except Exception as e:
            logger.error(f"Failed to load dataset: {e}")
            logger.info("Creating a sample dataset for testing...")
            self.dataset = self._create_sample_dataset()
    
    def _preprocess_dataset(self, dataset):
        """Preprocess dataset to ensure proper tokenization."""
        logger.info("Preprocessing dataset for training...")
        
        def tokenize_function(examples):
            # Use the text field for tokenization
            if "text" in examples:
                texts = examples["text"]
            elif "input_ids" in examples:
                # If already tokenized, return as is
                return {
                    "input_ids": examples["input_ids"],
                    "attention_mask": examples["attention_mask"]
                }
            else:
                raise ValueError("No text or input_ids field found in dataset")
            
            # Ensure texts are strings
            if isinstance(texts, list):
                texts = [str(text) if text is not None else "" for text in texts]
            else:
                texts = str(texts) if texts is not None else ""
            
            # Tokenize
            tokenized = self.tokenizer(
                texts,
                truncation=True,
                padding="max_length",
                max_length=MODEL_CONFIG["max_length"],
                return_tensors=None  # Return lists, not tensors
            )
            
            return {
                "input_ids": tokenized["input_ids"],
                "attention_mask": tokenized["attention_mask"]
            }
        
        # Apply tokenization to all splits
        processed_dataset = {}
        for split_name, split_data in dataset.items():
            logger.info(f"Processing {split_name} split...")
            processed_dataset[split_name] = split_data.map(
                tokenize_function,
                batched=True,
                remove_columns=split_data.column_names,
                desc=f"Tokenizing {split_name}"
            )
        
        from datasets import DatasetDict
        return DatasetDict(processed_dataset)
    
    def _create_sample_dataset(self):
        """Create a sample dataset for code completion tasks."""
        # Code examples focused on different patterns
        sample_codes = [
            # Function definitions
            "def calculate_factorial(n):\n    if n <= 1:\n        return 1\n    else:\n        return n * calculate_factorial(n - 1)",
            
            # Class definitions
            "class Stack:\n    def __init__(self):\n        self.items = []\n    \n    def push(self, item):\n        self.items.append(item)\n    \n    def pop(self):\n        return self.items.pop() if self.items else None",
            
            # Loop patterns
            "numbers = [1, 2, 3, 4, 5]\nfor num in numbers:\n    square = num ** 2\n    print(f'{num} squared is {square}')",
            
            # Error handling
            "try:\n    file = open('data.txt', 'r')\n    content = file.read()\n    file.close()\nexcept FileNotFoundError:\n    print('File not found')\nexcept Exception as e:\n    print(f'Error: {e}')",
            
            # Data structures
            "import collections\ndata = collections.defaultdict(list)\nfor item in items:\n    data[item.category].append(item.value)",
            
            # API calls
            "import requests\nresponse = requests.get('https://api.example.com/data')\nif response.status_code == 200:\n    data = response.json()\n    return data['results']",
            
            # Machine learning
            "from sklearn.model_selection import train_test_split\nfrom sklearn.linear_model import LinearRegression\nX_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)\nmodel = LinearRegression()\nmodel.fit(X_train, y_train)",
            
            # Web development
            "from flask import Flask, jsonify\napp = Flask(__name__)\n\n@app.route('/api/users')\ndef get_users():\n    users = database.get_all_users()\n    return jsonify([user.to_dict() for user in users])",
            
            # Algorithms
            "def binary_search(arr, target):\n    left, right = 0, len(arr) - 1\n    while left <= right:\n        mid = (left + right) // 2\n        if arr[mid] == target:\n            return mid\n        elif arr[mid] < target:\n            left = mid + 1\n        else:\n            right = mid - 1\n    return -1",
            
            # Data processing
            "import pandas as pd\ndf = pd.read_csv('sales_data.csv')\nmonthly_sales = df.groupby('month')['amount'].sum()\ntop_products = df.nlargest(10, 'sales_count')"
        ]
        
        # Tokenize samples
        tokenized_samples = []
        for code in sample_codes:
            tokens = self.tokenizer(
                code,
                truncation=True,
                padding="max_length",
                max_length=MODEL_CONFIG["max_length"],
                return_tensors="pt"
            )
            tokenized_samples.append({
                "input_ids": tokens["input_ids"].squeeze().tolist(),
                "attention_mask": tokens["attention_mask"].squeeze().tolist()
            })
        
        # Create dataset splits
        train_data = tokenized_samples * 12  # Duplicate for more training data
        val_data = tokenized_samples[:3]
        
        from datasets import DatasetDict, Dataset
        return DatasetDict({
            "train": Dataset.from_list(train_data),
            "validation": Dataset.from_list(val_data)
        })
    
    def setup_training_args(self, **kwargs):
        """Setup training arguments optimized for CodeGen."""
        default_args = {
            "output_dir": self.output_dir,
            "overwrite_output_dir": True,
            "num_train_epochs": TRAINING_CONFIG["num_epochs"],
            "per_device_train_batch_size": max(1, TRAINING_CONFIG["batch_size"] // 2),  # Smaller batch for larger model
            "per_device_eval_batch_size": max(1, TRAINING_CONFIG["batch_size"] // 2),
            "gradient_accumulation_steps": TRAINING_CONFIG["gradient_accumulation_steps"] * 2,
            "learning_rate": TRAINING_CONFIG["learning_rate"] * 0.5,  # Lower LR for CodeGen
            "warmup_steps": TRAINING_CONFIG["warmup_steps"],
            "logging_steps": TRAINING_CONFIG["logging_steps"],
            "save_steps": TRAINING_CONFIG["save_steps"],
            "eval_steps": TRAINING_CONFIG["eval_steps"],
            "eval_strategy": "steps",
            "save_strategy": "steps",
            "max_grad_norm": TRAINING_CONFIG["max_grad_norm"],
            "weight_decay": TRAINING_CONFIG["weight_decay"],
            "adam_epsilon": TRAINING_CONFIG["adam_epsilon"],
            "fp16": torch.cuda.is_available(),
            "dataloader_pin_memory": True,
            "dataloader_num_workers": 0,
            "remove_unused_columns": True,
            "load_best_model_at_end": True,
            "metric_for_best_model": "eval_loss",
            "greater_is_better": False,
            "save_total_limit": 3,
            "report_to": "none",
            "gradient_checkpointing": False,  # Disable for CPU stability
        }
        
        # Update with any provided kwargs
        default_args.update(kwargs)
        
        return TrainingArguments(**default_args)
    
    def compute_metrics(self, eval_pred):
        """Compute training metrics."""
        predictions, labels = eval_pred
        
        # Convert to PyTorch tensors if they're numpy arrays
        if isinstance(predictions, np.ndarray):
            predictions = torch.tensor(predictions)
        if isinstance(labels, np.ndarray):
            labels = torch.tensor(labels)
        
        # Calculate perplexity
        shift_logits = predictions[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        
        # Flatten for loss calculation
        shift_logits = shift_logits.view(-1, shift_logits.size(-1))
        shift_labels = shift_labels.view(-1)
        
        # Calculate cross entropy loss
        loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
        loss = loss_fct(shift_logits, shift_labels)
        
        perplexity = torch.exp(loss).item()
        
        # Calculate accuracy (next token prediction)
        predictions_flat = torch.argmax(shift_logits, dim=-1)
        labels_flat = shift_labels
        mask = labels_flat != -100
        
        if mask.sum() > 0:
            accuracy = (predictions_flat[mask] == labels_flat[mask]).float().mean().item()
        else:
            accuracy = 0.0
        
        return {
            "perplexity": perplexity,
            "accuracy": accuracy,
            "eval_loss": loss.item()
        }
    
    def train(self, **training_kwargs):
        """Start the training process."""
        logger.info("Starting CodeGen fine-tuning...")
        
        # Setup components
        self.setup_model_and_tokenizer()
        self.load_dataset()
        
        # Manually remove columns that are not needed for training
        if self.dataset:
            columns_to_remove = []
            for col in ["text", "language", "token_count", "truncated"]:
                if col in self.dataset["train"].column_names:
                    columns_to_remove.append(col)
            
            if columns_to_remove:
                self.dataset = self.dataset.remove_columns(columns_to_remove)

        # Setup data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,  # Causal LM, not masked LM
            pad_to_multiple_of=8 if torch.cuda.is_available() else None,
            return_tensors="pt"  # Ensure we get tensors
        )
        
        # Setup training arguments
        training_args = self.setup_training_args(**training_kwargs)
        
        # Initialize trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.dataset["train"],
            eval_dataset=self.dataset.get("validation"),
            tokenizer=self.tokenizer,
            data_collator=data_collator,
            compute_metrics=self.compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
        )
        
        # Resume from checkpoint if provided
        if self.resume_from_checkpoint:
            logger.info(f"Resuming from checkpoint: {self.resume_from_checkpoint}")
        
        # Start training
        logger.info("Training started...")
        train_result = trainer.train(resume_from_checkpoint=self.resume_from_checkpoint)
        
        # Save final model
        trainer.save_model()
        self.tokenizer.save_pretrained(self.output_dir)
        
        # Save training metrics
        metrics_file = Path(self.output_dir) / "training_metrics.json"
        with open(metrics_file, 'w') as f:
            json.dump(train_result.metrics, f, indent=2)
        
        logger.info(f"Training completed! Model saved to: {self.output_dir}")
        logger.info(f"Training metrics: {train_result.metrics}")
        
        return train_result
    
    def evaluate(self, test_dataset=None):
        """Evaluate the trained model."""
        if not self.model:
            logger.error("Model not loaded. Please train or load a model first.")
            return None
        
        eval_dataset = test_dataset or self.dataset.get("test") or self.dataset.get("validation")
        
        if not eval_dataset:
            logger.warning("No evaluation dataset available")
            return None
        
        # Setup data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False
        )
        
        # Create trainer for evaluation
        trainer = Trainer(
            model=self.model,
            tokenizer=self.tokenizer,
            data_collator=data_collator,
            compute_metrics=self.compute_metrics
        )
        
        # Run evaluation
        eval_results = trainer.evaluate(eval_dataset=eval_dataset)
        
        logger.info(f"Evaluation results: {eval_results}")
        return eval_results
    
    def generate_completion(self, prompt: str, max_length: int = 100, num_samples: int = 3):
        """Generate code completions using the trained CodeGen model."""
        if not self.model:
            logger.error("Model not loaded. Please train or load a model first.")
            return []
        
        self.model.eval()
        
        # Tokenize prompt
        inputs = self.tokenizer(prompt, return_tensors="pt")
        
        # Handle device placement properly
        device = next(self.model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Generate samples with safer parameters
        samples = []
        with torch.no_grad():
            for i in range(num_samples):
                try:
                    outputs = self.model.generate(
                        **inputs,
                        max_length=inputs["input_ids"].shape[1] + max_length,
                        do_sample=True,
                        temperature=1.0,  # Higher temperature for stability
                        top_p=0.95,       # Higher top_p for more diversity
                        top_k=50,         # Higher top_k
                        repetition_penalty=1.1,  # Slightly higher repetition penalty
                        pad_token_id=self.tokenizer.eos_token_id,
                        eos_token_id=self.tokenizer.eos_token_id,
                        bad_words_ids=[[self.tokenizer.unk_token_id]],
                        # Add safety parameters
                        use_cache=True,
                        return_dict_in_generate=True,
                        output_scores=False,  # Don't return scores to avoid issues
                        # Add numerical stability
                        renormalize_logits=True,
                        # Limit generation length
                        max_new_tokens=min(max_length, 50)
                    )
                    
                    # Decode generated text
                    if hasattr(outputs, 'sequences'):
                        generated_text = self.tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)
                    else:
                        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                    
                    completion = generated_text[len(prompt):].strip()
                    samples.append(completion)
                    
                except Exception as e:
                    logger.warning(f"Generation failed for sample {i+1}: {e}")
                    # Fallback to simple completion
                    samples.append("print('Hello, World!')")
        
        return samples
    
    def generate_completion_safe(self, prompt: str, max_length: int = 50):
        """Generate code completions using greedy decoding (more stable)."""
        if not self.model:
            logger.error("Model not loaded. Please train or load a model first.")
            return []
        
        self.model.eval()
        
        # Tokenize prompt
        inputs = self.tokenizer(prompt, return_tensors="pt")
        
        # Handle device placement properly
        device = next(self.model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        try:
            with torch.no_grad():
                # Use greedy decoding (more stable than sampling)
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_length,
                    do_sample=False,  # Greedy decoding
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    use_cache=True
                )
                
                # Decode generated text
                generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                completion = generated_text[len(prompt):].strip()
                
                return [completion]
                
        except Exception as e:
            logger.error(f"Safe generation failed: {e}")
            return ["print('Hello, World!')"]
    
    def analyze_code_quality(self, code: str) -> Dict[str, Any]:
        """Analyze the quality of generated code."""
        quality_metrics = {
            "length": len(code),
            "lines": len(code.split('\n')),
            "has_functions": bool('def ' in code),
            "has_classes": bool('class ' in code),
            "has_imports": bool('import ' in code or 'from ' in code),
            "has_docstrings": bool('"""' in code or "'''" in code),
            "has_comments": bool('#' in code),
            "syntax_elements": len([c for c in code if c in '(){}[]']),
        }
        
        # Calculate complexity score
        complexity_score = (
            quality_metrics["has_functions"] * 2 +
            quality_metrics["has_classes"] * 3 +
            quality_metrics["has_imports"] * 1 +
            quality_metrics["has_docstrings"] * 2 +
            quality_metrics["has_comments"] * 1 +
            min(quality_metrics["syntax_elements"] / 10, 5)  # Cap at 5
        )
        
        quality_metrics["complexity_score"] = complexity_score
        quality_metrics["quality_rating"] = (
            "excellent" if complexity_score >= 8 else
            "good" if complexity_score >= 5 else
            "fair" if complexity_score >= 3 else
            "basic"
        )
        
        return quality_metrics
    
    def interactive_completion(self):
        """Interactive code completion session."""
        if not self.model:
            logger.error("Model not loaded. Please train or load a model first.")
            return
        
        logger.info("Starting interactive CodeGen completion session...")
        logger.info("Type 'quit' to exit, 'clear' to reset context.")
        
        context = ""
        
        while True:
            try:
                user_input = input("\nEnter code prompt: ").strip()
                
                if user_input.lower() == 'quit':
                    break
                elif user_input.lower() == 'clear':
                    context = ""
                    logger.info("Context cleared.")
                    continue
                
                # Add to context
                full_prompt = context + user_input
                
                # Generate completions
                logger.info("Generating completions...")
                completions = self.generate_completion(full_prompt, max_length=150, num_samples=3)
                
                print("\n" + "="*60)
                for i, completion in enumerate(completions, 1):
                    print(f"\nCompletion {i}:")
                    print("-" * 30)
                    print(completion)
                    
                    # Analyze quality
                    quality = self.analyze_code_quality(completion)
                    print(f"Quality: {quality['quality_rating']} (score: {quality['complexity_score']:.1f})")
                
                print("="*60)
                
                # Ask if user wants to continue with one of the completions
                choice = input("\nUse completion as context? (1-3, or 'n' for no): ").strip()
                if choice in ['1', '2', '3']:
                    context = full_prompt + completions[int(choice)-1] + "\n"
                    logger.info(f"Added completion {choice} to context.")
                
            except KeyboardInterrupt:
                break
            except Exception as e:
                logger.error(f"Error: {e}")
        
        logger.info("Interactive session ended.")

def main():
    parser = argparse.ArgumentParser(description="Fine-tune CodeGen model")
    parser.add_argument("--model-name", default="Salesforce/codegen-350M-mono", 
                       help="Base CodeGen model name")
    parser.add_argument("--dataset-path", default=None,
                       help="Path to processed dataset")
    parser.add_argument("--output-dir", default="training/checkpoints/codegen",
                       help="Output directory for checkpoints")
    parser.add_argument("--resume-from-checkpoint", default=None,
                       help="Resume training from checkpoint")
    parser.add_argument("--epochs", type=int, default=3,
                       help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=4,
                       help="Training batch size")
    parser.add_argument("--learning-rate", type=float, default=2.5e-5,
                       help="Learning rate")
    parser.add_argument("--eval-only", action="store_true",
                       help="Only run evaluation")
    parser.add_argument("--test-generation", action="store_true",
                       help="Test code generation after training")
    parser.add_argument("--interactive", action="store_true",
                       help="Start interactive completion session")
    
    print("Check #0")
    args = parser.parse_args()
    
    # Setup logging
    log_file = Path("logs") / "training_codegen.log"
    log_file.parent.mkdir(exist_ok=True)
    logger.add(log_file, rotation="10 MB")
    
    # Initialize trainer
    trainer = CodeGenTrainer(
        model_name=args.model_name,
        dataset_path=args.dataset_path,
        output_dir=args.output_dir,
        resume_from_checkpoint=args.resume_from_checkpoint
    )

    print("Check #1")
    
    if args.interactive:
        print("Check #2")
        # Start interactive session
        trainer.setup_model_and_tokenizer()
        trainer.interactive_completion()
    elif args.eval_only:
        print("Check #3")
        # Only run evaluation
        trainer.setup_model_and_tokenizer()
        trainer.load_dataset()
        trainer.evaluate()
    else:
        print("Check #4")
        # Run training
        train_result = trainer.train(
            num_train_epochs=args.epochs,
            per_device_train_batch_size=args.batch_size,
            learning_rate=args.learning_rate
        )
        
        # Run evaluation
        trainer.evaluate()
    
    # Test generation if requested
    if args.test_generation:
        logger.info("Testing code generation...")
        test_prompts = [
            "def merge_sort(arr):",
            "class BinaryTree:",
            "import tensorflow as tf\nmodel = ",
            "async def fetch_data(url):",
            "with open('config.json', 'r') as f:"
        ]
        
        for prompt in test_prompts:
            logger.info(f"Prompt: {prompt}")
            samples = trainer.generate_completion(prompt, max_length=120)
            for i, sample in enumerate(samples, 1):
                logger.info(f"Sample {i}: {sample}")
                quality = trainer.analyze_code_quality(sample)
                logger.info(f"Quality: {quality['quality_rating']}")
            logger.info("-" * 60)

if __name__ == "__main__":
    main() 