"""
Main training orchestrator for CopilotMini model fine-tuning.
Supports training CodeGen model with configuration management.
"""

import os
import sys
import argparse
import yaml
import json
from pathlib import Path
from typing import Dict, Any, Optional
import logging
from datetime import datetime

import torch
from loguru import logger

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))
from config import MODEL_CONFIG, TRAINING_CONFIG

# Import model trainers
try:
    from training.scripts.train_codegen import CodeGenTrainer
except ImportError as e:
    logger.error(f"Failed to import trainers: {e}")
    sys.exit(1)

class ModelTrainingOrchestrator:
    """Orchestrate training of the CodeGen model."""

    def __init__(self):
        self.supported_models = {
            "codegen": {
                "trainer_class": CodeGenTrainer,
                "config_file": "training/configs/codegen_config.yaml",
                "default_model": "Salesforce/codegen-350M-mono"
            }
        }
        self.results = {}

    def load_config(self, config_path: str) -> Dict[str, Any]:
        """Load training configuration from YAML file."""
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            logger.info(f"Loaded config from: {config_path}")
            return config
        except Exception as e:
            logger.error(f"Failed to load config {config_path}: {e}")
            return {}

    def setup_logging(self, model_type: str, config: Dict[str, Any]):
        """Setup logging for training."""
        log_config = config.get("logging", {})
        log_dir = Path("training/logs")
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / f"{model_type}_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        logger.add(
            log_file,
            rotation="50 MB",
            level=log_config.get("log_level", "INFO"),
            format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}"
        )
        logger.info(f"Training logs will be saved to: {log_file}")

    def validate_environment(self, model_type: str, config: Dict[str, Any]) -> bool:
        """Validate training environment and requirements."""
        logger.info(f"Validating environment for {model_type} training...")
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            logger.info(f"✅ GPU available: {gpu_count} GPU(s), {gpu_memory:.1f}GB memory")
        else:
            logger.warning("⚠️ No GPU available, training will be slow")
        try:
            output_dir = Path(config.get("checkpointing", {}).get("output_dir", f"training/checkpoints/{model_type}"))
            output_dir.mkdir(parents=True, exist_ok=True)
            stat = os.statvfs(output_dir)
            free_space_gb = (stat.f_bavail * stat.f_frsize) / 1e9
            if free_space_gb < 10:
                logger.warning(f"⚠️ Low disk space: {free_space_gb:.1f}GB available")
            else:
                logger.info(f"✅ Disk space: {free_space_gb:.1f}GB available")
        except Exception as e:
            logger.warning(f"Could not check disk space: {e}")
        try:
            import psutil
            memory_gb = psutil.virtual_memory().total / 1e9
            if memory_gb < 16:
                logger.warning(f"⚠️ Low system memory: {memory_gb:.1f}GB")
            else:
                logger.info(f"✅ System memory: {memory_gb:.1f}GB")
        except ImportError:
            logger.info("psutil not available, skipping memory check")
        return True

    def prepare_dataset(self, model_type: str, config: Dict[str, Any]) -> bool:
        """Prepare or validate dataset for training."""
        dataset_config = config.get("dataset", {})
        dataset_path = dataset_config.get("processed_data_path", f"data/processed/processed_dataset_{model_type}")
        if Path(dataset_path).exists():
            logger.info(f"✅ Dataset found at: {dataset_path}")
            return True
        else:
            logger.warning(f"⚠️ Dataset not found at: {dataset_path}")
            logger.info("Creating sample dataset for training...")
            return True

    def train_single_model(
        self,
        model_type: str,
        config_path: Optional[str] = None,
        **override_args
    ) -> Dict[str, Any]:
        """Train a single model type."""
        logger.info(f"🚀 Starting {model_type} model training...")
        if config_path:
            config = self.load_config(config_path)
        else:
            config = self.load_config(self.supported_models[model_type]["config_file"])
        if not config:
            raise ValueError(f"Failed to load configuration for {model_type}")

        self.setup_logging(model_type, config)
        if not self.validate_environment(model_type, config):
            raise RuntimeError(f"Environment validation failed for {model_type}")
        if not self.prepare_dataset(model_type, config):
            raise RuntimeError(f"Dataset preparation failed for {model_type}")

        trainer_class = self.supported_models[model_type]["trainer_class"]
        model_config = config.get("model", {})
        
        trainer = trainer_class(
            model_name=model_config.get("name", self.supported_models[model_type]["default_model"]),
            dataset_path=config.get("dataset", {}).get("processed_data_path"),
            output_dir=config.get("checkpointing", {}).get("output_dir"),
            resume_from_checkpoint=config.get("checkpointing", {}).get("resume_from_checkpoint")
        )

        training_args = {}
        if "epochs" in override_args and override_args["epochs"] is not None:
            training_args["num_train_epochs"] = override_args["epochs"]
        if "batch_size" in override_args and override_args["batch_size"] is not None:
            training_args["per_device_train_batch_size"] = override_args["batch_size"]
        if "learning_rate" in override_args and override_args["learning_rate"] is not None:
            training_args["learning_rate"] = override_args["learning_rate"]

        try:
            start_time = datetime.now()
            logger.info(f"Training started at: {start_time}")
            train_result = trainer.train(**training_args)
            end_time = datetime.now()
            training_duration = end_time - start_time
            logger.info(f"✅ Training completed in: {training_duration}")
            eval_results = trainer.evaluate()
            results = {
                "model_type": model_type,
                "start_time": start_time.isoformat(),
                "end_time": end_time.isoformat(),
                "duration": str(training_duration),
                "train_metrics": train_result.metrics if hasattr(train_result, 'metrics') else {},
                "eval_metrics": eval_results if eval_results else {},
                "config": config,
                "success": True
            }
            self.save_training_results(model_type, results)
            return results
        except Exception as e:
            logger.error(f"❌ Training failed for {model_type}: {e}", exc_info=True)
            results = {"model_type": model_type, "error": str(e), "success": False}
            self.save_training_results(model_type, results)
            return results

    def save_training_results(self, model_type: str, results: Dict[str, Any]):
        """Save training results to a JSON file."""
        results_dir = Path("training/results")
        results_dir.mkdir(parents=True, exist_ok=True)
        results_file = results_dir / f"{model_type}_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        try:
            with open(results_file, 'w') as f:
                json.dump(results, f, indent=4, default=str)
            logger.info(f"✅ Training results saved to: {results_file}")
        except Exception as e:
            logger.error(f"Failed to save results: {e}")

    def generate_training_summary(self, all_results: Dict[str, Any]):
        """Generate and print a summary of all training runs."""
        summary = ["\n" + "="*80, "CopilotMini Fine-Tuning Training Summary", "="*80 + "\n"]
        for model_type, result in all_results.items():
            summary.append(f"Model: {model_type.upper()}")
            summary.append("-"*40)
            if result.get("success"):
                summary.append(f"  Status: ✅ SUCCESS")
                summary.append(f"  Duration: {result.get('duration', 'N/A')}")
                train_metrics = result.get('train_metrics', {})
                if train_metrics:
                    summary.append("  Training Metrics:")
                    summary.append(f"    - Epochs: {train_metrics.get('epoch', 'N/A'):.1f}")
                    summary.append(f"    - Loss: {train_metrics.get('train_loss', 'N/A'):.4f}")
                eval_metrics = result.get('eval_metrics', {})
                if eval_metrics:
                    summary.append("  Evaluation Metrics:")
                    summary.append(f"    - Loss: {eval_metrics.get('eval_loss', 'N/A'):.4f}")
                    summary.append(f"    - Perplexity: {eval_metrics.get('perplexity', 'N/A'):.2f}")
            else:
                summary.append(f"  Status: ❌ FAILED")
                summary.append(f"  Error: {result.get('error', 'Unknown error')}")
            summary.append("")
        summary.append("="*80)
        summary_str = "\n".join(summary)
        logger.info(summary_str)
        print(summary_str)

def main():
    """Main function to run the training orchestrator."""
    parser = argparse.ArgumentParser(description="CopilotMini Model Fine-Tuning")
    parser.add_argument(
        "--config",
        type=str,
        help="Path to the CodeGen YAML configuration file.",
        default=None
    )
    parser.add_argument("--epochs", type=int, help="Override number of training epochs.")
    parser.add_argument("--batch_size", type=int, help="Override training batch size.")
    parser.add_argument("--learning_rate", type=float, help="Override learning rate.")
    
    args = parser.parse_args()
    orchestrator = ModelTrainingOrchestrator()
    
    logger.info("Action: Train CodeGen Model")
    override_args = {k: v for k, v in vars(args).items() if v is not None}
    
    result = orchestrator.train_single_model("codegen", args.config, **override_args)
    
    if result and result.get("success"):
        orchestrator.generate_training_summary({"codegen": result})
    else:
        logger.error("Training failed. See logs for details.")

if __name__ == "__main__":
    main() 