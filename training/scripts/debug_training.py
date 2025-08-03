#!/usr/bin/env python3
"""
Debug script to test the training pipeline step by step.
This helps identify issues in the training process.
"""

import os
import sys
from pathlib import Path
from loguru import logger

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

def test_imports():
    """Test if all required modules can be imported."""
    logger.info("Testing imports...")
    
    try:
        import torch
        logger.info(f"✅ PyTorch: {torch.__version__}")
    except ImportError as e:
        logger.error(f"❌ PyTorch import failed: {e}")
        return False
    
    try:
        import transformers
        logger.info(f"✅ Transformers: {transformers.__version__}")
    except ImportError as e:
        logger.error(f"❌ Transformers import failed: {e}")
        return False
    
    try:
        from datasets import Dataset, DatasetDict
        logger.info("✅ Datasets imported successfully")
    except ImportError as e:
        logger.error(f"❌ Datasets import failed: {e}")
        return False
    
    try:
        from training.scripts.train_codegen import CodeGenTrainer
        logger.info("✅ CodeGenTrainer imported successfully")
    except ImportError as e:
        logger.error(f"❌ CodeGenTrainer import failed: {e}")
        return False
    
    return True

def test_model_loading():
    """Test if the model can be loaded."""
    logger.info("Testing model loading...")
    
    try:
        from training.scripts.train_codegen import CodeGenTrainer
        
        trainer = CodeGenTrainer(
            model_name="Salesforce/codegen-350M-mono",
            output_dir="training/checkpoints/debug_test"
        )
        
        trainer.setup_model_and_tokenizer()
        logger.info("✅ Model and tokenizer loaded successfully")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Model loading failed: {e}")
        return False

def test_dataset_loading():
    """Test if the dataset can be loaded and processed."""
    logger.info("Testing dataset loading...")
    
    try:
        from training.scripts.train_codegen import CodeGenTrainer
        
        trainer = CodeGenTrainer(
            model_name="Salesforce/codegen-350M-mono",
            output_dir="training/checkpoints/debug_test"
        )
        
        # Setup model first
        trainer.setup_model_and_tokenizer()
        
        # Load dataset
        trainer.load_dataset()
        
        if trainer.dataset:
            logger.info(f"✅ Dataset loaded successfully")
            logger.info(f"   Train samples: {len(trainer.dataset['train'])}")
            logger.info(f"   Validation samples: {len(trainer.dataset.get('validation', []))}")
            logger.info(f"   Test samples: {len(trainer.dataset.get('test', []))}")
            logger.info(f"   Columns: {trainer.dataset['train'].column_names}")
            return True
        else:
            logger.error("❌ Dataset doesn't exist")
            return False
        
    except Exception as e:
        logger.error(f"❌ Dataset loading failed: {e}")
        return False

def test_training_setup():
    """Test if training can be set up."""
    logger.info("Testing training setup...")
    
    try:
        from training.scripts.train_codegen import CodeGenTrainer
        from transformers import DataCollatorForLanguageModeling, TrainingArguments, Trainer
        
        trainer = CodeGenTrainer(
            model_name="Salesforce/codegen-350M-mono",
            output_dir="training/checkpoints/debug_test"
        )
        
        # Setup components
        trainer.setup_model_and_tokenizer()
        trainer.load_dataset()
        
        # Test data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=trainer.tokenizer,
            mlm=False,
            return_tensors="pt"
        )
        logger.info("✅ Data collator created successfully")
        
        # Test training arguments
        training_args = trainer.setup_training_args(
            num_train_epochs=1,
            per_device_train_batch_size=1,
            max_steps=5
        )
        logger.info("✅ Training arguments created successfully")
        
        # Test trainer initialization
        trainer_instance = Trainer(
            model=trainer.model,
            args=training_args,
            train_dataset=trainer.dataset["train"],
            eval_dataset=trainer.dataset.get("validation"),
            tokenizer=trainer.tokenizer,
            data_collator=data_collator
        )
        logger.info("✅ Trainer initialized successfully")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Training setup failed: {e}")
        return False

def main():
    """Run all debug tests."""
    logger.info("🔍 Starting training pipeline debug...")
    logger.info("=" * 50)
    
    tests = [
        ("Imports", test_imports),
        ("Model Loading", test_model_loading),
        ("Dataset Loading", test_dataset_loading),
        ("Training Setup", test_training_setup),
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        logger.info(f"\n📋 Running {test_name} test...")
        try:
            result = test_func()
            results[test_name] = result
            
            if result:
                logger.info(f"✅ {test_name} test PASSED")
            else:
                logger.error(f"❌ {test_name} test FAILED")
                
        except Exception as e:
            logger.error(f"❌ {test_name} test failed with exception: {e}")
            results[test_name] = False
    
    # Summary
    logger.info("\n" + "=" * 50)
    logger.info("📊 Debug Test Summary:")
    
    passed = sum(results.values())
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        logger.info(f"  {test_name}: {status}")
    
    logger.info(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 All tests passed! Training pipeline should work.")
    else:
        logger.info("⚠️ Some tests failed. Check the issues above.")

if __name__ == "__main__":
    main() 