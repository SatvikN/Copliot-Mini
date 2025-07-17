#!/usr/bin/env python3
"""
Quick training script to test the fixed training pipeline.
Runs a minimal training session to verify everything works.
"""

import os
import sys
from pathlib import Path
from loguru import logger

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

def main():
    """Run a quick training test."""
    logger.info("🚀 Starting quick training test...")
    
    try:
        from training.scripts.train_codegen import CodeGenTrainer
        
        # Initialize trainer
        trainer = CodeGenTrainer(
            model_name="Salesforce/codegen-350M-mono",
            output_dir="training/checkpoints/quick_test",
        )
        
        logger.info("✅ Trainer initialized")
        
        # Run minimal training
        logger.info("🏃‍♂️ Starting minimal training (5 steps)...")
        
        train_result = trainer.train(
            num_train_epochs=1,
            per_device_train_batch_size=1,
            max_steps=5,  # Only 5 steps for quick test
            eval_steps=5,
            save_steps=5,
            logging_steps=1,
            warmup_steps=0,
            learning_rate=1e-4,
            gradient_accumulation_steps=1
        )
        
        logger.info("✅ Training completed successfully!")
        logger.info(f"Training metrics: {train_result.metrics}")
        
        # Test generation
        logger.info("🎨 Testing code generation...")
        try:
            samples = trainer.generate_completion("def hello():", max_length=30, num_samples=2)
            
            for i, sample in enumerate(samples, 1):
                logger.info(f"Sample {i}: {sample}")
                
        except Exception as e:
            logger.warning(f"Sampling generation failed: {e}")
            logger.info("Trying safe generation method...")
            
            samples = trainer.generate_completion_safe("def hello():", max_length=30)
            for i, sample in enumerate(samples, 1):
                logger.info(f"Safe Sample {i}: {sample}")
        
        logger.info("🎉 All tests passed! Training pipeline is working.")
        return True
        
    except Exception as e:
        logger.error(f"❌ Training test failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 