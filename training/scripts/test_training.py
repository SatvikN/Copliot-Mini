"""
Quick test script to verify the training infrastructure works correctly.
Tests a small CodeParrot model with minimal data for rapid validation.
"""

import os
import sys
import asyncio
from pathlib import Path
from loguru import logger

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

# Import trainers
from training.scripts.train_codeparrot import CodeParrotTrainer
from training.scripts.train_models import ModelTrainingOrchestrator

async def test_codeparrot_training():
    """Test CodeParrot training with minimal setup."""
    logger.info("🧪 Testing CodeParrot training...")
    
    try:
        # Initialize trainer with small model
        trainer = CodeParrotTrainer(
            model_name="codeparrot/codeparrot-small",
            output_dir="training/checkpoints/test_codeparrot",
        )
        
        # Test model and tokenizer setup
        trainer.setup_model_and_tokenizer()
        logger.info("✅ Model and tokenizer setup successful")
        
        # Test dataset loading (will create sample data)
        trainer.load_dataset()
        logger.info("✅ Dataset loading successful")
        
        # Test a very short training run (1 step)
        logger.info("🏃‍♂️ Running minimal training test...")
        train_result = trainer.train(
            num_train_epochs=1,
            per_device_train_batch_size=1,
            max_steps=5,  # Only 5 steps for testing
            eval_steps=5,
            save_steps=5
        )
        
        logger.info("✅ Training test successful!")
        
        # Test evaluation
        eval_result = trainer.evaluate()
        logger.info("✅ Evaluation test successful!")
        
        # Test generation
        logger.info("🎨 Testing code generation...")
        samples = trainer.generate_sample("def hello_world():", max_length=50, num_samples=2)
        logger.info(f"Generated samples: {samples}")
        logger.info("✅ Generation test successful!")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Training test failed: {e}")
        return False

async def test_custom_inference():
    """Test the custom inference engine loading."""
    logger.info("🧪 Testing custom inference engine...")
    
    try:
        from backend.models.custom_inference import CustomInferenceEngine
        
        engine = CustomInferenceEngine()
        
        # Test initialization (should not find models yet)
        success = await engine.initialize()
        
        if success:
            logger.info("✅ Custom inference engine found trained models!")
            
            # Test model info
            info = engine.get_model_info()
            logger.info(f"Model info: {info}")
            
            # Test completion generation
            if info["total_models"] > 0:
                logger.info("🎨 Testing completion generation...")
                result = await engine.generate_completions(
                    "def fibonacci(n):",
                    "python",
                    max_suggestions=2
                )
                logger.info(f"Generated completions: {result}")
        else:
            logger.info("ℹ️ No custom models found (expected if no training has been done)")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Custom inference test failed: {e}")
        return False

def test_training_configs():
    """Test that training configurations load correctly."""
    logger.info("🧪 Testing training configurations...")
    
    try:
        orchestrator = ModelTrainingOrchestrator()
        
        # Test loading each config
        for model_type, model_info in orchestrator.supported_models.items():
            config_path = model_info["config_file"]
            config = orchestrator.load_config(config_path)
            
            if config:
                logger.info(f"✅ {model_type} config loaded successfully")
                
                # Validate required sections
                required_sections = ["model", "training", "dataset"]
                for section in required_sections:
                    if section not in config:
                        logger.warning(f"⚠️ Missing {section} section in {model_type} config")
                    else:
                        logger.info(f"  • {section}: ✓")
            else:
                logger.error(f"❌ Failed to load {model_type} config")
                return False
        
        logger.info("✅ All configs loaded successfully!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Config test failed: {e}")
        return False

def test_environment():
    """Test the training environment."""
    logger.info("🧪 Testing training environment...")
    
    try:
        import torch
        import transformers
        from datasets import Dataset
        
        logger.info(f"✅ PyTorch version: {torch.__version__}")
        logger.info(f"✅ Transformers version: {transformers.__version__}")
        logger.info(f"✅ CUDA available: {torch.cuda.is_available()}")
        
        if torch.cuda.is_available():
            logger.info(f"✅ GPU count: {torch.cuda.device_count()}")
            logger.info(f"✅ GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")
        
        # Test basic operations
        logger.info("🔬 Testing basic operations...")
        
        # Test tokenizer loading
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        logger.info("✅ Tokenizer loading works")
        
        # Test simple dataset creation
        sample_data = [{"text": "def hello(): print('hello')"}] * 10
        dataset = Dataset.from_list(sample_data)
        logger.info(f"✅ Dataset creation works: {len(dataset)} examples")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Environment test failed: {e}")
        return False

async def run_all_tests():
    """Run all tests."""
    logger.info("🚀 Running CopilotMini training infrastructure tests...")
    logger.info("=" * 60)
    
    tests = [
        ("Environment", test_environment),
        ("Training Configs", test_training_configs),
        ("Custom Inference", test_custom_inference),
        ("CodeParrot Training", test_codeparrot_training),
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        logger.info(f"\n📋 Running {test_name} test...")
        try:
            if asyncio.iscoroutinefunction(test_func):
                result = await test_func()
            else:
                result = test_func()
            
            results[test_name] = result
            
            if result:
                logger.info(f"✅ {test_name} test PASSED")
            else:
                logger.error(f"❌ {test_name} test FAILED")
                
        except Exception as e:
            logger.error(f"💥 {test_name} test CRASHED: {e}")
            results[test_name] = False
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("📊 TEST SUMMARY")
    logger.info("=" * 60)
    
    passed = sum(1 for r in results.values() if r)
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        logger.info(f"{test_name}: {status}")
    
    logger.info(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 All tests passed! Training infrastructure is ready.")
    else:
        logger.warning(f"⚠️ {total - passed} test(s) failed. Check the issues above.")
    
    return passed == total

def main():
    """Main test runner."""
    # Setup logging
    logger.add("training/logs/test_training.log", rotation="10 MB")
    
    try:
        # Run tests
        success = asyncio.run(run_all_tests())
        
        if success:
            logger.info("\n🎯 Ready to start training! Run:")
            logger.info("  python training/scripts/train_models.py --models codeparrot")
            sys.exit(0)
        else:
            logger.error("\n❌ Tests failed. Fix issues before training.")
            sys.exit(1)
            
    except KeyboardInterrupt:
        logger.info("\n⏹️ Tests interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"\n💥 Test runner crashed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 