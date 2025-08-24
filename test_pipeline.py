#!/usr/bin/env python3
"""
Sacred Pipeline Test Script
Test complete training pipeline before full H200 training.

Tests:
1. Memory cache preprocessing
2. Memory-optimized DataModule
3. SmokeyNet-like model
4. Lightning training (1 epoch)
5. H200 optimizations
6. Sacred objectives validation
"""

import os
import sys
import torch
import pytorch_lightning as pl
from pathlib import Path
import logging
from omegaconf import OmegaConf
import subprocess
import time

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.verifier.lightning_module import SmokeyNetLightningModule
from src.dataio.figlib_memory_datamodule import FIgLibMemoryDataModule

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class PipelineTester:
    """Comprehensive pipeline tester for sacred SmokeyNet implementation."""
    
    def __init__(self, config_path: str):
        self.config_path = config_path
        self.config = OmegaConf.load(config_path)
        self.test_results = {}
        
    def test_system_requirements(self) -> bool:
        """Test system requirements for H200 + memory cache."""
        logger.info("🔍 Testing system requirements...")
        
        tests = []
        
        # Test GPU
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            logger.info(f"✅ GPU: {gpu_name} ({gpu_memory:.1f}GB)")
            tests.append(("GPU", True, f"{gpu_name} with {gpu_memory:.1f}GB"))
            
            if "H200" in gpu_name:
                logger.info("✅ H200 detected - optimal for sacred training")
            else:
                logger.warning(f"⚠️ Non-H200 GPU detected: {gpu_name}")
        else:
            logger.error("❌ No CUDA GPU available")
            tests.append(("GPU", False, "No CUDA GPU"))
        
        # Test /dev/shm
        shm_path = Path("/dev/shm/sai_cache")
        if shm_path.exists():
            shm_usage = subprocess.check_output(['du', '-sh', str(shm_path)]).decode().split()[0]
            logger.info(f"✅ Memory cache: {shm_path} ({shm_usage})")
            tests.append(("Memory Cache", True, f"Available at {shm_path}"))
        else:
            logger.error(f"❌ Memory cache not found: {shm_path}")
            tests.append(("Memory Cache", False, "Not found"))
        
        # Test disk space
        disk_usage = subprocess.check_output(['df', '-h', '.']).decode().split('\n')[1].split()
        available_space = disk_usage[3]
        logger.info(f"✅ Disk space: {available_space} available")
        tests.append(("Disk Space", True, f"{available_space} available"))
        
        # Test dataset
        dataset_path = Path(self.config.data.cache_dir)
        if dataset_path.exists():
            logger.info(f"✅ Dataset cache: {dataset_path}")
            tests.append(("Dataset Cache", True, "Available"))
        else:
            logger.warning(f"⚠️ Dataset cache not found: {dataset_path}")
            tests.append(("Dataset Cache", False, "Not preprocessed"))
        
        self.test_results['system'] = tests
        all_passed = all(test[1] for test in tests)
        
        if all_passed:
            logger.info("✅ All system requirements satisfied")
        else:
            logger.error("❌ Some system requirements not met")
        
        return all_passed
    
    def test_preprocessing(self) -> bool:
        """Test dataset preprocessing to memory cache."""
        logger.info("🔍 Testing dataset preprocessing...")
        
        try:
            # Check if cache already exists
            cache_dir = Path(self.config.data.cache_dir)
            metadata_dir = cache_dir / "metadata"
            
            if not metadata_dir.exists() or not (metadata_dir / "master_cache_index.json").exists():
                logger.info("Cache not found, running preprocessing...")
                
                # Run preprocessing
                cmd = [
                    'python', 'scripts/preprocess_to_memory.py',
                    '--input', 'data/figlib_seq',
                    '--cache-dir', str(cache_dir),
                    '--compression-level', '6'
                ]
                
                logger.info(f"Running: {' '.join(cmd)}")
                result = subprocess.run(cmd, capture_output=True, text=True)
                
                if result.returncode == 0:
                    logger.info("✅ Preprocessing completed successfully")
                    self.test_results['preprocessing'] = ("Success", True, "Completed")
                    return True
                else:
                    logger.error(f"❌ Preprocessing failed: {result.stderr}")
                    self.test_results['preprocessing'] = ("Failed", False, result.stderr)
                    return False
            else:
                logger.info("✅ Cache already exists")
                self.test_results['preprocessing'] = ("Cached", True, "Already available")
                return True
                
        except Exception as e:
            logger.error(f"❌ Preprocessing test failed: {e}")
            self.test_results['preprocessing'] = ("Exception", False, str(e))
            return False
    
    def test_datamodule(self) -> bool:
        """Test memory-optimized DataModule."""
        logger.info("🔍 Testing memory-optimized DataModule...")
        
        try:
            # Create DataModule with test configuration
            dm = FIgLibMemoryDataModule(
                cache_dir=self.config.data.cache_dir,
                batch_size=2,  # Small batch for testing
                num_workers=4,  # Reduced for testing
                use_memory_cache=True,
                preload_all=False,  # Don't preload for test
                max_cache_size_gb=5  # Small limit for test
            )
            
            # Setup
            dm.setup('fit')
            
            # Test train dataloader
            train_loader = dm.train_dataloader()
            logger.info(f"✅ Train dataloader: {len(train_loader)} batches")
            
            # Test loading one batch
            start_time = time.time()
            batch = next(iter(train_loader))
            load_time = time.time() - start_time
            
            # Validate batch structure
            expected_keys = ['frames', 'tiles', 'label', 'metadata']
            if all(key in batch for key in expected_keys):
                logger.info("✅ Batch structure correct")
            else:
                logger.error(f"❌ Missing keys in batch: {set(expected_keys) - set(batch.keys())}")
                return False
            
            # Validate shapes
            frames_shape = batch['frames'].shape  # [B, L, C, H, W]
            tiles_shape = batch['tiles'].shape    # [B, L, num_tiles, C, H, W]
            
            logger.info(f"✅ Frames shape: {frames_shape}")
            logger.info(f"✅ Tiles shape: {tiles_shape}")
            logger.info(f"✅ Load time: {load_time:.3f}s")
            
            # Check sacred specifications
            B, L, C, H, W = frames_shape
            B2, L2, num_tiles, C2, H2, W2 = tiles_shape
            
            sacred_checks = [
                (L == 3, f"Temporal window L={L} (should be 3)"),
                (num_tiles == 45, f"Number of tiles={num_tiles} (should be 45)"),
                (H == W == 224, f"Tile size {H}x{W} (should be 224x224)"),
                (C == 3, f"Channels={C} (should be 3)")
            ]
            
            all_sacred_ok = True
            for check_result, message in sacred_checks:
                if check_result:
                    logger.info(f"✅ {message}")
                else:
                    logger.error(f"❌ {message}")
                    all_sacred_ok = False
            
            # Test cache performance
            stats = dm.get_dataset_stats()
            logger.info(f"✅ Dataset stats: {stats}")
            
            performance = dm.get_cache_performance()
            logger.info(f"✅ Cache performance: {performance}")
            
            self.test_results['datamodule'] = ("Success", all_sacred_ok, f"Load time: {load_time:.3f}s")
            return all_sacred_ok
            
        except Exception as e:
            logger.error(f"❌ DataModule test failed: {e}")
            self.test_results['datamodule'] = ("Exception", False, str(e))
            return False
    
    def test_model(self) -> bool:
        """Test SmokeyNet-like model."""
        logger.info("🔍 Testing SmokeyNet-like model...")
        
        try:
            # Create model
            model = SmokeyNetLightningModule(
                model_config=self.config.model,
                learning_rate=self.config.training.learning_rate,
                weight_decay=self.config.training.weight_decay,
                max_epochs=self.config.training.max_epochs
            )
            
            # Test forward pass
            batch_size = 2
            L, C, H, W = 3, 3, 224, 224
            test_input = torch.randn(batch_size, L, C, H, W)
            test_labels = torch.tensor([1.0, 0.0])
            
            test_batch = {
                'frames': test_input,
                'label': test_labels
            }
            
            # Forward pass
            start_time = time.time()
            with torch.no_grad():
                loss, metrics = model.shared_step(test_batch, 'val')
            forward_time = time.time() - start_time
            
            logger.info(f"✅ Forward pass successful: loss={loss:.4f}")
            logger.info(f"✅ Forward time: {forward_time:.3f}s")
            logger.info(f"✅ Metrics: {metrics}")
            
            # Test model parameters
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            
            logger.info(f"✅ Model parameters: {total_params:,} total, {trainable_params:,} trainable")
            
            # Test sacred architecture components
            sacred_components = [
                hasattr(model.model, 'tile_encoder'),
                hasattr(model.model, 'temporal_aggregator'),
                hasattr(model.model, 'spatial_aggregator'),
                hasattr(model.model, 'global_head')
            ]
            
            if all(sacred_components):
                logger.info("✅ All sacred architecture components present")
                self.test_results['model'] = ("Success", True, f"Forward time: {forward_time:.3f}s")
                return True
            else:
                logger.error("❌ Missing sacred architecture components")
                self.test_results['model'] = ("Missing Components", False, "Architecture incomplete")
                return False
            
        except Exception as e:
            logger.error(f"❌ Model test failed: {e}")
            self.test_results['model'] = ("Exception", False, str(e))
            return False
    
    def test_training_step(self) -> bool:
        """Test one training step."""
        logger.info("🔍 Testing training step...")
        
        try:
            # Setup model and data
            model = SmokeyNetLightningModule(
                model_config=self.config.model,
                learning_rate=self.config.training.learning_rate
            )
            
            dm = FIgLibMemoryDataModule(
                cache_dir=self.config.data.cache_dir,
                batch_size=2,
                num_workers=0,  # No workers for test
                use_memory_cache=True
            )
            
            dm.setup('fit')
            
            # Get one batch
            train_loader = dm.train_dataloader()
            batch = next(iter(train_loader))
            
            # Training step
            model.train()
            optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4)
            
            start_time = time.time()
            
            # Forward
            loss = model.training_step(batch, 0)
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # Sacred grad clipping
            optimizer.step()
            
            step_time = time.time() - start_time
            
            logger.info(f"✅ Training step successful: loss={loss:.4f}")
            logger.info(f"✅ Step time: {step_time:.3f}s")
            
            self.test_results['training_step'] = ("Success", True, f"Step time: {step_time:.3f}s")
            return True
            
        except Exception as e:
            logger.error(f"❌ Training step test failed: {e}")
            self.test_results['training_step'] = ("Exception", False, str(e))
            return False
    
    def test_full_pipeline(self) -> bool:
        """Test complete pipeline with Lightning trainer (1 epoch)."""
        logger.info("🔍 Testing full pipeline (1 epoch)...")
        
        try:
            # Setup model
            model = SmokeyNetLightningModule(
                model_config=self.config.model,
                learning_rate=self.config.training.learning_rate,
                weight_decay=self.config.training.weight_decay,
                max_epochs=1  # Only 1 epoch for test
            )
            
            # Setup data
            dm = FIgLibMemoryDataModule(
                cache_dir=self.config.data.cache_dir,
                batch_size=self.config.training.batch_size,
                num_workers=min(4, self.config.data.num_workers),  # Reduced for test
                use_memory_cache=True,
                max_cache_size_gb=10  # Conservative for test
            )
            
            # Setup trainer
            trainer = pl.Trainer(
                max_epochs=1,
                devices=1,
                accelerator='gpu' if torch.cuda.is_available() else 'cpu',
                precision=self.config.training.precision,
                gradient_clip_val=self.config.training.gradient_clip_val,
                accumulate_grad_batches=self.config.training.accumulate_grad_batches,
                enable_progress_bar=True,
                enable_model_summary=True,
                limit_train_batches=5,  # Only 5 batches for test
                limit_val_batches=2,    # Only 2 val batches for test
                logger=False,           # No logging for test
                enable_checkpointing=False
            )
            
            # Run training
            start_time = time.time()
            trainer.fit(model, dm)
            training_time = time.time() - start_time
            
            logger.info(f"✅ Full pipeline test successful")
            logger.info(f"✅ Training time (1 epoch, limited batches): {training_time:.1f}s")
            
            self.test_results['full_pipeline'] = ("Success", True, f"Training time: {training_time:.1f}s")
            return True
            
        except Exception as e:
            logger.error(f"❌ Full pipeline test failed: {e}")
            import traceback
            traceback.print_exc()
            self.test_results['full_pipeline'] = ("Exception", False, str(e))
            return False
    
    def run_all_tests(self) -> bool:
        """Run all pipeline tests."""
        logger.info("🔥 STARTING SACRED PIPELINE TESTS 🔥")
        
        tests = [
            ("System Requirements", self.test_system_requirements),
            ("Dataset Preprocessing", self.test_preprocessing),
            ("Memory DataModule", self.test_datamodule),
            ("SmokeyNet Model", self.test_model),
            ("Training Step", self.test_training_step),
            ("Full Pipeline", self.test_full_pipeline)
        ]
        
        results = []
        
        for test_name, test_func in tests:
            logger.info(f"\n{'='*60}")
            logger.info(f"TESTING: {test_name}")
            logger.info('='*60)
            
            try:
                result = test_func()
                results.append((test_name, result))
                
                if result:
                    logger.info(f"✅ {test_name}: PASSED")
                else:
                    logger.error(f"❌ {test_name}: FAILED")
                    
            except Exception as e:
                logger.error(f"❌ {test_name}: EXCEPTION - {e}")
                results.append((test_name, False))
        
        # Summary
        logger.info(f"\n{'='*60}")
        logger.info("SACRED PIPELINE TEST SUMMARY")
        logger.info('='*60)
        
        passed_tests = 0
        for test_name, result in results:
            status = "✅ PASS" if result else "❌ FAIL"
            logger.info(f"{test_name:<25}: {status}")
            if result:
                passed_tests += 1
        
        all_passed = passed_tests == len(results)
        
        logger.info(f"\nTotal: {passed_tests}/{len(results)} tests passed")
        
        if all_passed:
            logger.info("🔥🔥🔥 ALL SACRED TESTS PASSED! READY FOR TRAINING! 🔥🔥🔥")
        else:
            logger.error(f"❌ {len(results) - passed_tests} tests failed. Fix issues before training.")
        
        return all_passed


def main():
    config_path = "configs/smokeynet/train_config_h200_optimized.yaml"
    
    if not Path(config_path).exists():
        logger.error(f"Config file not found: {config_path}")
        sys.exit(1)
    
    tester = PipelineTester(config_path)
    success = tester.run_all_tests()
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()