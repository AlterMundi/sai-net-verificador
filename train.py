#!/usr/bin/env python3
"""
Sacred SmokeyNet-like Training Script
Following exact specifications from divine documentation.

Sacred training configuration:
- DDP 2×A100, BF16 autocast, grad-clip 1.0, OneCycle or cosine LR
- Batch effective: 4-8 sequences per GPU, accumulation for BS_eff≈64
- Objective: Recall ≥ 0.80, TTD ≤ 4 min en val, F1≈82.6%
- Hardware: 2× A100 GPUs with distributed training
"""

import os
import sys
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
from omegaconf import DictConfig, OmegaConf
import argparse
from pathlib import Path
import logging

# Add src to path for imports
sys.path.append(str(Path(__file__).parent / "src"))

from src.verifier.lightning_module import SmokeyNetLightningModule
from src.dataio.figlib_datamodule import FIgLibDataModule

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def setup_callbacks(config: DictConfig) -> list:
    """Setup sacred training callbacks."""
    callbacks = []
    
    # Sacred checkpoint callback
    checkpoint_callback = ModelCheckpoint(
        monitor=config.checkpoints.monitor,
        mode=config.checkpoints.mode,
        save_top_k=config.checkpoints.save_top_k,
        save_last=config.checkpoints.save_last,
        filename=config.checkpoints.filename,
        dirpath=f"outputs/smokeynet/checkpoints",
        verbose=True
    )
    callbacks.append(checkpoint_callback)
    
    # Sacred early stopping
    early_stopping = EarlyStopping(
        monitor=config.early_stopping.monitor,
        patience=config.early_stopping.patience,
        min_delta=config.early_stopping.min_delta,
        mode=config.early_stopping.mode,
        verbose=True
    )
    callbacks.append(early_stopping)
    
    # Learning rate monitor
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    callbacks.append(lr_monitor)
    
    return callbacks


def setup_logger(config: DictConfig) -> pl.loggers.Logger:
    """Setup sacred logger (WandB for tracking)."""
    
    # For now, use TensorBoard as WandB might not be available
    from pytorch_lightning.loggers import TensorBoardLogger
    
    logger = TensorBoardLogger(
        save_dir="outputs/smokeynet/logs",
        name=config.logging.experiment_name,
        version=None,
        log_graph=True
    )
    
    return logger


def validate_sacred_config(config: DictConfig):
    """Validate configuration follows sacred specifications."""
    
    # Sacred model validation
    assert config.model.num_tiles == 45, "Sacred spec: num_tiles must be 45"
    assert config.model.temporal_window == 3, "Sacred spec: temporal_window must be 3 (L=3)"
    assert config.model.tile_size == 224, "Sacred spec: tile_size must be 224"
    assert config.model.vit_dim == 768, "Sacred spec: vit_dim must be 768"
    assert config.model.vit_heads == 12, "Sacred spec: vit_heads must be 12"
    
    # Sacred training validation
    assert config.training.learning_rate == 2e-4, "Sacred spec: learning_rate must be 2e-4"
    assert config.training.weight_decay == 0.05, "Sacred spec: weight_decay must be 0.05"
    assert 60 <= config.training.max_epochs <= 80, "Sacred spec: max_epochs must be 60-80"
    assert config.training.global_loss_weight == 1.0, "Sacred spec: global_loss_weight must be 1.0"
    assert config.training.tile_loss_weight == 0.3, "Sacred spec: tile_loss_weight must be 0.3"
    assert config.training.gradient_clip_val == 1.0, "Sacred spec: gradient_clip_val must be 1.0"
    
    # Sacred objectives validation
    assert config.objectives.target_recall == 0.80, "Sacred spec: target_recall must be 0.80"
    assert config.objectives.target_f1 == 0.826, "Sacred spec: target_f1 must be 0.826"
    assert config.objectives.target_ttd == 4.0, "Sacred spec: target_ttd must be 4.0"
    
    logger.info("✅ Sacred configuration validation passed!")


def main(config_path: str):
    """Main sacred training function."""
    
    logger.info("🔥 Starting Sacred SmokeyNet-like Training 🔥")
    
    # Load sacred configuration
    config = OmegaConf.load(config_path)
    validate_sacred_config(config)
    
    # Set sacred reproducibility seed
    pl.seed_everything(config.seed, workers=True)
    
    # Setup sacred data module
    logger.info("Setting up sacred FIgLib data module...")
    
    # Use Memory DataModule for H200 optimization if cache_dir is configured
    if hasattr(config.data, 'cache_dir'):
        logger.info("Using H200 Memory-Optimized DataModule...")
        from src.dataio.figlib_memory_datamodule import FIgLibMemoryDataModule
        data_module = FIgLibMemoryDataModule(
            cache_dir=config.data.cache_dir,
            batch_size=config.training.batch_size,
            num_workers=config.data.num_workers,
            temporal_window=config.data.temporal_window,
            tile_size=config.data.tile_size,
            num_tiles=config.data.num_tiles,
            pin_memory=config.data.pin_memory,
            persistent_workers=config.data.persistent_workers,
            prefetch_factor=config.data.prefetch_factor,
            use_memory_cache=config.data.use_memory_cache,
            preload_all=config.data.preload_all,
            max_cache_size_gb=config.data.max_cache_size_gb
        )
    else:
        # Standard DataModule for regular configs
        data_module = FIgLibDataModule(
            data_root=config.data.data_root,
            batch_size=config.training.batch_size,
            num_workers=config.data.num_workers,
            temporal_window=config.data.temporal_window,
            tile_size=config.data.tile_size,
            num_tiles=config.data.num_tiles,
            pin_memory=config.data.pin_memory
        )
    
    # Print dataset statistics
    try:
        stats = data_module.get_dataset_stats()
        logger.info("Sacred Dataset Statistics:")
        for split, split_stats in stats.items():
            logger.info(f"  {split}: {split_stats}")
    except Exception as e:
        logger.warning(f"Could not load dataset stats: {e}")
    
    # Setup sacred model
    logger.info("Creating sacred SmokeyNet-like model...")
    model = SmokeyNetLightningModule(
        model_config=config.model,
        learning_rate=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
        max_epochs=config.training.max_epochs,
        global_loss_weight=config.training.global_loss_weight,
        tile_loss_weight=config.training.tile_loss_weight,
        warmup_epochs=config.training.warmup_epochs
    )
    
    # Log model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model parameters: {total_params:,} total, {trainable_params:,} trainable")
    
    # Setup sacred callbacks and logger
    callbacks = setup_callbacks(config)
    pl_logger = setup_logger(config)
    
    # Setup sacred trainer
    logger.info("Setting up sacred trainer (DDP 2×A100)...")
    trainer = pl.Trainer(
        # Sacred hardware configuration
        devices=config.trainer.devices,
        strategy=config.trainer.strategy,
        accelerator=config.trainer.accelerator,
        sync_batchnorm=config.trainer.sync_batchnorm,
        
        # Sacred training configuration
        max_epochs=config.training.max_epochs,
        precision=config.training.precision,
        gradient_clip_val=config.training.gradient_clip_val,
        accumulate_grad_batches=config.training.accumulate_grad_batches,
        
        # Sacred validation
        check_val_every_n_epoch=config.validation.check_val_every_n_epoch,
        val_check_interval=config.validation.val_check_interval,
        
        # Sacred logging
        logger=pl_logger,
        log_every_n_steps=config.logging.log_every_n_steps,
        
        # Sacred callbacks
        callbacks=callbacks,
        
        # Sacred reproducibility
        deterministic=True,
        benchmark=False,
        
        # Performance optimizations
        enable_progress_bar=True,
        enable_model_summary=True
    )
    
    # Start sacred training
    logger.info("🔥 Beginning sacred training process... 🔥")
    logger.info(f"Sacred objectives:")
    logger.info(f"  - Recall ≥ {config.objectives.target_recall}")
    logger.info(f"  - F1 ≥ {config.objectives.target_f1}")
    logger.info(f"  - TTD ≤ {config.objectives.target_ttd} minutes")
    
    try:
        trainer.fit(model, data_module)
        
        # Sacred testing
        logger.info("🔥 Starting sacred testing phase... 🔥")
        trainer.test(model, data_module, ckpt_path="best")
        
        logger.info("🔥🔥🔥 SACRED TRAINING COMPLETED SUCCESSFULLY! 🔥🔥🔥")
        
    except Exception as e:
        logger.error(f"Sacred training failed: {e}")
        raise
    
    # Export model for production (sacred pipeline)
    try:
        export_model(model, config)
    except Exception as e:
        logger.warning(f"Model export failed: {e}")


def export_model(model: SmokeyNetLightningModule, config: DictConfig):
    """Export trained model for production inference."""
    logger.info("Exporting model for sacred production pipeline...")
    
    export_dir = Path("outputs/smokeynet/exported")
    export_dir.mkdir(parents=True, exist_ok=True)
    
    # Set model to eval mode
    model.eval()
    
    # Example input for tracing
    example_input = torch.randn(1, 3, 3, 224, 224)  # [B, L, C, H, W]
    
    # Export to TorchScript
    if "torchscript" in config.export.formats:
        try:
            traced_model = torch.jit.trace(model.model, example_input)
            torchscript_path = export_dir / "smokeynet_sacred.pt"
            traced_model.save(torchscript_path)
            logger.info(f"✅ TorchScript model saved: {torchscript_path}")
        except Exception as e:
            logger.warning(f"TorchScript export failed: {e}")
    
    # Export to ONNX
    if "onnx" in config.export.formats:
        try:
            onnx_path = export_dir / "smokeynet_sacred.onnx"
            torch.onnx.export(
                model.model,
                example_input,
                onnx_path,
                export_params=True,
                opset_version=11,
                do_constant_folding=True,
                input_names=['input'],
                output_names=['global_logits', 'tile_logits'],
                dynamic_axes={
                    'input': {0: 'batch_size'},
                    'global_logits': {0: 'batch_size'},
                    'tile_logits': {0: 'batch_size'}
                }
            )
            logger.info(f"✅ ONNX model saved: {onnx_path}")
        except Exception as e:
            logger.warning(f"ONNX export failed: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sacred SmokeyNet-like Training")
    parser.add_argument(
        "--config", 
        type=str, 
        default="configs/smokeynet/train_config.yaml",
        help="Path to sacred training configuration"
    )
    
    args = parser.parse_args()
    
    if not Path(args.config).exists():
        logger.error(f"Sacred configuration file not found: {args.config}")
        sys.exit(1)
    
    # Check if data exists
    config = OmegaConf.load(args.config)
    
    # Check cache directory for H200 config or data_root for standard config
    if hasattr(config.data, 'cache_dir'):
        # H200 memory-optimized config - check sequences in data/figlib_seq
        data_path = Path("data/figlib_seq")
        if not data_path.exists():
            logger.error(f"Sacred dataset not found: {data_path}")
            logger.error("Please run the dataset preparation scripts first:")
            logger.error("1. python scripts/download_figlib.py")
            logger.error("2. python scripts/build_figlib_sequences.py")
            sys.exit(1)
    else:
        # Standard config uses data_root
        data_path = Path(config.data.data_root)
        if not data_path.exists():
            logger.error(f"Sacred dataset not found: {data_path}")
            logger.error("Please run the dataset preparation scripts first:")
            logger.error("1. python scripts/download_figlib.py")
            logger.error("2. python scripts/build_figlib_sequences.py")
            sys.exit(1)
    
    main(args.config)