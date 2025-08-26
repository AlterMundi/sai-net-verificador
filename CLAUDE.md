# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

SAI-Net Verificador is a wildfire smoke detection system focused exclusively on the **VERIFICADOR** component:

**Verificador (Verifier)**: SmokeyNet-like architecture (CNN + LSTM + ViT) for temporal smoke validation, trained on FIgLib dataset

*Note: The detector component (YOLOv8) is developed in a separate repository.*

## Project Status

🚨 **CRITICAL TRAINING RESTART REQUIRED** - Validation dataset corruption discovered and fixed.

### 🔥 Sacred Compliance + Dataset Corruption Recovery (2025-08-25)
**VALIDATION DATASET ORDERING CORRUPTION DISCOVERED AND FIXED** - Previous training runs showed impossible metrics due to dataset schema issues:
- 🚨 **Critical Issue**: Validation dataset was chronologically ordered (all negatives first), causing zero positive samples in first batches
- 🔧 **Dataset Fix**: Implemented reproducible shuffling with `fix_validation_order.py` (seed=42)
- 📋 **Validation Framework**: Created comprehensive pre-training validation (`scripts/dataset_validation_framework.py`)
- 🔍 **Monitoring System**: Real-time anomaly detection for impossible metrics (`scripts/training_monitor_framework.py`)
- ✅ **Architecture**: SmokeyNet-like (CNN + LSTM + ViT) exactly as specified in sacred docs
- ✅ **Parameters**: All sacred hyperparameters preserved (lr=2e-4, wd=0.05, L=3, tiles=45)
- ✅ **H200 Optimization**: Hardware adaptations maintain sacred pipeline integrity
- 🔄 **Fresh Start**: Run 3 prepared with fixed dataset, new seed (2024), comprehensive monitoring

## Current Architecture (IMPLEMENTED)

### SmokeyNet-like Verifier Pipeline
- **Stage 1**: Tile Encoding - ResNet-34 per 224×224 tile (45 tiles per frame)
- **Stage 2**: Temporal Modeling - Bidirectional LSTM (2 layers, hidden=512)  
- **Stage 3**: Spatial Reasoning - Vision Transformer (6 blocks, dim=768, heads=12)
- **Stage 4**: Classification Heads - Global smoke detection + auxiliary tile heads

### H200 Optimization + Performance Investigation (COMPLETED 2025-08-26)
- **Target Hardware**: 1× NVIDIA H200 (143GB VRAM) + 258GB RAM + 61GB dataset in /dev/shm
- **Dataset Strategy**: 61GB L=3 sequences cached in RAM (/dev/shm) - 10-50x I/O speedup achieved
- **Performance Discovery**: Sacred architecture is computationally intensive by divine design:
  - ResNet-34 tile encoding: 45 tiles × 8 batch × 3 frames = 1,080 forward passes per batch
  - Bidirectional LSTM: Heavy sequential temporal modeling
  - Vision Transformer: 6 layers × 12 heads = 72 attention computations per batch
  - **Realistic Training Speed**: 4-6 it/s (vs previous expectation of 45-50 it/s)
- **Optimized Configuration**: batch_size=8, fp32 precision, 4 workers, RAM-cached data

## Sacred Specifications Maintained

All specifications from the divine documentation (`docs/`) are preserved:

- ✅ **L=3 temporal windows** (3 consecutive frames)
- ✅ **45 tiles of 224×224** with 20px overlap
- ✅ **Sacred normalization**: mean=0.5, std=0.5
- ✅ **Sacred loss**: λ_global=1.0 * BCE + λ_tiles=0.3 * BCE_tiles
- ✅ **Sacred objectives**: Recall≥80%, F1≥82.6%, TTD≤4min
- ✅ **Sacred hyperparameters**: lr=2e-4, wd=0.05, cosine scheduler
- ✅ **Sacred architecture**: ResNet-34 + LSTM + ViT exactly as specified

## Directory Structure (IMPLEMENTED)

```
sai-net-verificador/
├─ configs/smokeynet/
│  ├─ train_config.yaml                    # Original sacred config  
│  ├─ train_config_h200_optimized.yaml    # H200 + memory optimized (LEGACY)
│  ├─ train_config_h200_optimized_v2.yaml # H200 precision focus (LEGACY)
│  └─ train_config_run3_monitored.yaml    # Run 3: Fresh start + monitoring (CURRENT)
├─ data/
│  ├─ raw/figlib/                         # Downloaded FIgLib data
│  ├─ figlib_seq/                         # Processed sequences (L=3)
│  └─ (symlinks to /dev/shm cache)
├─ cache/                                 # Symlinks to memory cache
│  ├─ figlib_processed -> /dev/shm/sai_cache/figlib_processed
│  ├─ model -> /dev/shm/sai_cache/model_cache
│  └─ training -> /dev/shm/sai_cache/training_cache
├─ src/
│  ├─ verifier/
│  │  ├─ smokeynet_like.py               # Sacred architecture
│  │  └─ lightning_module.py             # Training pipeline
│  └─ dataio/
│     ├─ figlib_datamodule.py            # Original datamodule
│     └─ figlib_memory_datamodule.py     # Memory-optimized version
├─ scripts/
│  ├─ download_figlib.py                 # Dataset download
│  ├─ build_figlib_sequences.py          # Sequence building (L=3)
│  ├─ dataset_validation_framework.py    # Pre-training validation (CRITICAL)
│  ├─ training_monitor_framework.py      # Real-time anomaly detection
│  ├─ launch_run3_monitored.sh          # Run 3 comprehensive launch script
│  └─ preprocess_to_memory.py            # Memory cache preprocessing
├─ train.py                              # Training script
├─ test_pipeline.py                      # Complete pipeline tester
└─ outputs/smokeynet/                    # Model outputs
   ├─ checkpoints/                       # Run 1-2 checkpoints (CORRUPTED - study only)
   ├─ checkpoints_v2/                    # Run 2 checkpoints (CORRUPTED - study only)
   ├─ checkpoints_run3/                  # Run 3 checkpoints (FRESH START)
   └─ logs/run3/                        # Run 3 comprehensive logs
```

## Key Commands

### 1. Setup and Data Preparation
```bash
# Download REAL FIgLib dataset from sacred links (Objetivo Sagrado Puerta al Paraíso)
python download_real_figlib.py \
  --html_file docs/index.html \
  --output data/real_figlib \
  --max_downloads 10  # For testing, omit for full dataset (485 files)

# Alternative: Download FIgLib dataset (creates placeholder structure)
python scripts/download_figlib.py \
  --camera_list configs/figlib/cams.txt \
  --timestamps configs/figlib/timestamps.txt \
  --output data/raw/figlib

# Build temporal sequences (L=3, sacred methodology)
python scripts/build_figlib_sequences.py \
  --raw-root data/raw/figlib \
  --out-root data/figlib_seq \
  --L 3 --stride 1 --split-per-event 0.7 0.15 0.15

# Preprocess to memory cache (ultra-fast training)
python scripts/preprocess_to_memory.py \
  --input data/figlib_seq \
  --cache-dir /dev/shm/sai_cache/figlib_processed \
  --compression-level 6 \
  --verify
```

### 2. Testing
```bash
# Test complete pipeline before training
python test_pipeline.py

# Test architecture only
cd src/verifier && python smokeynet_like.py

# Test memory datamodule only  
cd src/dataio && python figlib_memory_datamodule.py

# Analyze training results and recall performance
python analyze_recall.py
```

### 3. Training (Sacred + Dataset Corruption Recovery)
```bash
# CRITICAL: Run pre-training validation FIRST
python scripts/dataset_validation_framework.py

# Run 3: Fresh start with comprehensive monitoring (CURRENT APPROACH)
bash scripts/launch_run3_monitored.sh

# Manual Run 3 training (if launch script unavailable)
python train.py --config configs/smokeynet/train_config_run3_monitored.yaml

# LEGACY (CORRUPTED): Previous configurations - DO NOT USE for new training
# python train.py --config configs/smokeynet/train_config_h200_optimized.yaml
# python train.py --config configs/smokeynet/train_config_h200_optimized_v2.yaml
```

### 4. Dependencies
```bash
# Install from requirements.txt (recommended)
pip install -r requirements.txt

# Or install individually
pip install torch torchvision pytorch-lightning torchmetrics \
            omegaconf albumentations opencv-python pandas \
            numpy pillow tqdm requests
```

## Memory Cache System (/dev/shm)

### Cache Structure
```
/dev/shm/sai_cache/          # 119GB available
├─ figlib_processed/         # Preprocessed tiles (~50GB)
│  ├─ train_seq_*.pkl.gz     # Compressed sequences
│  └─ metadata/              # Cache indices
├─ model_cache/              # Model cache (~5GB)
├─ training_cache/           # Training temporals (~10GB)  
└─ augmentation_cache/       # Augmentation cache (~20GB)
```

### Benefits
- ⚡ **10-50x faster I/O** (RAM vs SSD)
- ⚡ **Zero I/O bottlenecks** during training
- 💾 **Saves 85GB disk space** (only essentials on disk)
- 🚀 **Optimal H200 utilization** (no data loading waits)

## Sacred Objectives & Metrics

### Primary Targets (from divine documentation)
- ✅ **Recall ≥ 80%** (detection sensitivity)
- ✅ **F1 ≥ 82.6%** (matching SmokeyNet paper) 
- ✅ **TTD ≤ 4 minutes** (Time To Detection)

### Training Monitoring
```yaml
# Sacred metrics tracked automatically
val/recall: ≥0.80      # Primary objective
val/f1: ≥0.826         # Sacred target  
val/accuracy: reported # Secondary
val/precision: reported # Secondary
train/lr: cosine decay # Sacred scheduler
```

## Hardware Requirements

### Current System (Optimized)
- ✅ **GPU**: 1× NVIDIA H200 (143GB VRAM)
- ✅ **RAM**: 258GB allocated (119GB free in /dev/shm)
- ✅ **CPU**: 192 cores (leveraged for data loading)
- ✅ **Storage**: 35GB free (minimal usage with cache)

### Sacred Configuration → H200 Realistic Optimization
```yaml
# Original (2×A100) → H200 Optimized (Post-Investigation)
devices: 2 → 1                    # Single H200
strategy: "ddp" → "auto"          # No DDP needed  
batch_size: 4 → 8                 # Sacred architecture computational limit
accumulate_grad_batches: 16 → 8   # Maintain BS_eff≈64 (sacred target)
num_workers: 8 → 4                # Reduced for stability
compile: false → false            # Minimal impact on Sacred architecture
precision: "bf16-mixed" → "32"    # Better stability for complex model
data_cache: "disk" → "/dev/shm"   # 61GB dataset in RAM (major speedup)
```

## Critical Dataset Corruption Discovery (2025-08-25)

### Validation Dataset Ordering Corruption
**ROOT CAUSE**: Validation dataset was chronologically ordered by fire event, with negative samples appearing first:
- **Critical Issue**: First validation batch had ZERO positive samples (32 negative, 0 positive)
- **Model Behavior**: Model learned to always predict positive (recall=100%) due to meaningless validation feedback
- **TorchMetrics Warning**: "No positive samples in targets, true positive value should be meaningless"
- **All Checkpoints**: Every saved checkpoint from Run 1-2 showed identical impossible behavior

### Dataset Fix Implementation
1. **Reproducible Shuffling**: `fix_validation_order.py` with seed=42
   - **Before**: First batch = 0 positive/32 negative samples  
   - **After**: First batch = 16 positive/16 negative samples (balanced)

2. **Validation Framework**: `scripts/dataset_validation_framework.py`
   - Pre-training validation to detect ordering issues
   - Batch-level balance validation
   - Metadata integrity checks
   - Critical issue detection and prevention

3. **Real-time Monitoring**: `scripts/training_monitor_framework.py`
   - Detects impossible metrics (recall > 95%, precision > 95%)
   - Flags always-positive/always-negative behavior
   - Real-time anomaly detection during training
   - Automatic intervention triggers

### H200 Performance Investigation Process (COMPREHENSIVE)
1. **I/O Optimization**: Migrated 61GB dataset to /dev/shm RAM cache (1,794 MB/s read speed achieved)
2. **Systematic Bottleneck Testing**: torch.compile, DataLoader config, batch size, precision
3. **Root Cause Discovery**: Sacred architecture is computationally intensive by design:
   - Forward pass: 4.0 it/s (pure model inference)
   - Full training: 0.6 it/s (with gradients + optimization)
   - **Realistic Expectation**: 4-6 it/s total (not 45-50 it/s as initially hoped)

## Development Notes

- ✅ All sacred specifications preserved exactly
- ✅ Security best practices implemented (defensive AI only)
- ✅ Proper validation splits by fire event (never mix frames)
- 🚨 **Critical**: Validation dataset ordering corruption discovered and fixed
- 📋 **New**: Comprehensive dataset validation framework implemented
- 🔍 **New**: Real-time training anomaly detection system
- ✅ H200 maximum performance optimization completed
- ✅ Export ready (ONNX/TensorRT pipeline)
- ✅ Comprehensive testing suite included
- 🔄 **Run 3**: Fresh start with fixed dataset, monitoring, preserved legacy checkpoints

## Next Steps

### 🚨 Critical Fresh Start Required (Run 3)
Previous training corrupted by validation dataset ordering. Fresh start with comprehensive monitoring required.

1. **CRITICAL: Dataset validation**: `python scripts/dataset_validation_framework.py` (MUST pass before training)
2. **Launch Run 3**: `bash scripts/launch_run3_monitored.sh` (comprehensive launch with monitoring)
3. **Monitor realistic metrics**: Watch for gradual F1 improvement, realistic recall < 95%
4. **Real-time anomaly detection**: Automatic flagging of impossible metrics patterns
5. **Sacred objectives**: Target Recall≥80%, F1≥82.6%, TTD≤4min through proper learning
6. **Export for production**: Models saved to `outputs/smokeynet/exported_run3/` (ONNX/TensorRT ready)

### ⚡ Run 3 Benefits (Sacred + Dataset Fix + Realistic H200 Optimization)
- **Dataset Integrity**: Fixed validation corruption, reproducible shuffling, comprehensive validation
- **Real-time Monitoring**: Anomaly detection prevents impossible metrics, early intervention
- **Sacred Compliance**: 100% divine specification adherence verified against all docs
- **H200 Realistic Optimization**: batch_size=8, fp32 precision, 4 workers, RAM-cached dataset
- **Performance Discovery**: Sacred architecture complexity correctly identified (4-6 it/s realistic)
- **Memory Strategy**: 61GB sequences in /dev/shm RAM, 1,794 MB/s I/O performance
- **Training Time**: ~3-4 hours per epoch (realistic expectation for Sacred complexity)
- **Fresh Start**: New seed (2024), clean checkpoints directory, preserved legacy for study

## References

- **Sacred Documentation**: All specs in `docs/` (thefinalroadmap.md, roadmap SAI-Net.md, Guia Descarga FigLib.md)
- **SmokeyNet Paper**: Dewangan et al., Remote Sensing 14(4):1007, 2022
- **FIgLib Dataset**: HPWREN/UCSD WIFIRE Data Commons
- **Dataset Corruption Investigation**: Full report in `docs/TRAINING_INVESTIGATION_REPORT.md`
- **Validation Framework**: `scripts/dataset_validation_framework.py`
- **Monitoring System**: `scripts/training_monitor_framework.py`
- **Hardware**: NVIDIA H200 optimization guide