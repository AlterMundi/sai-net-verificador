#!/usr/bin/env python3
"""
Test real performance to investigate suspicious 100% recall
"""

import torch
import numpy as np
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent))

from src.dataio.figlib_memory_datamodule import FIgLibMemoryDataModule
from src.verifier.lightning_module import SmokeyNetLightningModule
from omegaconf import OmegaConf

# Load config
config = OmegaConf.load("configs/smokeynet/train_config_h200_optimized.yaml")

# Load best model
model = SmokeyNetLightningModule.load_from_checkpoint(
    "outputs/smokeynet/checkpoints/smokeynet-h200-epochepoch=00-f1val/f1=0.8333.ckpt",
    model_config=config.model,
    learning_rate=config.training.learning_rate,
    weight_decay=config.training.weight_decay,
    max_epochs=config.training.max_epochs
)
model.eval()

# Setup data
dm = FIgLibMemoryDataModule(
    cache_dir=config.data.cache_dir,
    batch_size=1,  # Test one at a time
    num_workers=0
)
dm.setup('fit')  # This creates train and val datasets

# Test on validation set
val_loader = dm.val_dataloader()

print("\n🔍 ANALYZING SUSPICIOUS 100% RECALL\n")
print("="*60)

all_preds = []
all_labels = []
detailed_results = []

with torch.no_grad():
    for i, batch in enumerate(val_loader):
        # Get prediction
        logits = model(batch)
        probs = torch.sigmoid(logits['global_logits'])
        pred = (probs > 0.5).float()
        
        label = batch['label']
        
        all_preds.append(pred.item())
        all_labels.append(label.item())
        
        result = {
            'sample': i,
            'true_label': label.item(),
            'predicted': pred.item(),
            'probability': probs.item(),
            'correct': pred.item() == label.item()
        }
        detailed_results.append(result)
        
        print(f"Sample {i}: True={label.item():.0f}, Pred={pred.item():.0f}, "
              f"Prob={probs.item():.3f}, {'✅' if result['correct'] else '❌'}")

print("\n" + "="*60)
print("SUMMARY STATISTICS:")
print("="*60)

# Calculate metrics
all_preds = np.array(all_preds)
all_labels = np.array(all_labels)

# Confusion matrix
tp = ((all_preds == 1) & (all_labels == 1)).sum()
tn = ((all_preds == 0) & (all_labels == 0)).sum()
fp = ((all_preds == 1) & (all_labels == 0)).sum()
fn = ((all_preds == 0) & (all_labels == 1)).sum()

print(f"\nConfusion Matrix:")
print(f"  True Positives (TP): {tp}")
print(f"  True Negatives (TN): {tn}")
print(f"  False Positives (FP): {fp}")
print(f"  False Negatives (FN): {fn}")

# Metrics
accuracy = (tp + tn) / len(all_labels) if len(all_labels) > 0 else 0
precision = tp / (tp + fp) if (tp + fp) > 0 else 0
recall = tp / (tp + fn) if (tp + fn) > 0 else 1.0  # This is the suspicious one!
f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

print(f"\nMetrics:")
print(f"  Accuracy: {accuracy:.3f}")
print(f"  Precision: {precision:.3f}")
print(f"  Recall: {recall:.3f} {'⚠️ SUSPICIOUS!' if recall == 1.0 else ''}")
print(f"  F1: {f1:.3f}")

# Analyze predictions
print(f"\nPrediction Analysis:")
print(f"  Total samples: {len(all_labels)}")
print(f"  Positive labels: {(all_labels == 1).sum()} ({(all_labels == 1).sum()/len(all_labels)*100:.1f}%)")
print(f"  Negative labels: {(all_labels == 0).sum()} ({(all_labels == 0).sum()/len(all_labels)*100:.1f}%)")
print(f"  Model predicted positive: {(all_preds == 1).sum()} ({(all_preds == 1).sum()/len(all_preds)*100:.1f}%)")
print(f"  Model predicted negative: {(all_preds == 0).sum()} ({(all_preds == 0).sum()/len(all_preds)*100:.1f}%)")

# Check if model is just predicting all positive
if (all_preds == 1).all():
    print("\n⚠️ WARNING: Model is predicting ALL samples as POSITIVE (smoke)!")
    print("This explains the 100% recall but indicates the model is not discriminating.")
elif recall == 1.0:
    print("\n⚠️ WARNING: Model has 100% recall but is not predicting all positive.")
    print("This could indicate:")
    print("  - Very imbalanced/easy validation set")
    print("  - Data leakage between train/val")
    print("  - Overfitting to specific patterns")

# Check probability distribution
print(f"\nProbability Distribution:")
probs_array = [r['probability'] for r in detailed_results]
print(f"  Min probability: {min(probs_array):.3f}")
print(f"  Max probability: {max(probs_array):.3f}")
print(f"  Mean probability: {np.mean(probs_array):.3f}")
print(f"  Std probability: {np.std(probs_array):.3f}")

if np.std(probs_array) < 0.1:
    print("  ⚠️ Very low variance in predictions - model may be overconfident")

print("\n" + "="*60)
print("DIAGNOSIS:")
print("="*60)

if recall == 1.0 and (all_preds == 1).all():
    print("❌ Model is broken - predicting everything as positive")
    print("   This gives 100% recall but terrible precision")
    print("   Likely causes: ")
    print("   - Training collapsed to trivial solution")
    print("   - Loss weighting too aggressive for positive class")
    print("   - Dataset too small/simple")
elif recall == 1.0:
    print("⚠️ Perfect recall is suspicious but model is making some negative predictions")
    print("   Possible issues:")
    print("   - Validation set too easy/small (only 7 samples!)")
    print("   - Data leakage or similar samples in train/val")
else:
    print("✅ Recall is not perfect, metrics seem more realistic")

print("\n🔍 RECOMMENDATION: Train on larger, more diverse dataset")
print("   Current val set has only 7 samples - not statistically significant!")
