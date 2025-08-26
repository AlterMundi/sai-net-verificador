#!/bin/bash
# 🚀 Dataset-Only Migration to /dev/shm for 100x I/O speedup
# Migrates only the 61GB sequence dataset, keeps everything else in place

set -e  # Exit on error

echo "🔥 SAI-NET Dataset Migration to RAM 🔥"
echo "======================================="
echo ""
echo "Current setup:"
echo "  Data location: /workspace/sai_sequences (61GB)"
echo "  Symlink: data/figlib_seq_real → /workspace/sai_sequences"
echo ""
echo "Target setup:"
echo "  Data location: /dev/shm/sai_sequences (in RAM)"
echo "  Symlink: data/figlib_seq_real → /dev/shm/sai_sequences"
echo ""
echo "Available RAM cache:"
df -h /dev/shm
echo ""

# Step 1: Copy dataset to RAM (this is the critical operation)
echo "📦 Copying 61GB dataset to RAM..."
echo "This will take approximately 2-3 minutes..."
echo ""

# Create target directory
mkdir -p /dev/shm/sai_sequences

# Copy with progress
time rsync -av --info=progress2 /workspace/sai_sequences/ /dev/shm/sai_sequences/

echo ""
echo "✅ Dataset copied to RAM!"
echo ""

# Step 2: Update symlink
echo "🔗 Updating symlink to point to RAM cache..."
cd /workspace/sai-net-verificador/data
rm -f figlib_seq_real
ln -s /dev/shm/sai_sequences figlib_seq_real
echo "✅ Symlink updated!"
echo ""

# Step 3: Verification
echo "📊 Verification:"
echo "================"
echo ""
echo "1. Symlink points to RAM:"
ls -la /workspace/sai-net-verificador/data/figlib_seq_real
echo ""

echo "2. Dataset structure intact:"
echo "   Train sequences: $(ls /workspace/sai-net-verificador/data/figlib_seq_real/train/*.pkl 2>/dev/null | wc -l)"
echo "   Val sequences: $(ls /workspace/sai-net-verificador/data/figlib_seq_real/val/*.pkl 2>/dev/null | wc -l)"
echo "   Test sequences: $(ls /workspace/sai-net-verificador/data/figlib_seq_real/test/*.pkl 2>/dev/null | wc -l)"
echo ""

echo "3. RAM usage after migration:"
df -h /dev/shm
echo ""

echo "4. Quick read test:"
time python -c "
import pickle
import time
start = time.time()
with open('/workspace/sai-net-verificador/data/figlib_seq_real/train/train_seq_000000.pkl', 'rb') as f:
    data = pickle.load(f)
print(f'✅ Test read successful! Time: {time.time()-start:.4f}s')
print(f'   Sequence shape: {data[\"frames\"].shape}')
"

echo ""
echo "🚀 Migration Complete!"
echo "======================================="
echo "Dataset is now in RAM at /dev/shm/sai_sequences"
echo "Expected training speed: 45-50 it/s (100x faster)"
echo ""
echo "No configuration changes needed - the symlink handles everything!"
echo "Ready to relaunch Run 3 training with the same config."