#!/usr/bin/env python3
"""
Profile the Sacred SmokeyNet forward pass to identify bottlenecks
NEVER questioning the Sacred architecture - only optimizing execution
"""

import time
import torch
import torch.nn as nn
from omegaconf import OmegaConf
import numpy as np

# Set optimal tensor precision for H200
torch.set_float32_matmul_precision('high')

def profile_forward_pass():
    print("🔍 PROFILING SACRED SMOKEYNET FORWARD PASS")
    print("=" * 50)
    
    # Load sacred config
    config = OmegaConf.load("configs/smokeynet/test_performance_debug.yaml")
    
    # Import Sacred architecture
    from src.verifier.smokeynet_like import SmokeyNetLike
    
    # Create Sacred model
    print("🏗️  Creating Sacred SmokeyNet model...")
    model = SmokeyNetLike(
        num_tiles=config.model.num_tiles,              # 45 tiles (Sacred)
        temporal_window=config.model.temporal_window,   # L=3 (Sacred)
        tile_size=config.model.tile_size,              # 224 (Sacred)
        vit_dim=config.model.vit_dim,                  # 768 (Sacred)
        vit_depth=config.model.vit_depth,              # 6 (Sacred) 
        vit_heads=config.model.vit_heads,              # 12 (Sacred)
        use_tile_heads=config.model.use_tile_heads     # True (Sacred)
    )
    
    model = model.cuda()
    model.train()
    
    # Sacred input dimensions: batch_size=8, L=3, channels=3, height=1040, width=1856 
    batch_size = 8
    L = 3
    channels = 3
    height = 1040
    width = 1856
    
    print(f"📊 Sacred input shape: [{batch_size}, {L}, {channels}, {height}, {width}]")
    
    # Create sacred input tensor (original frame format)
    sacred_input = torch.randn(
        batch_size, L, channels, height, width,
        device='cuda', dtype=torch.float32
    )
    
    print(f"💾 Sacred input tensor size: {sacred_input.element_size() * sacred_input.numel() / (1024**3):.2f} GB")
    
    # Warm up GPU
    print("🔥 GPU warmup (5 iterations)...")
    with torch.no_grad():
        for i in range(5):
            _ = model(sacred_input)
            torch.cuda.synchronize()
    
    # Profile forward pass timing
    print("⚡ Profiling forward pass (10 iterations)...")
    times = []
    
    torch.cuda.synchronize()
    for i in range(10):
        start = time.time()
        
        with torch.no_grad():  # Disable gradients for pure forward timing
            output = model(sacred_input)
            torch.cuda.synchronize()
        
        iteration_time = time.time() - start
        times.append(iteration_time)
        print(f"  Iteration {i+1:2d}: {iteration_time:.3f}s ({1/iteration_time:.2f} it/s)")
    
    # Calculate statistics
    avg_time = np.mean(times)
    std_time = np.std(times)
    min_time = np.min(times)
    max_time = np.max(times)
    
    print()
    print("📈 Forward Pass Analysis:")
    print(f"  Average time: {avg_time:.3f}s ± {std_time:.3f}s")
    print(f"  Min time: {min_time:.3f}s ({1/min_time:.2f} it/s)")
    print(f"  Max time: {max_time:.3f}s ({1/max_time:.2f} it/s)")
    print(f"  Stable it/s: {1/avg_time:.2f}")
    
    # Memory usage
    memory_used = torch.cuda.max_memory_allocated() / (1024**3)
    print(f"  Peak GPU memory: {memory_used:.2f} GB")
    
    # Expected vs Actual
    expected_its = 45  # From H200 optimization claims
    actual_its = 1/avg_time
    performance_ratio = actual_its / expected_its
    
    print()
    print("🎯 Performance Analysis:")
    print(f"  Expected (H200 optimized): {expected_its:.1f} it/s")
    print(f"  Actual (Sacred forward): {actual_its:.2f} it/s") 
    print(f"  Performance ratio: {performance_ratio:.1%}")
    
    if performance_ratio < 0.1:
        print("🚨 CRITICAL: Sacred forward pass 10x+ slower than expected!")
        print("   Potential Sacred architecture bottlenecks to investigate:")
        print("   - ResNet-34 tile encoding (45 tiles × batch)")  
        print("   - Bidirectional LSTM temporal modeling")
        print("   - Vision Transformer spatial reasoning")
        print("   - Multi-head attention (12 heads)")
        print("   - Auxiliary tile classification heads")
    elif performance_ratio < 0.5:
        print("⚠️  WARNING: Sacred forward pass 2-10x slower than expected")
        print("   Some optimization opportunities may exist")
    else:
        print("✅ Sacred forward pass performance is reasonable")
    
    return avg_time, actual_its

if __name__ == "__main__":
    avg_time, its = profile_forward_pass()