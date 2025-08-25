"""
FIgLib Memory-Optimized DataModule - Sacred Implementation with /dev/shm Cache
Optimized for H200 + memory cache system for ultra-fast training.

Key optimizations:
- Loads preprocessed tiles from /dev/shm cache (zero I/O bottleneck)
- Maintains sacred specifications while maximizing performance
- Supports both disk and memory cache modes
- Optimized for single H200 GPU with 119GB /dev/shm available
"""

import os
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
import pandas as pd
import numpy as np
from pathlib import Path
import pickle
import gzip
import json
from typing import Dict, List, Tuple, Optional, Any
import logging
from concurrent.futures import ThreadPoolExecutor
import warnings

logger = logging.getLogger(__name__)


class FIgLibMemoryDataset(Dataset):
    """
    Memory-optimized FIgLib Dataset that loads from /dev/shm cache.
    
    Sacred specifications maintained:
    - L=3 temporal windows
    - 45 tiles of 224×224 per frame
    - Sacred normalization (mean=0.5, std=0.5)
    - Binary sequence labels (smoke/no-smoke)
    """
    
    def __init__(
        self,
        cache_index_path: str,
        split: str = 'train',
        use_memory_cache: bool = True,
        preload_all: bool = False,
        max_cache_size_gb: int = 50
    ):
        self.cache_index_path = Path(cache_index_path)
        self.split = split
        self.use_memory_cache = use_memory_cache
        self.preload_all = preload_all
        self.max_cache_size_gb = max_cache_size_gb
        
        # Load cache index
        self.cache_index = self._load_cache_index()
        self.sequences = self.cache_index['sequences']
        
        # Memory management
        self.memory_cache = {}
        self.cache_hits = 0
        self.cache_misses = 0
        
        # Preload all data if requested and feasible
        if self.preload_all:
            self._preload_all_sequences()
        
        logger.info(f"Initialized {split} dataset: {len(self.sequences)} sequences")
        if self.use_memory_cache:
            logger.info(f"Memory cache enabled, max size: {max_cache_size_gb}GB")
    
    def _load_cache_index(self) -> Dict[str, Any]:
        """Load cache index from preprocessed metadata."""
        if not self.cache_index_path.exists():
            raise FileNotFoundError(f"Cache index not found: {self.cache_index_path}")
        
        with open(self.cache_index_path, 'r') as f:
            cache_index = json.load(f)
        
        # Handle both individual split files and master index structure
        if 'splits' in cache_index:
            # Master cache index format
            if self.split not in cache_index['splits']:
                raise ValueError(f"Split '{self.split}' not found in cache index")
            return cache_index['splits'][self.split]
        else:
            # Individual split file format - return as-is
            return cache_index
    
    def _preload_all_sequences(self):
        """Preload all sequences into memory cache."""
        logger.info(f"Preloading all {len(self.sequences)} sequences to memory...")
        
        # Estimate memory usage
        if self.sequences:
            sample_path = self._get_cache_path(self.sequences[0])
            sample_size = sample_path.stat().st_size if sample_path.exists() else 1024*1024
            estimated_size_gb = (len(self.sequences) * sample_size) / (1024**3)
            
            if estimated_size_gb > self.max_cache_size_gb:
                logger.warning(f"Estimated cache size ({estimated_size_gb:.1f}GB) exceeds limit ({self.max_cache_size_gb}GB)")
                logger.warning("Skipping preload to avoid memory issues")
                return
        
        # Preload with progress tracking
        from tqdm import tqdm
        
        successful_preloads = 0
        for i, sequence_meta in enumerate(tqdm(self.sequences, desc="Preloading")):
            try:
                tensor = self._load_sequence_tensor(sequence_meta)
                if tensor is not None:
                    self.memory_cache[i] = tensor
                    successful_preloads += 1
            except Exception as e:
                logger.warning(f"Failed to preload sequence {i}: {e}")
        
        logger.info(f"Successfully preloaded {successful_preloads}/{len(self.sequences)} sequences")
    
    def _get_cache_path(self, sequence_meta: Dict[str, Any]) -> Path:
        """Get full cache path for sequence."""
        cache_dir = Path(self.cache_index['cache_dir'])
        cache_filename = sequence_meta['cache_path']
        return cache_dir / cache_filename
    
    def _load_sequence_tensor(self, sequence_meta: Dict[str, Any]) -> Optional[torch.Tensor]:
        """Load sequence tensor from compressed cache."""
        cache_path = self._get_cache_path(sequence_meta)
        
        if not cache_path.exists():
            logger.warning(f"Cache file not found: {cache_path}")
            return None
        
        try:
            with gzip.open(cache_path, 'rb') as f:
                tensor_np = pickle.load(f)
            
            # Convert back to torch tensor
            tensor = torch.from_numpy(tensor_np).float()  # Convert from float16 to float32
            
            # Expected shape: [L=3, num_tiles=45, C=3, H=224, W=224]
            expected_shape = (3, 45, 3, 224, 224)
            if tensor.shape != expected_shape:
                logger.warning(f"Unexpected tensor shape: {tensor.shape}, expected {expected_shape}")
            
            return tensor
            
        except Exception as e:
            logger.error(f"Failed to load cached tensor from {cache_path}: {e}")
            return None
    
    def __len__(self) -> int:
        return len(self.sequences)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get sequence sample from memory cache.
        
        Returns:
            Dict containing:
            - 'frames': [L, C, H, W] - Temporal sequence (not used, kept for compatibility)
            - 'tiles': [L, num_tiles, C, tile_H, tile_W] - Sacred tiled frames from cache
            - 'label': scalar - Sequence label (1=smoke, 0=no-smoke)
            - 'metadata': dict - Additional information
        """
        sequence_meta = self.sequences[idx]
        
        # Try memory cache first
        if idx in self.memory_cache:
            tiles_tensor = self.memory_cache[idx]
            self.cache_hits += 1
        else:
            # Load from compressed cache
            tiles_tensor = self._load_sequence_tensor(sequence_meta)
            self.cache_misses += 1
            
            if tiles_tensor is None:
                # Fallback: create dummy tensor with correct shape
                logger.warning(f"Creating fallback tensor for sequence {idx}")
                tiles_tensor = torch.zeros(3, 45, 3, 224, 224)
            
            # Cache in memory if space allows
            if self.use_memory_cache and len(self.memory_cache) < 1000:  # Reasonable limit
                self.memory_cache[idx] = tiles_tensor
        
        # Extract components
        L, num_tiles, C, H, W = tiles_tensor.shape
        
        # For compatibility with original interface, create 'frames' 
        # by taking the center tile from each frame
        center_tile_idx = num_tiles // 2  # Middle tile
        frames = tiles_tensor[:, center_tile_idx, :, :, :]  # [L, C, H, W]
        
        # Sacred label
        label = torch.tensor(sequence_meta['label'], dtype=torch.float32)
        
        return {
            'frames': frames,           # [L=3, C=3, H=224, W=224] - for compatibility
            'tiles': tiles_tensor,      # [L=3, 45, C=3, H=224, W=224] - sacred tiles
            'label': label,             # scalar - sacred binary label
            'metadata': {
                'sequence_id': sequence_meta['sequence_id'],
                'event_id': sequence_meta['event_id'],
                'start_offset': sequence_meta.get('start_offset', 0),
                'end_offset': sequence_meta.get('end_offset', 0),
                'cache_hit': idx in self.memory_cache
            }
        }
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache performance statistics."""
        total_requests = self.cache_hits + self.cache_misses
        hit_rate = self.cache_hits / total_requests if total_requests > 0 else 0
        
        return {
            'cache_hits': self.cache_hits,
            'cache_misses': self.cache_misses,
            'hit_rate': hit_rate,
            'memory_cached_sequences': len(self.memory_cache),
            'total_sequences': len(self.sequences)
        }


class FIgLibMemoryDataModule(pl.LightningDataModule):
    """
    Sacred Memory-Optimized DataModule for H200 system.
    
    Optimizations:
    - Loads from /dev/shm cache (zero I/O bottleneck)
    - Optimized batch sizes for H200 (143GB VRAM)
    - Enhanced worker configuration for 192 CPU cores
    - Maintains all sacred specifications
    """
    
    def __init__(
        self,
        cache_dir: str = "/dev/shm/sai_cache/figlib_processed",
        batch_size: int = 10,        # Optimized for H200
        num_workers: int = 16,       # Optimized for 192 cores
        temporal_window: int = 3,    # Sacred L=3
        tile_size: int = 224,        # Sacred specification
        num_tiles: int = 45,         # Sacred specification
        pin_memory: bool = True,
        persistent_workers: bool = True,
        prefetch_factor: int = 4,    # Optimized for abundant RAM
        use_memory_cache: bool = True,
        preload_all: bool = False,   # Can enable for smaller datasets
        max_cache_size_gb: int = 40  # Conservative limit for /dev/shm
    ):
        super().__init__()
        
        self.cache_dir = Path(cache_dir)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.temporal_window = temporal_window
        self.tile_size = tile_size
        self.num_tiles = num_tiles
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers
        self.prefetch_factor = prefetch_factor
        self.use_memory_cache = use_memory_cache
        self.preload_all = preload_all
        self.max_cache_size_gb = max_cache_size_gb
        
        # Cache index paths
        self.metadata_dir = self.cache_dir / "metadata"
        self.train_index = self.metadata_dir / "train_cache_index.json"
        self.val_index = self.metadata_dir / "val_cache_index.json"
        self.test_index = self.metadata_dir / "test_cache_index.json"
        self.master_index = self.metadata_dir / "master_cache_index.json"
        
        # Validate cache exists
        self._validate_cache()
    
    def _validate_cache(self):
        """Validate that cache directory and indices exist."""
        if not self.cache_dir.exists():
            raise FileNotFoundError(f"Cache directory not found: {self.cache_dir}")
        
        if not self.metadata_dir.exists():
            raise FileNotFoundError(f"Cache metadata directory not found: {self.metadata_dir}")
        
        missing_indices = []
        for split, index_path in [
            ('train', self.train_index),
            ('val', self.val_index),
            ('test', self.test_index)
        ]:
            if not index_path.exists():
                missing_indices.append(split)
        
        if missing_indices:
            raise FileNotFoundError(
                f"Missing cache indices for splits: {missing_indices}. "
                "Please run preprocess_to_memory.py first."
            )
        
        logger.info("✅ Cache validation passed")
    
    def setup(self, stage: Optional[str] = None):
        """Setup datasets for different stages."""
        
        if stage == 'fit' or stage is None:
            self.train_dataset = FIgLibMemoryDataset(
                cache_index_path=self.train_index,
                split='train',
                use_memory_cache=self.use_memory_cache,
                preload_all=self.preload_all,
                max_cache_size_gb=self.max_cache_size_gb
            )
            
            self.val_dataset = FIgLibMemoryDataset(
                cache_index_path=self.val_index,
                split='val',
                use_memory_cache=self.use_memory_cache,
                preload_all=self.preload_all,
                max_cache_size_gb=self.max_cache_size_gb // 4  # Smaller cache for val
            )
        
        if stage == 'test' or stage is None:
            self.test_dataset = FIgLibMemoryDataset(
                cache_index_path=self.test_index,
                split='test',
                use_memory_cache=self.use_memory_cache,
                preload_all=self.preload_all,
                max_cache_size_gb=self.max_cache_size_gb // 4  # Smaller cache for test
            )
    
    def train_dataloader(self) -> DataLoader:
        # Configure dataloader based on num_workers
        dataloader_kwargs = {
            'batch_size': self.batch_size,
            'shuffle': True,
            'num_workers': self.num_workers,
            'pin_memory': self.pin_memory,
            'drop_last': True
        }
        
        # Only add multiprocessing options if num_workers > 0
        if self.num_workers > 0:
            dataloader_kwargs.update({
                'persistent_workers': self.persistent_workers,
                'prefetch_factor': self.prefetch_factor
            })
        
        return DataLoader(self.train_dataset, **dataloader_kwargs)
    
    def val_dataloader(self) -> DataLoader:
        # Configure dataloader based on num_workers
        dataloader_kwargs = {
            'batch_size': self.batch_size,
            'shuffle': False,
            'num_workers': self.num_workers,
            'pin_memory': self.pin_memory
        }
        
        # Only add multiprocessing options if num_workers > 0
        if self.num_workers > 0:
            dataloader_kwargs.update({
                'persistent_workers': self.persistent_workers,
                'prefetch_factor': self.prefetch_factor
            })
        
        return DataLoader(self.val_dataset, **dataloader_kwargs)
    
    def test_dataloader(self) -> DataLoader:
        # Configure dataloader based on num_workers
        dataloader_kwargs = {
            'batch_size': self.batch_size,
            'shuffle': False,
            'num_workers': self.num_workers,
            'pin_memory': self.pin_memory
        }
        
        # Only add multiprocessing options if num_workers > 0
        if self.num_workers > 0:
            dataloader_kwargs.update({
                'persistent_workers': self.persistent_workers,
                'prefetch_factor': self.prefetch_factor
            })
        
        return DataLoader(self.test_dataset, **dataloader_kwargs)
    
    def get_dataset_stats(self) -> Dict[str, Any]:
        """Get comprehensive dataset statistics."""
        stats = {}
        
        if hasattr(self, 'master_index') and self.master_index.exists():
            try:
                with open(self.master_index, 'r') as f:
                    master_index = json.load(f)
                
                for split_name, split_data in master_index['splits'].items():
                    stats[split_name] = {
                        'total_sequences': split_data['total_sequences'],
                        'positive_sequences': sum(1 for seq in split_data['sequences'] if seq['label'] == 1),
                        'negative_sequences': sum(1 for seq in split_data['sequences'] if seq['label'] == 0),
                        'unique_events': len(set(seq['event_id'] for seq in split_data['sequences'])),
                    }
                    stats[split_name]['balance_ratio'] = (
                        stats[split_name]['positive_sequences'] / stats[split_name]['total_sequences']
                        if stats[split_name]['total_sequences'] > 0 else 0
                    )
                
                # Add cache information
                stats['cache_info'] = {
                    'cache_dir': str(self.cache_dir),
                    'cache_size_mb': master_index['global_stats']['cache_size_mb'],
                    'preprocessing_config': master_index['sacred_config']
                }
                
            except Exception as e:
                logger.warning(f"Could not load master index: {e}")
        
        return stats
    
    def get_cache_performance(self) -> Dict[str, Any]:
        """Get cache performance statistics from all datasets."""
        performance = {}
        
        for split, dataset in [
            ('train', getattr(self, 'train_dataset', None)),
            ('val', getattr(self, 'val_dataset', None)),
            ('test', getattr(self, 'test_dataset', None))
        ]:
            if dataset is not None:
                performance[split] = dataset.get_cache_stats()
        
        return performance


if __name__ == "__main__":
    # Test the memory-optimized datamodule
    cache_dir = "/dev/shm/sai_cache/figlib_processed"
    
    if Path(cache_dir).exists():
        try:
            dm = FIgLibMemoryDataModule(
                cache_dir=cache_dir,
                batch_size=2,
                num_workers=0,  # For testing
                preload_all=False
            )
            
            dm.setup('fit')
            
            # Test train dataloader
            train_loader = dm.train_dataloader()
            
            logger.info("Testing Memory-Optimized DataModule...")
            
            for batch_idx, batch in enumerate(train_loader):
                print(f"Sacred Memory DataModule Test:")
                print(f"Frames shape: {batch['frames'].shape}")
                print(f"Tiles shape: {batch['tiles'].shape}")
                print(f"Labels: {batch['label']}")
                print(f"Cache hits: {[meta['cache_hit'] for meta in batch['metadata']['cache_hit']]}")
                
                if batch_idx == 0:  # Test only first batch
                    break
            
            # Print performance stats
            performance = dm.get_cache_performance()
            print("\nCache Performance:")
            for split, stats in performance.items():
                print(f"{split}: {stats}")
            
            print("Sacred memory datamodule test completed successfully! 🔥")
            
        except FileNotFoundError as e:
            print(f"Cache not found: {e}")
            print("Please run: python scripts/preprocess_to_memory.py --input data/figlib_seq")
        except Exception as e:
            print(f"Test failed: {e}")
            import traceback
            traceback.print_exc()
    else:
        print(f"Cache directory not found: {cache_dir}")
        print("Please run the preprocessing script first.")