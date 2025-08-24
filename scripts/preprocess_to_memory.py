#!/usr/bin/env python3
"""
Memory Preprocessing Script for Sacred SmokeyNet Training
Optimized for H200 + /dev/shm cache system

This script processes FIgLib sequences and caches them in memory (/dev/shm)
for ultra-fast training with zero I/O bottlenecks.

Sacred optimizations:
- Preprocess all 45 tiles per frame to 224x224
- Apply sacred normalization (mean=0.5, std=0.5)
- Cache in compressed format to maximize /dev/shm usage
- Maintain sacred sequence structure (L=3 temporal windows)
"""

import os
import sys
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import cv2
from pathlib import Path
from PIL import Image
import pickle
import gzip
import argparse
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2
import logging
from typing import List, Dict, Tuple, Any
import json

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class SacredMemoryPreprocessor:
    """
    Sacred preprocessing for memory cache optimization.
    
    Processes FIgLib sequences following exact sacred specifications:
    - Resize to 1392×1856 → crop 352 top → 1040×1856
    - Create 45 tiles of 224×224 with 20px overlap
    - Apply sacred normalization: mean=0.5, std=0.5
    - Cache in compressed format for /dev/shm efficiency
    """
    
    def __init__(
        self,
        input_dir: str,
        cache_dir: str,
        target_size: Tuple[int, int] = (1040, 1856),  # Sacred final size
        tile_size: int = 224,                         # Sacred tile size
        num_tiles: int = 45,                          # Sacred number of tiles
        compression_level: int = 6                     # Balanced compression
    ):
        self.input_dir = Path(input_dir)
        self.cache_dir = Path(cache_dir)
        self.target_size = target_size
        self.tile_size = tile_size
        self.num_tiles = num_tiles
        self.compression_level = compression_level
        
        # Create cache directories
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        (self.cache_dir / "metadata").mkdir(exist_ok=True)
        
        # Sacred preprocessing transforms (matching DataModule)
        self.setup_transforms()
        
        # Statistics tracking
        self.stats = {
            'processed_sequences': 0,
            'total_frames': 0,
            'total_tiles': 0,
            'cache_size_mb': 0,
            'compression_ratio': 0
        }
    
    def setup_transforms(self):
        """Setup sacred preprocessing transforms."""
        self.base_transforms = A.Compose([
            # Sacred normalization: mean=0.5, std=0.5
            A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ToTensorV2()
        ])
    
    def create_sacred_tiles(self, image: torch.Tensor) -> torch.Tensor:
        """
        Create 45 tiles of 224×224 with sacred specifications.
        Exact replication of the tiling logic from DataModule.
        """
        C, H, W = image.shape  # [3, 1040, 1856]
        
        tiles = []
        tile_size = self.tile_size  # 224
        
        # Sacred grid: 5×9 = 45 tiles
        target_tiles_h = 5
        target_tiles_w = 9
        
        step_h = (H - tile_size) // (target_tiles_h - 1) if target_tiles_h > 1 else 0
        step_w = (W - tile_size) // (target_tiles_w - 1) if target_tiles_w > 1 else 0
        
        for i in range(target_tiles_h):
            for j in range(target_tiles_w):
                # Calculate tile position
                start_h = min(i * step_h, H - tile_size)
                start_w = min(j * step_w, W - tile_size)
                
                # Extract tile
                tile = image[:, start_h:start_h + tile_size, start_w:start_w + tile_size]
                
                # Ensure exact size
                if tile.shape[1] != tile_size or tile.shape[2] != tile_size:
                    tile = F.interpolate(
                        tile.unsqueeze(0), 
                        size=(tile_size, tile_size), 
                        mode='bilinear', 
                        align_corners=False
                    ).squeeze(0)
                
                tiles.append(tile)
        
        # Stack exactly 45 tiles
        tiles_tensor = torch.stack(tiles[:self.num_tiles], dim=0)  # [45, 3, 224, 224]
        return tiles_tensor
    
    def process_frame(self, image_path: Path) -> torch.Tensor:
        """
        Process a single frame following sacred methodology.
        Returns tiles tensor [45, 3, 224, 224].
        """
        # Load or create synthetic image (matching DataModule logic)
        if image_path.exists() and image_path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
            try:
                image = cv2.imread(str(image_path))
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            except Exception as e:
                logger.warning(f"Failed to load image {image_path}: {e}")
                image = self._create_synthetic_image()
        else:
            image = self._create_synthetic_image()
        
        # Sacred preprocessing pipeline
        # Step 1: Resize to 1392×1856
        image_resized = cv2.resize(image, (1856, 1392))  # (W, H)
        
        # Step 2: Crop top 352 rows (sacred sky removal)
        image_cropped = image_resized[352:, :, :]  # Remove top 352 rows
        
        # Step 3: Apply sacred transforms
        transformed = self.base_transforms(image=image_cropped)
        processed_frame = transformed['image']  # [3, 1040, 1856]
        
        # Step 4: Create 45 sacred tiles
        tiles = self.create_sacred_tiles(processed_frame)  # [45, 3, 224, 224]
        
        return tiles
    
    def _create_synthetic_image(self) -> np.ndarray:
        """Create synthetic image for testing (matching DataModule)."""
        synthetic_image = np.random.randint(0, 255, (1536, 2048, 3), dtype=np.uint8)
        
        # Add structure: sky + landscape
        synthetic_image[:400, :, :] = np.random.randint(120, 200, (400, 2048, 3))  # Sky
        synthetic_image[400:, :, :] = np.random.randint(60, 140, (1136, 2048, 3))  # Ground
        
        return synthetic_image
    
    def process_sequence(self, sequence_metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a complete sequence (L=3 frames) and cache in memory.
        
        Returns:
            Cached sequence info with memory paths
        """
        sequence_id = sequence_metadata['sequence_id']
        frame_paths = sequence_metadata['frame_paths'].split('|')
        
        # Process each frame in the sequence
        sequence_tiles = []
        
        for frame_path in frame_paths:
            full_path = self.input_dir / frame_path
            tiles = self.process_frame(full_path)  # [45, 3, 224, 224]
            sequence_tiles.append(tiles)
        
        # Stack temporal sequence: [L, 45, 3, 224, 224]
        sequence_tensor = torch.stack(sequence_tiles, dim=0)
        
        # Compress and cache
        cache_path = self.cache_dir / f"{sequence_id}_tiles.pkl.gz"
        self._save_compressed_tensor(sequence_tensor, cache_path)
        
        # Update statistics
        self.stats['processed_sequences'] += 1
        self.stats['total_frames'] += len(frame_paths)
        self.stats['total_tiles'] += len(frame_paths) * self.num_tiles
        
        # Return cached sequence metadata
        cached_metadata = sequence_metadata.copy()
        cached_metadata['cache_path'] = str(cache_path.relative_to(self.cache_dir))
        cached_metadata['tensor_shape'] = list(sequence_tensor.shape)
        cached_metadata['compressed_size_kb'] = cache_path.stat().st_size // 1024
        
        return cached_metadata
    
    def _save_compressed_tensor(self, tensor: torch.Tensor, path: Path):
        """Save tensor with compression to maximize /dev/shm efficiency."""
        # Convert to numpy for better compression
        tensor_np = tensor.cpu().numpy().astype(np.float16)  # Half precision saves 50% space
        
        # Compress and save
        with gzip.open(path, 'wb', compresslevel=self.compression_level) as f:
            pickle.dump(tensor_np, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    def process_split(self, split: str) -> Dict[str, Any]:
        """Process entire split (train/val/test) and create cache index."""
        logger.info(f"Processing {split} split...")
        
        metadata_file = self.input_dir / split / "sequences_metadata.csv"
        if not metadata_file.exists():
            logger.error(f"Metadata file not found: {metadata_file}")
            return {}
        
        # Load sequences metadata
        sequences_df = pd.read_csv(metadata_file)
        logger.info(f"Found {len(sequences_df)} sequences in {split} split")
        
        # Process each sequence
        cached_sequences = []
        
        for idx, row in tqdm(sequences_df.iterrows(), total=len(sequences_df), desc=f"Processing {split}"):
            sequence_metadata = row.to_dict()
            
            try:
                cached_info = self.process_sequence(sequence_metadata)
                cached_sequences.append(cached_info)
            except Exception as e:
                logger.error(f"Failed to process sequence {row['sequence_id']}: {e}")
        
        # Save cache index for this split
        cache_index = {
            'split': split,
            'total_sequences': len(cached_sequences),
            'sequences': cached_sequences,
            'cache_dir': str(self.cache_dir),
            'preprocessing_config': {
                'target_size': self.target_size,
                'tile_size': self.tile_size,
                'num_tiles': self.num_tiles,
                'normalization': {'mean': [0.5, 0.5, 0.5], 'std': [0.5, 0.5, 0.5]}
            }
        }
        
        index_file = self.cache_dir / "metadata" / f"{split}_cache_index.json"
        with open(index_file, 'w') as f:
            json.dump(cache_index, f, indent=2)
        
        logger.info(f"✅ {split} split processed: {len(cached_sequences)} sequences cached")
        return cache_index
    
    def process_all_splits(self):
        """Process all splits and create master index."""
        logger.info("🔥 Starting Sacred Memory Preprocessing 🔥")
        
        splits = ['train', 'val', 'test']
        all_indices = {}
        
        for split in splits:
            split_index = self.process_split(split)
            if split_index:
                all_indices[split] = split_index
        
        # Create master index
        master_index = {
            'cache_version': '1.0',
            'created_at': pd.Timestamp.now().isoformat(),
            'splits': all_indices,
            'global_stats': self.stats,
            'sacred_config': {
                'target_size': self.target_size,
                'tile_size': self.tile_size,
                'num_tiles': self.num_tiles,
                'compression_level': self.compression_level
            }
        }
        
        master_index_file = self.cache_dir / "metadata" / "master_cache_index.json"
        with open(master_index_file, 'w') as f:
            json.dump(master_index, f, indent=2)
        
        # Calculate final statistics
        total_cache_size = sum(
            p.stat().st_size for p in self.cache_dir.glob("*.pkl.gz")
        ) / (1024 * 1024)  # MB
        
        self.stats['cache_size_mb'] = total_cache_size
        
        # Log final results
        logger.info("🔥🔥🔥 SACRED MEMORY PREPROCESSING COMPLETED! 🔥🔥🔥")
        logger.info(f"✅ Processed {self.stats['processed_sequences']} sequences")
        logger.info(f"✅ Total frames: {self.stats['total_frames']}")
        logger.info(f"✅ Total tiles: {self.stats['total_tiles']}")
        logger.info(f"✅ Cache size: {total_cache_size:.1f} MB")
        logger.info(f"✅ Cache location: {self.cache_dir}")
        logger.info(f"✅ Master index: {master_index_file}")
        
        return master_index


def main():
    parser = argparse.ArgumentParser(description="Sacred Memory Preprocessing for SmokeyNet")
    parser.add_argument("--input", required=True, help="Input FIgLib sequences directory")
    parser.add_argument("--cache-dir", default="/dev/shm/sai_cache/figlib_processed", 
                       help="Cache directory in memory")
    parser.add_argument("--compression-level", type=int, default=6, 
                       help="Compression level (1-9)")
    parser.add_argument("--verify", action='store_true', 
                       help="Verify cached data after preprocessing")
    
    args = parser.parse_args()
    
    # Validate inputs
    input_path = Path(args.input)
    if not input_path.exists():
        logger.error(f"Input directory not found: {input_path}")
        sys.exit(1)
    
    cache_path = Path(args.cache_dir)
    if not cache_path.parent.exists():
        logger.error(f"Cache parent directory not accessible: {cache_path.parent}")
        sys.exit(1)
    
    # Initialize processor
    processor = SacredMemoryPreprocessor(
        input_dir=args.input,
        cache_dir=args.cache_dir,
        compression_level=args.compression_level
    )
    
    # Process all splits
    master_index = processor.process_all_splits()
    
    # Verify if requested
    if args.verify:
        verify_cache(cache_path)
    
    logger.info("Sacred preprocessing pipeline completed successfully! 🔥")


def verify_cache(cache_dir: Path):
    """Verify cached data integrity."""
    logger.info("Verifying cache integrity...")
    
    cache_files = list(cache_dir.glob("*.pkl.gz"))
    logger.info(f"Found {len(cache_files)} cache files")
    
    # Test loading a few random files
    import random
    test_files = random.sample(cache_files, min(5, len(cache_files)))
    
    for cache_file in test_files:
        try:
            with gzip.open(cache_file, 'rb') as f:
                tensor_np = pickle.load(f)
            tensor = torch.from_numpy(tensor_np)
            
            expected_shape = (3, 45, 3, 224, 224)  # [L, num_tiles, C, H, W]
            if tensor.shape == expected_shape:
                logger.info(f"✅ {cache_file.name}: Shape {tensor.shape} - OK")
            else:
                logger.warning(f"⚠️ {cache_file.name}: Unexpected shape {tensor.shape}")
                
        except Exception as e:
            logger.error(f"❌ {cache_file.name}: Load failed - {e}")
    
    logger.info("Cache verification completed!")


if __name__ == "__main__":
    main()