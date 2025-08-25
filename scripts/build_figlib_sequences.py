#!/usr/bin/env python3
"""
FIgLib Sequence Builder
Sacred implementation following the exact specifications from roadmap documentation.

Builds temporal sequences with L=3 frames (stride=1) and splits by event (70/15/15).
Critical: NEVER mix frames from the same fire event between train/val/test splits.
"""

import os
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
import csv
import logging
from typing import List, Dict, Tuple
import shutil
import re

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FIgLibSequenceBuilder:
    """
    Builds FIgLib sequences following sacred SmokeyNet methodology.
    
    Key requirements from documentation:
    - L=3 temporal window (3 consecutive frames)
    - stride=1 (sliding window)
    - Split by event (70/15/15) - NEVER mix frames from same fire
    - Balance positives (post-ignition) and negatives (pre-ignition)
    """
    
    def __init__(self, raw_root: str, out_root: str, L: int = 3, stride: int = 1):
        self.raw_root = Path(raw_root)
        self.out_root = Path(out_root)
        self.L = L  # Temporal window length
        self.stride = stride  # Stride for sliding window
        
        # Create output directories
        self.out_root.mkdir(parents=True, exist_ok=True)
        for split in ['train', 'val', 'test']:
            (self.out_root / split).mkdir(parents=True, exist_ok=True)
    
    def load_global_labels(self) -> pd.DataFrame:
        """Load global labels CSV from real FIgLib dataset."""
        labels_file = self.raw_root / "labels.csv"
        if not labels_file.exists():
            raise FileNotFoundError(f"Labels file not found: {labels_file}")
        
        df = pd.read_csv(labels_file)
        
        # Adapt real_figlib format to expected format
        df = self._adapt_real_figlib_format(df)
        
        logger.info(f"Loaded {len(df)} image labels from {labels_file}")
        return df
    
    def _adapt_real_figlib_format(self, df: pd.DataFrame) -> pd.DataFrame:
        """Adapt real FIgLib dataset format to expected format."""
        logger.info("Adapting real FIgLib dataset format...")
        
        # Map event_name to event_id
        df['event_id'] = df['event_name']
        
        # Extract offset_seconds from filename pattern: TIMESTAMP_±OFFSET.jpg
        def extract_offset(filename: str) -> int:
            match = re.search(r'_([-+]\d+)\.jpg$', filename)
            if match:
                return int(match.group(1))
            else:
                logger.warning(f"Could not extract offset from filename: {filename}")
                return 0
        
        df['offset_seconds'] = df['filename'].apply(extract_offset)
        logger.info(f"Extracted offsets range: {df['offset_seconds'].min()} to {df['offset_seconds'].max()} seconds")
        
        # 🔥 SACRED LABELING RULE from divine documentation (Guia Descarga FigLib.md):
        # "etiquete como no-smoke los frames con offset negativo y como smoke aquellos con offset ≥ 0"
        # "cualquier imagen con 'offset_0' o positivo tiene humo"
        def apply_sacred_labeling(offset_seconds: int) -> int:
            """Apply sacred temporal labeling based on ignition timing"""
            if offset_seconds < 0:
                return 0  # pre-ignition, no smoke visible yet
            else:
                return 1  # post-ignition, smoke present
        
        # Override existing labels with sacred temporal labeling
        df['label'] = df['offset_seconds'].apply(apply_sacred_labeling)
        
        # Log sacred labeling statistics
        negative_count = len(df[df['label'] == 0])
        positive_count = len(df[df['label'] == 1])
        logger.info(f"🔥 Sacred labeling applied:")
        logger.info(f"  - Pre-ignition (negative): {negative_count} frames")
        logger.info(f"  - Post-ignition (positive): {positive_count} frames")
        logger.info(f"  - Balance ratio: {positive_count / len(df):.3f}")
        
        return df
    
    def group_by_events(self, df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """Group images by fire events."""
        events = {}
        for event_id, group in df.groupby('event_id'):
            # Sort by offset_seconds for proper temporal order
            group_sorted = group.sort_values('offset_seconds')
            events[event_id] = group_sorted
            logger.info(f"Event {event_id}: {len(group_sorted)} frames, offset range: {group_sorted['offset_seconds'].min()}s to {group_sorted['offset_seconds'].max()}s")
        
        return events
    
    def create_sequences(self, event_data: pd.DataFrame, event_id: str) -> List[Dict]:
        """
        Create temporal sequences of length L with stride for one event.
        
        Returns list of sequences, each containing:
        - frames: list of L consecutive frames
        - label: sequence label (1 if any frame has smoke, 0 otherwise)
        - event_id: fire event identifier
        """
        sequences = []
        frames = event_data.to_dict('records')
        
        # Create sliding windows of L frames
        for i in range(0, len(frames) - self.L + 1, self.stride):
            sequence_frames = frames[i:i + self.L]
            
            # Sequence is positive if ANY frame in the sequence has smoke
            # Following sacred documentation: detect smoke in temporal context
            sequence_label = max([frame['label'] for frame in sequence_frames])
            
            sequence = {
                'frames': sequence_frames,
                'label': sequence_label,
                'event_id': event_id,
                'start_offset': sequence_frames[0]['offset_seconds'],
                'end_offset': sequence_frames[-1]['offset_seconds']
            }
            sequences.append(sequence)
        
        logger.info(f"Created {len(sequences)} sequences for event {event_id}")
        return sequences
    
    def split_events_by_ratio(self, events: Dict[str, pd.DataFrame], 
                             ratios: Tuple[float, float, float]) -> Dict[str, List[str]]:
        """
        Split events (not individual frames) into train/val/test.
        Sacred rule: NEVER mix frames from same fire between splits.
        """
        train_ratio, val_ratio, test_ratio = ratios
        assert abs(sum(ratios) - 1.0) < 1e-6, "Ratios must sum to 1.0"
        
        event_ids = list(events.keys())
        np.random.seed(42)  # Reproducible splits
        np.random.shuffle(event_ids)
        
        n_events = len(event_ids)
        n_train = int(n_events * train_ratio)
        n_val = int(n_events * val_ratio)
        
        splits = {
            'train': event_ids[:n_train],
            'val': event_ids[n_train:n_train + n_val],
            'test': event_ids[n_train + n_val:]
        }
        
        logger.info(f"Event splits: train={len(splits['train'])}, val={len(splits['val'])}, test={len(splits['test'])}")
        return splits
    
    def save_sequences(self, sequences: List[Dict], split: str):
        """Save sequences to split directory with metadata."""
        split_dir = self.out_root / split
        metadata_file = split_dir / "sequences_metadata.csv"
        
        # Save sequence metadata
        metadata_rows = []
        for idx, seq in enumerate(sequences):
            seq_id = f"{split}_seq_{idx:06d}"
            
            # Create sequence directory
            seq_dir = split_dir / seq_id
            seq_dir.mkdir(exist_ok=True)
            
            # Copy real wildfire image files from FIgLib dataset
            frame_paths = []
            for frame_idx, frame in enumerate(seq['frames']):
                # Handle different path structures in real_figlib
                relative_path = frame['relative_path']
                if relative_path.startswith('Data/HPWREN-FIgLib/HPWREN-FIgLib-Data/'):
                    # Path like: Data/HPWREN-FIgLib/HPWREN-FIgLib-Data/eventname/file.jpg
                    # This is relative to the event directory, not the root
                    src_path = self.raw_root / seq['event_id'] / relative_path
                else:
                    # Path like: eventname/file.jpg (from labels.csv)
                    # But actual file might be in nested structure: eventname/eventname/file.jpg
                    basic_path = self.raw_root / relative_path
                    
                    # If basic path doesn't exist, try nested structure
                    if not basic_path.exists():
                        # Extract filename from relative_path
                        path_parts = relative_path.split('/')
                        if len(path_parts) == 2:
                            event_name, filename = path_parts
                            # Try nested: eventname/eventname/filename
                            nested_path = self.raw_root / event_name / event_name / filename
                            if nested_path.exists():
                                src_path = nested_path
                            else:
                                src_path = basic_path  # Will fail, but log the correct attempted path
                        else:
                            src_path = basic_path
                    else:
                        src_path = basic_path
                dst_path = seq_dir / f"frame_{frame_idx:02d}.jpg"
                
                # Copy real wildfire image
                if src_path.exists():
                    shutil.copy2(src_path, dst_path)
                    frame_paths.append(dst_path.relative_to(self.out_root))
                else:
                    logger.warning(f"Source image not found: {src_path} (from {frame['relative_path']})")
                    # Create placeholder if source missing
                    dst_path.write_text(f"# Missing: {frame['filename']}")
                    frame_paths.append(dst_path.relative_to(self.out_root))
            
            # Record sequence metadata
            metadata_rows.append({
                'sequence_id': seq_id,
                'event_id': seq['event_id'],
                'label': seq['label'],
                'start_offset': seq['start_offset'],
                'end_offset': seq['end_offset'],
                'num_frames': len(seq['frames']),
                'frame_paths': '|'.join(map(str, frame_paths))
            })
        
        # Save metadata CSV
        df_metadata = pd.DataFrame(metadata_rows)
        df_metadata.to_csv(metadata_file, index=False)
        
        # Log statistics
        positive_seqs = df_metadata[df_metadata['label'] == 1]
        negative_seqs = df_metadata[df_metadata['label'] == 0]
        
        logger.info(f"Split {split}: {len(sequences)} sequences")
        logger.info(f"  - Positive (smoke): {len(positive_seqs)}")
        logger.info(f"  - Negative (no-smoke): {len(negative_seqs)}")
        logger.info(f"  - Saved to: {metadata_file}")
    
    def build_sequences(self, split_ratios: Tuple[float, float, float] = (0.7, 0.15, 0.15)):
        """Main function to build all sequences following sacred methodology."""
        logger.info("Starting FIgLib sequence building (Sacred SmokeyNet methodology)")
        logger.info(f"Parameters: L={self.L}, stride={self.stride}, ratios={split_ratios}")
        
        # Step 1: Load global labels
        df_labels = self.load_global_labels()
        
        # Step 2: Group by events
        events = self.group_by_events(df_labels)
        
        # Step 3: Split events (not frames!) into train/val/test
        event_splits = self.split_events_by_ratio(events, split_ratios)
        
        # Step 4: Create sequences for each split
        for split_name, event_ids in event_splits.items():
            logger.info(f"Processing {split_name} split with {len(event_ids)} events")
            
            all_sequences = []
            for event_id in event_ids:
                event_data = events[event_id]
                sequences = self.create_sequences(event_data, event_id)
                all_sequences.extend(sequences)
            
            # Step 5: Save sequences for this split
            self.save_sequences(all_sequences, split_name)
        
        # Step 6: Create summary statistics
        self.create_summary_stats()
        logger.info("FIgLib sequence building completed successfully!")
    
    def create_summary_stats(self):
        """Create summary statistics file."""
        summary_file = self.out_root / "dataset_summary.csv"
        
        summary_rows = []
        for split in ['train', 'val', 'test']:
            metadata_file = self.out_root / split / "sequences_metadata.csv"
            if metadata_file.exists():
                df = pd.read_csv(metadata_file)
                positive = len(df[df['label'] == 1])
                negative = len(df[df['label'] == 0])
                unique_events = df['event_id'].nunique()
                
                summary_rows.append({
                    'split': split,
                    'total_sequences': len(df),
                    'positive_sequences': positive,
                    'negative_sequences': negative,
                    'unique_events': unique_events,
                    'balance_ratio': positive / len(df) if len(df) > 0 else 0
                })
        
        df_summary = pd.DataFrame(summary_rows)
        df_summary.to_csv(summary_file, index=False)
        logger.info(f"Dataset summary saved to: {summary_file}")
        
        # Print summary
        print("\\n" + "="*60)
        print("FIGLIB DATASET SUMMARY (Sacred SmokeyNet Format)")
        print("="*60)
        print(df_summary.to_string(index=False))
        print("="*60)


def main():
    parser = argparse.ArgumentParser(description="Build FIgLib sequences (Sacred SmokeyNet methodology)")
    parser.add_argument("--raw-root", required=True, help="Path to raw FIgLib dataset")
    parser.add_argument("--out-root", required=True, help="Output directory for sequences")
    parser.add_argument("--L", type=int, default=3, help="Temporal window length (default: 3)")
    parser.add_argument("--stride", type=int, default=1, help="Sliding window stride (default: 1)")
    parser.add_argument("--split-per-event", nargs=3, type=float, default=[0.7, 0.15, 0.15],
                       help="Train/val/test ratios (default: 0.7 0.15 0.15)")
    
    args = parser.parse_args()
    
    # Validate split ratios
    if abs(sum(args.split_per_event) - 1.0) > 1e-6:
        raise ValueError("Split ratios must sum to 1.0")
    
    builder = FIgLibSequenceBuilder(
        raw_root=args.raw_root,
        out_root=args.out_root,
        L=args.L,
        stride=args.stride
    )
    
    builder.build_sequences(tuple(args.split_per_event))


if __name__ == "__main__":
    main()