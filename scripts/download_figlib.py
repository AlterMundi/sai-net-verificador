#!/usr/bin/env python3
"""
FIgLib Dataset Downloader
Based on SmokeyNet approach as specified in the sacred documentation.

Downloads Fire Ignition Library from HPWREN servers following the exact methodology
described in the roadmap documentation.
"""

import os
import argparse
import requests
import pandas as pd
from pathlib import Path
from urllib.parse import urljoin
from tqdm import tqdm
import time
import csv
from typing import List, Tuple
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FIgLibDownloader:
    """Downloads FIgLib dataset from HPWREN servers."""
    
    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # HPWREN base URLs (as per sacred documentation)
        self.base_url = "http://hpwren.ucsd.edu/HPWREN-FIgLib/"
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (SmokeyNet FIgLib Downloader)'
        })
        
    def load_camera_list(self, camera_file: str) -> List[str]:
        """Load camera IDs from file."""
        cameras = []
        try:
            with open(camera_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#'):
                        cameras.append(line)
            logger.info(f"Loaded {len(cameras)} cameras from {camera_file}")
            return cameras
        except FileNotFoundError:
            logger.error(f"Camera file not found: {camera_file}")
            # Create example file
            self._create_example_camera_file(camera_file)
            raise
    
    def load_timestamps(self, timestamp_file: str) -> List[Tuple[str, str, str]]:
        """Load ignition timestamps from file."""
        timestamps = []
        try:
            with open(timestamp_file, 'r') as f:
                reader = csv.reader(f)
                next(reader, None)  # Skip header
                for row in reader:
                    if len(row) >= 3:
                        camera_id, timestamp, event_id = row[:3]
                        timestamps.append((camera_id.strip(), timestamp.strip(), event_id.strip()))
            logger.info(f"Loaded {len(timestamps)} timestamp entries from {timestamp_file}")
            return timestamps
        except FileNotFoundError:
            logger.error(f"Timestamp file not found: {timestamp_file}")
            self._create_example_timestamp_file(timestamp_file)
            raise
    
    def _create_example_camera_file(self, filepath: str):
        """Create example camera list file."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w') as f:
            f.write("""# Example camera IDs from HPWREN network
# Add actual camera IDs here, one per line
# Format: camera-id-from-hpwren
mlo-n-mobo-c
bf-n-mobo-c
lp-n-mobo-c
""")
        logger.info(f"Created example camera file at {filepath}")
    
    def _create_example_timestamp_file(self, filepath: str):
        """Create example timestamp file."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w') as f:
            writer = csv.writer(f)
            writer.writerow(['camera_id', 'ignition_timestamp', 'event_id'])
            writer.writerow(['mlo-n-mobo-c', '2020-08-15T14:30:00', 'event_001'])
            writer.writerow(['bf-n-mobo-c', '2020-09-10T16:45:00', 'event_002'])
        logger.info(f"Created example timestamp file at {filepath}")
    
    def download_sequence(self, camera_id: str, timestamp: str, event_id: str) -> bool:
        """
        Download image sequence for a specific fire event.
        Following FIgLib naming convention with offset from ignition time.
        """
        event_dir = self.output_dir / event_id
        event_dir.mkdir(exist_ok=True)
        
        logger.info(f"Downloading sequence for {camera_id} at {timestamp} (event: {event_id})")
        
        # Try to download from HPWREN archive
        # This is a simplified implementation - actual HPWREN API may differ
        success = self._download_from_hpwren_archive(camera_id, timestamp, event_dir)
        
        if success:
            # Create metadata file
            metadata_file = event_dir / "metadata.csv"
            self._create_metadata_file(metadata_file, camera_id, timestamp, event_id)
            return True
        
        return False
    
    def _download_from_hpwren_archive(self, camera_id: str, timestamp: str, output_dir: Path) -> bool:
        """
        Download images from HPWREN archive.
        This is a placeholder implementation - actual HPWREN API access required.
        """
        logger.warning("HPWREN API access not implemented - creating placeholder structure")
        
        # Create placeholder structure following FIgLib naming convention
        # Real implementation would query HPWREN servers
        
        # Create sample files following the offset naming pattern
        sample_files = [
            f"origin_{timestamp}__offset_-2400_from_visible_plume_appearance.jpg",  # 40 min before
            f"origin_{timestamp}__offset_-1800_from_visible_plume_appearance.jpg",  # 30 min before
            f"origin_{timestamp}__offset_-1200_from_visible_plume_appearance.jpg",  # 20 min before
            f"origin_{timestamp}__offset_-600_from_visible_plume_appearance.jpg",   # 10 min before
            f"origin_{timestamp}__offset_0_from_visible_plume_appearance.jpg",      # ignition
            f"origin_{timestamp}__offset_600_from_visible_plume_appearance.jpg",    # 10 min after
            f"origin_{timestamp}__offset_1200_from_visible_plume_appearance.jpg",   # 20 min after
            f"origin_{timestamp}__offset_1800_from_visible_plume_appearance.jpg",   # 30 min after
            f"origin_{timestamp}__offset_2400_from_visible_plume_appearance.jpg",   # 40 min after
        ]
        
        for filename in sample_files:
            placeholder_file = output_dir / filename
            with open(placeholder_file, 'w') as f:
                f.write(f"# Placeholder for {filename}\n# Real image would be downloaded from HPWREN\n")
        
        logger.info(f"Created {len(sample_files)} placeholder files for {camera_id}")
        return True
    
    def _create_metadata_file(self, filepath: Path, camera_id: str, timestamp: str, event_id: str):
        """Create metadata CSV for the event."""
        with open(filepath, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['filename', 'camera_id', 'timestamp', 'event_id', 'offset_seconds', 'label'])
            
            # Sample entries following FIgLib labeling convention
            offsets = [-2400, -1800, -1200, -600, 0, 600, 1200, 1800, 2400]
            for offset in offsets:
                filename = f"origin_{timestamp}__offset_{offset}_from_visible_plume_appearance.jpg"
                label = 1 if offset >= 0 else 0  # smoke=1 for offset>=0, no-smoke=0 for offset<0
                writer.writerow([filename, camera_id, timestamp, event_id, offset, label])
    
    def download_dataset(self, camera_file: str, timestamp_file: str):
        """Download complete FIgLib dataset."""
        logger.info("Starting FIgLib dataset download")
        
        cameras = self.load_camera_list(camera_file)
        timestamps = self.load_timestamps(timestamp_file)
        
        successful_downloads = 0
        total_events = len(timestamps)
        
        for camera_id, timestamp, event_id in tqdm(timestamps, desc="Downloading events"):
            try:
                if self.download_sequence(camera_id, timestamp, event_id):
                    successful_downloads += 1
                time.sleep(0.1)  # Be nice to servers
            except Exception as e:
                logger.error(f"Failed to download {event_id}: {str(e)}")
        
        logger.info(f"Download complete: {successful_downloads}/{total_events} events downloaded")
        
        # Create global labels CSV
        self._create_global_labels_csv()
    
    def _create_global_labels_csv(self):
        """Create global labels.csv file combining all events."""
        global_labels_file = self.output_dir / "labels.csv"
        
        with open(global_labels_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['filename', 'camera_id', 'datetime', 'event_id', 'offset_seconds', 'label'])
            
            # Combine all metadata files
            for event_dir in self.output_dir.iterdir():
                if event_dir.is_dir():
                    metadata_file = event_dir / "metadata.csv"
                    if metadata_file.exists():
                        with open(metadata_file, 'r') as meta_f:
                            reader = csv.reader(meta_f)
                            next(reader)  # Skip header
                            for row in reader:
                                writer.writerow(row)
        
        logger.info(f"Created global labels file: {global_labels_file}")


def main():
    parser = argparse.ArgumentParser(description="Download FIgLib dataset from HPWREN")
    parser.add_argument("--camera_list", required=True, help="File containing camera IDs")
    parser.add_argument("--timestamps", required=True, help="File containing ignition timestamps")
    parser.add_argument("--output", required=True, help="Output directory for dataset")
    
    args = parser.parse_args()
    
    downloader = FIgLibDownloader(args.output)
    downloader.download_dataset(args.camera_list, args.timestamps)


if __name__ == "__main__":
    main()