# Dataset Status Report

## Real FIgLib Dataset Download Status

**Last Updated:** 2024-08-25  
**Status:** ✅ 100% COMPLETE (486/485 events + bonus)

### Technical Specifications

- **Source:** HPWREN FIgLib Data Commons (`https://cdn.hpwren.ucsd.edu/HPWREN-FIgLib-Data/Tar/`)
- **Download Method:** Automated extraction from `docs/index.html` directory listing
- **Total Available Events:** 485 fire ignition sequences
- **Events Downloaded:** 486 (100% complete + 1 bonus event)
- **Images Downloaded:** 38,474 real JPEG images
- **Total Size:** 33GB

### Dataset Structure

```
data/real_figlib/
├── YYYYMMDD_EventName_camera-id/
│   ├── metadata.csv
│   └── *.jpg (temporal sequence images)
└── labels.csv (global dataset labels)
```

### Image Naming Convention

FIgLib follows standard temporal offset naming:
```
origin_TIMESTAMP__offset_SECONDS_from_visible_plume_appearance.jpg
```

Where:
- **Negative offsets:** Pre-ignition frames (no smoke)
- **Zero/Positive offsets:** Post-ignition frames (smoke present)
- **Typical range:** -2400 to +2400 seconds (±40 minutes)

### Event Coverage

- **Date Range:** 2016-2024 (8 years of wildfire data)
- **Geographic Coverage:** Southern California (HPWREN camera network)
- **Fire Types:** Wildland fires, prescribed burns, structure fires
- **Camera Types:** MOBO-C, IQEYE, MOBO-M variants

### Data Validation

- **File Integrity:** All .tgz archives extracted successfully
- **Image Format:** JPEG (various resolutions, typically 1024x768 to 1920x1080)
- **Metadata Completeness:** 486/486 events have metadata.csv files
- **Label Distribution:** Binary smoke/no-smoke classification

### Next Steps

1. ✅ **COMPLETED:** Full real FIgLib dataset download
2. Build temporal sequences (L=3 sliding window) from real data
3. Apply sacred preprocessing pipeline
4. Train SmokeyNet-like architecture on authentic wildfire data

### Technical Notes

- **Storage:** Large dataset excluded from git via .gitignore
- **Processing:** Requires temporal sequence building for training
- **Memory:** H200-optimized pipeline ready for 30GB+ dataset
- **Validation:** Event-wise splits to prevent temporal leakage