# Unreal Engine Multi-View RGB-D Lunar Rock Dataset for 3D Segment Correspondence in Complex Scenes
DOI: [10.5281/zenodo.18917286](https://doi.org/10.5281/zenodo.18917286)
## Overview

This dataset contains multi-camera synthetic images of moon rocks captured in Unreal Engine. Each scene includes RGB color images, depth maps, and semantic segmentation masks from 8 camera viewpoints arranged in a circular configuration around the target rocks.

**Total Sets:** 2,377 timestamped scenes  
**Images per Set:** 24 (8 cameras × 3 image types)  
**Total Images:** 57,048  
**Format:** Mixed format (PNG + EXR)  
  - `Color`: PNG (`.png`)  
  - `Segment`: PNG (`.png`)  
  - `Depth`: OpenEXR (`.exr`)  
**Resolution:** [Image resolution from EXR headers]

---

## Dataset Structure

### File Naming Convention

Files follow the pattern: `SetNNNN_Type_C.<ext>`

- **SetNNNN**: Set number (0001–2377), zero-padded to 4 digits
- **Type**: Image type
  - `Color`: RGB color image
  - `Depth`: Depth map (distance from camera)
  - `Segment`: Semantic segmentation mask
- **C**: Camera index (0–7)

**Examples:**
- `Set0001_Color_0.png` — Set 1, RGB color, Camera 0
- `Set0042_Depth_3.exr` — Set 42, Depth map, Camera 3
- `Set2377_Segment_7.png` — Set 2377, Segmentation, Camera 7

### Original Timestamps

Each set corresponds to a unique timestamp from the original capture sequence. The mapping from set numbers to timestamps is preserved in the processing order (sorted chronologically).

---

## Camera Configuration

### Camera Array Layout

8 cameras arranged in a circular pattern around the scene origin, positioned at equal angular intervals (45° apart).

**Camera Positions (relative to origin):**

| Camera ID | Position (X, Y, Z) [cm] | Rotation (Pitch, Yaw, Roll) [deg] |
|-----------|-------------------------|-----------------------------------|
| 0 | (345.03, 1015.59, 329.10) | (-26.71, 91.44, 0.0) |
| 1 | (1076.78, 755.59, 329.10) | (-23.74, -211.70, 0.0) |
| 2 | (1343.34, 267.84, 329.10) | (-21.76, -178.34, 0.0) |
| 3 | (1065.47, -248.91, 329.10) | (-21.50, -142.46, 0.0) |
| 4 | (302.47, -499.34, 329.10) | (-25.13, -87.01, 0.0) |
| 5 | (-280.62, -174.59, 329.10) | (-24.27, -37.38, 0.0) |
| 6 | (-439.22, 306.06, 329.10) | (-26.05, 1.03, 0.0) |
| 7 | (-260.75, 710.91, 329.10) | (-25.67, 31.12, 0.0) |

**Scene Origin:** [367806.06, 410775.47, 920.90] (absolute coordinates in Unreal Engine world space)

All cameras are positioned at the same height (329.1 cm above origin) and oriented to look toward the center of the scene with a downward pitch angle (~21°–27°).

### Camera Intrinsics

All 8 cameras share the same pinhole intrinsics:

| Parameter | Value | Notes |
|-----------|-------|-------|
| Image resolution | 1920 × 1080 px | Width × Height |
| Horizontal FOV | 90° | Unreal Engine default |
| fx (focal length X) | 960.0 px | = width / (2 · tan(hFOV/2)) |
| fy (focal length Y) | 960.0 px | = fx (square pixels) |
| cx (principal point X) | 960.0 px | = width / 2 |
| cy (principal point Y) | 540.0 px | = height / 2 |

**Back-projection formula** (depth is z-distance, not radial):
```
X_cam = (u - cx) * depth_cm / fx
Y_cam = (v - cy) * depth_cm / fy
Z_cam = depth_cm
```

---

## Image Types

### 1. Color Images (`Color`)
- **Format:** RGB, 8-bit per channel (0–255)
- **Content:** Photorealistic rendered color images of the moon rock scene
- **Storage:** PNG (`.png`)

### 2. Depth Maps (`Depth`)
- **Format:** Single-channel float32
- **Units:** Centimeters (cm) from camera
- **Content:** Per-pixel distance from camera to surface
- **Channel:** Z-depth or R channel (depending on EXR export settings)
- **Range:** Typically 0–2000 cm (0–20 meters)
- **Storage:** OpenEXR (`.exr`)

### 3. Semantic Segmentation (`Segment`)
- **Format:** RGB, 8-bit per channel
- **Content:** Per-pixel class labels encoded as RGB colors
- **Classes:** 21 object classes (0–20)
- **Background:** Class 0 (RGB: 0, 0, 0 — black)
- **Storage:** PNG (`.png`)

**Class Color Mapping:**

| Class ID | Class Name | RGB Color |
|----------|------------|-----------|
| 0 | Background | (0, 0, 0) |
| 1 | Rock 1 | (255, 255, 0) |
| 2 | Rock 2 | (255, 255, 127) |
| 3 | Rock 3 | (255, 0, 0) |
| 4 | Rock 4 | (255, 0, 255) |
| 5 | Rock 5 | (0, 255, 0) |
| 6 | Rock 6 | (127, 254, 255) |
| 7 | Rock 7 | (0, 0, 255) |
| 8 | Rock 8 | (255, 191, 0) |
| 9 | Rock 9 | (255, 255, 191) |
| 10 | Rock 10 | (255, 127, 0) |
| 11 | Rock 11 | (255, 191, 127) |
| 12 | Rock 12 | (255, 127, 255) |
| 13 | Rock 13 | (191, 255, 255) |
| 14 | Rock 14 | (255, 63, 255) |
| 15 | Rock 15 | (63, 191, 0) |
| 16 | Rock 16 | (255, 63, 0) |
| 17 | Rock 17 | (255, 191, 254) |
| 18 | Rock 18 | (0, 255, 255) |
| 19 | Rock 19 | (255, 255, 63) |
| 20 | Rock 20 | (63, 255, 255) |

---


### Coordinate Systems

- **Camera positions:** Relative to scene origin, in centimeters
- **Depth values:** Distance from camera optical center, in centimeters
- **Rotations:** Pitch (up/down), Yaw (left/right), Roll (tilt) in degrees


## Requirements

These packages are needed to run `generate_segment_npz.py`:

```bash
pip install numpy opencv-python-headless openexr open3d Pillow
```

`open3d` is used for voxel downsampling during conversion. The rest of the ShadowCorr training and evaluation stack has no dependency on this script and does not require `opencv` or `openexr`.

---

## Citation

If you use this dataset, please cite the Zenodo record:

```bibtex
@dataset{ruan2026moonrock,
  author    = {Ruan, Yiyan and Komendera, Erik},
  title     = {Unreal Engine Multi-View {RGB-D} Lunar Rock Dataset for {3D} Segment Correspondence in Complex Scenes},
  year      = {2026},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.18917286},
  url       = {https://doi.org/10.5281/zenodo.18917286},
}
```

---

## Acknowledgments

This dataset was generated using Unreal Engine for synthetic lunar surface simulation research.
Real moon rock data are provided by NASA: https://ares.jsc.nasa.gov/astromaterials3d/index.htm
The Moon surface model is provided by: https://www.fab.com/listings/2378160c-3be6-4a0d-817e-df027d035e49
