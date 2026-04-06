# ShadowCorr

**ShadowCorr: Correspondence via Volumetric Consensus for Multi-View 3D Segments**
Yiyan Ruan · Erik Komendera · Virginia Tech

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![Dataset DOI](https://img.shields.io/badge/dataset-10.5281%2Fzenodo.18917286-blue)](https://doi.org/10.5281/zenodo.18917286)

> ShadowCorr infers multi-view segment correspondences by projecting each visible segment behind its surface into a shared 3D voxel grid. When segments from different viewpoints cast overlapping shadows into the same region, that volumetric consensus reveals correspondences — even when the visible surfaces do not overlap. A sparse 3D CNN learns these co-occurrence patterns; clustering its output embeddings identifies which segments belong to the same physical object.

## Pipeline

```
  Raw segments                Voxel NPZs             Trained model            Results
                                   
       │                           │                       │                     │
       ▼                           ▼                       ▼                     ▼
┌─────────────────┐     ┌─────────────────────┐   ┌───────────────────┐   ┌──────────────┐
│   PREPROCESS    │────▶│       TRAIN         │──▶│      EVAL         │──▶│ ARI / Purity │
│  (§ III  Data   │     │  (§ IV  Method)     │   │ (§ V  Experiments)│   │   Metrics    │
│   Preparation)  │     │                     │   │                   │   │              │
│                 │     │                     │   │                   │   │              │
│ • Ray-cast each │     │ • RockInstanceNet   │   │ • Load .pth       │   │ • ARI        │
│   segment into  │     │   Sparse (3D sparse │   │   checkpoint      │   │ • Rock purity│
│   confidence-   │     │   CNN + local       │   │   directly        │   │ • Cluster    │
│   weighted occ. │     │   attention)        │   │ • Forward pass on │   │   purity     │
│   voxels        │     │ • Multi-objective   │   │   voxel NPZs      │   │              │
│ • Word2Vec-style│     │   loss (discrim. +  │   │ • MeanShift       │   │              │
│   segment-ID    │     │   prototypical +    │   │   clustering of   │   │              │
│   embeddings    │     │   graph-based)      │   │   embeddings      │   │              │
│   per voxel     │     │ • Hyperparameter    │   │ • Merge tiny      │   │              │
│                 │     │   sweep             │   │   clusters        │   │              │
└─────────────────┘     └─────────────────────┘   └───────────────────┘   └──────────────┘
  preprocess_app.py          train_app.py               eval_app.py
  preprocess/scene.py        pipeline/sweep.py          pipeline/evaluator.py
  preprocess/voxel.py        pipeline/train_one.py      pipeline/metrics.py
  models/encoder.py          models/network.py
                             pipeline/losses.py
```

## Requirements

> **TorchSparse only builds on Linux.** macOS and Windows are not supported.

- Linux with a CUDA-capable GPU (tested: CUDA 12.8 / 12.9, RTX 5090)
- Python 3.10+
- PyTorch and TorchSparse matched to your CUDA version
- `torch-cluster` (optional, faster GPU k-NN during training)

```bash
cd /path/to/ShadowCorr
pip install -U pip setuptools wheel
pip install -r requirements.txt
pip install -e .
```

Then install PyTorch and build TorchSparse from source (example for CUDA 12.8):

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu128

git clone https://github.com/mit-han-lab/torchsparse.git /tmp/torchsparse
pip install /tmp/torchsparse --no-build-isolation
```

Conda users: create a Python 3.10 environment, then run the pip steps above.

## Usage

```bash
shadowcorr preprocess data.input_dir=/abs/path/to/in data.output_dir=/abs/path/to/out
shadowcorr train      data.train_dir=/abs/train data.valid_dir=/abs/valid
shadowcorr eval       input_dir=/abs/npz_or_dir model_path=/abs/checkpoint.pth
```

Or via Python module (`PYTHONPATH=. python -m shadowcorr.<preprocess|train|eval>_app ...`). Shell wrappers are in `scripts/`.

All configuration is in `shadowcorr/conf/` (Hydra). Any key can be overridden inline, e.g.:

```bash
shadowcorr train training.num_epochs=50 grid.learning_rate=[1e-4,5e-5]
```

Each run writes its log, resolved config, TensorBoard events, and checkpoints under `outputs/train/<date>/<time>/`. All training output goes to `shadowcorr.log` rather than the terminal.

## Data layout

```text
data/
  in_segments/            # raw segment NPZs — input to preprocess
    train_sample/         # 10 sample scenes (committed)
    valid_sample/         # 4  sample scenes (committed)
    test_sample/          # 5  sample scenes (committed)
    train/                # full training split   (gitignored)
    valid/                # full validation split (gitignored)
    test/                 # full test split       (gitignored)
  voxel_npz_scene/        # voxel NPZs output by preprocess (gitignored, created at runtime)
```

The `*_sample/` directories are committed for an immediate test run without downloading anything. For the full dataset see [Dataset](#dataset) below.

Use absolute paths for `data.train_dir`, `data.valid_dir`, `input_dir`, and `model_path` — Hydra changes the working directory on each run.

### Raw segment NPZ format

Each file corresponds to one multi-view scene:

| Key | Shape | dtype | Description |
|---|---|---|---|
| `rock_pcd_list` | `(N_rocks,)` | object | Per-rock list of `(N_pts, 3)` float64 arrays, one per visible view. |
| `cameras` | `(N_cams, 4, 4)` | float64 | Unique camera-to-world transforms (typically 4 per scene). |
| `cam_idx_list` | `(N_rocks,)` | object | Per-rock list of int indices into `cameras`. |

- 18–20 rocks per scene after occlusion filtering; 4 cameras at 90° intervals; ≈ 220 k points per scene before voxelisation.

```python
import numpy as np
d = np.load("data/in_segments/test_sample/218_stacked_segment.npz", allow_pickle=True)
pts  = d["rock_pcd_list"][0][0]              # (N_pts, 3) — first rock, first view
T_cw = d["cameras"][d["cam_idx_list"][0][0]] # (4, 4)     — matching camera matrix
```

### Processed voxel NPZ format

Output of `shadowcorr preprocess`, consumed by `train` and `eval`:

| Key | Shape | dtype | Description |
|---|---|---|---|
| `voxel_positions` | `(N_vox, 3)` | int32 | Integer grid indices `[x, y, z]`. |
| `voxel_labels` | `(N_vox,)` | int64 | Ground-truth rock instance ID (`-1` = background). |
| `voxel_confidences` | `(N_vox,)` | float32 | Beta-kernel confidence score ∈ (0, 1]. |
| `voxel_segment_embeddings` | `(N_vox, 12)` | float32 | Word2Vec-style segment-ID embedding per voxel. |

## Dataset

The **Unreal Engine Multi-View RGB-D Lunar Rock Dataset for 3D Segment Correspondence in Complex Scenes** (2,377 scenes, 57,048 images) is on Zenodo:

> DOI [10.5281/zenodo.18917286](https://doi.org/10.5281/zenodo.18917286)

Each scene: 8-camera RGBD captures of a synthetic lunar surface (Unreal Engine). See [`dataset/Dataset_README.md`](dataset/Dataset_README.md) for the class colour table, camera intrinsics, and coordinate conventions.

### Train / validation / test split

| Split      | Set range     | Scenes |
|------------|---------------|--------|
| Validation | Set 0001–0200 | 200    |
| Test       | Set 0201–0400 | 200    |
| Training   | Set 0401–2377 | 1,977  |

### Converting raw images to segment NPZs

`dataset/generate_segment_npz.py` converts raw RGBD captures to the stacked-segment NPZ format and splits them directly into `data/in_segments/train/`, `valid/`, and `test/`. Three sample scenes (`Set2001`–`Set2003`) are in `dataset/sample/`.

```bash
# Run on the included sample scenes (default 80/10/10 split → data/in_segments/)
python dataset/generate_segment_npz.py

# Full dataset with a custom split
python dataset/generate_segment_npz.py \
    --input-dir /path/to/raw/sets \
    --split 0.70 0.15 0.15

# Override output root, cameras, or seed
python dataset/generate_segment_npz.py \
    --input-dir /path/to/raw/sets \
    --output-dir /custom/output \
    --cameras 0 2 4 6 --seed 123

python dataset/generate_segment_npz.py --help
```

## Visualization

`scripts/visualize.py` opens three simultaneous Open3D windows for one scene:

| Window | Description |
|--------|-------------|
| Ground truth | each rock in a distinct colour |
| Predicted | each predicted cluster in a distinct colour |
| Correctness | **green** = correct assignment, **red** = wrong (Hungarian matching) |

```bash
python scripts/visualize.py \
    --npz data/in_segments/test_sample/218_stacked_segment.npz \
    --model-path checkpoints/shadowcorr_best.pth

# Save PLY files for MeshLab / CloudCompare (skip interactive windows)
python scripts/visualize.py \
    --npz data/in_segments/test_sample/218_stacked_segment.npz \
    --model-path checkpoints/shadowcorr_best.pth \
    --save-dir vis_out/ --no-show
```

Works with both raw segment NPZs (`rock_pcd_list`) and processed voxel NPZs (`voxel_positions`).

## Pretrained model

```bash
shadowcorr eval input_dir=/abs/path/to/voxel_npz_dir \
               model_path=/abs/path/to/checkpoints/shadowcorr_best.pth
```

`checkpoints/shadowcorr_best.pth` is committed to the repo. `in_channels` is inferred from the checkpoint automatically.

## Repository layout

```text
ShadowCorr/
  pyproject.toml / requirements.txt
  checkpoints/
    shadowcorr_best.pth
  data/
    in_segments/          # sample scenes committed; full splits gitignored
    voxel_npz_scene/      # created at runtime by preprocess
  dataset/
    Dataset_README.md
    generate_segment_npz.py
    sample/               # 3 scenes (Set2001–Set2003)
  scripts/                # train.sh, eval.sh, preprocess.sh
  shadowcorr/
    cli/main.py           # shadowcorr train|eval|preprocess entry point
    conf/                 # Hydra configs: defaults, train, eval, preprocess
    logging_utils/        # file logger + TensorBoard helpers
    models/
      encoder.py          # SegmentIDEncoder
      network.py          # RockInstanceNetSparse + MultiHeadLocalAttention
      features.py         # heatmap_to_sparse_tensor_with_geometry
      data.py             # RockVoxelDatasetPrecomputed, load_data_from_folder
    preprocess/
      voxel.py            # RockSegment, voxelisation
      scene.py            # batch voxel-table + segment embedding
    pipeline/
      losses.py           # discriminative, prototypical, graph-based
      metrics.py          # compute_clustering_ari, evaluate_segment_clustering
      postprocessing.py   # merge_small_clusters
      early_stopping.py   # backtick-key stop
      io.py
      train_one.py        # single-combination training loop
      sweep.py            # hyperparameter sweep (shared data loader)
      evaluator.py        # inference + metrics
    train_app.py          # Hydra entry → pipeline.sweep
    eval_app.py           # Hydra entry → pipeline.evaluator
    preprocess_app.py     # Hydra entry → preprocess.scene
```

## Acknowledgements

- **[PyTorch](https://pytorch.org/)** — deep learning framework
- **[TorchSparse](https://github.com/mit-han-lab/torchsparse)** — sparse 3D convolution
- **[Open3D](http://www.open3d.org/)** — point cloud processing and voxelisation
- **[scikit-learn](https://scikit-learn.org/)** — MeanShift clustering
- **[Hydra](https://hydra.cc/)** — configuration management
- **[NumPy](https://numpy.org/)** / **[SciPy](https://scipy.org/)** — numerical computing

## Notes

- **Ablations** — `training.use_confidence=false` or `training.use_segment=false` on the CLI; `in_channels` is recomputed automatically.
- **Sweep vs Hydra multirun** — `shadowcorr-train --multirun grid.learning_rate=1e-4,5e-5` runs independent processes per combination. The custom `pipeline/sweep.py` loads the dataset once and streams all combinations through it, which is faster on a single GPU when I/O is the bottleneck.
- **Best-loss checkpoint** — off by default. Enable with `training.save_best_loss=true`.
- **Per-epoch TensorBoard** — currently logs summary scalars at run end (train) and per-scene metrics (eval). Per-epoch hooks are not yet wired.
