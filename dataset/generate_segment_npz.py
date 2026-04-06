#!/usr/bin/env python3
"""
generate_segment_npz.py
Batch converter: multi-camera RGBD captures → stacked-segment NPZ files.

Input layout (--input-dir, default: ./raw_sample):
  SetNNNN_Segment_C.png   — instance mask  (PNG, RGB-encoded class)
  SetNNNN_Depth_C.exr     — depth map      (EXR, float32, cm)
  where C = camera index 0–7
  (Color PNGs are present in the dataset but not used by this converter.)

Output layout (--output-dir, default: ../data/in_segments):
  <output-dir>/train/NNNN_stacked_segment.npz
  <output-dir>/valid/NNNN_stacked_segment.npz
  <output-dir>/test/ NNNN_stacked_segment.npz

  All discovered sets are shuffled (--seed) then split into train/valid/test
  at the ratios given by --split (default: 0.8 0.1 0.1).

NPZ keys written (matches the format expected by shadowcorr preprocess):
  rock_pcd_list   (N_rocks,)      object   per-rock list of (N_pts, 3) float64 [cm]
  cameras         (N_cams, 4, 4)  float64  unique camera-to-world transforms    [cm]
  cam_idx_list    (N_rocks,)      object   per-rock list of int → cameras index

Usage:
  python generate_segment_npz.py
  python generate_segment_npz.py --input-dir /data/raw --split 0.7 0.15 0.15
  python generate_segment_npz.py --input-dir /data/raw --overwrite --seed 0
  python generate_segment_npz.py --dry-run
"""

import argparse
import glob
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import OpenEXR
import Imath
from PIL import Image

# Class colour table (embedded from README — 21 classes including background)

CLASS_COLORS: dict[tuple, dict] = {
    (  0,   0,   0): {'id':  0, 'name': 'Background'},
    (255, 255,   0): {'id':  1, 'name': 'Rock 1'},
    (255, 255, 127): {'id':  2, 'name': 'Rock 2'},
    (255,   0,   0): {'id':  3, 'name': 'Rock 3'},
    (255,   0, 255): {'id':  4, 'name': 'Rock 4'},
    (  0, 255,   0): {'id':  5, 'name': 'Rock 5'},
    (127, 254, 255): {'id':  6, 'name': 'Rock 6'},
    (  0,   0, 255): {'id':  7, 'name': 'Rock 7'},
    (255, 191,   0): {'id':  8, 'name': 'Rock 8'},
    (255, 255, 191): {'id':  9, 'name': 'Rock 9'},
    (255, 127,   0): {'id': 10, 'name': 'Rock 10'},
    (255, 191, 127): {'id': 11, 'name': 'Rock 11'},
    (255, 127, 255): {'id': 12, 'name': 'Rock 12'},
    (191, 255, 255): {'id': 13, 'name': 'Rock 13'},
    (255,  63, 255): {'id': 14, 'name': 'Rock 14'},
    ( 63, 191,   0): {'id': 15, 'name': 'Rock 15'},
    (255,  63,   0): {'id': 16, 'name': 'Rock 16'},
    (255, 191, 254): {'id': 17, 'name': 'Rock 17'},
    (  0, 255, 255): {'id': 18, 'name': 'Rock 18'},
    (255, 255,  63): {'id': 19, 'name': 'Rock 19'},
    ( 63, 255, 255): {'id': 20, 'name': 'Rock 20'},
}
BACKGROUND_RGB = (0, 0, 0)
_VALID_COLORS = np.array(list(CLASS_COLORS.keys()), dtype=np.float32)  # (21, 3)

# Default camera configuration (from README, positions in cm, angles in deg)

DEFAULT_CAMERAS = [
    {'id': 0, 'pos': np.array([ 345.03,  1015.59, 329.10]), 'rot': np.array([-26.71,  91.44, 0.0])},
    {'id': 1, 'pos': np.array([1076.78,   755.59, 329.10]), 'rot': np.array([-23.74, -211.70, 0.0])},
    {'id': 2, 'pos': np.array([1343.34,   267.84, 329.10]), 'rot': np.array([-21.76, -178.34, 0.0])},
    {'id': 3, 'pos': np.array([1065.47,  -248.91, 329.10]), 'rot': np.array([-21.50, -142.46, 0.0])},
    {'id': 4, 'pos': np.array([ 302.47,  -499.34, 329.10]), 'rot': np.array([-25.13,  -87.01, 0.0])},
    {'id': 5, 'pos': np.array([-280.62,  -174.59, 329.10]), 'rot': np.array([-24.27,  -37.38, 0.0])},
    {'id': 6, 'pos': np.array([-439.22,   306.06, 329.10]), 'rot': np.array([-26.05,    1.03, 0.0])},
    {'id': 7, 'pos': np.array([-260.75,   710.91, 329.10]), 'rot': np.array([-25.67,   31.12, 0.0])},
]

# Image loaders

def load_png_segment(png_path: str) -> np.ndarray | None:
    """Load a segment mask PNG → (H, W, 3) uint8 RGB array."""
    img = cv2.imread(png_path, cv2.IMREAD_COLOR)
    if img is None:
        print(f"  [!] Failed to load segment PNG: {png_path}")
        return None
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def load_exr_depth(exr_path: str) -> np.ndarray | None:
    """Load a float32 depth channel from an EXR file → (H, W) float32 in cm."""
    try:
        exr = OpenEXR.InputFile(exr_path)
        dw  = exr.header()['dataWindow']
        w   = dw.max.x - dw.min.x + 1
        h   = dw.max.y - dw.min.y + 1
        channels = list(exr.header()['channels'].keys())
        ch = 'Z' if 'Z' in channels else ('R' if 'R' in channels else channels[0])
        buf = exr.channel(ch, Imath.PixelType(Imath.PixelType.FLOAT))
        exr.close()
        return np.frombuffer(buf, dtype=np.float32).reshape(h, w)
    except Exception as e:
        print(f"  [!] Failed to load EXR depth {exr_path}: {e}")
        return None

# Geometry helpers

def round_colors_to_nearest(seg_np: np.ndarray, threshold: int = 15) -> np.ndarray:
    """Snap PNG segment colours to the nearest known class colour (handles JPEG/PNG artifacts)."""
    h, w, _ = seg_np.shape
    pixels   = seg_np.reshape(-1, 3).astype(np.float32)
    dists    = np.linalg.norm(pixels[:, np.newaxis, :] - _VALID_COLORS[np.newaxis, :, :], axis=2)
    nearest  = np.argmin(dists, axis=1)
    min_dist = dists[np.arange(len(pixels)), nearest]
    snapped  = np.where(
        min_dist[:, np.newaxis] <= threshold,
        _VALID_COLORS[nearest],
        pixels,
    )
    return snapped.reshape(h, w, 3).astype(np.uint8)

def create_rotation_matrix(pitch_deg: float, yaw_deg: float, roll_deg: float) -> np.ndarray:
    """Euler → 3×3 rotation matrix (Unreal Engine convention: yaw → pitch → roll)."""
    p, y, r = np.radians([pitch_deg, yaw_deg, roll_deg])
    Ry = np.array([[ np.cos(y), np.sin(y), 0],
                   [-np.sin(y), np.cos(y), 0],
                   [0,          0,         1]])
    Rp = np.array([[ np.cos(p), 0, -np.sin(p)],
                   [0,          1,  0         ],
                   [ np.sin(p), 0,  np.cos(p)]])
    Rr = np.array([[1, 0,          0         ],
                   [0, np.cos(r), -np.sin(r) ],
                   [0, np.sin(r),  np.cos(r) ]])
    return Ry @ Rp @ Rr

def camera_to_world_transform(cam: dict) -> np.ndarray:
    """Build a 4×4 camera-to-world matrix in cm (translation already in cm)."""
    R_o3d_to_unreal = np.array([[0,  0, -1],
                                 [-1, 0,  0],
                                 [0,  1,  0]], dtype=np.float64)
    R_cam = create_rotation_matrix(*cam['rot'])
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R_cam @ R_o3d_to_unreal
    T[:3,  3] = cam['pos']          # already in cm
    return T

def depth_to_pointcloud(
    depth_cm: np.ndarray,
    mask: np.ndarray,
    fx: float, fy: float,
    cx: float, cy: float,
    max_depth_cm: float = 2000.0,
) -> np.ndarray:
    """
    Back-project masked depth pixels to 3D points in camera space (cm).

    depth_cm : (H, W) float32, cm distance from camera
    mask     : (H, W) bool
    returns  : (N, 3) float64 [X, Y, Z] in cm, camera frame
    """
    h, w = depth_cm.shape
    vs, us = np.where(mask)
    d = depth_cm[vs, us].astype(np.float64)

    # Drop zero/invalid/too-far pixels
    valid = (d > 0) & (d <= max_depth_cm)
    us, vs, d = us[valid], vs[valid], d[valid]
    if len(d) == 0:
        return np.zeros((0, 3), dtype=np.float64)

    X = (us - cx) * d / fx
    Y = (vs - cy) * d / fy
    Z = d
    return np.column_stack([X, Y, Z])

def transform_points(pts_cam: np.ndarray, T_cw: np.ndarray) -> np.ndarray:
    """Apply a 4×4 camera-to-world transform to an (N, 3) array."""
    if len(pts_cam) == 0:
        return pts_cam
    ones  = np.ones((len(pts_cam), 1), dtype=np.float64)
    homog = np.hstack([pts_cam, ones])          # (N, 4)
    return (T_cw @ homog.T).T[:, :3]            # (N, 3)

# Set discovery

def find_all_sets(input_dir: str) -> list[str]:
    """Return sorted list of set IDs (e.g. '2001') found in input_dir."""
    pattern = os.path.join(input_dir, "Set*_Depth_0.exr")
    hits    = glob.glob(pattern)
    ids     = set()
    for p in hits:
        m = re.match(r".*Set(\d+)_Depth_0\.exr$", p)
        if m:
            ids.add(m.group(1))
    return sorted(ids)

# Per-set processing

def process_set(
    set_id: str,
    input_dir: str,
    cameras: list[dict],
    fx: float, fy: float,
    cx: float, cy: float,
    max_depth_cm: float,
    voxel_downsample_cm: float,
    debug: bool = False,
) -> dict | None:
    """
    Process one set of RGBD images.

    Returns a dict with:
      'rock_pcd_list'   list[list[ndarray(N,3)]]  per rock, per view, pts in world cm
      'transforms'      list[list[ndarray(4,4)]]  matching camera-to-world matrices
    or None if the set cannot be processed.
    """
    print(f"\n{'='*60}")
    print(f"  Set {set_id}")
    print(f"{'='*60}")

    import open3d as o3d

    rock_dict: dict[int, dict] = {}   # class_id → {'pcd': [], 'transform': []}

    for cam in cameras:
        c = cam['id']
        seg_path   = os.path.join(input_dir, f"Set{set_id}_Segment_{c}.png")
        depth_path = os.path.join(input_dir, f"Set{set_id}_Depth_{c}.exr")

        seg_np   = load_png_segment(seg_path)
        depth_cm = load_exr_depth(depth_path)

        if seg_np is None or depth_cm is None:
            print(f"  [!] Camera {c}: missing files, skipping view")
            continue

        T_cw = camera_to_world_transform(cam)   # (4, 4) in cm

        seg_np = round_colors_to_nearest(seg_np)
        unique_colors = np.unique(seg_np.reshape(-1, 3), axis=0)

        if debug and c == cameras[0]['id']:
            print(f"  [debug cam {c}] unique segment colours: "
                  f"{[tuple(int(x) for x in col) for col in unique_colors[:8]]}")

        rocks_this_view = 0
        for col in unique_colors:
            col_tuple = tuple(int(x) for x in col)
            if col_tuple == BACKGROUND_RGB:
                continue
            class_info = CLASS_COLORS.get(col_tuple)
            if class_info is None:
                continue

            class_id = class_info['id']
            mask     = np.all(seg_np == col, axis=2)

            # Split disconnected blobs of the same colour
            num_blobs, blob_labels = cv2.connectedComponents(mask.astype(np.uint8) * 255)
            for blob_id in range(1, num_blobs):
                blob_mask = blob_labels == blob_id
                if np.sum(blob_mask) < 50:
                    continue

                pts_cam = depth_to_pointcloud(depth_cm, blob_mask, fx, fy, cx, cy, max_depth_cm)
                if len(pts_cam) == 0:
                    continue

                pts_world = transform_points(pts_cam, T_cw)

                # Voxel downsample
                if voxel_downsample_cm > 0 and len(pts_world) > 50:
                    pcd = o3d.geometry.PointCloud()
                    pcd.points = o3d.utility.Vector3dVector(pts_world)
                    pcd = pcd.voxel_down_sample(voxel_size=voxel_downsample_cm)
                    pts_world = np.asarray(pcd.points)

                if len(pts_world) == 0:
                    continue

                if class_id not in rock_dict:
                    rock_dict[class_id] = {'pcd': [], 'transform': []}
                rock_dict[class_id]['pcd'].append(pts_world.astype(np.float64))
                rock_dict[class_id]['transform'].append(T_cw.copy())
                rocks_this_view += 1

        print(f"  Camera {c}: {rocks_this_view} rock segments")

    if not rock_dict:
        print(f"  [!] No rocks found in Set {set_id}")
        return None

    rock_pcd_list = []
    transforms    = []
    for class_id in sorted(rock_dict.keys()):
        rock_pcd_list.append(rock_dict[class_id]['pcd'])
        transforms.append(rock_dict[class_id]['transform'])

    print(f"  → {len(rock_pcd_list)} unique rocks")
    return {'rock_pcd_list': rock_pcd_list, 'transforms': transforms}

# NPZ writer — new compressed format

def save_to_npz(rock_data: dict, output_path: str, atol: float = 1e-6) -> None:
    """
    Write compressed NPZ in the canonical ShadowCorr format:
      rock_pcd_list  (N_rocks,)      object
      cameras        (N_cams, 4,4)   float64  — unique camera-to-world matrices
      cam_idx_list   (N_rocks,)      object   — per-rock list[int] into cameras
    """
    rock_pcd_list = rock_data['rock_pcd_list']
    transforms    = rock_data['transforms']
    n_rocks       = len(rock_pcd_list)

    # Build deduplicated camera table
    unique_cams:    list[np.ndarray] = []
    cam_idx_list_py: list[list[int]] = []

    for r in range(n_rocks):
        indices = []
        for T in transforms[r]:
            match = next(
                (i for i, uc in enumerate(unique_cams) if np.allclose(T, uc, atol=atol)),
                -1,
            )
            if match == -1:
                match = len(unique_cams)
                unique_cams.append(T)
            indices.append(match)
        cam_idx_list_py.append(indices)

    cameras_arr = np.stack(unique_cams) if unique_cams else np.zeros((0, 4, 4), dtype=np.float64)

    pcd_arr = np.empty(n_rocks, dtype=object)
    for r in range(n_rocks):
        pcd_arr[r] = rock_pcd_list[r]   # list of (N,3) arrays

    idx_arr = np.empty(n_rocks, dtype=object)
    for r in range(n_rocks):
        idx_arr[r] = cam_idx_list_py[r]

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    np.savez_compressed(
        output_path,
        rock_pcd_list=pcd_arr,
        cameras=cameras_arr,
        cam_idx_list=idx_arr,
    )
    size_kb = os.path.getsize(output_path + '.npz' if not output_path.endswith('.npz') else output_path) / 1024
    print(f"  [OK] Saved {output_path}  ({size_kb:.0f} KB, "
          f"{n_rocks} rocks, {len(unique_cams)} unique cameras)")

# CLI entry point

def parse_args() -> argparse.Namespace:
    here = Path(__file__).parent
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument('--input-dir',  default=str(here / 'sample'),
                    help='Folder containing Set*_Segment/Depth files (default: ./sample)')
    ap.add_argument('--output-dir', default=str(here.parent / 'data' / 'in_segments'),
                    help='Output root; NPZs go to <output-dir>/train|valid|test/ '
                         '(default: ../data/in_segments)')
    ap.add_argument('--split', nargs=3, type=float, default=[0.8, 0.1, 0.1],
                    metavar=('TRAIN', 'VALID', 'TEST'),
                    help='Train / valid / test split ratios, must sum to 1 (default: 0.8 0.1 0.1)')
    ap.add_argument('--seed', type=int, default=42,
                    help='Random seed used to shuffle sets before splitting (default: 42)')
    ap.add_argument('--cameras', nargs='+', type=int, default=[0, 2, 4, 6],
                    metavar='C', help='Camera indices to use (default: 0 2 4 6)')
    # Camera intrinsics — defaults for UE4 1920×1080, 90° hFOV
    ap.add_argument('--fx', type=float, default=960.0, help='Focal length X in pixels (default: 960)')
    ap.add_argument('--fy', type=float, default=960.0, help='Focal length Y in pixels (default: 960)')
    ap.add_argument('--cx', type=float, default=960.0, help='Principal point X in pixels (default: 960)')
    ap.add_argument('--cy', type=float, default=540.0, help='Principal point Y in pixels (default: 540)')
    ap.add_argument('--max-depth', type=float, default=2000.0,
                    help='Maximum valid depth in cm (default: 2000)')
    ap.add_argument('--voxel-size', type=float, default=1.0,
                    help='Voxel downsample size in cm applied to each segment before saving '
                         '(0 = disabled, default: 1.0)')
    ap.add_argument('--overwrite', action='store_true',
                    help='Re-process sets whose output NPZ already exists')
    ap.add_argument('--dry-run',   action='store_true',
                    help='Discover sets, report split counts, write nothing')
    ap.add_argument('--debug',     action='store_true',
                    help='Print extra diagnostics for the first set processed')
    return ap.parse_args()

def main() -> None:
    import random

    args = parse_args()

    train_r, valid_r, test_r = args.split
    if abs(train_r + valid_r + test_r - 1.0) > 1e-6:
        print(f"[!] --split values must sum to 1.0 (got {train_r+valid_r+test_r:.4f})", file=sys.stderr)
        sys.exit(1)

    print("=" * 60)
    print("ShadowCorr — Segment NPZ Generator")
    print("=" * 60)
    print(f"  Input   : {args.input_dir}")
    print(f"  Output  : {args.output_dir}  (→ train/ valid/ test/)")
    print(f"  Split   : train={train_r:.0%}  valid={valid_r:.0%}  test={test_r:.0%}  (seed={args.seed})")
    print(f"  Cameras : {args.cameras}")
    print(f"  FX/FY   : {args.fx} / {args.fy}   CX/CY: {args.cx} / {args.cy}")
    print(f"  Max depth: {args.max_depth} cm   Voxel: {args.voxel_size} cm")

    # Select cameras
    cameras = [cam for cam in DEFAULT_CAMERAS if cam['id'] in args.cameras]
    if not cameras:
        print(f"[!] No cameras matched IDs {args.cameras}", file=sys.stderr)
        sys.exit(1)
    print(f"  Using {len(cameras)} camera(s): {[c['id'] for c in cameras]}")

    # Discover sets
    set_ids = find_all_sets(args.input_dir)
    if not set_ids:
        print(f"[!] No Set*_Depth_0.exr files found in {args.input_dir}", file=sys.stderr)
        sys.exit(1)
    print(f"\nFound {len(set_ids)} set(s): {set_ids[:5]}{'...' if len(set_ids) > 5 else ''}")

    # Shuffle then partition into train / valid / test
    shuffled = list(set_ids)
    random.Random(args.seed).shuffle(shuffled)
    n = len(shuffled)
    n_train = round(n * train_r)
    n_valid = round(n * valid_r)
    # test gets the remainder to avoid rounding errors dropping a scene
    n_test  = n - n_train - n_valid

    split_map: dict[str, str] = {}
    for sid in shuffled[:n_train]:
        split_map[sid] = "train"
    for sid in shuffled[n_train:n_train + n_valid]:
        split_map[sid] = "valid"
    for sid in shuffled[n_train + n_valid:]:
        split_map[sid] = "test"

    print(f"  → {n_train} train  |  {n_valid} valid  |  {n_test} test")

    if args.dry_run:
        print("\n[dry-run] No files written.")
        return

    for split in ("train", "valid", "test"):
        os.makedirs(os.path.join(args.output_dir, split), exist_ok=True)

    ok_counts   = {"train": 0, "valid": 0, "test": 0}
    skip_counts = {"train": 0, "valid": 0, "test": 0}
    failed = 0

    for idx, set_id in enumerate(set_ids):
        split    = split_map[set_id]
        out_path = os.path.join(args.output_dir, split, f"{set_id}_stacked_segment.npz")
        print(f"\n[{idx+1}/{len(set_ids)}] Set {set_id}  [{split}]", end="")

        if os.path.exists(out_path) and not args.overwrite:
            print("  [SKIP] output exists")
            skip_counts[split] += 1
            continue

        print()
        try:
            rock_data = process_set(
                set_id,
                args.input_dir,
                cameras,
                fx=args.fx, fy=args.fy, cx=args.cx, cy=args.cy,
                max_depth_cm=args.max_depth,
                voxel_downsample_cm=args.voxel_size,
                debug=(args.debug and idx == 0),
            )
            if rock_data is None:
                failed += 1
                continue
            save_to_npz(rock_data, out_path)
            ok_counts[split] += 1
        except Exception as exc:
            import traceback
            print(f"  [!] ERROR: {exc}")
            traceback.print_exc()
            failed += 1

    total_ok      = sum(ok_counts.values())
    total_skipped = sum(skip_counts.values())
    print("\n" + "=" * 60)
    print(f"Done — {total_ok} saved, {total_skipped} skipped, {failed} failed")
    print(f"  train : {ok_counts['train']} saved  ({skip_counts['train']} skipped)")
    print(f"  valid : {ok_counts['valid']} saved  ({skip_counts['valid']} skipped)")
    print(f"  test  : {ok_counts['test']} saved  ({skip_counts['test']} skipped)")
    print(f"Output root: {os.path.abspath(args.output_dir)}")

if __name__ == "__main__":
    main()
