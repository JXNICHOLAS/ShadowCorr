#!/usr/bin/env python3
"""
Visualize inference results for one NPZ scene.

Three Open3D windows open simultaneously:
  1. Ground truth  — each rock in a distinct colour
  2. Predicted     — each predicted cluster in a distinct colour
  3. Correctness   — green = correct assignment, red = wrong (via Hungarian matching)

Usage:
  python scripts/visualize.py \
      --npz data/in_segments/test_sample/216_stacked_segment.npz \
      --model-path checkpoints/shadowcorr_best.pth

  # Save PLY files instead of (or in addition to) interactive windows:
  python scripts/visualize.py --npz ... --model-path ... --save-dir vis_out/
  python scripts/visualize.py --npz ... --model-path ... --save-dir vis_out/ --no-show
"""

from __future__ import annotations

import argparse
import multiprocessing
import shutil
import sys
import tempfile
from colorsys import hsv_to_rgb
from pathlib import Path

import numpy as np
import open3d as o3d

# Ensure the repo root is importable when run as a script.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import torch
from shadowcorr.pipeline.evaluator import (
    assign_segments_to_clusters,
    load_model_once,
    process_rock_npz_to_voxels,
    run_inference_on_voxels,
)

VOXEL_SIZE = 8  # mm per voxel unit (for display scaling only)


# Colour utilities

def distinct_colors(n: int) -> list[np.ndarray]:
    """Generate n perceptually distinct colours using HSV spacing."""
    np.random.seed(42)
    return [np.array(hsv_to_rgb(i / max(n, 1), 0.8, 0.9)) for i in range(n)]


def hungarian_matching(gt: np.ndarray, pred: np.ndarray) -> np.ndarray:
    """Map predicted cluster IDs to GT rock IDs via linear-sum assignment."""
    from scipy.optimize import linear_sum_assignment

    mapped = np.full_like(pred, -1)
    valid = pred != -1
    if valid.sum() == 0:
        return mapped

    gt_ids = np.unique(gt)
    pred_ids = np.unique(pred[valid])
    gi = {int(g): i for i, g in enumerate(gt_ids)}
    pi = {int(p): i for i, p in enumerate(pred_ids)}

    conf = np.zeros((len(gt_ids), len(pred_ids)), dtype=np.int64)
    for g, p in zip(gt[valid], pred[valid]):
        conf[gi[int(g)], pi[int(p)]] += 1

    rows, cols = linear_sum_assignment(-conf)
    for c, r in zip(cols, rows):
        mapped[pred == int(pred_ids[c])] = int(gt_ids[r])
    return mapped


def colorize_correctness(gt: np.ndarray, mapped_pred: np.ndarray) -> np.ndarray:
    """Return (N,3) float32 RGB: green=correct, red=wrong, gray=unassigned."""
    rgb = np.full((len(gt), 3), 0.55, dtype=np.float32)  # gray default
    valid = mapped_pred != -1
    rgb[valid & (gt == mapped_pred)] = [0.0, 0.9, 0.0]   # green
    rgb[valid & (gt != mapped_pred)] = [0.9, 0.0, 0.0]   # red
    return rgb


# Point-cloud builders (voxel format)

def _voxel_positions_mm(voxel_positions):
    return np.array(list(voxel_positions), dtype=np.float64) * VOXEL_SIZE


def build_gt_pcd(voxel_positions, voxel_gt_labels) -> o3d.geometry.PointCloud:
    rocks = sorted(r for r in set(voxel_gt_labels.values()) if r != -1)
    color_map = {r: c for r, c in zip(rocks, distinct_colors(len(rocks)))}
    pts = _voxel_positions_mm(voxel_positions)
    rgb = np.array([color_map.get(voxel_gt_labels.get(p, -1), [0.5, 0.5, 0.5])
                    for p in voxel_positions])
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    pcd.colors = o3d.utility.Vector3dVector(rgb)
    return pcd


def build_pred_pcd(voxel_positions, cluster_labels,
                   segment_assignments, voxel_to_segments) -> o3d.geometry.PointCloud:
    clusters = sorted({a['predicted_cluster'] for a in segment_assignments.values()
                       if a['predicted_cluster'] != -1})
    color_map = {c: col for c, col in zip(clusters, distinct_colors(len(clusters)))}
    seg_to_cluster = {idx: a['predicted_cluster'] for idx, a in segment_assignments.items()}
    pts, rgb = [], []
    for pos in voxel_positions:
        segs = voxel_to_segments.get(pos, set())
        cid = seg_to_cluster.get(next(iter(segs), None), -1) if segs else -1
        pts.append(np.array(pos, dtype=np.float64) * VOXEL_SIZE)
        rgb.append(color_map.get(cid, [0.5, 0.5, 0.5]))
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.array(pts))
    pcd.colors = o3d.utility.Vector3dVector(np.array(rgb))
    return pcd


def build_correctness_pcd(voxel_positions, cluster_labels,
                          voxel_gt_labels) -> o3d.geometry.PointCloud:
    gt = np.array([voxel_gt_labels.get(p, -1) for p in voxel_positions], dtype=np.int64)
    mapped = hungarian_matching(gt, cluster_labels)
    rgb = colorize_correctness(gt, mapped)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(_voxel_positions_mm(voxel_positions))
    pcd.colors = o3d.utility.Vector3dVector(rgb.astype(np.float64))
    return pcd


# Point-cloud builders (raw segment format)

def build_gt_pcd_segments(rock_segments, segment_to_rock) -> o3d.geometry.PointCloud:
    rocks = sorted(set(segment_to_rock))
    color_map = {r: c for r, c in zip(rocks, distinct_colors(len(rocks)))}
    pts, rgb = [], []
    for seg_idx, rock_id in enumerate(segment_to_rock):
        if seg_idx >= len(rock_segments):
            continue
        p = np.asarray(rock_segments[seg_idx].point_cloud.points)
        pts.append(p)
        rgb.append(np.tile(color_map.get(rock_id, [0.5, 0.5, 0.5]), (len(p), 1)))
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.vstack(pts))
    pcd.colors = o3d.utility.Vector3dVector(np.vstack(rgb))
    return pcd


def build_pred_pcd_segments(rock_segments, segment_to_rock,
                             segment_assignments) -> o3d.geometry.PointCloud:
    clusters = sorted({a['predicted_cluster'] for a in segment_assignments.values()
                       if a['predicted_cluster'] != -1})
    color_map = {c: col for c, col in zip(clusters, distinct_colors(len(clusters)))}
    pts, rgb = [], []
    for seg_idx in range(len(rock_segments)):
        a = segment_assignments.get(seg_idx)
        if a is None or a['predicted_cluster'] == -1:
            continue
        p = np.asarray(rock_segments[seg_idx].point_cloud.points)
        pts.append(p)
        rgb.append(np.tile(color_map.get(a['predicted_cluster'], [0.5, 0.5, 0.5]), (len(p), 1)))
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.vstack(pts))
    pcd.colors = o3d.utility.Vector3dVector(np.vstack(rgb))
    return pcd


def build_correctness_pcd_segments(rock_segments, segment_to_rock,
                                    segment_assignments) -> o3d.geometry.PointCloud:
    gt_arr, pred_arr, seg_indices = [], [], []
    for seg_idx, segment in enumerate(rock_segments):
        if not segment.is_valid or seg_idx >= len(segment_to_rock):
            continue
        a = segment_assignments.get(seg_idx)
        if a is None or a['predicted_cluster'] == -1:
            continue
        gt_arr.append(segment_to_rock[seg_idx])
        pred_arr.append(a['predicted_cluster'])
        seg_indices.append(seg_idx)

    mapped = hungarian_matching(np.array(gt_arr), np.array(pred_arr))
    pts, rgb = [], []
    for idx, seg_idx in enumerate(seg_indices):
        p = np.asarray(rock_segments[seg_idx].point_cloud.points)
        color = [0.0, 0.9, 0.0] if mapped[idx] == gt_arr[idx] else [0.9, 0.0, 0.0]
        pts.append(p)
        rgb.append(np.tile(color, (len(p), 1)))

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.vstack(pts))
    pcd.colors = o3d.utility.Vector3dVector(np.vstack(rgb))
    return pcd


# Window management (multiprocessing so all three open simultaneously)

def _show_window(ply_path: str, title: str, point_size: float) -> None:
    pcd = o3d.io.read_point_cloud(ply_path)
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=title, width=900, height=700)
    vis.add_geometry(pcd)
    vis.get_render_option().point_size = point_size
    vis.run()
    vis.destroy_window()


def show_windows(pcds: dict[str, o3d.geometry.PointCloud],
                 point_size: float = 2.0) -> None:
    """Open one window per entry in pcds dict {title: pcd}, all simultaneously."""
    tmp = Path(tempfile.mkdtemp())
    procs = []
    try:
        for title, pcd in pcds.items():
            p = tmp / f"{title.replace(' ', '_')}.ply"
            o3d.io.write_point_cloud(str(p), pcd)
            proc = multiprocessing.Process(target=_show_window,
                                           args=(str(p), title, point_size))
            proc.start()
            procs.append(proc)
        for proc in procs:
            proc.join()
    finally:
        shutil.rmtree(tmp)


# CLI

def parse_args():
    p = argparse.ArgumentParser(description="Visualize ShadowCorr inference results")
    p.add_argument("--npz", required=True, help="Path to a processed voxel or raw segment NPZ")
    p.add_argument("--model-path", required=True, help="Path to checkpoint (.pth)")
    p.add_argument("--bandwidth", type=float, default=0.7, help="MeanShift bandwidth")
    p.add_argument("--save-dir", default=None,
                   help="If set, save gt.ply / pred.ply / correctness.ply here")
    p.add_argument("--no-show", action="store_true",
                   help="Skip interactive windows (useful with --save-dir)")
    p.add_argument("--point-size", type=float, default=2.0)
    p.add_argument("--embed-dim", type=int, default=12)
    p.add_argument("--embed-seed", type=int, default=42)
    p.add_argument("--score-threshold", type=float, default=50.0)
    p.add_argument("--voxel-size", type=float, default=8.0,
                   help="Voxel edge length in mm — must match preprocessing (default: 8)")
    p.add_argument("--expansion-rate", type=float, default=3.0,
                   help="Ellipsoid expansion rate — must match preprocessing (default: 3.0)")
    return p.parse_args()


def main():
    args = parse_args()
    npz_path = Path(args.npz)
    model_path = Path(args.model_path)

    # Align display scale with the voxel size used during preprocessing.
    global VOXEL_SIZE
    VOXEL_SIZE = args.voxel_size

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model, params = load_model_once(device, model_path=model_path)
    print(f"Model loaded: {model_path.name}  in_channels={params.get('in_channels')}")

    print(f"Processing: {npz_path.name}")
    (voxel_occupancy, voxel_to_segments, segment_to_rock,
     rock_segments, segment_embeddings, voxel_gt_labels,
     _, _, _) = process_rock_npz_to_voxels(
        npz_path,
        embed_dim=args.embed_dim,
        embed_seed=args.embed_seed,
        score_threshold_percentile=args.score_threshold,
        voxel_size=args.voxel_size,
        expansion_rate=args.expansion_rate,
        device=device,
    )

    _, cluster_labels, _ = run_inference_on_voxels(
        voxel_occupancy, segment_embeddings, model,
        bandwidth=args.bandwidth, device=device,
    )

    voxel_positions = list(voxel_occupancy.keys())
    segment_assignments = assign_segments_to_clusters(
        voxel_positions, cluster_labels, voxel_to_segments,
        rock_segments, segment_to_rock,
    )

    raw_mode = len(rock_segments) > 0
    print(f"Format: {'raw segments' if raw_mode else 'processed voxels'}")

    if raw_mode:
        pcd_gt   = build_gt_pcd_segments(rock_segments, segment_to_rock)
        pcd_pred = build_pred_pcd_segments(rock_segments, segment_to_rock, segment_assignments)
        pcd_corr = build_correctness_pcd_segments(rock_segments, segment_to_rock, segment_assignments)
    else:
        pcd_gt   = build_gt_pcd(voxel_positions, voxel_gt_labels)
        pcd_pred = build_pred_pcd(voxel_positions, cluster_labels, segment_assignments, voxel_to_segments)
        pcd_corr = build_correctness_pcd(voxel_positions, cluster_labels, voxel_gt_labels)

    pcds = {
        f"1. Ground Truth — {npz_path.stem}": pcd_gt,
        f"2. Predicted — {npz_path.stem}":    pcd_pred,
        f"3. Correctness (green=correct, red=wrong) — {npz_path.stem}": pcd_corr,
    }

    if args.save_dir:
        out = Path(args.save_dir)
        out.mkdir(parents=True, exist_ok=True)
        o3d.io.write_point_cloud(str(out / f"{npz_path.stem}_gt.ply"),          pcd_gt)
        o3d.io.write_point_cloud(str(out / f"{npz_path.stem}_pred.ply"),        pcd_pred)
        o3d.io.write_point_cloud(str(out / f"{npz_path.stem}_correctness.ply"), pcd_corr)
        print(f"Saved PLY files to {out}/")

    if not args.no_show:
        print("Opening three windows — close all to exit.")
        show_windows(pcds, point_size=args.point_size)


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)
    main()
