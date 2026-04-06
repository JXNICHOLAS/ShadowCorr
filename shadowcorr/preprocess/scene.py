#!/usr/bin/env python3
"""
Batch voxel label generator with Word2Vec-style segment-ID encoder.

Each segment ID gets a learned embedding. A voxel's embedding is the mean of
its segment IDs' embeddings, trained per-scene via co-occurrence-weighted
contrastive loss (InfoNCE). Segment pairs sharing more voxels get pulled
harder, so e.g. {1,2} and {3,4} become related when {2,3} co-occurs with both.
"""

import argparse
import importlib.util
import sys
import time
import traceback
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
try:
    from tqdm import tqdm
except ImportError:
    class _FakeTqdm:
        def __init__(self, iterable, desc="", unit="", leave=True, ncols=100, **kwargs):
            self._it = iterable
        def __iter__(self):
            return iter(self._it)
        def set_postfix(self, **kwargs):
            pass
    tqdm = _FakeTqdm

from shadowcorr.models.encoder import create_segment_encoder

# NPZ format helpers

def load_segments_from_npz(npz_path) -> list:
    """Return a flat list of (rock_idx, points, camera_transform) tuples.

    Expected NPZ layout:
      - ``rock_pcd_list``  (N_rocks,) object — per-rock list of (N_pts, 3) arrays
      - ``cameras``        (N_cams, 4, 4)    — unique camera-to-world transforms
      - ``cam_idx_list``   (N_rocks,) object — per-rock list of int indices into cameras
    """
    with np.load(npz_path, allow_pickle=True) as data:
        rock_pcd_list = data["rock_pcd_list"]
        cameras       = data["cameras"]       # (N_cams, 4, 4)
        cam_ids       = data["cam_idx_list"]  # (N_rocks,) of lists of int

    segments = []
    for rock_idx, (pcd_views, view_ids) in enumerate(zip(rock_pcd_list, cam_ids)):
        for pts, cam_idx in zip(pcd_views, view_ids):
            segments.append((rock_idx, pts, np.array(cameras[int(cam_idx)], dtype=np.float64)))
    return segments

# ============================================================================
# CONFIGURATION — defaults for this repository (override via CLI).
# ============================================================================

SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = SCRIPT_DIR.parent.parent  # repository root (two levels up from shadowcorr/preprocess/)

INPUT_DIR = _REPO_ROOT / "data" / "in_segments"
OUTPUT_DIR = _REPO_ROOT / "data" / "voxel_npz_scene"
SINGLE_SCRIPT = SCRIPT_DIR / "voxel.py"  # sibling file in the same preprocess/ folder

SCORE_THRESHOLD_PERCENTILE = 50
EMBED_DIM = 12
EMBED_SEED = 42
OVERWRITE_EXISTING = False

# Per-scene overfit: train one encoder per scene (no cross-scene). 0 = no training (random init).
TRAIN_ENCODER_STEPS = 200
TRAIN_ENCODER_LR = 1e-3
TRAIN_ENCODER_TEMPERATURE = 0.1

# ============================================================================

def load_single_label_module(script_path: Path, quiet: bool = True):
    spec = importlib.util.spec_from_file_location(
        "single_label_generation", script_path
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[arg-type]
    if quiet and hasattr(module, "_quiet"):
        module._quiet = True  # Suppress per-segment prints (faster, less I/O)
    return module

class SegmentSetEncoderScene:
    """
    Wrapper around SegmentIDEncoder (word-embedding style).
    Each segment ID has a learned vector; voxel embedding = mean of its segment embeddings.
    """

    def __init__(
        self,
        max_segments: int,
        embed_dim: int = 12,
        seed: int = 42,
        **kwargs,
    ):
        if max_segments <= 0:
            raise ValueError("max_segments must be positive for segment encoding.")

        self.embed_dim = embed_dim
        self.max_segments = max_segments
        self.seed = seed

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)

        self.encoder = create_segment_encoder(max_segments, embed_dim=embed_dim)
        self.encoder.eval()

    @torch.inference_mode()
    def encode(self, segment_lists: Sequence[Sequence[int]]) -> torch.Tensor:
        if not segment_lists:
            return torch.zeros((0, self.embed_dim), dtype=torch.float32)
        return self.encoder(segment_lists)

def _build_cooccurrence(segment_lists: List[List[int]], device: torch.device):
    """
    Precompute co-occurrence COUNTS (called once per scene, reused every step).
    
    cooccur_counts[i, j] = number of voxels where segment i and segment j
    both appear. More voxels sharing the pair -> stronger pull during training.
    
    Returns: seg_ids_tensor (K,), cooccur_counts (K, K float), has_pos (K, bool).
    """
    all_seg_ids = set()
    for seg_list in segment_lists:
        all_seg_ids.update(seg_list)
    seg_ids_list = sorted(all_seg_ids)
    K = len(seg_ids_list)
    seg_to_idx = {seg: i for i, seg in enumerate(seg_ids_list)}
    cooccur_counts = torch.zeros(K, K, device=device, dtype=torch.float32)
    for seg_list in segment_lists:
        if len(seg_list) < 2:
            continue
        for i, seg_i in enumerate(seg_list):
            for seg_j in seg_list[i + 1:]:
                idx_i, idx_j = seg_to_idx[seg_i], seg_to_idx[seg_j]
                cooccur_counts[idx_i, idx_j] += 1.0
                cooccur_counts[idx_j, idx_i] += 1.0
    seg_ids_tensor = torch.tensor(seg_ids_list, dtype=torch.long, device=device)
    has_pos = cooccur_counts.sum(dim=1) > 0
    return seg_ids_tensor, cooccur_counts, has_pos

def _contrastive_loss_segment_ids(
    encoder_module,
    seg_ids_tensor: torch.Tensor,
    cooccur_counts: torch.Tensor,
    has_pos: torch.Tensor,
    temperature: float = 0.1,
) -> torch.Tensor:
    """
    Weighted contrastive loss on segment IDs.
    
    For each segment ID i, the "positive score" toward j is weighted by
    cooccur_counts[i, j] — i.e. how many voxels they share. This means
    segment pairs that co-occur in MORE voxels get pulled proportionally harder.
    
    The denominator sums over ALL other segments (positives + negatives).
    
    Loss = -mean_i [ log( sum_j(weight_ij * exp(sim_ij/T)) / sum_k(exp(sim_ik/T)) ) ]
    """
    if seg_ids_tensor.size(0) < 2 or has_pos.sum() == 0:
        return torch.tensor(0.0, device=seg_ids_tensor.device)

    seg_embs = encoder_module.segment_embedding(seg_ids_tensor)  # (K, embed_dim)
    seg_embs = F.normalize(seg_embs, p=2, dim=1)

    sim = seg_embs @ seg_embs.T  # (K, K) cosine similarity
    sim.fill_diagonal_(float("-inf"))
    logits = sim / temperature
    exp_logits = torch.exp(logits)

    denom = exp_logits.sum(dim=1)  # (K,)
    # Weight each positive pair by co-occurrence count
    num = (exp_logits * cooccur_counts).sum(dim=1)  # (K,)

    log_prob = torch.log(num[has_pos] + 1e-8) - torch.log(denom[has_pos] + 1e-8)
    return -log_prob.mean()

def _train_encoder_on_scene(
    encoder_module,  # raw nn.Module from create_segment_encoder (SegmentIDEncoder)
    segment_lists: List[List[int]],
    device: torch.device,
    steps: int,
    lr: float = 1e-3,
    temperature: float = 0.1,
) -> None:
    """Overfit this one scene: train segment ID embeddings for `steps` steps, in place."""
    if not segment_lists or len(segment_lists) < 2 or steps <= 0:
        return
    encoder_module.train()
    encoder_module.to(device)

    # Precompute co-occurrence once (no Python loops during training)
    t0 = time.perf_counter()
    seg_ids_tensor, cooccur, has_pos = _build_cooccurrence(segment_lists, device)
    K = seg_ids_tensor.size(0)
    print(f"  Co-occurrence built in {time.perf_counter()-t0:.2f}s: K={K} segment IDs, K×K={K*K}")

    optimizer = torch.optim.Adam(encoder_module.parameters(), lr=lr)
    pbar = tqdm(range(steps), desc="Encoder overfit", unit="step", leave=False, ncols=100)
    for step in pbar:
        optimizer.zero_grad()
        loss = _contrastive_loss_segment_ids(encoder_module, seg_ids_tensor, cooccur, has_pos, temperature=temperature)
        if loss.requires_grad and loss.item() > 0:
            loss.backward()
            optimizer.step()
        pbar.set_postfix(loss=f"{loss.item():.4f}")
    encoder_module.eval()

def build_voxel_tables(rock_segments) -> Tuple[Dict, Dict[Tuple[float, float, float], set]]:
    voxel_occupancy: Dict[Tuple[float, float, float], Dict[int, dict]] = {}
    voxel_to_segments: Dict[Tuple[float, float, float], set] = {}

    for segment_idx, segment in enumerate(rock_segments):
        for pos_key, score_data in segment.voxel_scores.items():
            voxel_to_segments.setdefault(pos_key, set()).add(segment_idx)
            rock_scores = voxel_occupancy.setdefault(pos_key, {})
            rock_idx = segment.complete_rock_idx
            if rock_idx not in rock_scores:
                entry = score_data.copy()
                entry["view_count"] = 1
                rock_scores[rock_idx] = entry
            else:
                existing = rock_scores[rock_idx]
                existing["score"] += score_data["score"]
                existing["confidence"] += score_data["confidence"]
                existing["view_count"] += 1
    return voxel_occupancy, voxel_to_segments

def compute_segment_embeddings(
    voxel_to_segments: Dict[Tuple[float, float, float], set],
    embed_dim: int = 12,
    seed: int = 42,
    train_encoder_steps: int = 0,
    train_encoder_lr: float = 1e-3,
    train_encoder_temperature: float = 0.1,
    device: Optional[torch.device] = None,
) -> Dict[Tuple[float, float, float], np.ndarray]:
    """
    Precompute segment-ID embeddings for this scene (word-embedding style).
    If train_encoder_steps > 0: create one encoder for this scene, overfit it on
    this scene's co-occurrence contrastive loss, then encode. No cross-scene data.
    """
    max_segment_id = -1
    for segments in voxel_to_segments.values():
        if segments:
            max_segment_id = max(max_segment_id, max(segments))

    if max_segment_id < 0:
        return {}

    max_segments = max_segment_id + 1
    encoder_wrapper = SegmentSetEncoderScene(
        max_segments,
        embed_dim=embed_dim,
        seed=seed,
    )
    encoder_module = encoder_wrapper.encoder

    voxel_positions = []
    segment_lists = []
    for pos_key, segments in sorted(voxel_to_segments.items()):
        voxel_positions.append(pos_key)
        segment_lists.append(sorted(int(s) for s in segments))

    dev = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
    encoder_module.to(dev)
    print(f"  Encoder on device: {dev}")

    if train_encoder_steps > 0:
        _train_encoder_on_scene(
            encoder_module,
            segment_lists,
            dev,
            steps=train_encoder_steps,
            lr=train_encoder_lr,
            temperature=train_encoder_temperature,
        )

    embeddings = encoder_wrapper.encode(segment_lists).detach().cpu().numpy().astype(np.float32)
    return {pos: emb for pos, emb in zip(voxel_positions, embeddings)}

def serialize_segments(segment_ids: Sequence[int]) -> str:
    if not segment_ids:
        return "-1"
    return ",".join(str(int(seg)) for seg in sorted(segment_ids))

def save_npz_results(
    output_path: Path,
    voxel_best_rock: Dict[Tuple[float, float, float], int],
    voxel_occupancy: Dict[Tuple[float, float, float], Dict[int, dict]],
    voxel_to_segments: Dict[Tuple[float, float, float], set],
    segment_embeddings: Dict[Tuple[float, float, float], np.ndarray],
    segment_to_rock: Sequence[int],
    voxel_size: float,
    embed_dim: int,
    embed_seed: int,
    score_threshold_percentile: float = SCORE_THRESHOLD_PERCENTILE,
    train_encoder_steps: int = 0,
) -> int:
    if not voxel_best_rock:
        return 0

    voxel_scores_list = []
    for pos_key in voxel_best_rock.keys():
        rock_idx = voxel_best_rock[pos_key]
        rock_scores = voxel_occupancy.get(pos_key, {})
        if rock_idx in rock_scores:
            score = float(rock_scores[rock_idx].get("score", 0.0))
            voxel_scores_list.append((pos_key, score))
        else:
            voxel_scores_list.append((pos_key, 0.0))

    if voxel_scores_list and score_threshold_percentile > 0:
        scores_only = [score for _, score in voxel_scores_list]
        threshold_score = np.percentile(scores_only, 100 - score_threshold_percentile)
        print(f"  Score threshold ({score_threshold_percentile}th percentile): {threshold_score:.4f}")
        print(f"  Score range: [{min(scores_only):.4f}, {max(scores_only):.4f}]")
    else:
        threshold_score = 0.0

    assigned_positions = []
    assigned_labels = []
    assigned_confidences = []
    assigned_segment_labels = []
    assigned_embeddings = []
    zero_embed = np.zeros(embed_dim, dtype=np.float32)
    filtered_count = 0

    for pos_key in sorted(voxel_best_rock.keys()):
        rock_idx = voxel_best_rock[pos_key]
        rock_scores = voxel_occupancy.get(pos_key, {})
        score = 0.0
        confidence = 0.0
        if rock_idx in rock_scores:
            score = float(rock_scores[rock_idx].get("score", 0.0))
            confidence = float(rock_scores[rock_idx].get("confidence", 0.0))
        if score < threshold_score:
            filtered_count += 1
            continue
        assigned_positions.append(pos_key)
        assigned_labels.append(rock_idx)
        assigned_confidences.append(confidence)
        segment_ids = voxel_to_segments.get(pos_key, set())
        assigned_segment_labels.append(serialize_segments(segment_ids))
        embed_vec = segment_embeddings.get(pos_key, zero_embed)
        assigned_embeddings.append(embed_vec)

    if filtered_count > 0:
        print(f"  Filtered out {filtered_count} voxels below threshold (kept {len(assigned_positions)}/{len(voxel_best_rock)})")

    voxel_positions = np.array(assigned_positions, dtype=np.float32)
    voxel_labels = np.array(assigned_labels, dtype=np.int64)
    voxel_confidences = np.array(assigned_confidences, dtype=np.float32)
    voxel_segment_labels = np.array(assigned_segment_labels)
    voxel_segment_embeddings = (
        np.vstack(assigned_embeddings) if assigned_embeddings else np.zeros((0, embed_dim), dtype=np.float32)
    )

    num_rocks = max(segment_to_rock) + 1 if segment_to_rock else 0
    metadata = {
        "voxel_size": voxel_size,
        "num_rocks": num_rocks,
        "num_segments": len(segment_to_rock),
        "segment_embedding": {
            "type": "segment_id_encoder",
            "dim": embed_dim,
            "train_encoder_steps": train_encoder_steps,
            "seed": embed_seed,
        },
    }

    np.savez(
        output_path,
        voxel_positions=voxel_positions,
        voxel_labels=voxel_labels,
        voxel_confidences=voxel_confidences,
        voxel_segment_labels=voxel_segment_labels,
        voxel_segment_embeddings=voxel_segment_embeddings,
        metadata=metadata,
    )
    return len(assigned_positions)

def process_scene(
    npz_path: Path,
    output_dir: Path,
    module,
    embed_dim: int = 12,
    embed_seed: int = 42,
    score_threshold_percentile: float = SCORE_THRESHOLD_PERCENTILE,
    overwrite: bool = False,
    train_encoder_steps: int = 0,
    train_encoder_lr: float = 1e-3,
    train_encoder_temperature: float = 0.1,
) -> None:
    scene_name = npz_path.stem
    output_path = output_dir / f"{scene_name}_labels.npz"

    if output_path.exists() and not overwrite:
        print(f"[SKIP] {scene_name} (existing output)")
        return

    raw_segments = load_segments_from_npz(npz_path)

    rock_segments = []
    segment_to_rock = []
    skipped_segments = 0
    for rock_idx, points, transform in raw_segments:
        segment = module.RockSegment(points, transform, rock_idx)
        if segment.is_valid:
            rock_segments.append(segment)
            segment_to_rock.append(rock_idx)
        else:
            skipped_segments += 1

    if not rock_segments:
        print(f"[WARN] {scene_name}: no valid segments (skipped_segments={skipped_segments})")
        return

    voxel_occupancy, voxel_to_segments = build_voxel_tables(rock_segments)
    voxel_best_rock = module.simple_voxel_assignment(voxel_occupancy)

    segment_embeddings = compute_segment_embeddings(
        voxel_to_segments,
        embed_dim=embed_dim,
        seed=embed_seed,
        train_encoder_steps=train_encoder_steps,
        train_encoder_lr=train_encoder_lr,
        train_encoder_temperature=train_encoder_temperature,
    )

    assigned_count = save_npz_results(
        output_path,
        voxel_best_rock,
        voxel_occupancy,
        voxel_to_segments,
        segment_embeddings,
        segment_to_rock,
        module.VOXEL_SIZE,
        embed_dim,
        embed_seed,
        score_threshold_percentile=score_threshold_percentile,
        train_encoder_steps=train_encoder_steps,
    )

    if assigned_count == 0:
        print(f"[WARN] {scene_name}: no voxels assigned, nothing saved.")
    else:
        print(
            f"[OK] {scene_name}: {assigned_count} voxels saved "
            f"(segments={len(segment_to_rock)}, skipped_segments={skipped_segments})"
        )

def batch_process(
    input_dir: Path,
    output_dir: Path,
    module,
    embed_dim: int,
    embed_seed: int,
    score_threshold_percentile: float,
    overwrite: bool,
    train_encoder_steps: int = 0,
    train_encoder_lr: float = 1e-3,
    train_encoder_temperature: float = 0.1,
    limit: Optional[int] = None,
    npz_files: Optional[List[Path]] = None,
) -> None:
    if npz_files is None:
        npz_files = sorted(input_dir.glob("*.npz"))
    if not npz_files:
        print(f"No NPZ files found in {input_dir}")
        return
    if limit is not None:
        npz_files = npz_files[:limit]
        print(f"Processing only first {len(npz_files)} scene(s) (--limit {limit})")
    output_dir.mkdir(parents=True, exist_ok=True)
    for npz_file in tqdm(npz_files, desc="Scenes", unit="scene", ncols=100):
        try:
            process_scene(
                npz_file,
                output_dir,
                module,
                embed_dim=embed_dim,
                embed_seed=embed_seed,
                score_threshold_percentile=score_threshold_percentile,
                overwrite=overwrite,
                train_encoder_steps=train_encoder_steps,
                train_encoder_lr=train_encoder_lr,
                train_encoder_temperature=train_encoder_temperature,
            )
        except Exception as exc:
            print(f"[ERROR] Failed to process {npz_file.name}: {exc}")
            traceback.print_exc()

def parse_args():
    parser = argparse.ArgumentParser(
        description="Batch voxel label generator (SCENE-AWARE encoder experiment)"
    )
    parser.add_argument("--input-dir", type=Path, default=INPUT_DIR, help="Input NPZ directory")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help="Output directory (default: voxel_npz_8_3_scene)",
    )
    parser.add_argument("--single-script", type=Path, default=SINGLE_SCRIPT, help="Single-scene script path")
    parser.add_argument("--embed-dim", type=int, default=EMBED_DIM, help="Embedding dimension")
    parser.add_argument("--embed-seed", type=int, default=EMBED_SEED, help="Random seed for encoder")
    parser.add_argument("--score-threshold", type=float, default=SCORE_THRESHOLD_PERCENTILE, help="Score threshold percentile")
    parser.add_argument("--overwrite", action="store_true", default=OVERWRITE_EXISTING, help="Overwrite existing outputs")
    parser.add_argument("--train-encoder-steps", type=int, default=TRAIN_ENCODER_STEPS, help="Overfit one encoder per scene for this many steps (0 = no training)")
    parser.add_argument("--train-encoder-lr", type=float, default=TRAIN_ENCODER_LR, help="LR for per-scene encoder overfit")
    parser.add_argument("--train-encoder-temperature", type=float, default=TRAIN_ENCODER_TEMPERATURE, help="Contrastive temperature for per-scene overfit")
    parser.add_argument("--limit", type=int, default=None, help="Process only this many scenes (e.g. 1 for single-scene run)")
    return parser.parse_args()

def main():
    args = parse_args()
    module = load_single_label_module(args.single_script.resolve())
    print(f"\n{'='*60}")
    print("BATCH PROCESSING (SCENE-AWARE ENCODER EXPERIMENT)")
    print(f"{'='*60}")
    print(f"Input directory:      {args.input_dir}")
    print(f"Output directory:     {args.output_dir}")
    print(f"Score threshold:      Keep top {args.score_threshold}% of voxels")
    print(f"Embedding dimension:  {args.embed_dim}")
    print(f"Encoder:              SegmentIDEncoder (word-embedding style, co-occurrence weighted)")
    print(f"Per-scene overfit:     {args.train_encoder_steps} steps (one encoder per scene, no cross-scene)")
    if args.limit is not None:
        print(f"Limit:                 {args.limit} scene(s) only")
    print(f"{'='*60}\n")
    batch_process(
        args.input_dir.resolve(),
        args.output_dir.resolve(),
        module,
        embed_dim=args.embed_dim,
        embed_seed=args.embed_seed,
        score_threshold_percentile=args.score_threshold,
        overwrite=args.overwrite,
        train_encoder_steps=args.train_encoder_steps,
        train_encoder_lr=args.train_encoder_lr,
        train_encoder_temperature=args.train_encoder_temperature,
        limit=args.limit,
    )

if __name__ == "__main__":
    main()
