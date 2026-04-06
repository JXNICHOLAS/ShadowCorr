"""
Inference and batch-evaluation pipeline for segment-to-rock assignment.

Supports two NPZ input formats:
  FORMAT A  raw segment files  (rock_pcd_list, cameras, cam_idx_list)
  FORMAT B  processed voxel files (voxel_positions, voxel_labels, ...)

load_model_once          – build and load model from a checkpoint (.pth)
process_single_file      – end-to-end pipeline for one NPZ scene
process_batch            – batch processing across a directory of NPZ files
"""

import sys
import time
import traceback
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from sklearn.cluster import MeanShift
from torchsparse import SparseTensor

import shadowcorr.preprocess.scene as _scene_mod
import shadowcorr.preprocess.voxel as _voxel_mod
from shadowcorr.models.features import heatmap_to_sparse_tensor_with_geometry
from shadowcorr.models.network import RockInstanceNetSparse
from shadowcorr.pipeline.io import (
    _npz_metadata_segment_embedding,
    convert_to_json_serializable,
    load_model_from_json,
    save_cumulative_results,
    save_single_result,
)
from shadowcorr.pipeline.metrics import evaluate_segment_clustering
from shadowcorr.pipeline.postprocessing import merge_tiny_clusters

build_voxel_tables = _scene_mod.build_voxel_tables
compute_segment_embeddings = _scene_mod.compute_segment_embeddings
RockSegment = _voxel_mod.RockSegment


# Model loading

def load_model_once(
    device: torch.device,
    model_path: Path,
    use_confidence: bool = True,
    use_segment: bool = True,
    params: Optional[Dict] = None,
):
    """Load a checkpoint and build the model.

    ``in_channels`` is inferred automatically from the first weight tensor in
    the checkpoint, so no separate config file is needed.
    """
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {model_path}")

    params = params or {}
    if 'in_channels' not in params:
        sd = torch.load(model_path, map_location='cpu', weights_only=False)
        sd = sd.get('model_state_dict', sd)
        params['in_channels'] = next(iter(sd.values())).shape[0]

    in_channels_base = 3
    if use_confidence:
        in_channels_base += 1
    if use_segment:
        in_channels_base += 12

    in_channels = params.get('in_channels', in_channels_base)
    model = RockInstanceNetSparse(
        in_channels=in_channels,
        instance_embed_dim=params.get('instance_embed_dim', 32),
        attn_k1=params.get('attn_k1', params.get('attn_k', 32)),
        attn_k2=params.get('attn_k2', params.get('attn_k', 32)),
        num_heads1=params.get('num_heads1', 4),
        num_heads2=params.get('num_heads2', 8),
    ).to(device)

    full_ckpt = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(full_ckpt.get('model_state_dict', full_ckpt))
    model.eval()
    return model, params


# Data preparation

def process_rock_npz_to_voxels(
    npz_path: Path,
    embed_dim: int,
    embed_seed: int,
    score_threshold_percentile: float,
    voxel_size: float,
    expansion_rate: float = 2.0,
    train_encoder_steps: int = 200,
    train_encoder_lr: float = 1e-3,
    train_encoder_temperature: float = 0.1,
    device: Optional[torch.device] = None,
):
    """
    Load an NPZ scene and return voxel data structures ready for inference.

    Automatically detects FORMAT A (raw segments) or FORMAT B (preprocessed voxels).
    """
    with np.load(npz_path, allow_pickle=True) as data:
        if 'voxel_positions' in data and 'voxel_labels' in data and 'voxel_confidences' in data:
            # FORMAT B – preprocessed voxels
            voxel_positions = data['voxel_positions']
            voxel_labels = data['voxel_labels']
            voxel_scores = data['voxel_confidences']

            segment_embeddings_dict = {}
            if 'voxel_segment_embeddings' in data:
                seg_emb = data['voxel_segment_embeddings']
                for i, pos in enumerate(voxel_positions):
                    segment_embeddings_dict[tuple(pos)] = torch.tensor(seg_emb[i], dtype=torch.float32)
            else:
                segment_embeddings_dict = None

            voxel_occupancy_simple = {
                tuple(pos): float(score) for pos, score in zip(voxel_positions, voxel_scores)
            }

            if 'voxel_segment_labels' in data:
                t0 = time.time()
                voxel_to_segments = {}
                for i, pos in enumerate(voxel_positions):
                    seg_label = data['voxel_segment_labels'][i]
                    if isinstance(seg_label, str) and seg_label != "-1":
                        voxel_to_segments[tuple(pos)] = {int(s.strip()) for s in seg_label.split(',')}
                    elif isinstance(seg_label, (int, np.integer)) and seg_label != -1:
                        voxel_to_segments[tuple(pos)] = {int(seg_label)}
                    else:
                        voxel_to_segments[tuple(pos)] = set()

                instance_labels = voxel_labels.astype(np.int64)
                unique_segments = {s for segs in voxel_to_segments.values() for s in segs}
                segment_to_rock = {}
                for i, pos in enumerate(voxel_positions):
                    rock_id = int(instance_labels[i])
                    if rock_id != -1:
                        for seg_id in voxel_to_segments[tuple(pos)]:
                            segment_to_rock.setdefault(seg_id, rock_id)

                max_seg_id = max(unique_segments) if unique_segments else -1
                segment_to_rock_list = [segment_to_rock.get(i, -1) for i in range(max_seg_id + 1)]
                t_seg_labels = time.time() - t0

                if segment_embeddings_dict is None:
                    t_embed_start = time.time()
                    meta_se = _npz_metadata_segment_embedding(data)
                    steps = int(meta_se.get("train_encoder_steps", train_encoder_steps))
                    seed_use = int(meta_se.get("seed", embed_seed))
                    raw_embs = compute_segment_embeddings(
                        voxel_to_segments, embed_dim=embed_dim, seed=seed_use,
                        train_encoder_steps=steps, train_encoder_lr=train_encoder_lr,
                        train_encoder_temperature=train_encoder_temperature, device=device,
                    )
                    segment_embeddings_dict = {
                        pos: torch.from_numpy(e) if isinstance(e, np.ndarray) else e
                        for pos, e in raw_embs.items()
                    }
                    t_segment_embed = time.time() - t_embed_start
                else:
                    t_segment_embed = 0.0

                voxel_gt_labels_dict = {
                    tuple(pos): int(label) for pos, label in zip(voxel_positions, instance_labels)
                }
                return (voxel_occupancy_simple, voxel_to_segments,
                        segment_to_rock_list, [], segment_embeddings_dict, voxel_gt_labels_dict,
                        t_seg_labels, t_segment_embed, 0.0)
            else:
                if segment_embeddings_dict is None:
                    raise ValueError("Cannot compute segment embeddings: missing voxel_segment_labels")
                voxel_to_segments = {tuple(pos): set() for pos in voxel_positions}
                voxel_gt_labels_dict = {tuple(pos): -1 for pos in voxel_positions}
                return (voxel_occupancy_simple, voxel_to_segments,
                        [], [], segment_embeddings_dict, voxel_gt_labels_dict,
                        0.0, 0.0, 0.0)

        elif 'rock_pcd_list' in data:
            # FORMAT A – raw segment file (new or old layout)
            _voxel_mod.VOXEL_SIZE = voxel_size
            _voxel_mod.expansion_rate = expansion_rate

            t0 = time.time()
            rock_segments, segment_to_rock = [], []
            skipped = 0
            for rock_idx, points, transform in _scene_mod.load_segments_from_npz(npz_path):
                seg = RockSegment(points, transform, rock_idx)
                if seg.is_valid:
                    rock_segments.append(seg)
                    segment_to_rock.append(rock_idx)
                else:
                    skipped += 1

            voxel_occupancy, voxel_to_segments = build_voxel_tables(rock_segments)
            voxel_best_rock = _voxel_mod.simple_voxel_assignment(voxel_occupancy)
            t_seg_labels = time.time() - t0

            if score_threshold_percentile > 0:
                voxel_scores_list = [(pos, float(voxel_occupancy[pos][voxel_best_rock[pos]].get("score", 0.0)))
                                     for pos in voxel_best_rock]
                threshold = np.percentile([s for _, s in voxel_scores_list], 100 - score_threshold_percentile)
                filtered_occ, filtered_to_seg, filtered_best = {}, {}, {}
                for pos, score in voxel_scores_list:
                    if score >= threshold:
                        filtered_occ[pos] = voxel_occupancy[pos]
                        filtered_to_seg[pos] = voxel_to_segments[pos]
                        filtered_best[pos] = voxel_best_rock[pos]
                voxel_occupancy, voxel_to_segments, voxel_best_rock = filtered_occ, filtered_to_seg, filtered_best

            t_embed_start = time.time()
            raw_embs = compute_segment_embeddings(
                voxel_to_segments, embed_dim=embed_dim, seed=embed_seed,
                train_encoder_steps=train_encoder_steps, train_encoder_lr=train_encoder_lr,
                train_encoder_temperature=train_encoder_temperature,
            )
            segment_embeddings_dict = {
                pos: torch.from_numpy(raw_embs[pos]) if isinstance(raw_embs[pos], np.ndarray) else raw_embs[pos]
                for pos in voxel_occupancy
            }
            t_segment_embed = time.time() - t_embed_start

            t_conf_start = time.time()
            voxel_occupancy_simple = {
                pos: float(voxel_occupancy[pos][voxel_best_rock[pos]].get("confidence", 0.0))
                for pos in voxel_occupancy
            }
            t_confidence = time.time() - t_conf_start

            voxel_gt_labels_dict = {pos: voxel_best_rock[pos] for pos in voxel_occupancy}
            return (voxel_occupancy_simple, voxel_to_segments,
                    segment_to_rock, rock_segments, segment_embeddings_dict, voxel_gt_labels_dict,
                    t_seg_labels, t_segment_embed, t_confidence)
        else:
            raise ValueError(
                f"NPZ file {npz_path} does not contain expected format. "
                "Expected 'voxel_positions' (processed) or 'rock_pcd_list' (raw)."
            )


# Inference

def run_inference_on_voxels(
    voxel_occupancy: Dict,
    segment_embeddings: Dict,
    model: torch.nn.Module,
    bandwidth: float,
    device: torch.device,
    use_confidence: bool = True,
    use_segment: bool = True,
    enable_post_processing: bool = True,
    min_cluster_size: int = 5,
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Forward pass + MeanShift clustering on one scene's voxels.

    Returns (embeddings_np, cluster_labels, timing_dict).
    """
    t0 = time.time()
    coords, feats = heatmap_to_sparse_tensor_with_geometry(
        voxel_occupancy,
        precomputed_segment_embeddings=segment_embeddings,
        use_confidence=use_confidence,
        use_segment=use_segment,
    )
    coords, feats = coords.to(device), feats.to(device)
    stensor = SparseTensor(coords=coords, feats=feats)
    with torch.no_grad():
        embeddings = model(stensor)
    embeddings_np = embeddings.detach().cpu().numpy()
    t_inference = time.time() - t0

    t_clust = time.time()
    np.random.seed(42)
    cluster_labels_raw = MeanShift(bandwidth=bandwidth).fit_predict(embeddings_np)
    if enable_post_processing:
        cluster_labels, _ = merge_tiny_clusters(cluster_labels_raw, embeddings_np, min_cluster_size)
    else:
        cluster_labels = cluster_labels_raw
    t_clustering = time.time() - t_clust

    return embeddings_np, cluster_labels, {'inference': t_inference, 'clustering': t_clustering}


# Segment assignment

def assign_segments_to_clusters(
    voxel_positions, cluster_labels, voxel_to_segments, rock_segments, segment_to_rock
):
    """
    Assign each segment to a predicted cluster by majority vote over its voxels.
    Ground truth is stored in the output for evaluation only (not used for prediction).
    """
    pos_to_cluster = {pos: cluster_labels[i] for i, pos in enumerate(voxel_positions)}
    all_segments = {s for segs in voxel_to_segments.values() for s in segs}
    num_segments = len(rock_segments) if len(rock_segments) > 0 else (max(all_segments) + 1 if all_segments else 0)

    segment_assignments = {}
    for segment_idx in range(num_segments):
        segment_rock = segment_to_rock[segment_idx] if segment_idx < len(segment_to_rock) else -1
        votes = [pos_to_cluster[pos] for pos, segs in voxel_to_segments.items() if segment_idx in segs and pos in pos_to_cluster]
        if not votes:
            segment_assignments[segment_idx] = {'predicted_cluster': -1, 'ground_truth_rock': segment_rock, 'num_voxels': 0, 'cluster_votes': {}}
            continue
        vote_counts = Counter(votes)
        segment_assignments[segment_idx] = {
            'predicted_cluster': vote_counts.most_common(1)[0][0],
            'ground_truth_rock': segment_rock,
            'num_voxels': len(votes),
            'cluster_votes': dict(vote_counts),
        }
    return segment_assignments


# Single-file pipeline

def process_single_file(
    npz_path: Path,
    model,
    params: Dict,
    bandwidth: float,
    output_dir: Path,
    embed_dim: int,
    embed_seed: int,
    score_threshold_percentile: float,
    voxel_size: float,
    expansion_rate: float,
    train_encoder_steps: int,
    train_encoder_lr: float,
    train_encoder_temperature: float,
    device: torch.device,
    use_confidence: bool = True,
    use_segment: bool = True,
    enable_post_processing: bool = True,
    min_cluster_size: int = 5,
):
    """Complete pipeline for one NPZ scene: data prep -> inference -> evaluation."""
    start_time = time.time()
    print(f"\n--- {npz_path.name} ---")

    t_pre = time.time()
    (voxel_occupancy, voxel_to_segments, segment_to_rock,
     rock_segments, segment_embeddings, voxel_gt_labels,
     t_seg_labels, t_segment_embed, t_confidence) = process_rock_npz_to_voxels(
        npz_path, embed_dim, embed_seed, score_threshold_percentile, voxel_size,
        expansion_rate, train_encoder_steps, train_encoder_lr, train_encoder_temperature, device,
    )
    t_preprocess = time.time() - t_pre

    embeddings, cluster_labels, inf_timing = run_inference_on_voxels(
        voxel_occupancy, segment_embeddings, model, bandwidth, device,
        use_confidence=use_confidence, use_segment=use_segment,
        enable_post_processing=enable_post_processing, min_cluster_size=min_cluster_size,
    )

    t_post_start = time.time()
    voxel_positions = list(voxel_occupancy.keys())
    segment_assignments = assign_segments_to_clusters(
        voxel_positions, cluster_labels, voxel_to_segments, rock_segments, segment_to_rock
    )
    metrics = evaluate_segment_clustering(
        segment_assignments, segment_to_rock, voxel_to_segments,
        voxel_positions, cluster_labels, voxel_gt_labels, rock_segments,
    )
    t_postprocess = time.time() - t_post_start

    elapsed = time.time() - start_time
    metrics.update({
        'processing_time_seconds': elapsed,
        'preprocessing_time_seconds': t_preprocess,
        'preprocessing_time_no_seg_labels_seconds': t_preprocess - t_seg_labels,
        'segment_label_calculation_time_seconds': t_seg_labels,
        'segment_embedding_time_seconds': t_segment_embed,
        'confidence_extraction_time_seconds': t_confidence,
        'inference_time_seconds': inf_timing['inference'],
        'clustering_time_seconds': inf_timing['clustering'],
        'postprocessing_time_seconds': t_postprocess,
        'file_type': 'processed_voxel' if len(rock_segments) == 0 else 'raw_segment',
        'preprocessing_without_segment_embed_seconds': t_preprocess - t_segment_embed,
        'preprocessing_without_confidence_seconds': t_preprocess - t_confidence,
        'preprocessing_without_both_seconds': t_preprocess - t_segment_embed - t_confidence,
    })

    processing_params = {
        'bandwidth': bandwidth, 'voxel_size': voxel_size, 'expansion_rate': expansion_rate,
        'score_threshold_percentile': score_threshold_percentile,
        'embed_dim': embed_dim, 'embed_seed': embed_seed,
        'train_encoder_steps': train_encoder_steps,
        'train_encoder_lr': train_encoder_lr,
        'train_encoder_temperature': train_encoder_temperature,
    }
    result_entry = save_single_result(npz_path, segment_assignments, metrics, params, processing_params)
    return metrics, result_entry


# Batch pipeline

def process_batch(
    input_dir: Path,
    model_path: Path,
    bandwidth: float,
    output_dir: Path,
    embed_dim: int,
    embed_seed: int,
    score_threshold_percentile: float,
    voxel_size: float,
    expansion_rate: float,
    train_encoder_steps: int,
    train_encoder_lr: float,
    train_encoder_temperature: float,
    max_files: int = 0,
    use_confidence: bool = True,
    use_segment: bool = True,
    enable_post_processing: bool = True,
    min_cluster_size: int = 5,
):
    """Process all NPZ files in a directory."""
    npz_files = sorted(list(input_dir.glob('*.npz')))
    if not npz_files:
        print(f"No NPZ files found in {input_dir}")
        return [], []
    if max_files > 0:
        npz_files = npz_files[:max_files]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, params = load_model_once(
        device, model_path=model_path,
        use_confidence=use_confidence, use_segment=use_segment,
    )
    print(f"Model loaded on {device} ({len(npz_files)} files to process)")

    all_results, result_entries = [], []
    success_count = failed_count = 0

    for idx, npz_path in enumerate(npz_files, 1):
        print(f"[{idx}/{len(npz_files)}] {npz_path.name}")
        try:
            metrics, result_entry = process_single_file(
                npz_path=npz_path, model=model, params=params, bandwidth=bandwidth,
                output_dir=output_dir, embed_dim=embed_dim, embed_seed=embed_seed,
                score_threshold_percentile=score_threshold_percentile,
                voxel_size=voxel_size, expansion_rate=expansion_rate,
                train_encoder_steps=train_encoder_steps, train_encoder_lr=train_encoder_lr,
                train_encoder_temperature=train_encoder_temperature, device=device,
                use_confidence=use_confidence, use_segment=use_segment,
                enable_post_processing=enable_post_processing, min_cluster_size=min_cluster_size,
            )
            all_results.append({'file': npz_path.name, 'metrics': metrics, 'success': True})
            result_entries.append(result_entry)
            success_count += 1
            print(f"  total={metrics['processing_time_seconds']*1000:.1f}ms")
        except Exception as e:
            print(f"  error: {e}", file=sys.stderr)
            traceback.print_exc()
            all_results.append({'file': npz_path.name, 'error': str(e), 'success': False})
            failed_count += 1

    if result_entries:
        output_dir.mkdir(parents=True, exist_ok=True)
        cumulative_json = output_dir / f'segment_evaluation_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        processing_params = {
            'bandwidth': bandwidth, 'voxel_size': voxel_size, 'expansion_rate': expansion_rate,
            'score_threshold_percentile': score_threshold_percentile,
            'embed_dim': embed_dim, 'embed_seed': embed_seed,
            'train_encoder_steps': train_encoder_steps,
            'train_encoder_lr': train_encoder_lr,
            'train_encoder_temperature': train_encoder_temperature,
        }
        save_cumulative_results(
            cumulative_json, result_entries,
            model_path=model_path,
            processing_params=processing_params,
        )
        print(f"Saved results: {cumulative_json} ({len(result_entries)} scenes)")

    print(f"Done: {success_count}/{len(npz_files)} succeeded")
    return all_results, result_entries
