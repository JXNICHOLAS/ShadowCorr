"""
I/O helpers for model checkpoints and evaluation result JSON files.

load_model_from_json      – read best model path + params from grid-search JSON
save_single_result        – create a single scene result entry (dict)
save_cumulative_results   – append results and summary stats to a JSON file
convert_to_json_serializable – deep-convert numpy types for JSON
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

# NPZ metadata helpers


def _npz_metadata_segment_embedding(data) -> Dict[str, Any]:
    """
    Read optional metadata['segment_embedding'] from a label NPZ.
    Used to match train_encoder_steps / seed when recomputing embeddings.
    """
    if "metadata" not in data:
        return {}
    md = data["metadata"]
    if isinstance(md, np.ndarray) and md.ndim == 0:
        md = md.item()
    if not isinstance(md, dict):
        return {}
    se = md.get("segment_embedding")
    return dict(se) if isinstance(se, dict) else {}

# Model loading from grid-search JSON


def load_model_from_json(json_file: Path, use_best: str = 'ari', verbose: bool = True):
    """
    Load best model path and parameters from a grid-search result JSON.

    Returns:
        model_path (Path), params (dict)
    """
    if not json_file.exists():
        raise FileNotFoundError(f"JSON file not found: {json_file}")

    with open(json_file, 'r') as f:
        grid_results = json.load(f)

    if use_best.lower() == 'loss':
        if 'best_loss_result' not in grid_results:
            raise ValueError(f"No best_loss_result found in {json_file}")
        result = grid_results['best_loss_result']
        score_key, score_name = 'avg_loss', 'Loss'
    elif use_best.lower() == 'ari':
        if 'best_ari_result' not in grid_results:
            raise ValueError(f"No best_ari_result found in {json_file}")
        result = grid_results['best_ari_result']
        score_key, score_name = 'avg_ari', 'ARI'
    else:
        raise ValueError(f"Invalid use_best: {use_best}. Must be 'loss' or 'ari'")

    model_path = Path(result['model_file'])
    params = result['parameters']
    score = result[score_key]

    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    if verbose:
        print(f"Best {score_name} model: {model_path}")
        print(f"{score_name} score: {score:.6f}")

    return model_path, params

# JSON serialization helpers


def convert_to_json_serializable(obj):
    """Recursively convert numpy types to Python native types for JSON."""
    if isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {
            str(convert_to_json_serializable(k)): convert_to_json_serializable(v)
            for k, v in obj.items()
        }
    elif isinstance(obj, (list, tuple)):
        return [convert_to_json_serializable(item) for item in obj]
    elif isinstance(obj, Path):
        return str(obj)
    return obj

# Result persistence


def save_single_result(
    npz_path: Path,
    segment_assignments: Dict,
    metrics: Dict,
    params: Dict,
    processing_params: Optional[Dict] = None,
) -> Dict:
    """Create a single scene result entry (not written to disk yet)."""
    result = {
        'timestamp': datetime.now().isoformat(),
        'input_file': str(npz_path.name),
        'input_file_full_path': str(npz_path),
        'model_params': convert_to_json_serializable(params),
        'metrics': convert_to_json_serializable(metrics),
        'num_segments': len(segment_assignments),
        'segment_assignments': convert_to_json_serializable(segment_assignments),
    }
    if processing_params:
        result['processing_params'] = convert_to_json_serializable(processing_params)
    return result


def save_cumulative_results(
    output_json: Path,
    new_results: List[Dict],
    model_path: Optional[Path] = None,
    processing_params: Optional[Dict] = None,
) -> Dict:
    """
    Append new_results to a cumulative JSON file and recompute summary stats.
    Creates the file if it does not exist.
    """
    if output_json.exists():
        with open(output_json, 'r') as f:
            cumulative = json.load(f)
    else:
        cumulative = {
            'metadata': {
                'created': datetime.now().isoformat(),
                'model_file': str(model_path) if model_path else None,
                'processing_params': convert_to_json_serializable(processing_params) if processing_params else {},
                'description': 'Segment-level evaluation results for multiple scenes',
            },
            'all_results': [],
        }

    cumulative['metadata']['last_updated'] = datetime.now().isoformat()
    cumulative['all_results'].extend(new_results)

    if cumulative['all_results']:
        def _collect(key):
            return [r['metrics'][key] for r in cumulative['all_results'] if key in r['metrics']]

        voxel_aris = _collect('voxel_ari')
        segment_aris = _collect('segment_ari')
        rock_purities = _collect('avg_rock_purity')
        cluster_purities = _collect('avg_cluster_purity')
        perfect_1to1 = _collect('perfect_1to1_accuracy')

        def _stats(vals):
            if not vals:
                return {'mean': None, 'std': None, 'min': None, 'max': None}
            return {
                'mean': float(np.mean(vals)),
                'std': float(np.std(vals)),
                'min': float(np.min(vals)),
                'max': float(np.max(vals)),
            }

        cumulative['summary_stats'] = {
            'total_scenes': len(cumulative['all_results']),
            'voxel_ari': _stats(voxel_aris),
            'segment_ari': _stats(segment_aris),
            'avg_rock_purity': {'mean': float(np.mean(rock_purities)) if rock_purities else None,
                                'std': float(np.std(rock_purities)) if rock_purities else None},
            'avg_cluster_purity': {'mean': float(np.mean(cluster_purities)) if cluster_purities else None,
                                   'std': float(np.std(cluster_purities)) if cluster_purities else None},
            'perfect_1to1_accuracy': {'mean': float(np.mean(perfect_1to1)) if perfect_1to1 else None,
                                      'std': float(np.std(perfect_1to1)) if perfect_1to1 else None},
        }

        if voxel_aris:
            best_idx = int(np.argmax(voxel_aris))
            cumulative['best_voxel_ari'] = {
                'value': float(voxel_aris[best_idx]),
                'scene': cumulative['all_results'][best_idx]['input_file'],
            }
        if segment_aris:
            best_idx = int(np.argmax(segment_aris))
            cumulative['best_segment_ari'] = {
                'value': float(segment_aris[best_idx]),
                'scene': cumulative['all_results'][best_idx]['input_file'],
            }

    with open(output_json, 'w') as f:
        json.dump(cumulative, f, indent=2)

    return cumulative
