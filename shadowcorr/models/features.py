"""
Per-voxel feature construction for TorchSparse input.

Builds (coords, feats) tensors from a voxel occupancy dict.
Features: optional confidence (1D), scene-normalized spatial (3D),
optional segment embedding (12D).
"""

import numpy as np
import torch


def heatmap_to_sparse_tensor_with_geometry(
    voxel_occupancy,
    batch_idx=0,
    segment_data=None,
    segment_encoder=None,
    precomputed_segment_embeddings=None,
    use_confidence=True,
    use_segment=True,
):
    """
    Build [batch,x,y,z] coords and per-voxel feature rows for TorchSparse.

    Features: optional confidence (1), spatial in [-1,1] (3), optional segment embedding (12).
    segment_data / segment_encoder are legacy kwargs (ignored); use precomputed_segment_embeddings.
    """
    feat_dim = 3
    if use_confidence:
        feat_dim += 1
    if use_segment:
        feat_dim += 12

    if len(voxel_occupancy) == 0:
        coords = torch.tensor([], dtype=torch.int32).reshape(0, 4)
        feats = torch.tensor([], dtype=torch.float32).reshape(0, feat_dim)
        return coords, feats

    coords = []
    confidence_scores_raw = []
    spatial_coords_raw = []
    segment_embeddings = []

    all_positions = np.array(list(voxel_occupancy.keys()))
    all_scores = np.array([score for _, score in voxel_occupancy.items()])

    scene_min = all_positions.min(axis=0)
    scene_max = all_positions.max(axis=0)
    scene_center = (scene_min + scene_max) / 2
    scene_size = (scene_max - scene_min).max() + 1e-6

    score_min = all_scores.min()
    score_max = all_scores.max()
    score_range = score_max - score_min + 1e-6

    for pos, score in voxel_occupancy.items():
        coords.append([batch_idx, pos[0], pos[1], pos[2]])
        score_normalized = (score - score_min) / score_range
        confidence_scores_raw.append([score_normalized])
        pos_normalized = (np.array(pos) - scene_center) / (scene_size / 2)
        spatial_coords_raw.append(pos_normalized.tolist())

        if precomputed_segment_embeddings is not None and pos in precomputed_segment_embeddings:
            emb = precomputed_segment_embeddings[pos]
            if not isinstance(emb, torch.Tensor):
                emb = torch.tensor(emb, dtype=torch.float32)
            segment_embeddings.append(emb)
        else:
            if use_segment:
                raise ValueError(
                    f"Pre-computed segment embeddings missing for voxel {pos}. "
                    "NPZ should include 'voxel_segment_embeddings' from preprocess (batch_label_generation_scene)."
                )
            segment_embeddings.append(torch.zeros(12, dtype=torch.float32))

    spatial_feats = torch.tensor(spatial_coords_raw, dtype=torch.float32)

    feature_components = []
    if use_confidence:
        feature_components.append(
            torch.tensor(confidence_scores_raw, dtype=torch.float32)
        )
    feature_components.append(spatial_feats)
    if use_segment:
        feature_components.append(torch.stack(segment_embeddings))

    feats = torch.cat(feature_components, dim=1)
    coords = torch.from_numpy(np.array(coords, dtype=np.int32))
    return coords, feats
