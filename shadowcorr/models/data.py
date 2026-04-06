"""
Dataset and DataLoader utilities for voxel NPZ files.

Loads pre-processed NPZ scenes (voxel_positions, voxel_labels,
voxel_confidences, voxel_segment_embeddings) into a PyTorch Dataset
suitable for the grid-search training loop.
"""

import os
import warnings

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from shadowcorr.models.features import heatmap_to_sparse_tensor_with_geometry


class RockVoxelDatasetPrecomputed(Dataset):
    """Loads scenes as (voxel_occupancy, labels, precomputed segment embeddings per voxel)."""

    def __init__(self, data_list, use_confidence=True, use_segment=True):
        self.data_list = data_list
        self.use_confidence = use_confidence
        self.use_segment = use_segment

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        voxel_occupancy, instance_labels, precomputed_embeddings = self.data_list[idx]
        coords, feats = heatmap_to_sparse_tensor_with_geometry(
            voxel_occupancy,
            batch_idx=idx,
            precomputed_segment_embeddings=precomputed_embeddings,
            use_confidence=self.use_confidence,
            use_segment=self.use_segment,
        )
        instance_labels = torch.tensor(instance_labels, dtype=torch.long)
        return coords, feats, instance_labels


def collate_variable_size_scenes(batch):
    coords_list = [item[0] for item in batch]
    feats_list = [item[1] for item in batch]
    labels_list = [item[2] for item in batch]
    return coords_list, feats_list, labels_list


def load_data_from_folder(
    folder_path,
    include_gaps=False,
    seed=None,
    batch_size=1,
    use_confidence=True,
    use_segment=True,
):
    """
    Load all *.npz under folder_path with voxel_positions, voxel_labels, voxel_confidences
    and optional voxel_segment_embeddings.
    """
    del include_gaps, seed  # unused, kept for call-site compatibility

    folder_path = os.path.abspath(os.path.expanduser(str(folder_path)))
    if not os.path.isdir(folder_path):
        raise FileNotFoundError(
            f"load_data_from_folder: not a directory or missing: {folder_path}\n"
            "Set TRAIN_DIR / VALID_DIR to folders that exist on this machine."
        )

    data_list = []
    total_voxels = 0
    total_rocks = 0
    npz_count = 0
    skipped_files = []

    for file_name in sorted(os.listdir(folder_path)):
        if not file_name.endswith(".npz"):
            continue
        npz_count += 1
        file_path = os.path.join(folder_path, file_name)
        with np.load(file_path, allow_pickle=True) as data:
            if (
                "voxel_positions" in data
                and "voxel_labels" in data
                and "voxel_confidences" in data
            ):
                voxel_positions = data["voxel_positions"]
                voxel_labels = data["voxel_labels"]
                voxel_scores = data["voxel_confidences"]
                precomputed_embeddings = {}
                if "voxel_segment_embeddings" in data:
                    segment_embeddings = data["voxel_segment_embeddings"]
                    for i, pos in enumerate(voxel_positions):
                        precomputed_embeddings[tuple(pos)] = segment_embeddings[i]
                voxel_occupancy = {
                    tuple(pos): float(score) for pos, score in zip(voxel_positions, voxel_scores)
                }
                instance_labels = voxel_labels.astype(np.int64)
                data_list.append((voxel_occupancy, instance_labels, precomputed_embeddings))
                num_rocks = len(np.unique(instance_labels[instance_labels != -1]))
                total_voxels += len(voxel_positions)
                total_rocks += num_rocks
            else:
                missing = [
                    k
                    for k in ("voxel_positions", "voxel_labels", "voxel_confidences")
                    if k not in data
                ]
                reason = f"missing keys {missing}" if missing else "unknown format"
                skipped_files.append((file_name, reason))
                warnings.warn(f"{file_name}: missing required arrays ({reason}), skipping")

    if len(data_list) == 0:
        hint = ""
        if npz_count == 0:
            hint = "No .npz files in this folder. Add label NPZs from preprocess (batch_label_generation_scene)."
        else:
            hint = (
                f"Found {npz_count} .npz file(s) but none had required arrays "
                "(voxel_positions, voxel_labels, voxel_confidences)."
            )
        skip_detail = ""
        if skipped_files:
            examples = skipped_files[:5]
            skip_detail = " Examples: " + "; ".join(f"{n}: {r}" for n, r in examples)
            if len(skipped_files) > 5:
                skip_detail += f" ... (+{len(skipped_files) - 5} more)"
        raise ValueError(
            f"load_data_from_folder: dataset is empty (0 scenes) for directory:\n  {folder_path}\n"
            f"{hint}{skip_detail}"
        )

    dataset = RockVoxelDatasetPrecomputed(
        data_list, use_confidence=use_confidence, use_segment=use_segment
    )
    num_workers = min(8, os.cpu_count() or 1)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_variable_size_scenes,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=bool(num_workers),
    )
    return loader, None
