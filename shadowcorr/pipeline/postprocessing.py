"""
Post-processing utilities for predicted voxel / segment clusters.

merge_small_clusters  – used during ARI evaluation (merges by centroid in embedding space)
merge_tiny_clusters   – used during segment-level evaluation (same logic, richer stats)
"""

import numpy as np


def merge_small_clusters(embeddings, cluster_labels, min_cluster_size=5, merge_threshold=0.5):
    """
    Merge clusters smaller than min_cluster_size into the nearest larger cluster
    by centroid distance in embedding space.
    """
    _ = merge_threshold  # unused; kept for backward compatibility
    unique_labels = np.unique(cluster_labels)
    merged_labels = cluster_labels.copy()
    small_clusters = [
        label for label in unique_labels
        if np.sum(cluster_labels == label) < min_cluster_size
    ]
    if not small_clusters:
        return merged_labels

    centroids = {}
    for label in unique_labels:
        if label not in small_clusters:
            mask = cluster_labels == label
            centroids[label] = np.mean(embeddings[mask], axis=0)

    for small_label in small_clusters:
        small_mask = cluster_labels == small_label
        small_embeddings = embeddings[small_mask]
        if len(centroids) == 0:
            largest_cluster = max(
                unique_labels, key=lambda x: np.sum(cluster_labels == x) if x not in small_clusters else 0
            )
            merged_labels[small_mask] = largest_cluster
        else:
            nearest_labels = None
            min_distances = None
            for label, centroid in centroids.items():
                dist = np.linalg.norm(small_embeddings - centroid, axis=1)
                if nearest_labels is None:
                    nearest_labels = np.full(len(small_embeddings), label)
                    min_distances = dist
                else:
                    closer = dist < min_distances
                    nearest_labels[closer] = label
                    min_distances[closer] = dist[closer]
            merged_labels[small_mask] = nearest_labels

    return merged_labels


def merge_tiny_clusters(cluster_labels, embeddings, min_cluster_size=10):
    """
    Merge tiny MeanShift clusters into their nearest larger cluster by
    embedding centroid distance.

    Returns:
        merged_labels: (N,) numpy array of merged cluster labels
        merge_stats:   dict with merge statistics
    """
    unique_labels = np.unique(cluster_labels)
    merged_labels = cluster_labels.copy()

    tiny_clusters = []
    normal_clusters = []
    for label in unique_labels:
        if np.sum(cluster_labels == label) < min_cluster_size:
            tiny_clusters.append(label)
        else:
            normal_clusters.append(label)

    if len(tiny_clusters) == 0:
        return merged_labels, {'num_tiny': 0, 'num_merged': 0, 'tiny_sizes': [], 'merge_targets': []}

    if len(normal_clusters) == 0:
        return merged_labels, {
            'num_tiny': len(tiny_clusters),
            'num_merged': 0,
            'tiny_sizes': [np.sum(cluster_labels == label) for label in tiny_clusters],
            'merge_targets': [],
        }

    cluster_centroids = {
        label: embeddings[cluster_labels == label].mean(axis=0)
        for label in unique_labels
    }

    merge_stats = {'num_tiny': len(tiny_clusters), 'num_merged': 0, 'tiny_sizes': [], 'merge_targets': []}

    for tiny_label in tiny_clusters:
        tiny_size = np.sum(cluster_labels == tiny_label)
        tiny_centroid = cluster_centroids[tiny_label]

        min_distance = float('inf')
        nearest_cluster = None
        for normal_label in normal_clusters:
            distance = np.linalg.norm(tiny_centroid - cluster_centroids[normal_label])
            if distance < min_distance:
                min_distance = distance
                nearest_cluster = normal_label

        if nearest_cluster is not None:
            merged_labels[cluster_labels == tiny_label] = nearest_cluster
            merge_stats['num_merged'] += 1
            merge_stats['tiny_sizes'].append(tiny_size)
            merge_stats['merge_targets'].append(nearest_cluster)

    return merged_labels, merge_stats
