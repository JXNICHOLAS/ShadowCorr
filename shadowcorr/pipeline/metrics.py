"""
Clustering evaluation metrics for ShadowCorr.

compute_clustering_ari       – voxel-level ARI with MeanShift + optional merge
evaluate_segment_clustering  – full segment-level + voxel-level metrics
print_batch_summary          – console summary table for batch evaluation runs
"""

from collections import Counter, defaultdict

import numpy as np
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

from shadowcorr.pipeline.postprocessing import merge_small_clusters


# Voxel-level ARI (used during training)

def compute_clustering_ari(
    embeddings, instance_labels, bandwidth=0.8, enable_merge=True, min_cluster_size=5
):
    """MeanShift on embeddings, optional small-cluster merge, then ARI vs instance_labels."""
    try:
        from sklearn.cluster import MeanShift

        embeddings_np = (
            embeddings.detach().cpu().numpy()
            if embeddings.requires_grad
            else embeddings.cpu().numpy()
        )
        labels_np = (
            instance_labels.detach().cpu().numpy()
            if instance_labels.requires_grad
            else instance_labels.cpu().numpy()
        )
        fg_mask = labels_np != -1
        if np.sum(fg_mask) == 0:
            return {"ari_score": 0.0, "num_predicted": 0, "num_ground_truth": 0}

        fg_embeddings = embeddings_np[fg_mask]
        fg_labels = labels_np[fg_mask]
        np.random.seed(42)
        cluster_labels = MeanShift(bandwidth=bandwidth).fit_predict(fg_embeddings)
        if enable_merge:
            cluster_labels = merge_small_clusters(fg_embeddings, cluster_labels, min_cluster_size)
        ari_score = adjusted_rand_score(fg_labels, cluster_labels)
        return {
            "ari_score": float(ari_score),
            "num_predicted": int(len(np.unique(cluster_labels))),
            "num_ground_truth": int(len(np.unique(fg_labels))),
        }
    except Exception as e:
        return {"ari_score": 0.0, "num_predicted": 0, "num_ground_truth": 0, "error": str(e)}


# Segment-level evaluation (used during evaluation)

def evaluate_segment_clustering(
    segment_assignments,
    segment_to_rock,
    voxel_to_segments,
    voxel_positions,
    cluster_labels,
    voxel_gt_labels,
    rock_segments,
):
    """
    Evaluate how well segments are clustered versus ground-truth rock labels.

    Returns a metrics dict with voxel_ari, segment_ari, rock_purity,
    cluster_purity, perfect_1to1_matches, and timing breakdowns.
    """
    valid_segments = [
        seg_idx for seg_idx, assignment in segment_assignments.items()
        if assignment['predicted_cluster'] != -1
    ]

    if len(valid_segments) == 0:
        print("No valid segments to evaluate.")
        return {}

    predicted_clusters_weighted = []
    ground_truth_rocks_weighted = []

    for seg_idx in valid_segments:
        pred_cluster = segment_assignments[seg_idx]['predicted_cluster']
        gt_rock = segment_assignments[seg_idx]['ground_truth_rock']

        if len(rock_segments) > 0 and seg_idx < len(rock_segments):
            num_points = len(rock_segments[seg_idx].point_cloud.points)
        else:
            num_points = segment_assignments[seg_idx]['num_voxels']

        predicted_clusters_weighted.extend([pred_cluster] * num_points)
        ground_truth_rocks_weighted.extend([gt_rock] * num_points)

    predicted_clusters_weighted = np.array(predicted_clusters_weighted)
    ground_truth_rocks_weighted = np.array(ground_truth_rocks_weighted)

    segment_ari = adjusted_rand_score(ground_truth_rocks_weighted, predicted_clusters_weighted)
    segment_nmi = normalized_mutual_info_score(ground_truth_rocks_weighted, predicted_clusters_weighted)

    pos_to_cluster = {pos: cluster_labels[i] for i, pos in enumerate(voxel_positions)}
    voxel_gt_labels_list = []
    voxel_pred_labels_list = []
    for pos in voxel_positions:
        gt_rock = voxel_gt_labels.get(pos, -1)
        if gt_rock != -1:
            voxel_gt_labels_list.append(gt_rock)
            voxel_pred_labels_list.append(pos_to_cluster[pos])

    if len(voxel_gt_labels_list) > 0:
        voxel_ari = adjusted_rand_score(voxel_gt_labels_list, voxel_pred_labels_list)
        voxel_nmi = normalized_mutual_info_score(voxel_gt_labels_list, voxel_pred_labels_list)
    else:
        voxel_ari = 0.0
        voxel_nmi = 0.0

    cluster_to_rocks = defaultdict(list)
    rock_to_clusters = defaultdict(list)
    for seg_idx in valid_segments:
        gt_rock = segment_assignments[seg_idx]['ground_truth_rock']
        pred_cluster = segment_assignments[seg_idx]['predicted_cluster']
        cluster_to_rocks[pred_cluster].append(gt_rock)
        rock_to_clusters[gt_rock].append(pred_cluster)

    rock_purities = [
        Counter(clusters).most_common(1)[0][1] / len(clusters)
        for clusters in rock_to_clusters.values()
    ]
    cluster_purities = [
        Counter(rocks).most_common(1)[0][1] / len(rocks)
        for rocks in cluster_to_rocks.values()
    ]

    avg_rock_purity = np.mean(rock_purities) if rock_purities else 0.0
    avg_cluster_purity = np.mean(cluster_purities) if cluster_purities else 0.0
    perfect_rock_matches = sum(1 for p in rock_purities if p == 1.0)

    perfect_matches = 0
    for rock_id, clusters in rock_to_clusters.items():
        unique_clusters = set(clusters)
        if len(unique_clusters) == 1:
            cluster_id = list(unique_clusters)[0]
            if len(set(cluster_to_rocks[cluster_id])) == 1:
                perfect_matches += 1

    overseg_rocks = sum(1 for clusters in rock_to_clusters.values() if len(set(clusters)) > 1)
    underseg_clusters = sum(1 for rocks in cluster_to_rocks.values() if len(set(rocks)) > 1)

    print(f"\nVoxel-level  ARI: {voxel_ari:.4f}")
    print(f"Segment-level ARI (point-weighted): {segment_ari:.4f}")
    print(f"  Rock purity (avg):    {avg_rock_purity:.4f}  ({avg_rock_purity*100:.1f}%)")
    print(f"  Cluster purity (avg): {avg_cluster_purity:.4f}  ({avg_cluster_purity*100:.1f}%)")
    print(f"  Perfect 1-to-1 matches: {perfect_matches}/{len(rock_to_clusters)}")

    if overseg_rocks > 0 or underseg_clusters > 0:
        print(f"\nClustering errors:")
        print(f"  Over-segmentation:  {overseg_rocks}/{len(rock_to_clusters)} rocks split across clusters")
        if overseg_rocks > 0:
            rock_id_to_purity = {
                rock_id: rock_purities[i]
                for i, rock_id in enumerate(sorted(rock_to_clusters.keys()))
            }
            overseg_examples = [
                (rock_id, rock_id_to_purity[rock_id])
                for rock_id in rock_to_clusters.keys()
                if len(set(rock_to_clusters[rock_id])) > 1
            ]
            overseg_examples.sort(key=lambda x: x[1])
            print("    Worst 3: " + ", ".join(f"Rock {rid} (purity={pur:.2f})" for rid, pur in overseg_examples[:3]))
        print(f"  Under-segmentation: {underseg_clusters}/{len(cluster_to_rocks)} clusters contain multiple rocks")
        if underseg_clusters > 0:
            cluster_id_to_purity = {
                cluster_id: cluster_purities[i]
                for i, cluster_id in enumerate(sorted(cluster_to_rocks.keys()))
            }
            underseg_examples = [
                (cluster_id, cluster_id_to_purity[cluster_id])
                for cluster_id in cluster_to_rocks.keys()
                if len(set(cluster_to_rocks[cluster_id])) > 1
            ]
            underseg_examples.sort(key=lambda x: x[1])
            print("    Worst 3: " + ", ".join(f"Cluster {cid} (purity={pur:.2f})" for cid, pur in underseg_examples[:3]))

    return {
        'voxel_ari': float(voxel_ari),
        'segment_ari': float(segment_ari),
        'avg_rock_purity': float(avg_rock_purity),
        'avg_cluster_purity': float(avg_cluster_purity),
        'perfect_1to1_matches': int(perfect_matches),
        'perfect_1to1_accuracy': float(perfect_matches / len(rock_to_clusters)) if len(rock_to_clusters) > 0 else 0.0,
        'voxel_nmi': float(voxel_nmi),
        'segment_nmi': float(segment_nmi),
        'num_valid_voxels': len(voxel_gt_labels_list),
        'num_valid_segments': len(valid_segments),
        'num_total_segments': len(segment_assignments),
        'num_predicted_clusters': len(cluster_to_rocks),
        'num_ground_truth_rocks': len(rock_to_clusters),
        'overseg_rocks': int(overseg_rocks),
        'underseg_clusters': int(underseg_clusters),
        'perfect_rock_matches': int(perfect_rock_matches),
        'perfect_cluster_matches': int(sum(1 for p in cluster_purities if p == 1.0)),
        'rock_purities': [float(p) for p in rock_purities],
        'cluster_purities': [float(p) for p in cluster_purities],
    }


# Batch summary printing

def print_batch_summary(results):
    """Print aggregate statistics for a batch evaluation run."""
    successful_results = [r for r in results if r['success']]
    if len(successful_results) == 0:
        print("No successful results to summarize.")
        return

    print(f"\n{'='*70}")
    print(f"AGGREGATE STATISTICS ({len(successful_results)} scenes)")
    print(f"{'='*70}")

    voxel_aris = [r['metrics']['voxel_ari'] for r in successful_results]
    segment_aris = [r['metrics']['segment_ari'] for r in successful_results]
    rock_purities = [r['metrics']['avg_rock_purity'] for r in successful_results]
    cluster_purities = [r['metrics']['avg_cluster_purity'] for r in successful_results]

    print(f"\nVoxel-Level ARI:   {np.mean(voxel_aris):.4f} +/- {np.std(voxel_aris):.4f}  "
          f"(best {max(voxel_aris):.4f} / worst {min(voxel_aris):.4f})")
    print(f"Segment-Level ARI: {np.mean(segment_aris):.4f} +/- {np.std(segment_aris):.4f}  "
          f"(best {max(segment_aris):.4f} / worst {min(segment_aris):.4f})")
    print(f"Rock purity:       {np.mean(rock_purities):.2%} +/- {np.std(rock_purities):.2%}")
    print(f"Cluster purity:    {np.mean(cluster_purities):.2%} +/- {np.std(cluster_purities):.2%}")

    processing_times = [r['metrics']['processing_time_seconds'] for r in successful_results]
    preprocess_times = [r['metrics']['preprocessing_time_seconds'] for r in successful_results]
    inference_times = [r['metrics']['inference_time_seconds'] for r in successful_results]
    clustering_times = [r['metrics']['clustering_time_seconds'] for r in successful_results]
    postprocess_times = [r['metrics']['postprocessing_time_seconds'] for r in successful_results]
    segment_embed_times = [r['metrics']['segment_embedding_time_seconds'] for r in successful_results]
    confidence_times = [r['metrics']['confidence_extraction_time_seconds'] for r in successful_results]
    preprocess_no_segment = [r['metrics']['preprocessing_without_segment_embed_seconds'] for r in successful_results]
    preprocess_no_confidence = [r['metrics']['preprocessing_without_confidence_seconds'] for r in successful_results]
    preprocess_no_both = [r['metrics']['preprocessing_without_both_seconds'] for r in successful_results]

    print(f"\nTiming (avg per file):")
    print(f"  Preprocess:  {np.mean(preprocess_times)*1000:.1f}ms +/- {np.std(preprocess_times)*1000:.1f}ms")
    print(f"    segment embed: {np.mean(segment_embed_times)*1000:.1f}ms")
    print(f"    confidence:    {np.mean(confidence_times)*1000:.1f}ms")
    print(f"  Inference:   {np.mean(inference_times)*1000:.1f}ms +/- {np.std(inference_times)*1000:.1f}ms")
    print(f"  Clustering:  {np.mean(clustering_times)*1000:.1f}ms +/- {np.std(clustering_times)*1000:.1f}ms")
    print(f"  Postprocess: {np.mean(postprocess_times)*1000:.1f}ms +/- {np.std(postprocess_times)*1000:.1f}ms")
    print(f"  Total:       {np.mean(processing_times)*1000:.1f}ms +/- {np.std(processing_times)*1000:.1f}ms")
    print(f"  Total (all scenes): {sum(processing_times):.2f}s ({sum(processing_times)/60:.1f} min)")

    print(f"\nPreprocess ablation (w/o segment embed / confidence / both):")
    print(f"  {np.mean(preprocess_no_segment)*1000:.1f}ms / "
          f"{np.mean(preprocess_no_confidence)*1000:.1f}ms / "
          f"{np.mean(preprocess_no_both)*1000:.1f}ms")

    print(f"\n{'='*70}")
    print(f"{'Scene':<40} {'Vox ARI':>8} {'Seg ARI':>8} {'Rock%':>7} {'Clus%':>7} {'Time':>7}")
    print(f"{'-'*85}")
    for r in sorted(successful_results, key=lambda x: x['metrics']['voxel_ari'], reverse=True):
        m = r['metrics']
        print(f"{r['file']:<40} {m['voxel_ari']:>8.4f} {m['segment_ari']:>8.4f} "
              f"{m['avg_rock_purity']:>7.1%} {m['avg_cluster_purity']:>7.1%} "
              f"{m['processing_time_seconds']:>6.2f}s")
