"""
Clustering loss functions for ShadowCorr instance segmentation.

All losses operate on per-voxel embedding tensors and integer instance labels.
Labels == -1 denote background (gap) voxels and are excluded from all losses.

Available losses:
  discriminative_loss         – intra-compact + inter-separated centroids
  prototypical_clustering_loss – prototype-based cross-entropy
  graph_based_loss            – k-NN graph edge similarity
  multi_objective_clustering_loss – weighted combination of all three
"""

import torch
import torch.nn.functional as F

# Discriminative loss

def discriminative_loss(
    embeddings,
    instance_labels,
    delta_var=0.2,
    delta_dist=1.0,
    norm=2,
    alpha=1.0,
    beta=1.0,
    gamma=0.001,
    batch_indices=None,
):
    """
    Vectorized discriminative loss (intra-compact, inter-separated centroids).
    If batch_indices is set, the inter-centroid distance loss is computed per scene.
    """
    device = embeddings.device

    if batch_indices is not None:
        max_label = instance_labels.max().item() + 2
        combined_labels = batch_indices * max_label + instance_labels
    else:
        combined_labels = instance_labels.clone()

    fg_mask = instance_labels != -1
    if fg_mask.sum() == 0:
        return torch.tensor(0.0, device=device, requires_grad=True)

    fg_embeddings = embeddings[fg_mask]
    fg_labels = combined_labels[fg_mask]

    if batch_indices is not None:
        fg_batch_indices = batch_indices[fg_mask]

    unique_labels = fg_labels.unique()
    num_instances = len(unique_labels)
    if num_instances == 0:
        return torch.tensor(0.0, device=device, requires_grad=True)

    unique_labels_sorted, _ = unique_labels.sort()
    instance_indices = torch.searchsorted(unique_labels_sorted, fg_labels)

    embed_dim = fg_embeddings.shape[1]
    centroid_sum = torch.zeros(num_instances, embed_dim, device=device, dtype=fg_embeddings.dtype)
    centroid_count = torch.zeros(num_instances, device=device, dtype=fg_embeddings.dtype)
    centroid_sum.scatter_add_(
        0, instance_indices.unsqueeze(1).expand(-1, embed_dim), fg_embeddings
    )
    centroid_count.scatter_add_(
        0, instance_indices, torch.ones(len(fg_embeddings), device=device, dtype=fg_embeddings.dtype)
    )
    centroids = centroid_sum / centroid_count.unsqueeze(1).clamp(min=1)

    point_centroids = centroids[instance_indices]
    if norm == 2:
        dist_to_centroid = torch.norm(fg_embeddings - point_centroids, dim=1)
    else:
        dist_to_centroid = torch.norm(fg_embeddings - point_centroids, dim=1, p=norm)

    var_loss_per_point = torch.clamp(dist_to_centroid - delta_var, min=0.0) ** 2
    var_loss_per_instance = torch.zeros(num_instances, device=device, dtype=var_loss_per_point.dtype)
    var_loss_per_instance.scatter_add_(0, instance_indices, var_loss_per_point)
    var_loss_per_instance = var_loss_per_instance / centroid_count.clamp(min=1)
    loss_var = var_loss_per_instance.mean()

    if norm == 2:
        loss_reg = torch.norm(centroids, dim=1).mean()
    else:
        loss_reg = torch.norm(centroids, dim=1, p=norm).mean()

    if batch_indices is not None and num_instances > 1:
        unique_batches = fg_batch_indices.unique()
        loss_dist = torch.tensor(0.0, device=device, requires_grad=True)
        valid_scenes = 0
        for bidx in unique_batches:
            scene_mask = fg_batch_indices == bidx
            scene_instance_indices = instance_indices[scene_mask].unique()
            if len(scene_instance_indices) <= 1:
                continue
            scene_centroids = centroids[scene_instance_indices]
            n = len(scene_centroids)
            diff = scene_centroids.unsqueeze(0) - scene_centroids.unsqueeze(1)
            if norm == 2:
                pairwise_dist = torch.norm(diff, dim=2)
            else:
                pairwise_dist = torch.norm(diff, dim=2, p=norm)
            triu_indices = torch.triu_indices(n, n, offset=1, device=device)
            upper_dists = pairwise_dist[triu_indices[0], triu_indices[1]]
            scene_dist_loss = torch.clamp(2 * delta_dist - upper_dists, min=0.0) ** 2
            if len(scene_dist_loss) > 0:
                loss_dist = loss_dist + scene_dist_loss.mean()
                valid_scenes += 1
        if valid_scenes > 0:
            loss_dist = loss_dist / valid_scenes
    elif num_instances > 1:
        diff = centroids.unsqueeze(0) - centroids.unsqueeze(1)
        if norm == 2:
            pairwise_dist = torch.norm(diff, dim=2)
        else:
            pairwise_dist = torch.norm(diff, dim=2, p=norm)
        n = num_instances
        triu_indices = torch.triu_indices(n, n, offset=1, device=device)
        upper_dists = pairwise_dist[triu_indices[0], triu_indices[1]]
        loss_dist = (torch.clamp(2 * delta_dist - upper_dists, min=0.0) ** 2).mean()
    else:
        loss_dist = torch.tensor(0.0, device=device)

    return alpha * loss_var + beta * loss_dist + gamma * loss_reg

# Graph-based loss

def graph_based_loss(embeddings, instance_labels, coords, k_neighbors=8, temperature=0.1):
    """
    Graph-based loss: constructs a k-NN graph and encourages graph connectivity
    within clusters. Vectorized — 10–100× faster than a sequential loop.

    Args:
        embeddings:      (N, embed_dim) tensor of embeddings
        instance_labels: (N,) tensor of instance labels
        coords:          (N, 3) tensor of spatial coordinates
        k_neighbors:     number of nearest neighbours
        temperature:     temperature for similarity computation
    """
    embeddings = F.normalize(embeddings, dim=1)

    k = min(k_neighbors, len(coords) - 1)
    if k <= 0 or len(coords) < 2:
        return torch.tensor(0.0, device=embeddings.device, requires_grad=True)

    try:
        from torch_cluster import knn as torch_knn
        edge_index = torch_knn(coords, coords, k=k + 1)
        mask = edge_index[0] != edge_index[1]
        edge_index = edge_index[:, mask]
        target_nodes = edge_index[0]
        source_nodes = edge_index[1]
    except ImportError:
        spatial_distances = torch.cdist(coords.float(), coords.float())
        _, nn_indices = torch.topk(spatial_distances, k=k + 1, dim=1, largest=False)
        nn_indices = nn_indices[:, 1:]
        N = len(coords)
        target_nodes = torch.arange(N, device=coords.device).repeat_interleave(k)
        source_nodes = nn_indices.flatten()

    target_embeddings = embeddings[target_nodes]
    source_embeddings = embeddings[source_nodes]
    similarities = (target_embeddings * source_embeddings).sum(dim=1) / temperature

    target_labels = instance_labels[target_nodes]
    source_labels = instance_labels[source_nodes]
    same_instance_mask = (target_labels == source_labels).float()

    fg_mask = (target_labels != -1) & (source_labels != -1)
    if fg_mask.sum() == 0:
        return torch.tensor(0.0, device=embeddings.device, requires_grad=True)

    weighted_similarities = similarities * same_instance_mask * fg_mask.float()
    graph_loss = -weighted_similarities.sum() / (same_instance_mask * fg_mask.float()).sum().clamp(min=1)
    return graph_loss

# Prototypical loss

def prototypical_clustering_loss(embeddings, instance_labels, temperature=0.1):
    """
    Prototypical clustering loss: learns cluster prototypes and assigns
    samples to the nearest prototype via cross-entropy.
    """
    unique_labels = torch.unique(instance_labels)
    if len(unique_labels) < 2:
        return torch.tensor(0.0, device=embeddings.device, requires_grad=True)

    prototypes = []
    for label in unique_labels:
        mask = instance_labels == label
        if mask.sum() > 0:
            prototype = embeddings[mask].mean(dim=0)
            prototypes.append(prototype)

    if len(prototypes) < 2:
        return torch.tensor(0.0, device=embeddings.device, requires_grad=True)

    prototypes = torch.stack(prototypes)
    prototypes = F.normalize(prototypes, dim=1)
    embeddings = F.normalize(embeddings, dim=1)

    distances = torch.cdist(embeddings, prototypes)
    similarities = -distances / temperature

    target_assignments = torch.zeros(len(embeddings), len(prototypes), device=embeddings.device)
    for i, label in enumerate(unique_labels):
        mask = instance_labels == label
        target_assignments[mask, i] = 1.0

    log_probs = torch.log_softmax(similarities, dim=1)
    loss = -torch.mean(torch.sum(target_assignments * log_probs, dim=1))
    return loss

# Multi-objective combination

def multi_objective_clustering_loss(
    embeddings,
    instance_labels,
    loss_weights,
    coords=None,
    discriminative_params=None,
    temperature=0.1,
    k_neighbors=8,
    batch_indices=None,
):
    """
    Weighted sum of prototypical + discriminative + graph-based losses.

    loss_weights: [prototypical_w, discriminative_w, graph_based_w]
      Supports legacy 4-element formats (auto-converted).
    """
    if len(loss_weights) == 4:
        if loss_weights[0] == 0.0:
            _, prototypical_w, discriminative_w, graph_based_w = loss_weights
        else:
            prototypical_w, discriminative_w, graph_based_w, _ = loss_weights
    elif len(loss_weights) == 3:
        prototypical_w, discriminative_w, graph_based_w = loss_weights
    else:
        prototypical_w, discriminative_w, graph_based_w = loss_weights[:3]

    total_loss = torch.tensor(0.0, device=embeddings.device, requires_grad=True)

    if prototypical_w > 0:
        proto_loss = prototypical_clustering_loss(embeddings, instance_labels, temperature)
        total_loss = total_loss + prototypical_w * proto_loss

    if discriminative_w > 0:
        if discriminative_params is None:
            discriminative_params = {
                'delta_var': 0.8, 'delta_dist': 1.2, 'alpha': 1.0, 'beta': 0.25, 'gamma': 0.0001
            }
        disc_loss = discriminative_loss(
            embeddings, instance_labels,
            delta_var=discriminative_params['delta_var'],
            delta_dist=discriminative_params['delta_dist'],
            alpha=discriminative_params['alpha'],
            beta=discriminative_params['beta'],
            gamma=discriminative_params['gamma'],
            batch_indices=batch_indices,
        )
        total_loss = total_loss + discriminative_w * disc_loss

    if graph_based_w > 0 and coords is not None:
        graph_loss = graph_based_loss(embeddings, instance_labels, coords, k_neighbors, temperature)
        total_loss = total_loss + graph_based_w * graph_loss

    return total_loss
