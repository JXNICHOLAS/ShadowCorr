"""
Single-combination training loop (train_one.py).

Called once per hyperparameter combination by sweep.py.

train_model_focused             – full epoch loop for one param dict
evaluate_model_on_scene_focused – ARI evaluation on one NPZ scene
get_validation_files            – collect all validation NPZ paths
"""

import csv
import logging
import os
from pathlib import Path

import numpy as np
import torch
from torchsparse import SparseTensor
from tqdm import tqdm

from shadowcorr.models.features import heatmap_to_sparse_tensor_with_geometry
from shadowcorr.models.network import RockInstanceNetSparse
from shadowcorr.pipeline.early_stopping import should_stop
from shadowcorr.pipeline.losses import multi_objective_clustering_loss
from shadowcorr.pipeline.metrics import compute_clustering_ari

_log = logging.getLogger(__name__)

# Helpers

def get_validation_files(valid_dir: str):
    """Return (eval_files, scene_names) lists from the configured validation directory."""
    import glob
    validation_files = glob.glob(os.path.join(valid_dir, "*.npz"))
    eval_files, scene_names = [], []
    for vf in sorted(validation_files):
        eval_files.append(vf)
        scene_names.append(os.path.basename(vf).replace('.npz', ''))
    _log.info("Validation files: %d total", len(validation_files))
    return eval_files, scene_names

def evaluate_model_on_scene_focused(
    model, scene_path, bandwidth, device, use_confidence=True, use_segment=True
):
    """
    Evaluate a trained model on one NPZ scene.

    Returns a dict with keys: success, ari_score, num_predicted, num_ground_truth (or error).
    """
    try:
        with np.load(scene_path, allow_pickle=True) as data:
            if 'voxel_positions' in data and 'voxel_labels' in data and 'voxel_confidences' in data:
                voxel_positions = data['voxel_positions']
                voxel_labels = data['voxel_labels']
                voxel_scores = data['voxel_confidences']
                voxel_occupancy = {tuple(pos): float(score) for pos, score in zip(voxel_positions, voxel_scores)}
                precomputed_segment_embeddings = {}
                if 'voxel_segment_embeddings' in data:
                    segment_embeddings = data['voxel_segment_embeddings']
                    for i, pos in enumerate(voxel_positions):
                        precomputed_segment_embeddings[tuple(pos)] = torch.tensor(
                            segment_embeddings[i], dtype=torch.float32
                        )
                instance_labels = voxel_labels.astype(np.int64)
            else:
                return {'success': False, 'error': 'NPZ file does not contain required format'}

        coords, feats = heatmap_to_sparse_tensor_with_geometry(
            voxel_occupancy,
            precomputed_segment_embeddings=precomputed_segment_embeddings,
            use_confidence=use_confidence,
            use_segment=use_segment,
        )
        coords, feats = coords.to(device), feats.to(device)

        stensor = SparseTensor(coords=coords, feats=feats)
        with torch.no_grad():
            instance_embed = model(stensor)

        ari_result = compute_clustering_ari(
            instance_embed.detach(),
            torch.tensor(instance_labels, dtype=torch.long),
            bandwidth=bandwidth,
        )
        return {
            'success': True,
            'num_predicted': ari_result['num_predicted'],
            'num_ground_truth': ari_result['num_ground_truth'],
            'ari_score': ari_result['ari_score'],
        }
    except Exception as e:
        return {'success': False, 'error': str(e)}

# Training loop

def _build_batch(coords_batch, feats_batch, labels_batch, device):
    """Combine a list of per-scene tensors into a single batched SparseTensor."""
    all_coords, all_feats, all_labels = [], [], []
    label_offset = 0
    valid_scenes = 0

    for coords, feats, instance_labels in zip(coords_batch, feats_batch, labels_batch):
        fg_mask = instance_labels != -1
        if fg_mask.sum() == 0:
            continue
        coords_rebatched = coords.clone()
        coords_rebatched[:, 0] = valid_scenes
        instance_labels_offset = instance_labels.clone()
        instance_labels_offset[fg_mask] = instance_labels[fg_mask] + label_offset
        if fg_mask.any():
            label_offset += instance_labels[fg_mask].max().item() + 1
        all_coords.append(coords_rebatched)
        all_feats.append(feats)
        all_labels.append(instance_labels_offset)
        valid_scenes += 1

    if valid_scenes == 0:
        return None, None, None, 0

    batched_coords = torch.cat(all_coords, dim=0).to(device)
    batched_feats = torch.cat(all_feats, dim=0).to(device)
    batched_labels = torch.cat(all_labels, dim=0).to(device)
    return batched_coords, batched_feats, batched_labels, valid_scenes

def train_model_focused(
    params, train_loader, valid_loader, device,
    eval_files=None, scene_names=None,
    checkpoint_dir=None, log_file=None,
):
    """
    Train one model for the given hyperparameter combination.

    Returns a result dict with model, losses, best states, and epoch history.
    """
    seed = params.get('seed', 42)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    if checkpoint_dir is not None:
        os.makedirs(checkpoint_dir, exist_ok=True)
        checkpoint_path = os.path.join(checkpoint_dir, 'checkpoint_last.pth')
    else:
        checkpoint_path = None

    log_writer = log_csv = None
    if log_file is not None:
        log_exists = os.path.exists(log_file)
        log_writer = open(log_file, 'a', newline='')
        log_csv = csv.writer(log_writer)
        if not log_exists:
            log_csv.writerow([
                'epoch', 'train_loss', 'valid_loss', 'lr',
                'best_valid_loss', 'best_epoch',
                'best_ari', 'best_ari_epoch',
                'prototypical', 'discriminative', 'graph',
            ])

    use_confidence = params.get('use_confidence', True)
    use_segment = params.get('use_segment', True)
    batch_size = params.get('batch_size', 1)
    _log.info("  Batch size: %d  use_confidence=%s  use_segment=%s  in_channels=%s",
              batch_size, use_confidence, use_segment, params.get("in_channels"))

    model = RockInstanceNetSparse(
        in_channels=params['in_channels'],
        instance_embed_dim=params['instance_embed_dim'],
        attn_k1=params.get('attn_k1', params.get('attn_k', 16)),
        attn_k2=params.get('attn_k2', params.get('attn_k', 16)),
        num_heads1=params.get('num_heads1', 4),
        num_heads2=params.get('num_heads2', 8),
    ).to(device)

    if params.get('use_pretrained', False):
        pretrained_path = params.get('pretrained_model_path', '')
        if os.path.exists(pretrained_path):
            try:
                model.load_state_dict(torch.load(pretrained_path, map_location=device, weights_only=False))
                _log.info("  Loaded pretrained model from: %s", pretrained_path)
            except Exception as e:
                _log.warning("  Failed to load pretrained model: %s — using random init", e)
        else:
            _log.warning("  Pretrained model not found at %s — using random init", pretrained_path)
    else:
        _log.info("  Using random initialization")

    optimizer = torch.optim.Adam(model.parameters(), lr=params['learning_rate'])

    lr_scheduler_type = params.get('lr_scheduler', 'cosine').lower()
    lr_scheduler_params = params.get('lr_scheduler_params', {})
    scheduler = None
    if lr_scheduler_type == 'cosine':
        T_max = lr_scheduler_params.get('T_max', params['num_epochs'])
        eta_min = lr_scheduler_params.get('eta_min', params['learning_rate'] * 0.01)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max, eta_min=eta_min)
        _log.info("  LR scheduler: CosineAnnealingLR (T_max=%d, eta_min=%.2e)", T_max, eta_min)
    elif lr_scheduler_type == 'step':
        step_size = lr_scheduler_params.get('step_size', params['num_epochs'] // 3)
        gamma = lr_scheduler_params.get('gamma', 0.5)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
        _log.info("  LR scheduler: StepLR (step_size=%d, gamma=%.2f)", step_size, gamma)
    elif lr_scheduler_type == 'exponential':
        gamma = lr_scheduler_params.get('gamma', 0.95)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)
        _log.info("  LR scheduler: ExponentialLR (gamma=%.2f)", gamma)
    else:
        _log.info("  LR scheduler: none (fixed learning rate)")

    scaler = torch.amp.GradScaler('cuda')

    # Checkpoint resume
    start_epoch = 0
    best_valid_loss = float('inf')
    best_model_state = None
    best_epoch = None
    best_ari = -1.0
    best_ari_model_state = None
    best_ari_epoch = 0
    epoch_loss_components = []

    if params.get('resume_from_checkpoint', False):
        resume_path = params.get('resume_checkpoint_path', None)
        ckpt_file = resume_path if resume_path and os.path.exists(resume_path) else checkpoint_path
        if ckpt_file and os.path.exists(ckpt_file):
            try:
                ckpt = torch.load(ckpt_file, map_location=device, weights_only=False)
                model.load_state_dict(ckpt['model_state_dict'])
                optimizer.load_state_dict(ckpt['optimizer_state_dict'])
                if scheduler is not None and 'scheduler_state_dict' in ckpt:
                    scheduler.load_state_dict(ckpt['scheduler_state_dict'])
                if 'scaler_state_dict' in ckpt:
                    scaler.load_state_dict(ckpt['scaler_state_dict'])
                start_epoch = ckpt.get('epoch', 0)
                best_valid_loss = ckpt.get('best_valid_loss', float('inf'))
                best_model_state = {k: v.clone() for k, v in ckpt['best_model_state'].items()} if ckpt.get('best_model_state') else None
                best_epoch = ckpt.get('best_epoch', None)
                best_ari = ckpt.get('best_ari', -1.0)
                best_ari_model_state = {k: v.clone() for k, v in ckpt['best_ari_model_state'].items()} if ckpt.get('best_ari_model_state') else None
                best_ari_epoch = ckpt.get('best_ari_epoch', 0)
                epoch_loss_components = ckpt.get('epoch_loss_components', [])
                _log.info("  Resumed from checkpoint: epoch %d/%d", start_epoch, params["num_epochs"])
            except Exception as e:
                _log.warning("  Failed to load checkpoint: %s — starting fresh", e)

    delta_var = params.get('delta_var', 0.2)
    delta_dist = params.get('delta_dist', 1.0)
    alpha = params.get('alpha', 1.0)
    beta = params.get('beta', 0.25)
    gamma = params.get('gamma', 0.0001)
    loss_weights = params.get('loss_weights', [0.0, 0.0, 1.0, 0.0])
    k_neighbors = params.get('k_neighbors', 8)
    run_ari_per_epoch = params.get('run_ari_per_epoch', False)
    discriminative_params = {
        'delta_var': delta_var, 'delta_dist': delta_dist,
        'alpha': alpha, 'beta': beta, 'gamma': gamma,
    }

    model.train()
    assert model.training
    total_loss = 0.0
    batch_count = 0
    early_stopped = False

    for epoch in range(start_epoch, params['num_epochs']):
        if should_stop():
            _log.info("  Early stop at epoch %d", epoch + 1)
            early_stopped = True
            break

        _log.debug("  Training epoch %d/%d ...", epoch + 1, params["num_epochs"])
        epoch_train_loss = 0.0
        epoch_batch_count = 0
        model.train()
        optimizer.zero_grad()
        torch.cuda.empty_cache()

        for scene_data in tqdm(train_loader, desc=f"Epoch {epoch+1} Train", leave=False, ncols=100):
            if should_stop():
                early_stopped = True
                break
            coords_batch, feats_batch, labels_batch = scene_data
            batched_coords, batched_feats, batched_labels, n_scenes = _build_batch(
                coords_batch, feats_batch, labels_batch, device
            )
            if n_scenes == 0:
                continue

            stensor = SparseTensor(coords=batched_coords, feats=batched_feats)
            with torch.amp.autocast('cuda'):
                instance_embed = model(stensor)
                spatial_coords = batched_coords[:, 1:4].float()
                loss = multi_objective_clustering_loss(
                    instance_embed, batched_labels, loss_weights,
                    coords=spatial_coords,
                    discriminative_params=discriminative_params,
                    temperature=params.get('temperature', 0.1),
                    k_neighbors=k_neighbors,
                    batch_indices=batched_coords[:, 0],
                )

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            loss_value = loss.item()
            epoch_train_loss += loss_value
            epoch_batch_count += n_scenes
            total_loss += loss_value
            batch_count += n_scenes

            del loss, instance_embed, stensor, batched_coords, batched_feats, batched_labels, spatial_coords
            if batch_count % 10 == 0:
                torch.cuda.empty_cache()

        model.eval()
        assert not model.training
        epoch_valid_loss = 0.0
        valid_batch_count = 0

        with torch.no_grad():
            for scene_data in tqdm(valid_loader, desc=f"Epoch {epoch+1} Valid", leave=False, ncols=100):
                coords_batch, feats_batch, labels_batch = scene_data
                batched_coords, batched_feats, batched_labels, n_scenes = _build_batch(
                    coords_batch, feats_batch, labels_batch, device
                )
                if n_scenes == 0:
                    continue
                stensor = SparseTensor(coords=batched_coords, feats=batched_feats)
                with torch.amp.autocast('cuda'):
                    instance_embed = model(stensor)
                    spatial_coords = batched_coords[:, 1:4].float()
                    valid_loss = multi_objective_clustering_loss(
                        instance_embed, batched_labels, loss_weights,
                        coords=spatial_coords,
                        discriminative_params=discriminative_params,
                        temperature=params.get('temperature', 0.1),
                        k_neighbors=k_neighbors,
                        batch_indices=batched_coords[:, 0],
                    )
                epoch_valid_loss += valid_loss.item()
                valid_batch_count += n_scenes

        avg_train_loss = epoch_train_loss / max(epoch_batch_count, 1)
        avg_valid_loss = epoch_valid_loss / max(valid_batch_count, 1)

        epoch_components = {'epoch': epoch + 1, 'train_loss': float(avg_train_loss), 'valid_loss': float(avg_valid_loss)}

        # Sample one validation batch for per-component logging
        with torch.no_grad():
            model.eval()
            for scene_data in valid_loader:
                coords_batch, feats_batch, labels_batch = scene_data
                batched_coords, batched_feats, batched_labels, n_scenes = _build_batch(
                    [coords_batch[0]], [feats_batch[0]], [labels_batch[0]], device
                )
                if n_scenes == 0:
                    break
                stensor = SparseTensor(coords=batched_coords, feats=batched_feats)
                with torch.amp.autocast('cuda'):
                    instance_embed = model(stensor)
                embed_fp32 = instance_embed.float()
                from shadowcorr.pipeline.losses import (
                    prototypical_clustering_loss, discriminative_loss as disc_loss_fn, graph_based_loss
                )
                lw = loss_weights
                pw = lw[0] if len(lw) >= 1 else 0.0
                dw = lw[1] if len(lw) >= 2 else 0.0
                gw = lw[2] if len(lw) >= 3 else 0.0
                if pw > 0:
                    epoch_components['prototypical'] = float(
                        prototypical_clustering_loss(embed_fp32, batched_labels, params.get('temperature', 0.1)).item()
                    )
                if dw > 0:
                    epoch_components['discriminative'] = float(
                        disc_loss_fn(embed_fp32, batched_labels,
                                     delta_var=delta_var, delta_dist=delta_dist,
                                     alpha=alpha, beta=beta, gamma=gamma,
                                     batch_indices=batched_coords[:, 0]).item()
                    )
                if gw > 0:
                    epoch_components['graph'] = float(
                        graph_based_loss(embed_fp32, batched_labels, batched_coords[:, 1:4].float(),
                                         k_neighbors, params.get('temperature', 0.1)).item()
                    )
                break
            model.train()

        epoch_loss_components.append(epoch_components)

        if avg_valid_loss < best_valid_loss:
            best_valid_loss = avg_valid_loss
            best_model_state = {k: v.clone() for k, v in model.state_dict().items()}
            best_epoch = epoch + 1

        if scheduler is not None:
            scheduler.step()
            current_lr = optimizer.param_groups[0]['lr']
            _log.info(
                "  Epoch %d/%d — Train: %.4f  Valid: %.4f  LR: %.2e",
                epoch + 1, params["num_epochs"], avg_train_loss, avg_valid_loss, current_lr,
            )
        else:
            _log.info(
                "  Epoch %d/%d — Train: %.4f  Valid: %.4f",
                epoch + 1, params["num_epochs"], avg_train_loss, avg_valid_loss,
            )

        if run_ari_per_epoch and eval_files:
            model.eval()
            with torch.no_grad():
                epoch_aris = []
                for scene_idx in tqdm(range(len(eval_files)), desc=f"Epoch {epoch+1} ARI", leave=False, ncols=100):
                    result = evaluate_model_on_scene_focused(
                        model, eval_files[scene_idx], params['bandwidth'], device,
                        use_confidence=use_confidence, use_segment=use_segment,
                    )
                    epoch_aris.append(result['ari_score'] if result['success'] else 0.0)
            if epoch_aris:
                avg_ari = np.mean(epoch_aris)
                _log.info(
                    "  Epoch %d ARI (%d scenes): %.3f | Min: %.3f | Max: %.3f",
                    epoch + 1, len(epoch_aris), avg_ari, np.min(epoch_aris), np.max(epoch_aris),
                )
                if avg_ari > best_ari:
                    best_ari = avg_ari
                    best_ari_model_state = {k: v.clone() for k, v in model.state_dict().items()}
                    best_ari_epoch = epoch + 1
                    _log.info("  NEW BEST ARI: %.3f at epoch %d", avg_ari, epoch + 1)
            model.train()

        if checkpoint_path is not None:
            try:
                ckpt = {
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_valid_loss': best_valid_loss,
                    'best_model_state': best_model_state,
                    'best_epoch': best_epoch,
                    'best_ari': best_ari,
                    'best_ari_model_state': best_ari_model_state,
                    'best_ari_epoch': best_ari_epoch,
                    'epoch_loss_components': epoch_loss_components,
                    'params': params,
                }
                if scheduler is not None:
                    ckpt['scheduler_state_dict'] = scheduler.state_dict()
                ckpt['scaler_state_dict'] = scaler.state_dict()
                torch.save(ckpt, checkpoint_path)
            except Exception as e:
                _log.warning("  Failed to save checkpoint: %s", e)

        if log_writer is not None:
            try:
                current_lr = optimizer.param_groups[0]['lr']
                log_csv.writerow([
                    epoch + 1, avg_train_loss, avg_valid_loss, current_lr,
                    best_valid_loss, best_epoch if best_epoch is not None else '',
                    best_ari if best_ari > 0 else '',
                    best_ari_epoch if best_ari_epoch > 0 else '',
                    epoch_components.get('prototypical', ''),
                    epoch_components.get('discriminative', ''),
                    epoch_components.get('graph', ''),
                ])
                log_writer.flush()
            except Exception as e:
                _log.warning("  Failed to write CSV log: %s", e)

    if not run_ari_per_epoch and best_model_state is not None:
        best_ari_model_state = {k: v.clone() for k, v in best_model_state.items()}
        best_ari_epoch = best_epoch

    avg_loss = total_loss / batch_count if batch_count > 0 else 0.0
    if log_writer is not None:
        log_writer.close()

    return {
        'success': True,
        'model': model,
        'avg_loss': avg_valid_loss if 'avg_valid_loss' in dir() else avg_loss,
        'best_valid_loss': best_valid_loss,
        'best_model_state': best_model_state,
        'best_epoch': best_epoch,
        'best_ari': best_ari,
        'best_ari_model_state': best_ari_model_state,
        'best_ari_epoch': best_ari_epoch,
        'early_stopped': early_stopped,
        'delta_params': {'delta_var': delta_var, 'delta_dist': delta_dist, 'alpha': alpha, 'beta': beta, 'gamma': gamma},
        'epoch_loss_components': epoch_loss_components,
    }
