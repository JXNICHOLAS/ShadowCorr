"""
Hyperparameter sweep orchestrator (sweep.py).

Loads training/validation data once, then runs every hyperparameter
combination defined in ``shadowcorr/conf/train.yaml`` sequentially.
Sharing the data loader across combinations avoids re-loading the dataset
on every run — the main advantage over Hydra's built-in ``--multirun``.

Entry point: run_focused_grid_search(cfg, logger)
"""

from __future__ import annotations

import json
import logging
import os
import time
from datetime import datetime
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from shadowcorr.models.data import load_data_from_folder
from shadowcorr.pipeline.early_stopping import (
    reset_stop_flag, should_stop, start_keyboard_monitoring, stop_keyboard_monitoring,
)
from shadowcorr.pipeline.train_one import (
    evaluate_model_on_scene_focused, get_validation_files, train_model_focused,
)


def run_focused_grid_search(
    cfg: DictConfig,
    logger: logging.Logger | None = None,
) -> dict[str, Any]:
    """Run the full grid search driven by *cfg* (a Hydra DictConfig).

    All hyperparameters come from ``cfg.grid`` and ``cfg.training``; nothing is
    hard-coded in this module.  Returns a dict with ``best_loss_params`` and
    ``best_ari_params``.
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    reset_stop_flag()
    start_keyboard_monitoring()

    # Convert OmegaConf → plain Python so itertools.product and json.dump work.
    raw: dict[str, Any] = OmegaConf.to_container(cfg, resolve=True)
    raw_training: dict = raw["training"]
    raw_grid: dict = dict(raw["grid"])        # shallow copy — we pop from it below
    train_dir: str = raw["data"]["train_dir"]
    valid_dir: str = raw["data"]["valid_dir"]

    # ── Fixed training params ────────────────────────────────────────────────
    use_confidence: bool = raw_training["use_confidence"]
    use_segment: bool    = raw_training["use_segment"]
    save_best_loss: bool = raw_training.get("save_best_loss", False)
    in_channels: int     = 3 + (1 if use_confidence else 0) + (12 if use_segment else 0)

    fixed_params: dict = {
        "num_epochs":             raw_training["num_epochs"],
        "in_channels":            in_channels,
        "instance_embed_dim":     raw_training["instance_embed_dim"],
        "num_heads1":             raw_training["num_heads1"],
        "num_heads2":             raw_training["num_heads2"],
        "use_confidence":         use_confidence,
        "use_segment":            use_segment,
        "seed":                   raw_training["seed"],
        "use_pretrained":         raw_training.get("use_pretrained", False),
        "pretrained_model_path":  raw_training.get("pretrained_model_path"),
        "resume_from_checkpoint": raw_training.get("resume_from_checkpoint", False),
        "resume_checkpoint_path": raw_training.get("resume_checkpoint_path"),
    }

    # ── Grid params ──────────────────────────────────────────────────────────
    # loss_weight_combinations is treated separately; everything else is swept.
    loss_weight_combinations: list = raw_grid.pop(
        "loss_weight_combinations", [[1.0, 1.0, 1.0]]
    )

    param_grids: dict[str, list] = {
        k: (v if isinstance(v, list) else [v]) for k, v in raw_grid.items()
    }

    base_param_names   = list(param_grids.keys())
    base_combinations  = list(product(*param_grids.values()))
    total_combinations = len(base_combinations) * len(loss_weight_combinations)

    logger.info("ShadowCorr Grid Search — Discriminative + Prototypical + Graph Loss")
    logger.info("=" * 70)
    logger.info("Press ` (backtick) to STOP training early and save best model")
    logger.info("=" * 70)
    logger.info("Search space: %d combinations", total_combinations)
    logger.info(
        "INPUT: use_confidence=%s  use_segment=%s  in_channels=%d",
        use_confidence, use_segment, in_channels,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    eval_files, scene_names = get_validation_files(valid_dir)
    logger.info("Validation: %d scenes", len(eval_files))

    batch_size = (param_grids.get("batch_size") or [1])[0]
    logger.info("Pre-loading data (batch_size=%d)...", batch_size)
    train_loader, _ = load_data_from_folder(
        train_dir, batch_size=batch_size,
        use_confidence=use_confidence, use_segment=use_segment,
    )
    valid_loader, _ = load_data_from_folder(
        valid_dir, batch_size=batch_size,
        use_confidence=use_confidence, use_segment=use_segment,
    )
    logger.info("%d train batches, %d valid batches", len(train_loader), len(valid_loader))

    best_avg_loss      = float("inf")
    best_loss_params   = best_loss_model_state = best_loss_epoch = best_loss_combination = None
    best_avg_ari       = 0.0
    best_ari_params    = best_ari_model_state = best_ari_epoch = best_ari_combination = None

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if not use_confidence and not use_segment:
        feature_suffix = "_spatial_only"
    elif not use_confidence:
        feature_suffix = "_no_confidence"
    elif not use_segment:
        feature_suffix = "_no_segment"
    else:
        feature_suffix = ""

    json_results_file  = f"pt/grid_search_complete_{timestamp}{feature_suffix}.json"
    best_ari_model_file = f"pt/rock_model_best_ari_{timestamp}{feature_suffix}.pth"

    grid_search_results: dict = {
        "timestamp": timestamp,
        "search_config": {
            "total_combinations":    total_combinations,
            "base_combinations":     len(base_combinations),
            "all_validation_scenes": scene_names,
            "param_grids":           param_grids,
            "fixed_params":          fixed_params,
        },
        "all_results":   [],
        "best_loss_result": None,
        "best_ari_result":  None,
        "summary_stats": {},
        "csv_data": {"header": [], "rows": []},
    }

    header = (
        base_param_names
        + ["prototypical_weight", "discriminative_weight", "graph_based_weight"]
        + ["avg_loss"]
        + [f"{name}_ari" for name in scene_names]
        + ["avg_ari", "min_ari", "max_ari", "training_time_seconds"]
    )
    grid_search_results["csv_data"]["header"] = list(header)

    start_time      = time.time()
    combination_idx = 0

    for base_combination in base_combinations:
        for loss_weight_combination in loss_weight_combinations:
            combination_idx += 1

            params = dict(zip(base_param_names, base_combination))

            # Flatten lr_scheduler_gamma → lr_scheduler_params dict for trainer.
            if "lr_scheduler_gamma" in params:
                params["lr_scheduler_params"] = {"gamma": params.pop("lr_scheduler_gamma")}

            params.update(fixed_params)

            lw = loss_weight_combination
            if len(lw) == 4:
                if lw[0] == 0.0:
                    _, prototypical_w, discriminative_w, graph_based_w = lw
                else:
                    prototypical_w, discriminative_w, graph_based_w, _ = lw
            else:
                prototypical_w, discriminative_w, graph_based_w = lw[0], lw[1], lw[2]
            params["loss_weights"] = [prototypical_w, discriminative_w, graph_based_w]

            run_ari_per_epoch = params.get("run_ari_per_epoch", False)

            logger.info(
                "[%3d/%d] lr=%.0e bw=%.2f batch=%d loss=[P%.1f,D%.1f,G%.1f]",
                combination_idx, total_combinations,
                params["learning_rate"], params["bandwidth"], params.get("batch_size", 1),
                prototypical_w, discriminative_w, graph_based_w,
            )

            checkpoint_dir = os.path.join("pt", "checkpoints", f"combo_{combination_idx:04d}")
            log_file       = os.path.join("pt", "logs", f"training_log_combo_{combination_idx:04d}.csv")
            os.makedirs(os.path.dirname(log_file), exist_ok=True)

            combo_start     = time.time()
            training_result = train_model_focused(
                params, train_loader, valid_loader, device, eval_files, scene_names,
                checkpoint_dir=checkpoint_dir, log_file=log_file,
            )
            training_time = time.time() - combo_start

            if not training_result["success"]:
                logger.warning("Training failed for combination %d", combination_idx)
                continue

            model                 = training_result["model"]
            avg_loss              = training_result["avg_loss"]
            best_valid_loss_combo = training_result["best_valid_loss"]
            best_model_state_combo = training_result["best_model_state"]
            best_epoch_combo      = training_result["best_epoch"]
            delta_params          = training_result["delta_params"]

            if not run_ari_per_epoch and training_result.get("best_model_state") is not None:
                model.load_state_dict(training_result["best_model_state"])

            model.eval()
            all_aris: list[float] = []
            all_results: list[str] = []
            for idx, scene_path in tqdm(
                enumerate(eval_files), total=len(eval_files),
                desc="Final Eval", leave=False, ncols=100,
            ):
                eval_result = evaluate_model_on_scene_focused(
                    model, scene_path, params["bandwidth"], device,
                    use_confidence=params.get("use_confidence", True),
                    use_segment=params.get("use_segment", True),
                )
                if eval_result["success"]:
                    all_aris.append(eval_result["ari_score"])
                    all_results.append(f"{scene_names[idx]}: {eval_result['ari_score']:.3f}")
                else:
                    all_aris.append(0.0)
                    all_results.append(f"{scene_names[idx]}: FAIL")

            if not all_aris:
                logger.warning("No valid evaluations for combination %d", combination_idx)
                if should_stop():
                    break
                continue

            avg_ari = float(np.mean(all_aris))
            min_ari = float(np.min(all_aris))
            max_ari = float(np.max(all_aris))
            logger.info(
                "  Loss: %.3f | ARI: %.3f | Min: %.3f | Max: %.3f | Time: %.1fs",
                avg_loss, avg_ari, min_ari, max_ari, training_time,
            )

            is_best_loss = best_valid_loss_combo < best_avg_loss
            if is_best_loss and save_best_loss:
                best_avg_loss        = best_valid_loss_combo
                best_loss_params     = {**params, "delta_params": delta_params}
                best_loss_model_state = {k: v.clone() for k, v in best_model_state_combo.items()}
                best_loss_epoch      = best_epoch_combo
                best_loss_combination = combination_idx
                logger.info(
                    "  NEW BEST LOSS: %.3f at epoch %d", best_valid_loss_combo, best_epoch_combo
                )

            best_epoch_ari = training_result.get("best_ari", avg_ari) if run_ari_per_epoch else avg_ari
            is_best_ari    = best_epoch_ari > best_avg_ari
            if is_best_ari:
                best_avg_ari    = best_epoch_ari
                best_ari_params = {**params, "delta_params": delta_params}
                if training_result.get("best_ari_model_state") is not None:
                    best_ari_model_state = {
                        k: v.clone() for k, v in training_result["best_ari_model_state"].items()
                    }
                    best_ari_epoch = training_result["best_ari_epoch"]
                else:
                    best_ari_model_state = {k: v.clone() for k, v in model.state_dict().items()}
                    best_ari_epoch       = params["num_epochs"]
                best_ari_combination = combination_idx
                try:
                    torch.save(best_ari_model_state, best_ari_model_file)
                    logger.info(
                        "  NEW BEST ARI: %.3f — saved to %s", best_avg_ari, best_ari_model_file
                    )
                except Exception as exc:
                    logger.error("  Error saving ARI model: %s", exc)

            row = (
                list(base_combination)
                + [float(prototypical_w), float(discriminative_w), float(graph_based_w)]
                + [float(avg_loss)]
                + [float(a) for a in all_aris]
                + [avg_ari, min_ari, max_ari, float(training_time)]
            )
            grid_search_results["csv_data"]["rows"].append(list(row))
            grid_search_results["all_results"].append({
                "combination_id": combination_idx,
                "parameters":     dict(zip(base_param_names, base_combination)),
                "delta_params":   {k: float(v) for k, v in delta_params.items()},
                "loss_weights": {
                    "prototypical":  float(prototypical_w),
                    "discriminative": float(discriminative_w),
                    "graph_based":   float(graph_based_w),
                },
                "avg_loss":          float(avg_loss),
                "all_scene_aris":    dict(zip(scene_names, [float(a) for a in all_aris])),
                "avg_ari":           avg_ari,
                "min_ari":           min_ari,
                "max_ari":           max_ari,
                "training_time_seconds": float(training_time),
                "is_best_loss":      bool(is_best_loss),
                "is_best_ari":       bool(is_best_ari),
                "epoch_loss_components": training_result.get("epoch_loss_components", []),
            })

            if should_stop():
                logger.info("Grid search stopped early after %d combinations", combination_idx)
                break

        if should_stop():
            break

    # ── Summary & persistence ────────────────────────────────────────────────
    elapsed = time.time() - start_time
    if grid_search_results["all_results"]:
        ari_summary  = [r["avg_ari"]  for r in grid_search_results["all_results"]]
        loss_summary = [r["avg_loss"] for r in grid_search_results["all_results"]]
        time_summary = [r["training_time_seconds"] for r in grid_search_results["all_results"]]
        grid_search_results["summary_stats"] = {
            "total_runtime_minutes":             elapsed / 60,
            "combinations_tested":               len(grid_search_results["all_results"]),
            "early_stopped":                     should_stop(),
            "avg_ari_overall":                   float(np.mean(ari_summary)),
            "best_ari_achieved":                 float(max(ari_summary)),
            "avg_loss_overall":                  float(np.mean(loss_summary)),
            "best_loss_achieved":                float(min(loss_summary)),
            "avg_training_time_per_combination": float(np.mean(time_summary)),
        }
        if best_loss_params:
            grid_search_results["best_loss_result"] = {
                "avg_loss":   float(best_avg_loss),
                "parameters": best_loss_params,
                "epoch":      int(best_loss_epoch),
                "combination": int(best_loss_combination),
                "model_file": f"pt/rock_model_best_loss_{timestamp}{feature_suffix}.pth",
            }
        if best_ari_params:
            grid_search_results["best_ari_result"] = {
                "avg_ari":    float(best_avg_ari),
                "parameters": best_ari_params,
                "epoch":      int(best_ari_epoch),
                "combination": int(best_ari_combination),
                "model_file": best_ari_model_file,
            }

    try:
        os.makedirs("pt", exist_ok=True)
        with open(json_results_file, "w") as fh:
            json.dump(grid_search_results, fh, indent=2)
        logger.info("Grid search complete — results: %s", json_results_file)
        logger.info("Total time: %.1f minutes", elapsed / 60)
    except Exception as exc:
        logger.error("Error saving JSON results: %s", exc)

    if save_best_loss and best_loss_params and best_loss_model_state is not None:
        best_loss_model_file = f"pt/rock_model_best_loss_{timestamp}{feature_suffix}.pth"
        try:
            torch.save(best_loss_model_state, best_loss_model_file)
            logger.info("Best loss model saved: %s (loss=%.3f)", best_loss_model_file, best_avg_loss)
        except Exception as exc:
            logger.error("Error saving best loss model: %s", exc)

    if best_ari_params:
        logger.info("Best ARI model: %s (ARI=%.3f)", best_ari_model_file, best_avg_ari)

    return {"best_loss_params": best_loss_params, "best_ari_params": best_ari_params}


def main() -> None:
    """Stand-alone CLI shim — delegates to the proper Hydra entry point."""
    import sys
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logging.getLogger(__name__).error(
        "Direct invocation is no longer supported. "
        "Use:  shadowcorr-train data.train_dir=... data.valid_dir=..."
    )
    sys.exit(1)


if __name__ == "__main__":
    main()
