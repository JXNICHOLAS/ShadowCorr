"""
Hydra entry for segment evaluation.

Usage::

  shadowcorr eval input_dir=/path/to/npz_dir model_path=/path/to/checkpoint.pth

Single NPZ::

  shadowcorr eval input_dir=/path/to/scene.npz model_path=/path/to/checkpoint.pth
"""

from __future__ import annotations

import statistics
from pathlib import Path

import hydra
import torch
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig

from shadowcorr.logging_utils import close_run_logging, log_metrics_to_tensorboard, setup_run_logging
import shadowcorr.pipeline.evaluator as ev

_CONFIG_DIR = (Path(__file__).resolve().parent / "conf").as_posix()


@hydra.main(version_base=None, config_path=_CONFIG_DIR, config_name="eval")
def main(cfg: DictConfig) -> None:
    run_dir = Path(HydraConfig.get().runtime.output_dir)
    logger, writer = setup_run_logging(
        run_dir,
        cfg,
        log_filename=cfg.logging.log_filename,
        level=cfg.logging.level,
        tensorboard=cfg.logging.tensorboard,
    )
    try:
        input_path = Path(cfg.input_dir)
        output_dir = Path(cfg.output_dir)
        if not input_path.exists():
            raise FileNotFoundError(f"input_dir not found: {input_path}")

        # Resolve checkpoint path.
        model_path = Path(cfg.model_path) if cfg.get("model_path") else None
        if model_path is None:
            raise ValueError("Provide model_path=/path/to/checkpoint.pth on the CLI.")
        if not model_path.exists():
            raise FileNotFoundError(f"model_path not found: {model_path}")

        if input_path.is_file():
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model, params = ev.load_model_once(device, model_path=model_path)
            metrics, _ = ev.process_single_file(
                npz_path=input_path,
                model=model,
                params=params,
                bandwidth=float(cfg.bandwidth),
                output_dir=output_dir,
                embed_dim=int(cfg.embed_dim),
                embed_seed=int(cfg.embed_seed),
                score_threshold_percentile=float(cfg.score_threshold_percentile),
                voxel_size=float(cfg.voxel_size),
                expansion_rate=float(cfg.expansion_rate),
                train_encoder_steps=int(cfg.train_encoder_steps),
                train_encoder_lr=float(cfg.train_encoder_lr),
                train_encoder_temperature=float(cfg.train_encoder_temperature),
                device=device,
            )
            log_metrics_to_tensorboard(
                writer,
                {
                    "eval/voxel_ari": float(metrics.get("voxel_ari", 0.0)),
                    "eval/segment_ari": float(metrics.get("segment_ari", 0.0)),
                    "eval/avg_rock_purity": float(metrics.get("avg_rock_purity", 0.0)),
                    "eval/avg_cluster_purity": float(metrics.get("avg_cluster_purity", 0.0)),
                    "eval/processing_time_seconds": float(metrics.get("processing_time_seconds", 0.0)),
                },
                0,
            )
            logger.info("Single-file eval done. voxel_ari=%.4f", metrics.get("voxel_ari", 0.0))
        else:
            results, _ = ev.process_batch(
                input_dir=input_path,
                model_path=model_path,
                bandwidth=float(cfg.bandwidth),
                output_dir=output_dir,
                embed_dim=int(cfg.embed_dim),
                embed_seed=int(cfg.embed_seed),
                score_threshold_percentile=float(cfg.score_threshold_percentile),
                voxel_size=float(cfg.voxel_size),
                expansion_rate=float(cfg.expansion_rate),
                train_encoder_steps=int(cfg.train_encoder_steps),
                train_encoder_lr=float(cfg.train_encoder_lr),
                train_encoder_temperature=float(cfg.train_encoder_temperature),
                max_files=int(cfg.max_files),
            )
            ok = [r for r in results if r.get("success")]
            if ok:
                vox = [float(r["metrics"]["voxel_ari"]) for r in ok]
                seg = [float(r["metrics"]["segment_ari"]) for r in ok]
                rp = [float(r["metrics"]["avg_rock_purity"]) for r in ok]
                cp = [float(r["metrics"]["avg_cluster_purity"]) for r in ok]
                for i, r in enumerate(ok):
                    m = r["metrics"]
                    log_metrics_to_tensorboard(
                        writer,
                        {
                            "eval/per_scene/voxel_ari": float(m["voxel_ari"]),
                            "eval/per_scene/segment_ari": float(m["segment_ari"]),
                        },
                        i,
                    )
                log_metrics_to_tensorboard(
                    writer,
                    {
                        "eval/mean/voxel_ari": float(statistics.mean(vox)),
                        "eval/mean/segment_ari": float(statistics.mean(seg)),
                        "eval/mean/rock_purity": float(statistics.mean(rp)),
                        "eval/mean/cluster_purity": float(statistics.mean(cp)),
                        "eval/num_scenes": float(len(ok)),
                    },
                    len(ok),
                )
                logger.info(
                    "Batch eval: %d scenes | mean voxel ARI %.4f | mean seg ARI %.4f",
                    len(ok),
                    statistics.mean(vox),
                    statistics.mean(seg),
                )
            else:
                logger.warning("No successful evaluations in batch.")

    finally:
        close_run_logging(logger, writer)


if __name__ == "__main__":
    main()
