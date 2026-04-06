"""Hydra entry: NPZ voxel preprocessing (batch scene encoder + slider pipeline)."""

from __future__ import annotations

from pathlib import Path

import hydra
from hydra.core.hydra_config import HydraConfig
from hydra.utils import get_original_cwd
from omegaconf import DictConfig

import shadowcorr
from shadowcorr.preprocess.scene import batch_process, load_single_label_module
from shadowcorr.logging_utils import close_run_logging, setup_run_logging

_CONFIG_DIR = (Path(__file__).resolve().parent / "conf").as_posix()


def _resolve_path(p: str) -> Path:
    path = Path(p).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (Path(get_original_cwd()) / path).resolve()


@hydra.main(version_base=None, config_path=_CONFIG_DIR, config_name="preprocess")
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
        pkg = Path(shadowcorr.__file__).resolve().parent
        if cfg.single_script is None:
            single_path = (pkg / "preprocess" / "voxel.py").resolve()
        else:
            single_path = _resolve_path(str(cfg.single_script))

        logger.info("Single-label script: %s", single_path)
        mod = load_single_label_module(single_path)

        in_dir = _resolve_path(str(cfg.data.input_dir))
        out_dir = _resolve_path(str(cfg.data.output_dir))
        logger.info("Input directory:  %s", in_dir)
        logger.info("Output directory: %s", out_dir)

        limit_n = None if cfg.limit is None else int(cfg.limit)

        batch_process(
            in_dir,
            out_dir,
            mod,
            embed_dim=int(cfg.embed_dim),
            embed_seed=int(cfg.embed_seed),
            score_threshold_percentile=float(cfg.score_threshold),
            overwrite=bool(cfg.overwrite),
            train_encoder_steps=int(cfg.train_encoder_steps),
            train_encoder_lr=float(cfg.train_encoder_lr),
            train_encoder_temperature=float(cfg.train_encoder_temperature),
            limit=limit_n,
        )
        logger.info("Preprocess batch finished.")
    finally:
        close_run_logging(logger, writer)


if __name__ == "__main__":
    main()
