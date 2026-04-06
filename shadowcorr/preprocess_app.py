"""Hydra entry: NPZ voxel preprocessing (batch scene encoder + slider pipeline)."""

from __future__ import annotations

import random
from pathlib import Path
from typing import List, Optional

import hydra
from hydra.core.hydra_config import HydraConfig
from hydra.utils import get_original_cwd
from omegaconf import DictConfig, OmegaConf

import shadowcorr
from shadowcorr.preprocess.scene import batch_process, load_single_label_module
from shadowcorr.logging_utils import close_run_logging, setup_run_logging

_CONFIG_DIR = (Path(__file__).resolve().parent / "conf").as_posix()


def _resolve_path(p: str) -> Path:
    path = Path(p).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (Path(get_original_cwd()) / path).resolve()


def _run_split(
    split_name: str,
    npz_files: List[Path],
    in_dir: Path,
    out_dir: Path,
    mod,
    cfg: DictConfig,
    logger,
    limit_n: Optional[int],
) -> None:
    split_out = out_dir / split_name
    logger.info("  %s: %d scenes → %s", split_name, len(npz_files), split_out)
    batch_process(
        input_dir=in_dir,
        output_dir=split_out,
        module=mod,
        embed_dim=int(cfg.embed_dim),
        embed_seed=int(cfg.embed_seed),
        score_threshold_percentile=float(cfg.score_threshold),
        overwrite=bool(cfg.overwrite),
        train_encoder_steps=int(cfg.train_encoder_steps),
        train_encoder_lr=float(cfg.train_encoder_lr),
        train_encoder_temperature=float(cfg.train_encoder_temperature),
        limit=limit_n,
        npz_files=npz_files,
    )


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

        # Patch voxel globals so preprocess and eval use the same geometry parameters.
        mod.VOXEL_SIZE     = float(cfg.voxel_size)
        mod.expansion_rate = float(cfg.expansion_rate)
        logger.info("Voxel size: %s  expansion rate: %s", cfg.voxel_size, cfg.expansion_rate)

        in_dir  = _resolve_path(str(cfg.data.input_dir))
        out_dir = _resolve_path(str(cfg.data.output_dir))
        logger.info("Input directory:  %s", in_dir)
        logger.info("Output directory: %s", out_dir)

        limit_n = None if cfg.limit is None else int(cfg.limit)
        split   = OmegaConf.select(cfg, "data.split")   # None or [train_r, valid_r, test_r]

        if split is not None:
            # Split segment NPZs into train/valid/test, preprocess each separately.
            ratios = list(split)
            if len(ratios) != 3 or abs(sum(ratios) - 1.0) > 1e-6:
                raise ValueError(
                    f"data.split must be a list of 3 ratios summing to 1.0, got {ratios}"
                )
            seed = int(OmegaConf.select(cfg, "data.split_seed") or 42)
            all_files = sorted(in_dir.glob("*.npz"))
            if not all_files:
                logger.warning("No NPZ files found in %s", in_dir)
                return

            shuffled = list(all_files)
            random.Random(seed).shuffle(shuffled)
            n = len(shuffled)
            n_train = round(n * ratios[0])
            n_valid = round(n * ratios[1])
            train_files = shuffled[:n_train]
            valid_files = shuffled[n_train:n_train + n_valid]
            test_files  = shuffled[n_train + n_valid:]

            logger.info(
                "Split (seed=%d): %d train | %d valid | %d test",
                seed, len(train_files), len(valid_files), len(test_files),
            )
            _run_split("train", train_files, in_dir, out_dir, mod, cfg, logger, limit_n)
            _run_split("valid", valid_files, in_dir, out_dir, mod, cfg, logger, limit_n)
            _run_split("test",  test_files,  in_dir, out_dir, mod, cfg, logger, limit_n)
        else:
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
