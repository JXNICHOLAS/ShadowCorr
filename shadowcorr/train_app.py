"""
Hydra entry point for grid-search training.

Usage::

  shadowcorr-train data.train_dir=/path/to/train data.valid_dir=/path/to/valid

Or::

  python -m shadowcorr.train_app data.train_dir=... data.valid_dir=...

All hyperparameters live in ``shadowcorr/conf/train.yaml`` and can be
overridden on the CLI, e.g.::

  shadowcorr-train training.num_epochs=50 grid.learning_rate=[1e-4,5e-5]
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig

from shadowcorr.logging_utils import close_run_logging, log_metrics_to_tensorboard, setup_run_logging
from shadowcorr.pipeline.sweep import run_focused_grid_search

_CONFIG_DIR = (Path(__file__).resolve().parent / "conf").as_posix()


def _flatten_for_tb(obj: Any, prefix: str = "") -> dict[str, float]:
    """Best-effort flatten nested dicts to scalar floats for TensorBoard."""
    out: dict[str, float] = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            key = f"{prefix}/{k}" if prefix else str(k)
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                out[key.replace(".", "/")] = float(v)
            elif isinstance(v, dict):
                out.update(_flatten_for_tb(v, key))
    return out


@hydra.main(version_base=None, config_path=_CONFIG_DIR, config_name="train")
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
        result = run_focused_grid_search(cfg, logger)
        logger.info("run_focused_grid_search finished.")

        if isinstance(result, dict) and writer is not None:
            flat = _flatten_for_tb(result, "train")
            log_metrics_to_tensorboard(writer, flat, 0)
            if flat:
                logger.info("Logged %d scalar groups to TensorBoard (step 0).", len(flat))
    finally:
        close_run_logging(logger, writer)


if __name__ == "__main__":
    main()
