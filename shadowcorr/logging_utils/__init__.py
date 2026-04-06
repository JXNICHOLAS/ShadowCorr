import logging
import sys
from pathlib import Path
from typing import Any, Optional, TextIO

from omegaconf import DictConfig, OmegaConf

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    SummaryWriter = None  # type: ignore[misc, assignment]


class _Tee(TextIO):
    def __init__(self, *streams: TextIO):
        self._streams = streams

    def write(self, s: str) -> int:
        for st in self._streams:
            st.write(s)
            st.flush()
        return len(s)

    def flush(self) -> None:
        for st in self._streams:
            st.flush()


_stdout_tee_file: Optional[Any] = None


def setup_run_logging(
    run_dir: Path,
    cfg: DictConfig,
    *,
    log_filename: str = "shadowcorr.log",
    level: str = "INFO",
    tensorboard: bool = True,
) -> tuple[logging.Logger, Optional[Any]]:
    global _stdout_tee_file
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / log_filename

    try:
        (run_dir / "config_resolved.yaml").write_text(OmegaConf.to_yaml(cfg), encoding="utf-8")
    except Exception as e:  # pragma: no cover
        print(f"Warning: could not write config_resolved.yaml: {e}", file=sys.stderr)

    _stdout_tee_file = open(log_path, "a", encoding="utf-8", buffering=1)
    sys.stdout = _Tee(sys.__stdout__, _stdout_tee_file)  # type: ignore[assignment]

    logger = logging.getLogger("shadowcorr")
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    logger.handlers.clear()
    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    writer = None
    if tensorboard and SummaryWriter is not None:
        tb_dir = run_dir / "tensorboard"
        tb_dir.mkdir(parents=True, exist_ok=True)
        writer = SummaryWriter(log_dir=str(tb_dir))
        writer.add_text("config/hydra_yaml", f"```yaml\n{OmegaConf.to_yaml(cfg)}\n```", 0)
        logger.info("TensorBoard: tensorboard --logdir %s", tb_dir.resolve())
    elif tensorboard:
        logger.warning("TensorBoard requested but torch.utils.tensorboard is not available")

    logger.info("Run directory: %s", run_dir.resolve())
    logger.info("Log file: %s", log_path.resolve())
    return logger, writer


def close_run_logging(logger: logging.Logger, writer: Optional[Any]) -> None:
    global _stdout_tee_file
    for h in list(logger.handlers):
        h.close()
        logger.removeHandler(h)
    if writer is not None:
        try:
            writer.flush()
            writer.close()
        except Exception:
            pass
    if isinstance(sys.stdout, _Tee):
        sys.stdout = sys.__stdout__
    if _stdout_tee_file is not None:
        try:
            _stdout_tee_file.flush()
            _stdout_tee_file.close()
        except Exception:
            pass
        _stdout_tee_file = None


def log_metrics_to_tensorboard(writer: Optional[Any], scalars: dict[str, float], step: int) -> None:
    if writer is None:
        return
    for k, v in scalars.items():
        if isinstance(v, (int, float)) and v == v:
            try:
                writer.add_scalar(k, float(v), step)
            except Exception:
                pass
