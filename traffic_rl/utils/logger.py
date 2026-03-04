"""
logger.py – Experiment logging to TensorBoard and structured JSON files.

Features
--------
* TensorBoard integration (scalar, histogram, image logging).
* JSON-line flat log for offline analysis.
* Automatic directory creation with timestamp-based run names.
* Reproducibility snapshot (saves config + git hash if available).
"""

from __future__ import annotations

import json
import os
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

try:
    from torch.utils.tensorboard import SummaryWriter
    HAS_TB = True
except ImportError:
    HAS_TB = False


class ExperimentLogger:
    """Structured logger for RL training runs.

    Parameters
    ----------
    log_dir:
        Root directory for TensorBoard summaries.
    run_name:
        Sub-folder name.  Defaults to a timestamp string.
    config_dict:
        Experiment configuration dict; saved as ``config.json`` at startup.
    use_tensorboard:
        Whether to write TensorBoard events.
    """

    def __init__(
        self,
        log_dir: str = "runs",
        run_name: Optional[str] = None,
        config_dict: Optional[Dict[str, Any]] = None,
        use_tensorboard: bool = True,
    ) -> None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_name = run_name or f"run_{stamp}"
        self.run_dir = os.path.join(log_dir, self.run_name)
        os.makedirs(self.run_dir, exist_ok=True)

        # TensorBoard writer
        self.writer: Optional[Any] = None
        if use_tensorboard and HAS_TB:
            self.writer = SummaryWriter(log_dir=self.run_dir)

        # CSV / JSON log path
        self.jsonl_path = os.path.join(self.run_dir, "metrics.jsonl")
        self._jsonl_file = open(self.jsonl_path, "a", encoding="utf-8")

        # Save config
        if config_dict is not None:
            cfg_path = os.path.join(self.run_dir, "config.json")
            with open(cfg_path, "w") as f:
                json.dump(config_dict, f, indent=2, default=str)

        self._step: int = 0
        self._episode: int = 0
        self._start_time = time.time()

        print(f"[Logger] Run directory: {self.run_dir}")

    # ------------------------------------------------------------------
    # Core logging API
    # ------------------------------------------------------------------

    def log_scalar(self, tag: str, value: float, step: Optional[int] = None) -> None:
        """Log a scalar metric.

        Parameters
        ----------
        tag:    Metric name (e.g. ``"train/episode_reward"``).
        value:  Numeric value.
        step:   Global step; uses internal counter if omitted.
        """
        step = step if step is not None else self._step
        if self.writer is not None:
            self.writer.add_scalar(tag, value, global_step=step)
        self._write_jsonl({"type": "scalar", "tag": tag, "value": value, "step": step})

    def log_scalars(self, tag_prefix: str, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        """Log multiple scalars at once.

        Parameters
        ----------
        tag_prefix:
            Prefix added to each key, e.g. ``"train"``.
        metrics:
            Dict of ``{name: value}``.
        """
        step = step if step is not None else self._step
        for name, value in metrics.items():
            full_tag = f"{tag_prefix}/{name}" if tag_prefix else name
            self.log_scalar(full_tag, float(value), step=step)

    def log_episode(self, metrics: Dict[str, Any], episode: Optional[int] = None) -> None:
        """Log end-of-episode metrics.

        Parameters
        ----------
        metrics:
            Dict of metric names to values.
        episode:
            Episode number; uses internal counter if omitted.
        """
        ep = episode if episode is not None else self._episode
        self._episode += 1
        if self.writer is not None:
            for k, v in metrics.items():
                if isinstance(v, (int, float)):
                    self.writer.add_scalar(f"episode/{k}", float(v), global_step=ep)
        metrics["_episode"] = ep
        metrics["_wall_time"] = time.time() - self._start_time
        self._write_jsonl({"type": "episode", **metrics})

    def log_training(self, update_metrics: Dict[str, float], step: Optional[int] = None) -> None:
        """Log one set of training-update metrics."""
        step = step if step is not None else self._step
        self.log_scalars("train", update_metrics, step=step)

    def set_step(self, step: int) -> None:
        """Manually update the internal global step counter."""
        self._step = step

    def increment_step(self, n: int = 1) -> None:
        """Advance the internal step counter by *n*."""
        self._step += n

    # ------------------------------------------------------------------
    # Histograms (actor/critic weight distributions)
    # ------------------------------------------------------------------

    def log_histogram(self, tag: str, values, step: Optional[int] = None) -> None:
        """Log a histogram of tensor values (TensorBoard only)."""
        if self.writer is None:
            return
        step = step if step is not None else self._step
        try:
            self.writer.add_histogram(tag, values, global_step=step)
        except Exception:
            pass  # ignore histogram errors in edge cases

    # ------------------------------------------------------------------
    # Text / custom logging
    # ------------------------------------------------------------------

    def log_text(self, tag: str, text: str, step: Optional[int] = None) -> None:
        """Log a text annotation (TensorBoard and JSONL)."""
        step = step if step is not None else self._step
        if self.writer is not None:
            self.writer.add_text(tag, text, global_step=step)
        self._write_jsonl({"type": "text", "tag": tag, "text": text, "step": step})

    def log_image(self, tag: str, img_array, step: Optional[int] = None) -> None:
        """Log an RGB image as a numpy array (H, W, 3) uint8."""
        if self.writer is None:
            return
        import numpy as np
        step = step if step is not None else self._step
        img = np.transpose(img_array, (2, 0, 1))  # CHW
        self.writer.add_image(tag, img, global_step=step)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _write_jsonl(self, record: Dict[str, Any]) -> None:
        try:
            self._jsonl_file.write(json.dumps(record, default=str) + "\n")
            self._jsonl_file.flush()
        except Exception:
            pass

    def print_summary(self, metrics: Dict[str, float], step: int) -> None:
        """Pretty-print a summary line to stdout."""
        elapsed = time.time() - self._start_time
        parts = [f"step={step:>8,}", f"elapsed={elapsed:6.0f}s"]
        for k, v in metrics.items():
            parts.append(f"{k}={v:.4f}")
        print("  ".join(parts))

    def close(self) -> None:
        """Flush and close all file handles."""
        if self.writer is not None:
            self.writer.close()
        self._jsonl_file.close()

    def __enter__(self) -> "ExperimentLogger":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()
