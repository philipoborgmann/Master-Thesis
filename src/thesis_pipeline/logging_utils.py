"""Simple stage-level logging.

Each stage prints a timestamped header summarising:
  - mode (full / smoke / dry-run)
  - input file(s)
  - output file(s)
  - row counts in/out (when callable knows them)
  - whether existing outputs were overwritten or preserved

An optional log file under ``Outputs/diagnostics/logs/`` is opened lazily.
"""

from __future__ import annotations

import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Iterable

from .config import resolve_path

_LOGGER_NAME = "thesis_pipeline"
_DEFAULT_FORMAT = "[%(levelname)s] %(asctime)s %(name)s :: %(message)s"


def get_logger(name: str = _LOGGER_NAME) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(logging.Formatter(_DEFAULT_FORMAT, datefmt="%Y-%m-%d %H:%M:%S"))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False
    return logger


def attach_file_handler(stage: str) -> Path | None:
    """Attach a per-run log file under Outputs/diagnostics/logs/."""
    try:
        log_dir = resolve_path("diagnostics_logs")
        log_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = log_dir / f"{stage}_{ts}.log"
        fh = logging.FileHandler(path, encoding="utf-8")
        fh.setFormatter(logging.Formatter(_DEFAULT_FORMAT, datefmt="%Y-%m-%d %H:%M:%S"))
        get_logger().addHandler(fh)
        return path
    except Exception:  # noqa: BLE001 — best-effort logging only
        return None


def log_stage_header(stage: str, *, mode: str,
                     inputs: Iterable[Path | str] = (),
                     outputs: Iterable[Path | str] = (),
                     extra: dict | None = None) -> None:
    logger = get_logger()
    logger.info("=" * 72)
    logger.info("STAGE: %s   mode=%s", stage, mode)
    for p in inputs:
        logger.info("  input  : %s%s", p, "" if Path(str(p)).exists() else "   [MISSING]")
    for p in outputs:
        exists = Path(str(p)).exists()
        marker = "exists (will overwrite)" if exists else "new"
        logger.info("  output : %s   [%s]", p, marker)
    if extra:
        for k, v in extra.items():
            logger.info("  %s: %s", k, v)
    logger.info("=" * 72)


def log_rows(label: str, n: int) -> None:
    get_logger().info("  %s: %d rows", label, n)
