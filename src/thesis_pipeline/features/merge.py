"""Merge price + sentiment features — wraps Merge_Features.py.

Produces ``Data/Final/features_<horizon>.parquet`` and
``Data/Final/merge_report.csv``. Coverage threshold, fill values, and join
semantics are preserved from the original script.
"""

from __future__ import annotations

from ..config import load_config, resolve_path
from ..logging_utils import get_logger, log_stage_header
from ..utils import run_script_main


def run(*, horizon: str | None = None, smoke: bool = False,
        dry_run: bool = False) -> int:
    horizons_to_emit = [horizon] if horizon else load_config("horizons")["final_model_horizons"]

    inputs = []
    for h in horizons_to_emit:
        inputs.append(resolve_path("price_features_pattern", horizon=h))
        inputs.append(resolve_path("sentiment_features_pattern", horizon=h))
    inputs.append(resolve_path("sentiment_coverage"))

    outputs = [resolve_path("final_features_pattern", horizon=h) for h in horizons_to_emit]
    outputs.append(resolve_path("merge_report"))

    log_stage_header(
        "merge_features",
        mode="dry-run" if dry_run else ("smoke" if smoke else "full"),
        inputs=inputs,
        outputs=outputs,
        extra={"horizon": horizon or "(all)"},
    )

    if dry_run:
        get_logger().info("dry-run: not invoking Merge_Features.py")
        return 0

    # Merge_Features.py reads from hardcoded directories. We don't override
    # those (it would risk drift); we just call it. TODO(refactor_log): when
    # we move logic into this module, accept --horizon / --feature_dir /
    # --output_dir flags directly.
    return run_script_main("merge_features", [])
