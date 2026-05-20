"""Modeling stage — wraps Run_Models.py.

Walk-forward, logistic-ridge, StandardScaler-on-train-only, and benchmark
behaviour are all preserved. See ``configs/model_specs.yaml`` for the
exact spec extracted from Run_Models.py.
"""

from __future__ import annotations

from typing import Sequence

from ..config import resolve_path, smoke_coins
from ..logging_utils import get_logger, log_stage_header
from ..utils import run_script_main


def run(*, horizon: str | None = None, set_id: str | None = None,
        coins: Sequence[str] | None = None,
        sentiment_model: str | None = None,
        smoke: bool = False, dry_run: bool = False) -> int:
    if smoke and not coins:
        coins = smoke_coins()
    if smoke and not horizon:
        horizon = "1d"
    if smoke and not set_id:
        set_id = "B1"

    inputs = []
    if horizon:
        inputs.append(resolve_path("final_features_pattern", horizon=horizon))
    inputs.append(resolve_path("feature_sets_xlsx"))

    outputs: list = []
    if horizon and set_id:
        outputs.append(resolve_path("signals_pattern", horizon=horizon, set_id=set_id))
    outputs.append(resolve_path("signals_metrics"))

    log_stage_header(
        "run_models",
        mode="dry-run" if dry_run else ("smoke" if smoke else "full"),
        inputs=inputs,
        outputs=outputs,
        extra={
            "horizon": horizon or "(all)",
            "set_id": set_id or "(all)",
            "coins": list(coins) if coins else "(all)",
            "sentiment_model": sentiment_model or "(per set_id default)",
        },
    )

    if dry_run:
        get_logger().info("dry-run: not invoking Run_Models.py")
        return 0

    argv: list[str] = []
    if horizon:
        argv += ["--horizon", horizon]
    if set_id:
        argv += ["--set_id", set_id]
    if sentiment_model:
        argv += ["--sentiment_model", sentiment_model]
    if coins:
        for c in coins:
            argv += ["--coin", c]

    return run_script_main("run_models", argv)
