"""Stationarity testing — wraps Sentiment_Stationarity_Test.py."""

from __future__ import annotations

from typing import Sequence

from ..config import resolve_path, smoke_coins
from ..logging_utils import get_logger, log_stage_header
from ..utils import run_script_main


def run(*, horizon: str | None = None,
        coins: Sequence[str] | None = None,
        smoke: bool = False, no_plots: bool = False,
        dry_run: bool = False) -> int:
    if smoke and not coins:
        coins = smoke_coins()
    if smoke and not horizon:
        horizon = "1d"

    inputs  = [resolve_path("features_root")]
    outputs = [
        resolve_path("stationarity_records"),
        resolve_path("stationarity_results"),
        resolve_path("stationarity_plots"),
    ]

    log_stage_header(
        "stationarity",
        mode="dry-run" if dry_run else ("smoke" if smoke else "full"),
        inputs=inputs,
        outputs=outputs,
        extra={
            "horizon": horizon or "(all)",
            "coins":   list(coins) if coins else "(all)",
            "no_plots": no_plots or smoke,
        },
    )

    if dry_run:
        get_logger().info("dry-run: not invoking Sentiment_Stationarity_Test.py")
        return 0

    argv: list[str] = ["--feature_dir", str(resolve_path("features_root"))]
    if horizon:
        argv += ["--horizon", horizon]
    if coins:
        for c in coins:
            argv += ["--ticker", c]
    if no_plots or smoke:
        argv += ["--no_plots"]
    return run_script_main("stationarity", argv)
