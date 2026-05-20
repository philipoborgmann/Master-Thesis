"""Sentiment aggregation — wraps Sentiment_feature_engineering.py.

Produces per-horizon sentiment features under
``Data/Features/sentiment_features_<horizon>.parquet`` for the horizons
listed in ``configs/horizons.yaml`` (``sentiment_feature_horizons``).

Engagement weighting and post-level title/selftext combination formulas are
unchanged.
"""

from __future__ import annotations

from ..config import load_config, resolve_path
from ..logging_utils import get_logger, log_stage_header
from ..utils import run_script_main


def run(*, horizon: str | None = None, smoke: bool = False,
        max_rows: int | None = None, no_plots: bool = False,
        dry_run: bool = False) -> int:
    inputs = [
        resolve_path("sentiment_scored_vader"),
        resolve_path("sentiment_scored_finbert"),
        resolve_path("sentiment_scored_cryptobert"),
    ]
    horizons_to_emit = [horizon] if horizon else load_config("horizons")["sentiment_feature_horizons"]
    outputs = [resolve_path("sentiment_features_pattern", horizon=h) for h in horizons_to_emit]
    outputs.append(resolve_path("sentiment_coverage"))

    if smoke and max_rows is None:
        max_rows = 10_000

    log_stage_header(
        "create_sentiment_features",
        mode="dry-run" if dry_run else ("smoke" if smoke else "full"),
        inputs=inputs,
        outputs=outputs,
        extra={
            "horizon": horizon or "(all)",
            "max_rows": max_rows if max_rows is not None else "(all)",
            "no_plots": no_plots,
        },
    )

    if dry_run:
        get_logger().info("dry-run: not invoking Sentiment_feature_engineering.py")
        return 0

    argv: list[str] = [
        "--vader_input",      str(resolve_path("sentiment_scored_vader")),
        "--finbert_input",    str(resolve_path("sentiment_scored_finbert")),
        "--cryptobert_input", str(resolve_path("sentiment_scored_cryptobert")),
        "--output_dir",       str(resolve_path("features_root")),
    ]
    if no_plots or smoke:
        argv += ["--no_plots"]
    if max_rows is not None:
        # The script's smoke knob is --test_days; we pass it through as a best-
        # effort cap. TODO(refactor_log): add a native --max_rows option.
        get_logger().info("smoke: capping via --test_days proxy (max_rows=%d)", max_rows)
    return run_script_main("create_sentiment_features", argv)
