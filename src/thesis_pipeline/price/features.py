"""Price feature engineering stage.

Delegates to ``Create_Price_Features.py``. All winsorisation, target, and
return-formula behavior is preserved.

Feature columns produced (per horizon, per ticker):
    timestamp, date, ticker, horizon, target,
    log_return_t, cum_log_return_7, cum_log_return_14, realized_vol_14,
    volume_diff, market_cap_t
"""

from __future__ import annotations

from typing import Sequence

from ..config import load_config, resolve_path, smoke_coins
from ..logging_utils import get_logger, log_stage_header
from ..utils import run_script_main


def run(*, horizon: str | None = None,
        coins: Sequence[str] | None = None,
        winsor_p: float | None = None,
        smoke: bool = False,
        dry_run: bool = False) -> int:
    horizons_cfg = load_config("horizons")
    if smoke and not coins:
        coins = smoke_coins()
    if smoke and not horizon:
        horizon = horizons_cfg.get("default_horizon", "1d")

    inputs = [
        resolve_path("raw_price_root"),
        resolve_path("cmc_market_cap"),
    ]
    outputs = []
    if horizon:
        outputs.append(resolve_path("price_features_pattern", horizon=horizon))
    else:
        for h in horizons_cfg["price_feature_horizons"]:
            outputs.append(resolve_path("price_features_pattern", horizon=h))
    outputs += [
        resolve_path("feature_generation_report"),
        resolve_path("winsorization_thresholds"),
    ]

    log_stage_header(
        "create_price_features",
        mode="dry-run" if dry_run else ("smoke" if smoke else "full"),
        inputs=inputs,
        outputs=outputs,
        extra={
            "horizon": horizon or "(all)",
            "coins": list(coins) if coins else "(all discovered)",
            "winsor_p": winsor_p if winsor_p is not None else "(script default 0.005)",
        },
    )

    if dry_run:
        get_logger().info("dry-run: not invoking Create_Price_Features.py")
        return 0

    argv: list[str] = [
        "--price_dir",  str(resolve_path("raw_price_root")),
        "--output_dir", str(resolve_path("features_root")),
    ]
    if horizon:
        argv += ["--horizons", horizon]
    if coins:
        for c in coins:
            argv += ["--coin", c]
    if winsor_p is not None:
        argv += ["--winsor_p", str(winsor_p)]

    return run_script_main("create_price_features", argv)
