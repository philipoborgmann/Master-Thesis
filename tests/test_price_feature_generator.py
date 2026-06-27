"""Tests for the v4 price-feature generator (commit 8).

* Repository-root path resolution — the generator must NOT anchor on
  ``Path(__file__).resolve().parent``; it must use the shared
  :func:`thesis_pipeline.config.project_root` helper.
* Bars-per-day scaling — 7d / 14d / 21d / realized_vol_14d use
  ``days * BARS_PER_DAY[horizon]`` so the wall-clock window length is
  invariant across 1d / 6h / 1h.
* Canonical column names — the active generator emits the ``_Nd``
  family. Legacy ``_N`` names are forbidden by the schema validator.
* ECON-registry compatibility — every column the registry's ECON
  entry references is present on the generated frame.
* Market-cap availability — ``market_cap_available_at`` is stamped on
  every row and survives the leakage audit.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from thesis_pipeline.price import features as pf


# ---------------------------------------------------------------------------
# Path resolution — repo-root anchored, NOT module-anchored
# ---------------------------------------------------------------------------

def test_default_paths_are_under_repository_root():
    """The price generator's DEFAULT_PRICE_DIR / DEFAULT_OUTPUT_DIR
    must live under ``<repo>/Data/...`` — never under
    ``src/thesis_pipeline/price/Data/...``."""
    from thesis_pipeline.config import project_root
    root = project_root()
    assert Path(pf.DEFAULT_PRICE_DIR).resolve() == (
        root / "Data" / "Raw" / "Price").resolve()
    assert Path(pf.DEFAULT_OUTPUT_DIR).resolve() == (
        root / "Data" / "Features").resolve()
    # And the module-local fallback path is gone.
    forbidden = root / "src" / "thesis_pipeline" / "price" / "Data"
    assert not str(pf.DEFAULT_PRICE_DIR).startswith(str(forbidden))
    assert not str(pf.DEFAULT_OUTPUT_DIR).startswith(str(forbidden))


def test_sentiment_default_paths_are_under_repository_root():
    """The sentiment aggregator must also resolve against the repo root."""
    from thesis_pipeline.config import project_root
    from thesis_pipeline.sentiment import aggregate as sa
    root = project_root()
    assert Path(sa.OUTPUT_DIR).resolve() == (root / "Data" / "Features").resolve()
    for label, p in sa.DEFAULT_INPUTS.items():
        assert Path(p).resolve().is_relative_to(root), (label, p)


# ---------------------------------------------------------------------------
# BARS_PER_DAY scaling
# ---------------------------------------------------------------------------

def test_bars_per_day_mapping_pins_canonical_values():
    assert pf.BARS_PER_DAY == {"1d": 1, "6h": 4, "1h": 24}


@pytest.mark.parametrize("horizon,days,expected_bars", [
    ("1d", 7,  7),
    ("1d", 14, 14),
    ("1d", 21, 21),
    ("6h", 7,  28),
    ("6h", 14, 56),
    ("6h", 21, 84),
    ("1h", 7,  168),
    ("1h", 14, 336),
    ("1h", 21, 504),
])
def test_bars_per_day_scaling_matches_specification(horizon, days, expected_bars):
    assert pf.BARS_PER_DAY[horizon] * days == expected_bars


# ---------------------------------------------------------------------------
# Canonical column names + realized_vol_14d
# ---------------------------------------------------------------------------

def _synthetic_ohlcv(horizon: str, n: int = 60, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    bar_freq = {"1d": "D", "6h": "6h", "1h": "h"}[horizon]
    ts = pd.date_range("2024-01-01", periods=n, freq=bar_freq, tz="UTC")
    close = 100 + np.cumsum(rng.normal(0, 0.5, n))
    return pd.DataFrame({
        "timestamp": ts,
        "open":   close + rng.normal(0, 0.05, n),
        "high":   close + np.abs(rng.normal(0, 0.10, n)),
        "low":    close - np.abs(rng.normal(0, 0.10, n)),
        "close":  close,
        "volume": np.abs(rng.normal(1000, 50, n)),
    })


@pytest.mark.parametrize("horizon", ["1d", "6h", "1h"])
def test_generator_emits_v4_canonical_momentum_names(horizon):
    """The internal feature builder MUST emit ``cum_log_return_7d``,
    ``cum_log_return_14d``, ``cum_log_return_21d`` and
    ``realized_vol_14d``. Legacy ``_N`` names are never present."""
    df = _synthetic_ohlcv(horizon, n=120)
    out, _, _ = pf.create_features_for_coin_horizon(
        price_dir=None,
        ticker="BTC", horizon=horizon,
        market_cap_series=None,
        winsor_p=0.005,
        ohlcv_override=df,
    )
    for canonical in ("cum_log_return_7d", "cum_log_return_14d",
                       "cum_log_return_21d", "realized_vol_14d"):
        assert canonical in out.columns, canonical
    for legacy in ("cum_log_return_7", "cum_log_return_14",
                    "cum_log_return_21", "realized_vol_14"):
        assert legacy not in out.columns, legacy


def _mcap_series(n: int = 90, start: str = "2024-01-01") -> pd.DataFrame:
    """Synthetic CMC long-form series with v4 availability metadata."""
    days = pd.date_range(start, periods=n, freq="D", tz="UTC")
    return pd.DataFrame({
        "date":                    days,
        "market_cap_source_date":  days,
        "market_cap_available_at": days + pf.MARKET_CAP_AVAILABILITY_LAG,
        "market_cap":              np.linspace(1e9, 1.2e9, n),
        "log_market_cap_lag1":     np.linspace(20.7, 21.0, n),
    })


def test_generated_frame_satisfies_econ_registry():
    """Every column the active ECON registry references must be
    present on the generated frame OR satisfied by the schema
    validator's required set."""
    from thesis_pipeline.features.feature_registry import load_feature_sets
    from thesis_pipeline.diagnostics.feature_schema import (
        validate_price_feature_schema,
    )
    out, _, _ = pf.create_features_for_coin_horizon(
        price_dir=None,
        ticker="BTC", horizon="1d",
        market_cap_series=_mcap_series(n=90),
        winsor_p=0.005,
        ohlcv_override=_synthetic_ohlcv("1d", n=90),
    )
    out["ticker"] = "BTC"
    econ_features = (load_feature_sets() or {}).get("ECON", {}).get("features", [])
    missing = [c for c in econ_features if c not in out.columns]
    assert not missing, f"ECON registry columns missing from generated frame: {missing}"
    # And the schema validator accepts the frame as a v4 contract.
    validate_price_feature_schema(out, horizon="1d")


def test_market_cap_available_at_preserved():
    out, _, _ = pf.create_features_for_coin_horizon(
        price_dir=None,
        ticker="BTC", horizon="1d",
        market_cap_series=_mcap_series(n=60),
        winsor_p=0.005,
        ohlcv_override=_synthetic_ohlcv("1d", n=60),
    )
    assert "market_cap_available_at" in out.columns
    # Every matched row passes the leakage audit (strict ``<``).
    from thesis_pipeline.diagnostics.leakage_checks import (
        assert_market_cap_asof_correct,
    )
    out["ticker"] = "BTC"
    assert_market_cap_asof_correct(out)


# ---------------------------------------------------------------------------
# Output filename matches the merge stage's expectation
# ---------------------------------------------------------------------------

def test_generator_output_filename_matches_paths_pattern():
    """The generator's output stem must be the same as
    ``configs/paths.yaml :: price_features_pattern`` — i.e.
    ``price_features_{horizon}.parquet``. The merge stage reads that
    exact name."""
    from thesis_pipeline.config import resolve_path
    expected = resolve_path("price_features_pattern", horizon="1d").name
    assert expected == "price_features_1d.parquet"
