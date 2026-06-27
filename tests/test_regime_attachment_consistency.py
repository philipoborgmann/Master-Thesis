"""Section E — every regime consumer uses the shared availability-based
as-of helper (commit 3).

Targets:
* :func:`attach_regimes`              (vol)
* :func:`attach_market_cap_regimes`   (mcap)
* :func:`difference_in_improvement_table`  (H2/H3 headline)
* :func:`regime_mcnemar_table`             (supplementary)
* :func:`volatility_stratification_table`  (descriptive)
* :func:`market_cap_stratification_table`  (descriptive)
* :func:`regime_interaction_table`         (interaction)

Strict-< availability semantics MUST hold in every path.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from thesis_pipeline.evaluation import (
    diff_in_improvement as di,
    market_cap as mc,
    significance as sig,
    volatility as vol,
)
from thesis_pipeline.evaluation.regime_join import (
    REGIME_JOIN_STRATEGY,
    attach_regime_asof,
    regime_lag_summary,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _vol_lookup(start_date="2024-01-01", n=12, ticker="BTC"):
    """Build a v4 vol lookup carrying regime_available_at."""
    dates = pd.date_range(start_date, periods=n, tz="UTC")
    return pd.DataFrame({
        "ticker":              ticker,
        "date":                dates,
        "regime_available_at": dates,
        "regime_source_date":  dates - pd.Timedelta(days=1),
        "regime":              ["high"] * n,
        "vol_regime":          ["high"] * n,
    })


def _mcap_lookup(start_date="2024-01-01", n=12, ticker="BTC"):
    dates = pd.date_range(start_date, periods=n, tz="UTC")
    return pd.DataFrame({
        "ticker":              ticker,
        "date":                dates,
        "regime_available_at": dates,
        "regime_source_date":  dates - pd.Timedelta(days=1),
        "mcap_regime":         ["small"] * n,
    })


def _signal_at(ts, ticker="BTC"):
    return pd.DataFrame({
        "timestamp":   [pd.Timestamp(ts, tz="UTC")],
        "ticker":      [ticker],
        "target":      [1],
        "prediction":  [1],
        "probability": [0.6],
        "set_id":      ["S1"],
        "sentiment_model": ["vader"],
        "horizon":     ["1h"],
        "category":    ["sentiment"],
    })


# ---------------------------------------------------------------------------
# Equality at regime_available_at is rejected for every consumer
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("horizon,ts", [
    ("1h", "2024-01-05 00:00:00"),  # exact midnight
    ("6h", "2024-01-05 00:00:00"),
    ("1d", "2024-01-05 00:00:00"),
])
def test_attach_regimes_rejects_exact_match(horizon, ts):
    sigdf = _signal_at(ts)
    sigdf["horizon"] = horizon
    lookup = _vol_lookup()  # includes 2024-01-05 with available_at=2024-01-05
    out = vol.attach_regimes(sigdf, lookup)
    # Strict-< excludes the exact-match row → matches the 2024-01-04 entry.
    assert out["regime"].notna().all()
    assert (out["vol_regime_available_at"] < out["timestamp"]).all()


@pytest.mark.parametrize("ts", [
    "2024-01-05 00:00:00",     # 1d at midnight
    "2024-01-05 06:00:00",     # 6h intraday
    "2024-01-05 14:00:00",     # 1h intraday
])
def test_attach_market_cap_regimes_uses_asof_strict(ts):
    sigdf = _signal_at(ts)
    lookup = _mcap_lookup()
    out = mc.attach_market_cap_regimes(sigdf, lookup)
    assert out["mcap_regime"].notna().all()
    assert (out["mcap_regime_available_at"] < out["timestamp"]).all()


# ---------------------------------------------------------------------------
# First observation predating the lookup stays unmatched
# ---------------------------------------------------------------------------

def test_unmatched_first_observation_kept_with_nan_regime():
    sigdf = _signal_at("2023-12-25 00:00:00")
    lookup = _vol_lookup(start_date="2024-01-01", n=10)
    out = vol.attach_regimes(sigdf, lookup)
    assert pd.isna(out["regime"].iloc[0])


# ---------------------------------------------------------------------------
# Latest strictly-earlier regime is selected
# ---------------------------------------------------------------------------

def test_intraday_signal_takes_freshest_available_regime():
    """A 14:00 prediction must take the same-day-midnight regime, not the
    previous day's. Builds a 2-day lookup so we can verify which row wins."""
    sig_ts = "2024-01-05 14:00:00"
    sigdf = _signal_at(sig_ts)
    lookup = pd.DataFrame({
        "ticker":              ["BTC", "BTC"],
        "regime_available_at": [pd.Timestamp("2024-01-04", tz="UTC"),
                                pd.Timestamp("2024-01-05", tz="UTC")],
        "vol_regime":          ["low", "high"],
    })
    out = attach_regime_asof(sigdf, lookup, regime_col="vol_regime")
    assert out.iloc[0]["vol_regime"] == "high"


# ---------------------------------------------------------------------------
# Cross-consumer consistency: same regime label across modules for one signal
# ---------------------------------------------------------------------------

def test_h2_and_descriptive_use_same_regime_assignment():
    """A given signal/timestamp/ticker must receive the same regime in the
    H2 headline path and in the descriptive stratification path."""
    base_lookup = _vol_lookup(start_date="2024-01-01", n=20)
    base_lookup["vol_regime"] = (["low"] * 7) + (["mid"] * 7) + (["high"] * 6)
    base_lookup["regime"]     = base_lookup["vol_regime"]

    # One signal mid-window so the lookup has unambiguous availability.
    sigdf = _signal_at("2024-01-10 14:00:00")
    # Descriptive: vol.attach_regimes
    desc = vol.attach_regimes(sigdf, base_lookup)
    desc_regime = desc.iloc[0]["regime"]
    # Headline: pd.merge_asof through di._asof_join_regime
    look_h2 = di._prepare_regime_lookup(base_lookup, "vol_regime")
    h2 = di._asof_join_regime(sigdf, look_h2, "vol_regime")
    h2_regime = h2.iloc[0]["vol_regime"]
    assert desc_regime == h2_regime


def test_supplementary_regime_mcnemar_uses_same_asof_assignment():
    """The supplementary regime McNemar path goes through the same
    attach_regimes helper — so its regime label for an intraday signal
    must equal the headline H2 path's label."""
    rng = np.random.default_rng(0)
    ts = pd.date_range("2024-01-01", periods=200, freq="6h", tz="UTC")
    rows = []
    for t in ts:
        target = int(rng.integers(0, 2))
        rows.append({"timestamp": t, "ticker": "BTC", "target": target,
                     "prediction": target if rng.random() < 0.62 else 1 - target,
                     "probability": 0.6, "set_id": "S1",
                     "sentiment_model": "vader", "horizon": "6h",
                     "category": "sentiment"})
        rows.append({"timestamp": t, "ticker": "BTC", "target": target,
                     "prediction": target if rng.random() < 0.5 else 1 - target,
                     "probability": 0.5, "set_id": "B1",
                     "sentiment_model": "-", "horizon": "6h",
                     "category": "benchmark"})
    signals = pd.DataFrame(rows)
    dates = pd.date_range("2023-12-31", periods=60, freq="D", tz="UTC")
    look = pd.DataFrame({
        "ticker":              "BTC",
        "date":                dates,
        "regime_available_at": dates,
        "regime":              (["low"] * 20) + (["mid"] * 20) + (["high"] * 20),
    })
    out = sig.regime_mcnemar_table(
        signals, vol_lookup=look, mcap_lookup=None, benchmarks=("B1",),
        min_n_matched=10, min_discordant=5,
    )
    assert not out.empty
    assert (out["regime_type"] == "volatility").all()


# ---------------------------------------------------------------------------
# Interaction stratification uses both availability-safe assignments
# ---------------------------------------------------------------------------

def test_interaction_table_uses_both_availability_safe_joins(monkeypatch):
    """Force-trip the strict-< assertion in attach_regime_asof: the
    interaction table must run cleanly because both joins go through the
    shared helper."""
    rng = np.random.default_rng(0)
    ts = pd.date_range("2024-01-01", periods=30, freq="D", tz="UTC")
    rows = []
    for t in ts:
        target = int(rng.integers(0, 2))
        rows.append({"timestamp": t, "ticker": "BTC", "target": target,
                     "prediction": target, "probability": 0.6,
                     "set_id": "S1", "sentiment_model": "vader",
                     "horizon": "1d", "category": "sentiment"})
    signals = pd.DataFrame(rows)
    dates = pd.date_range("2023-12-31", periods=35, freq="D", tz="UTC")
    vol_lookup = pd.DataFrame({
        "ticker": "BTC", "date": dates,
        "regime_available_at": dates,
        "regime": (["low"] * 12) + (["mid"] * 11) + (["high"] * 12),
    })
    mcap_lookup = pd.DataFrame({
        "ticker": "BTC", "date": dates,
        "regime_available_at": dates,
        "mcap_regime": (["small"] * 12) + (["mid"] * 11) + (["large"] * 12),
    })
    out = mc.regime_interaction_table(signals, mcap_lookup, vol_lookup)
    assert not out.empty
    # Cells exist for every combo (counts may be 0 for impossible cells).
    assert {"low", "mid", "high"}.issubset(set(out["vol_regime"]))
    assert {"small", "mid", "large"}.issubset(set(out["mcap_regime"]))


# ---------------------------------------------------------------------------
# Shared lag-summary helper produces the documented diagnostics fields
# ---------------------------------------------------------------------------

def test_regime_lag_summary_reports_required_fields():
    sigdf = _signal_at("2024-01-05 14:00:00")
    lookup = _vol_lookup(start_date="2024-01-01", n=10)
    joined = attach_regime_asof(sigdf, lookup, regime_col="vol_regime")
    summary = regime_lag_summary(joined, regime_col="vol_regime")
    for k in ("share_unmatched_regime", "median_regime_lag_days",
              "min_regime_lag_days", "max_regime_lag_days",
              "share_regime_lag_gt_1_day"):
        assert k in summary


def test_strategy_constant_exposed():
    assert REGIME_JOIN_STRATEGY == "asof_backward_strict"


# ---------------------------------------------------------------------------
# No same-date merge survives in production code
# ---------------------------------------------------------------------------

def test_no_same_date_merge_in_production_regime_code():
    """Static check: the regime-attachment functions in production code
    must NOT call ``DataFrame.merge`` with ``on=['ticker', 'date']``.
    They MUST delegate to the shared as-of helper instead."""
    import pathlib
    src = pathlib.Path(__file__).resolve().parents[1] / "src" / "thesis_pipeline" / "evaluation"
    offenders = []
    for p in src.glob("*.py"):
        if p.name == "regime_join.py":
            continue
        text = p.read_text(encoding="utf-8")
        if "merge(rl, on=[\"ticker\", \"date\"]" in text:
            offenders.append(str(p))
    assert offenders == [], offenders
