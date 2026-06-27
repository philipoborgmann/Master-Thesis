"""Tests for availability-based regime matching (commit 2 Section E).

Replaces the previous fixed D-1 calendar shift with a per-ticker
``pd.merge_asof`` keyed on ``regime_available_at`` and using strict-<
matching via ``allow_exact_matches=False``.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from thesis_pipeline.evaluation import diff_in_improvement as di
from thesis_pipeline.evaluation import volatility as vol_mod
from thesis_pipeline.evaluation import market_cap as mc_mod


# ---------------------------------------------------------------------------
# Lookup-builder contract — regime_source_date + regime_available_at present
# ---------------------------------------------------------------------------

def test_volatility_lookup_carries_availability_columns(monkeypatch, tmp_path):
    """build_regime_lookup must emit regime_source_date + regime_available_at."""
    n = 30
    rng = np.random.default_rng(0)
    df = pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=n, freq="D"),
        "open":  20000 + rng.normal(0, 50, n),
        "high":  20100 + rng.normal(0, 50, n),
        "low":   19900 + rng.normal(0, 50, n),
        "close": 20000 + rng.normal(0, 50, n),
        "volume": np.abs(rng.normal(1000, 50, n)),
    })
    p = tmp_path / "BTCUSDT_1d.parquet"
    df.to_parquet(p, index=False)
    monkeypatch.setattr(vol_mod, "_resolve_ohlcv_path",
                        lambda tk: p if tk.upper() == "BTC" else None)
    lookup = vol_mod.build_regime_lookup(["BTC"])

    assert {"regime_source_date", "regime_available_at"}.issubset(lookup.columns)
    # The shifted lookup has source = date - 1, available = source + 1 = date.
    pd.testing.assert_series_equal(
        lookup["regime_available_at"].reset_index(drop=True),
        lookup["date"].reset_index(drop=True),
        check_names=False,
    )
    expected_source = (lookup["date"] - pd.Timedelta(days=1)).reset_index(drop=True)
    pd.testing.assert_series_equal(
        lookup["regime_source_date"].reset_index(drop=True),
        expected_source,
        check_names=False,
    )


def test_market_cap_lookup_carries_availability_columns(tmp_path):
    dates = pd.date_range("2024-01-01", periods=10, tz="UTC")
    wide = pd.DataFrame({
        "date": dates,
        "000000000001_BTC": np.linspace(400, 450, 10),
        "000000001027_ETH": np.linspace(200, 250, 10),
        "000000000052_XRP": np.linspace(100, 150, 10),
    })
    path = tmp_path / "market_cap.parquet"
    wide.to_parquet(path, index=False)

    lookup = mc_mod.build_market_cap_lookup(path)
    assert {"regime_source_date", "regime_available_at"}.issubset(lookup.columns)
    expected_source = (lookup["date"] - pd.Timedelta(days=1)).reset_index(drop=True)
    pd.testing.assert_series_equal(
        lookup["regime_source_date"].reset_index(drop=True),
        expected_source, check_names=False,
    )
    pd.testing.assert_series_equal(
        lookup["regime_available_at"].reset_index(drop=True),
        lookup["date"].reset_index(drop=True),
        check_names=False,
    )


# ---------------------------------------------------------------------------
# _asof_join_regime: strict-< matching, per-ticker
# ---------------------------------------------------------------------------

def _signals(timestamps, ticker="BTC"):
    return pd.DataFrame({
        "timestamp": pd.to_datetime(timestamps, utc=True),
        "ticker":    ticker,
        "d":         np.zeros(len(timestamps), dtype=float),
    })


def _vol_lookup(dates_d, ticker="BTC", regimes=("low", "mid", "high", "low", "high")):
    dates = pd.to_datetime(dates_d, utc=True)
    return pd.DataFrame({
        "ticker":               ticker,
        "regime_available_at":  dates,
        "vol_regime":           list(regimes)[: len(dates)],
    })


def test_asof_join_uses_strict_less_than():
    """A signal at D 00:00 UTC must NOT match the regime row whose
    regime_available_at is exactly D 00:00 UTC — strict-< semantics."""
    pred_ts = ["2024-01-04 00:00:00", "2024-01-04 06:00:00"]
    sig = _signals(pred_ts)
    look = _vol_lookup(
        ["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"],
        regimes=("low", "mid", "high", "low"),
    )
    look = di._prepare_regime_lookup(look, "vol_regime")

    joined = di._asof_join_regime(sig, look, "vol_regime")
    # First signal (D=Jan 4 00:00) → exact-match excluded → must take
    # the previous regime (Jan 3 = "mid"), NOT the Jan 4 row ("high").
    assert joined.iloc[0]["vol_regime"] == "mid"
    # Second signal (D=Jan 4 06:00) → Jan 4 00:00 IS strictly before
    # 06:00 → must match "high".
    assert joined.iloc[1]["vol_regime"] == "high"


def test_asof_join_separates_per_ticker():
    """A BTC regime must never leak into an ETH signal."""
    sig = pd.concat([_signals(["2024-01-05 12:00:00"], ticker="BTC"),
                     _signals(["2024-01-05 12:00:00"], ticker="ETH")],
                    ignore_index=True)
    btc_lk = _vol_lookup(["2024-01-04"], ticker="BTC", regimes=("high",))
    eth_lk = _vol_lookup(["2024-01-04"], ticker="ETH", regimes=("low",))
    look = di._prepare_regime_lookup(
        pd.concat([btc_lk, eth_lk], ignore_index=True), "vol_regime",
    )
    joined = di._asof_join_regime(sig, look, "vol_regime")
    btc_row = joined[joined["ticker"] == "BTC"].iloc[0]
    eth_row = joined[joined["ticker"] == "ETH"].iloc[0]
    assert btc_row["vol_regime"] == "high"
    assert eth_row["vol_regime"] == "low"


def test_asof_join_unmatched_signal_keeps_row_with_nan_regime():
    """A signal earlier than every regime row stays in the frame with
    NaN regime so the diagnostic share_unmatched is computable."""
    sig = _signals(["2024-01-01 00:00:00"])
    look = _vol_lookup(["2024-02-01"], regimes=("high",))
    look = di._prepare_regime_lookup(look, "vol_regime")
    joined = di._asof_join_regime(sig, look, "vol_regime")
    assert len(joined) == 1
    assert pd.isna(joined.iloc[0]["vol_regime"])


def test_asof_join_preserves_signal_row_ordering():
    """The signal frame's row order MUST survive the join — the
    diff-in-improvement caller uses positional alignment downstream."""
    pred_ts = ["2024-01-05 12:00:00", "2024-01-04 06:00:00",
               "2024-01-06 00:00:00"]
    sig = _signals(pred_ts)
    sig["__pos__"] = np.arange(len(sig))
    look = _vol_lookup(["2024-01-03", "2024-01-05"],
                       regimes=("low", "high"))
    look = di._prepare_regime_lookup(look, "vol_regime")
    joined = di._asof_join_regime(sig, look, "vol_regime")
    assert list(joined["__pos__"]) == [0, 1, 2]


# ---------------------------------------------------------------------------
# Strict assertion in the table builder
# ---------------------------------------------------------------------------

def _synth_signals(n=200):
    rng = np.random.default_rng(0)
    ts = pd.date_range("2024-01-01", periods=n, freq="D", tz="UTC")
    # Target sequence MUST be shared between ECON and the augmented set
    # (they predict the same y at the same (ticker, timestamp)).
    shared_target = rng.integers(0, 2, n)
    rows = []
    for sid, sm, edge in (("ECON",       "-",     0.50),
                          ("ECON_VAD_F", "vader", 0.62)):
        tgt = shared_target.copy()
        flip = rng.random(n) > edge
        pred = np.where(flip, 1 - tgt, tgt)
        rows.append(pd.DataFrame({
            "timestamp":  ts,
            "ticker":     "BTC",
            "target":     tgt,
            "prediction": pred,
            "probability": rng.uniform(0.4, 0.7, size=n),
            "horizon":    "1d",
            "set_id":     sid,
            "sentiment_model": sm,
            "model_type": "panel_logit",
            "panel_mode": "ticker_fixed_effects",
            "hpo_variant": "log_loss",
            "hpo_objective": "log_loss",
            "train_window_mode": "rolling_fixed",
            "rolling_window_days": 180.0,
            "rolling_window_timestamps": np.nan,
        }))
    return pd.concat(rows, ignore_index=True)


def _matched_benchmark():
    return {"ECON_VAD_F": "ECON"}


def test_diff_in_improvement_uses_asof_and_emits_lag_diagnostics():
    sig = _synth_signals()
    dates = pd.date_range("2024-01-01", periods=200, tz="UTC")
    n = len(dates)
    third = n // 3
    regimes = ["low"] * third + ["mid"] * third + ["high"] * (n - 2 * third)
    vol_lk = pd.DataFrame({
        "ticker":               "BTC",
        "regime_available_at":  dates,
        "vol_regime":           regimes,
    })
    out = di.difference_in_improvement_table(
        signals=sig,
        matched_benchmark=_matched_benchmark(),
        regime_lookup=vol_lk,
        regime_col="vol_regime",
        treatment_value="high",
        control_value="low",
        hypothesis_family="H2_volatility",
    )
    assert not out.empty
    r = out.iloc[0]
    # Diagnostic columns present and populated for the matched row.
    assert r["regime_match_strategy"] == "asof_backward_strict"
    for col in ("median_regime_lag_days", "min_regime_lag_days",
                "max_regime_lag_days", "share_unmatched_regime"):
        assert col in out.columns
    # Strict-< means every matched row has min_regime_lag_days > 0.
    assert r["min_regime_lag_days"] > 0


def test_diff_in_improvement_falls_back_to_date_column():
    """A lookup carrying only ``date`` (legacy v3 layout) must still be
    accepted — _prepare_regime_lookup derives regime_available_at = date."""
    sig = _synth_signals()
    dates = pd.date_range("2024-01-01", periods=200, tz="UTC")
    n = len(dates)
    third = n // 3
    regimes = ["low"] * third + ["mid"] * third + ["high"] * (n - 2 * third)
    vol_lk = pd.DataFrame({
        "ticker":     "BTC",
        "date":       dates,
        "vol_regime": regimes,
    })
    out = di.difference_in_improvement_table(
        signals=sig,
        matched_benchmark=_matched_benchmark(),
        regime_lookup=vol_lk,
        regime_col="vol_regime",
        treatment_value="high",
        control_value="low",
        hypothesis_family="H2_volatility",
    )
    assert not out.empty
    assert out.iloc[0]["regime_match_strategy"] == "asof_backward_strict"


def test_strict_assertion_never_admits_exact_match():
    """Even with adversarially-aligned timestamps (lookup row available
    exactly at the prediction instant), strict-< MUST exclude that row
    from the match."""
    # Build a tiny signal where the lookup's only row is exactly at the
    # prediction timestamp — merge_asof with allow_exact_matches=False
    # must skip it, leaving the row unmatched.
    sig = _signals(["2024-01-05 00:00:00"])
    look = _vol_lookup(["2024-01-05"], regimes=("high",))
    look = di._prepare_regime_lookup(look, "vol_regime")
    joined = di._asof_join_regime(sig, look, "vol_regime")
    assert pd.isna(joined.iloc[0]["vol_regime"])
