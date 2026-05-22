"""Unit tests for the economic / backtest module.

Synthetic returns + positions. No real OHLC required.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from thesis_pipeline.evaluation import economic as ec


# ---------------------------------------------------------------------------
# Position logic
# ---------------------------------------------------------------------------

def _df(prob, pred):
    return pd.DataFrame({
        "probability": prob,
        "prediction":  pred,
    })


def test_build_positions_default_threshold_trades_everything():
    df = _df([0.40, 0.60, 0.55], [0, 1, 1])
    pos = ec.build_positions(df, threshold=0.5)
    # threshold=0.5 → no neutral band, sign of prediction wins
    assert pos.tolist() == [-1, 1, 1]


def test_build_positions_threshold_creates_flat_zone():
    df = _df([0.10, 0.95, 0.55, 0.45, 0.99], [0, 1, 1, 0, 1])
    pos = ec.build_positions(df, threshold=0.6, flat_middle_zone=True)
    # 0.10 < 0.4 → short, 0.95 > 0.6 → long, 0.55 ∈ band → flat,
    # 0.45 ∈ band → flat, 0.99 > 0.6 → long
    assert pos.tolist() == [-1, 1, 0, 0, 1]


def test_build_positions_long_only_collapses_shorts_to_flat():
    df = _df([0.10, 0.90], [0, 1])
    pos = ec.build_positions(df, threshold=0.5, long_short_enabled=False)
    assert pos.tolist() == [0, 1]


def test_build_positions_flat_middle_zone_false_uses_prediction():
    df = _df([0.55, 0.45], [1, 0])  # both in the 0.6 neutral band
    pos = ec.build_positions(df, threshold=0.6, flat_middle_zone=False)
    # With flat_middle_zone=False we fall back to sign-of-prediction.
    assert pos.tolist() == [1, -1]


# ---------------------------------------------------------------------------
# Turnover + costs
# ---------------------------------------------------------------------------

def test_compute_turnover_half_abs_diff():
    pos = pd.Series([1, 1, -1, -1, 0, 1], dtype=int)
    to  = ec.compute_turnover(pos)
    # Initial entry: |1-0|/2 = 0.5; then 0 / 1 (full flip) / 0 / 0.5 / 0.5
    assert to.tolist() == pytest.approx([0.5, 0.0, 1.0, 0.0, 0.5, 0.5])


def test_apply_transaction_costs_deducts_turnover_times_cost():
    gross = pd.Series([0.01, -0.005, 0.02])
    to    = pd.Series([1.0, 0.5, 0.0])
    net   = ec.apply_transaction_costs(gross, to, cost_bps=10.0)
    # 10 bps = 0.001
    assert net.iloc[0] == pytest.approx(0.01 - 1.0 * 0.001)
    assert net.iloc[1] == pytest.approx(-0.005 - 0.5 * 0.001)
    assert net.iloc[2] == pytest.approx(0.02)


def test_compute_strategy_returns_signs_correctly():
    pos = pd.Series([1, -1, 0])
    fwd = pd.Series([0.02, -0.01, 0.05])
    r = ec.compute_strategy_returns(pos, fwd)
    assert r.tolist() == pytest.approx([0.02, 0.01, 0.0])


# ---------------------------------------------------------------------------
# Risk + return metrics
# ---------------------------------------------------------------------------

def test_compute_drawdown_simple():
    # Cumulative: 0.05 (peak) → 0.03 → 0.01 → 0.06 (new peak) → 0.04
    # Drawdowns: 0, -0.02, -0.04, 0, -0.02 → max_dd = -0.04
    rets = pd.Series([0.05, -0.02, -0.02, 0.05, -0.02])
    dd, max_dd = ec.compute_drawdown(rets)
    assert max_dd == pytest.approx(-0.04)


def test_compute_sharpe_positive_for_positive_excess():
    rng = np.random.default_rng(0)
    rets = pd.Series(rng.normal(0.001, 0.01, 500))
    sh = ec.compute_sharpe(rets, periods_per_year=365, risk_free_rate=0.0)
    # Mean > 0, so Sharpe should be positive
    assert sh > 0


def test_compute_sharpe_returns_nan_on_zero_volatility():
    rets = pd.Series([0.001, 0.001, 0.001, 0.001])
    sh = ec.compute_sharpe(rets, periods_per_year=365)
    assert np.isnan(sh)


def test_compute_sortino_uses_downside_deviation_only():
    rets = pd.Series([0.02, 0.03, -0.01, 0.04, 0.01, -0.005, 0.02])
    so = ec.compute_sortino(rets, periods_per_year=365)
    sh = ec.compute_sharpe(rets, periods_per_year=365)
    # Sortino > Sharpe when upside variance dominates downside variance.
    assert so > sh


# ---------------------------------------------------------------------------
# Cumulative + summarisers via _backtest_row
# ---------------------------------------------------------------------------

def test_backtest_row_cumulative_return_matches_sum_of_net():
    rng = np.random.default_rng(1)
    n = 100
    pos = pd.Series(rng.choice([-1, 1], size=n), dtype=int)
    fwd = pd.Series(rng.normal(0, 0.02, n))
    ts  = pd.Series(pd.date_range("2024-01-01", periods=n, tz="UTC"))
    row = ec._backtest_row(pos, fwd, ts, cost_bps=0.0, cfg={
        "crypto_calendar_days": 365, "risk_free_rate": 0.0,
    })
    # With cost=0, net == gross == pos * fwd
    expected_cum = float((pos.astype(float) * fwd).sum())
    assert row["cumulative_return"] == pytest.approx(expected_cum)
    assert row["n_obs"] == n
    assert row["n_trades"] == n  # every observation traded


def test_backtest_row_cost_reduces_cumulative_return():
    rng = np.random.default_rng(2)
    n = 100
    pos = pd.Series(rng.choice([-1, 1], size=n), dtype=int)
    fwd = pd.Series(rng.normal(0, 0.02, n))
    ts  = pd.Series(pd.date_range("2024-01-01", periods=n, tz="UTC"))
    cfg = {"crypto_calendar_days": 365, "risk_free_rate": 0.0}
    r0  = ec._backtest_row(pos, fwd, ts, cost_bps=0.0,  cfg=cfg)
    r10 = ec._backtest_row(pos, fwd, ts, cost_bps=10.0, cfg=cfg)
    # 10 bps cost on every position-change must reduce cumulative return.
    assert r10["cumulative_return"] < r0["cumulative_return"]


def test_summarize_threshold_backtest_handles_flat_zone():
    # 3 bars, S1 set, varying probabilities
    sig = pd.DataFrame({
        "timestamp":   pd.date_range("2024-01-01", periods=3, tz="UTC"),
        "ticker":      "BTC",
        "target":      [1, 0, 1],
        "prediction":  [1, 0, 1],
        "probability": [0.95, 0.05, 0.55],   # last one in 0.6 neutral band
        "set_id":      "S1",
        "sentiment_model": "vader",
        "horizon":     "1d",
        "category":    "sentiment",
        "label":       "test",
    })
    fwd = pd.DataFrame({
        "ticker":    "BTC",
        "timestamp": sig["timestamp"],
        "forward_log_return": [0.01, -0.01, 0.01],
    })
    out = ec.summarize_threshold_backtest(sig, fwd, {
        "thresholds": [0.6], "transaction_cost_bps": [0],
        "flat_middle_zone": True, "long_short_enabled": True,
        "crypto_calendar_days": 365, "risk_free_rate": 0.0,
    })
    assert not out.empty
    row = out.iloc[0]
    # 2 trades (first + second observation), the third is flat.
    assert row["n_trades"] == 2
    # Coverage < 1 because of the flat middle zone.
    assert row["coverage"] < 1


def test_load_backtest_config_defaults_when_file_absent(tmp_path):
    cfg = ec.load_backtest_config(tmp_path / "nope.yaml")
    # No file → returns defaults
    assert cfg["annualization_mode"] == "crypto"
    assert cfg["crypto_calendar_days"] == 365


def test_load_backtest_config_reads_real_file(tmp_path):
    import yaml
    cfg_path = tmp_path / "bt.yaml"
    cfg_path.write_text(yaml.safe_dump({
        "crypto_calendar_days": 252,
        "transaction_cost_bps": [0, 25],
    }))
    cfg = ec.load_backtest_config(cfg_path)
    assert cfg["crypto_calendar_days"] == 252
    assert cfg["transaction_cost_bps"] == [0, 25]
