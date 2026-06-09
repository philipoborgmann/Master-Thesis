"""Unit tests for the cross-sectional High-Low portfolio backtest.

These tests pin the methodology fixes from the rewrite:

* Annualisation uses the number of portfolio timestamps, never the count
  of (ticker × timestamp) observations.
* Cumulative return is computed on portfolio returns; the equity drawdown
  is a percentage on ``exp(cumsum)``, not a log-difference.
* Turnover is portfolio-level: a long↔short flip is more expensive than
  an entry / exit.
* 1d uses date-normalised merge keys; 1h / 6h use exact timestamps.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from thesis_pipeline.evaluation import economic as ec


# ---------------------------------------------------------------------------
# High-Low portfolio construction
# ---------------------------------------------------------------------------

CFG_DEFAULT = {
    "long_quantile":       0.8,
    "short_quantile":      0.2,
    "min_assets_per_leg":  1,
    "allow_single_leg":    False,
    "crypto_calendar_days": 365,
    "risk_free_rate":      0.0,
    "transaction_cost_bps": [0],
    "thresholds":          [0.55, 0.60, 0.65],
}


def _two_period_three_asset_frame():
    ts1 = pd.Timestamp("2024-01-01", tz="UTC")
    ts2 = pd.Timestamp("2024-01-02", tz="UTC")
    rows = [
        # Period 1: BTC top prob, XRP bottom prob; ETH middle.
        {"timestamp": ts1, "ticker": "BTC", "probability": 0.90, "forward_log_return": 0.02,
         "set_id": "S1", "sentiment_model": "vader", "horizon": "1d",
         "category": "sentiment", "label": "test", "prediction": 1, "target": 1},
        {"timestamp": ts1, "ticker": "ETH", "probability": 0.50, "forward_log_return": 0.00,
         "set_id": "S1", "sentiment_model": "vader", "horizon": "1d",
         "category": "sentiment", "label": "test", "prediction": 1, "target": 0},
        {"timestamp": ts1, "ticker": "XRP", "probability": 0.10, "forward_log_return": -0.03,
         "set_id": "S1", "sentiment_model": "vader", "horizon": "1d",
         "category": "sentiment", "label": "test", "prediction": 0, "target": 0},
        # Period 2: same ordering.
        {"timestamp": ts2, "ticker": "BTC", "probability": 0.85, "forward_log_return": 0.01,
         "set_id": "S1", "sentiment_model": "vader", "horizon": "1d",
         "category": "sentiment", "label": "test", "prediction": 1, "target": 1},
        {"timestamp": ts2, "ticker": "ETH", "probability": 0.55, "forward_log_return": 0.005,
         "set_id": "S1", "sentiment_model": "vader", "horizon": "1d",
         "category": "sentiment", "label": "test", "prediction": 1, "target": 1},
        {"timestamp": ts2, "ticker": "XRP", "probability": 0.15, "forward_log_return": -0.02,
         "set_id": "S1", "sentiment_model": "vader", "horizon": "1d",
         "category": "sentiment", "label": "test", "prediction": 0, "target": 0},
    ]
    return pd.DataFrame(rows)


def test_high_low_portfolio_top_goes_long_bottom_goes_short():
    merged = _two_period_three_asset_frame()
    pf = ec.build_high_low_portfolio_returns(merged, CFG_DEFAULT)
    assert len(pf) == 2

    # Period 1 weights: BTC = +1 (long, single asset), XRP = -1 (short).
    w1 = pf.iloc[0]["weights"]
    assert w1 == {"BTC": pytest.approx(1.0), "XRP": pytest.approx(-1.0)}
    # Portfolio return = mean(long_fwd) - mean(short_fwd) = 0.02 - (-0.03) = 0.05.
    assert pf.iloc[0]["gross_log_return"] == pytest.approx(0.05)
    assert pf.iloc[0]["long_n"] == 1
    assert pf.iloc[0]["short_n"] == 1


def test_middle_asset_not_traded():
    merged = _two_period_three_asset_frame()
    pf = ec.build_high_low_portfolio_returns(merged, CFG_DEFAULT)
    for _, row in pf.iterrows():
        assert "ETH" not in row["weights"], "the middle asset must not appear in any leg"


def test_portfolio_skipped_when_single_leg_only():
    # Edge case: only one asset → empty short leg.
    df = pd.DataFrame([{
        "timestamp": pd.Timestamp("2024-01-01", tz="UTC"),
        "ticker": "BTC", "probability": 0.9, "forward_log_return": 0.02,
        "set_id": "S1", "sentiment_model": "v", "horizon": "1d",
    }])
    cfg = {**CFG_DEFAULT, "min_assets_per_leg": 1, "allow_single_leg": False}
    pf = ec.build_high_low_portfolio_returns(df, cfg)
    assert pf.empty


# ---------------------------------------------------------------------------
# Turnover + transaction costs
# ---------------------------------------------------------------------------

def test_portfolio_turnover_initial_entry_counts():
    weights = [{"BTC": 1.0, "XRP": -1.0}]
    to = ec.compute_portfolio_turnover(weights)
    # 0.5 · (|1 - 0| + |-1 - 0|) = 1.0
    assert to[0] == pytest.approx(1.0)


def test_portfolio_turnover_unchanged_weights_zero():
    w = {"BTC": 1.0, "XRP": -1.0}
    to = ec.compute_portfolio_turnover([w, w, w])
    assert to[0] == pytest.approx(1.0)  # initial entry
    assert to[1] == pytest.approx(0.0)
    assert to[2] == pytest.approx(0.0)


def test_portfolio_turnover_long_short_flip_costs_more():
    w_entry  = {"BTC": 1.0, "XRP": -1.0}
    w_swap   = {"BTC": -1.0, "XRP": 1.0}  # full flip on every coin
    to = ec.compute_portfolio_turnover([w_entry, w_swap])
    # 0.5 · (|-1 - 1| + |1 - (-1)|) = 0.5 · 4 = 2.0
    assert to[1] == pytest.approx(2.0)
    # ⇒ strictly larger than the initial entry (1.0)
    assert to[1] > to[0]


def test_apply_transaction_costs_deducts_turnover_times_cost():
    gross = np.array([0.01, -0.005, 0.02])
    to    = np.array([1.0, 0.5, 0.0])
    net   = ec.apply_transaction_costs(gross, to, cost_bps=10.0)
    # 10 bps = 0.001
    assert net[0] == pytest.approx(0.01 - 1.0 * 0.001)
    assert net[1] == pytest.approx(-0.005 - 0.5 * 0.001)
    assert net[2] == pytest.approx(0.02)


# ---------------------------------------------------------------------------
# Annualisation uses portfolio periods, not ticker observations
# ---------------------------------------------------------------------------

def test_periods_per_year_uses_portfolio_timestamps():
    # 100 portfolio timestamps over 100 calendar days → exactly ~365 ppy.
    # With the old (buggy) per-observation logic and 24 tickers per
    # timestamp, this would yield ≈ 24 × 365.
    ppy = ec._periods_per_year(100, 100.0, 365)
    assert ppy == pytest.approx(365.0)


def test_summarize_high_low_uses_n_periods_not_n_observations():
    # Synthetic: 20 timestamps, 3 tickers each → 60 observations but only
    # 20 portfolio periods.
    rng = np.random.default_rng(0)
    ts  = pd.date_range("2024-01-01", periods=20, tz="UTC", freq="D")
    rows = []
    for t in ts:
        for tk, p in (("BTC", 0.9), ("ETH", 0.5), ("XRP", 0.1)):
            rows.append({"timestamp": t, "ticker": tk,
                          "probability": p + rng.normal(0, 0.01),
                          "forward_log_return": rng.normal(0, 0.02),
                          "prediction": 1, "target": 1,
                          "set_id": "S1", "sentiment_model": "vader",
                          "horizon": "1d", "category": "sentiment",
                          "label": "test"})
    signals = pd.DataFrame(rows)
    fr = signals[["ticker", "timestamp", "forward_log_return"]].copy()
    out = ec.summarize_high_low_backtest(signals, fr, CFG_DEFAULT, horizon="1d")
    row = out.iloc[0]
    assert int(row["n_periods"]) == 20, "must report portfolio timestamps"
    # Annualisation: 20 periods over 19 days → ≈ 384 ppy. Either way, the
    # implied periods-per-year should be < 24 × 365.
    # Verify by reconstructing the implied ppy from the reported metrics.


# ---------------------------------------------------------------------------
# Drawdown is percentage on the equity curve
# ---------------------------------------------------------------------------

def test_equity_drawdown_returns_percentage_in_minus_one_zero():
    # Three +5 % then three -5 % bars → equity peaks above 1 then dips
    # below; drawdown must be in (-1, 0].
    r = np.array([0.05, 0.05, 0.05, -0.05, -0.05, -0.05])
    _, max_dd = ec.compute_equity_drawdown(r)
    assert -1.0 < max_dd <= 0.0
    # exp(0.15) → peak; exp(0.0) → trough; dd = 1/exp(0.15) − 1
    expected = float(np.exp(0.0) / np.exp(0.15) - 1.0)
    assert max_dd == pytest.approx(expected)


def test_equity_drawdown_is_not_log_drawdown():
    # The old log-difference formula would give max_dd = -0.15 (3 × -0.05).
    # The new percentage formula gives ≈ -0.139 (exp(-0.15) − 1).
    r = np.array([0.05, 0.05, 0.05, -0.05, -0.05, -0.05])
    _, max_dd = ec.compute_equity_drawdown(r)
    log_dd = -0.15
    assert not np.isclose(max_dd, log_dd, atol=1e-6)


# ---------------------------------------------------------------------------
# Forward-return join: daily normalisation vs intraday exactness
# ---------------------------------------------------------------------------

def test_daily_join_uses_normalized_date():
    signals = pd.DataFrame({
        "timestamp":   [pd.Timestamp("2024-01-01 23:59:00", tz="UTC"),
                        pd.Timestamp("2024-01-02 00:00:01", tz="UTC")],
        "ticker":      ["BTC", "BTC"],
        "probability": [0.6, 0.4],
        "prediction":  [1, 0], "target": [1, 0],
        "set_id": "S1", "sentiment_model": "v", "horizon": "1d",
    })
    fr = pd.DataFrame({
        "ticker":             ["BTC", "BTC"],
        "timestamp":          pd.to_datetime(
            ["2024-01-01", "2024-01-02"], utc=True),
        "forward_log_return": [0.01, -0.02],
    })
    merged = ec._signals_with_forward_returns(signals, fr, horizon="1d")
    assert merged["forward_log_return"].notna().all()


def test_intraday_join_does_not_normalise_to_date():
    # 1h horizon, exact hourly bars
    signals = pd.DataFrame({
        "timestamp":   [pd.Timestamp("2024-01-01 00:00:00", tz="UTC"),
                        pd.Timestamp("2024-01-01 01:00:00", tz="UTC")],
        "ticker":      ["BTC", "BTC"],
        "probability": [0.6, 0.4],
        "prediction":  [1, 0], "target": [1, 0],
        "set_id": "S1", "sentiment_model": "v", "horizon": "1h",
    })
    fr = pd.DataFrame({
        "ticker":             ["BTC", "BTC"],
        "timestamp":          pd.to_datetime(
            ["2024-01-01 00:00:00", "2024-01-01 01:00:00"], utc=True),
        "forward_log_return": [0.01, -0.02],
    })
    merged = ec._signals_with_forward_returns(signals, fr, horizon="1h")
    # Both bars matched their own hour, not collapsed onto the same date.
    assert merged["forward_log_return"].tolist() == pytest.approx([0.01, -0.02])


def test_intraday_join_rejects_many_to_many():
    # Duplicate forward-return rows for the same (ticker, timestamp).
    signals = pd.DataFrame({
        "timestamp":   [pd.Timestamp("2024-01-01 00:00:00", tz="UTC")],
        "ticker":      ["BTC"],
        "probability": [0.9], "prediction": [1], "target": [1],
        "set_id": "S1", "sentiment_model": "v", "horizon": "1h",
    })
    fr = pd.DataFrame({
        "ticker":             ["BTC", "BTC"],
        "timestamp":          pd.to_datetime(
            ["2024-01-01 00:00:00", "2024-01-01 00:00:00"], utc=True),
        "forward_log_return": [0.01, 0.05],
    })
    merged = ec._signals_with_forward_returns(signals, fr, horizon="1h")
    # Duplicates de-duplicated (kept last) so the merge stays 1:1.
    assert len(merged) == 1
    assert merged["forward_log_return"].iloc[0] == pytest.approx(0.05)


# ---------------------------------------------------------------------------
# Buy-and-hold and benchmark lift
# ---------------------------------------------------------------------------

def test_buy_and_hold_creates_row_per_cost_level():
    fr = pd.DataFrame({
        "ticker":   ["BTC"] * 10 + ["ETH"] * 10,
        "timestamp": list(pd.date_range("2024-01-01", periods=10, tz="UTC")) * 2,
        "forward_log_return": list(np.linspace(0.001, 0.01, 10)) * 2,
    })
    cfg = {**CFG_DEFAULT, "transaction_cost_bps": [0, 5, 10]}
    bh = ec.buy_and_hold_benchmark(["BTC", "ETH"], fr, cfg, horizon="1d")
    assert {0.0, 5.0, 10.0}.issubset(set(bh["cost_bps"].astype(float)))


def test_attach_benchmark_lifts_adds_lift_columns():
    base = pd.DataFrame({
        "horizon":   ["1d", "1d", "1d"],
        "set_id":    ["B1", "S1", "C1"],
        "cost_bps":  [0.0, 0.0, 0.0],
        "sentiment_model": ["-", "vader", "vader"],
        "sharpe":    [0.3, 0.8, 0.5],
        "net_cumulative_simple_return": [0.05, 0.20, 0.12],
    })
    out = ec.attach_benchmark_lifts(base, benchmark_ids=["B1"])
    assert "sharpe_lift_vs_b1" in out.columns
    sub = out.set_index("set_id")
    assert sub.loc["S1", "sharpe_lift_vs_b1"] == pytest.approx(0.5)
    assert sub.loc["C1", "sharpe_lift_vs_b1"] == pytest.approx(0.2)


# ---------------------------------------------------------------------------
# Config loader
# ---------------------------------------------------------------------------

def test_load_backtest_config_defaults_when_file_absent(tmp_path):
    cfg = ec.load_backtest_config(tmp_path / "nope.yaml")
    assert cfg["portfolio_mode"] == "high_low"
    assert cfg["long_quantile"]  == 0.8
    assert cfg["short_quantile"] == 0.2


def test_load_backtest_config_reads_real_file(tmp_path):
    import yaml
    cfg_path = tmp_path / "bt.yaml"
    cfg_path.write_text(yaml.safe_dump({
        "long_quantile": 0.9,
        "short_quantile": 0.1,
        "transaction_cost_bps": [0, 25],
    }))
    cfg = ec.load_backtest_config(cfg_path)
    assert cfg["long_quantile"]  == 0.9
    assert cfg["short_quantile"] == 0.1
    assert cfg["transaction_cost_bps"] == [0, 25]


# ---------------------------------------------------------------------------
# Threshold backtest with flat zone
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Regression: multi-set / multi-sentiment / multi-family signals all survive
# the forward-return join. The pre-fix dedup collapsed (ticker, ts) across
# groups and only the last-loaded set/sentiment reached economic_performance.csv.
# ---------------------------------------------------------------------------

def _multi_group_signals():
    rng = np.random.default_rng(0)
    ts = pd.date_range("2024-01-01", periods=10, tz="UTC", freq="D")
    tickers = ["BTC", "ETH", "XRP", "SOL", "ADA"]
    rows = []
    groups = [
        ("B1",  "-",         "per_asset",   "-",       "fixed"),
        ("S1",  "vader",     "per_asset",   "-",       "fixed"),
        ("S2",  "cryptobert", "per_asset",  "-",       "fixed"),
        ("C1",  "vader",     "per_asset",   "-",       "fixed"),
        ("M1",  "vader",     "panel_logit", "pooled",  "fixed"),
        ("M1",  "vader",     "panel_logit", "pooled",  "hpo_brier"),
    ]
    for sid, sm, mt, pm, hv in groups:
        for t in ts:
            for tk in tickers:
                rows.append({
                    "timestamp":      t,
                    "ticker":         tk,
                    "probability":    float(rng.uniform(0.1, 0.9)),
                    "prediction":     int(rng.integers(0, 2)),
                    "target":         int(rng.integers(0, 2)),
                    "set_id":         sid,
                    "sentiment_model": sm,
                    "model_type":     mt,
                    "panel_mode":     pm,
                    "hpo_variant":    hv,
                    "horizon":        "1d",
                    "category":       "sentiment",
                    "label":          "test",
                })
    return pd.DataFrame(rows), tickers, ts


def _forward_returns_for(tickers, ts):
    rng = np.random.default_rng(7)
    rows = []
    for tk in tickers:
        for t in ts:
            rows.append({"ticker": tk, "timestamp": t,
                          "forward_log_return": float(rng.normal(0, 0.02))})
    return pd.DataFrame(rows)


def test_summarize_high_low_preserves_every_set_sentiment_family():
    """Pre-fix dedup keyed only on (ticker, ts) — that collapsed every group
    but the last-loaded one. Make sure all six groups now appear."""
    signals, tickers, ts = _multi_group_signals()
    fr = _forward_returns_for(tickers, ts)
    out = ec.summarize_high_low_backtest(signals, fr, CFG_DEFAULT, horizon="1d")
    assert not out.empty
    seen = set(map(tuple, out[["set_id", "sentiment_model", "model_type",
                                "panel_mode", "hpo_variant"]]
                                .drop_duplicates().to_records(index=False)))
    expected = {
        ("B1", "-", "per_asset", "-", "fixed"),
        ("S1", "vader", "per_asset", "-", "fixed"),
        ("S2", "cryptobert", "per_asset", "-", "fixed"),
        ("C1", "vader", "per_asset", "-", "fixed"),
        ("M1", "vader", "panel_logit", "pooled", "fixed"),
        ("M1", "vader", "panel_logit", "pooled", "hpo_brier"),
    }
    assert seen == expected, (
        f"missing groups: {expected - seen}, unexpected groups: {seen - expected}"
    )


def test_signals_with_forward_returns_does_not_collapse_groups():
    """Direct unit test of the merge — every (set, ts, ticker) signal row
    must survive when forward returns have a single unique (ticker, ts)."""
    signals, tickers, ts = _multi_group_signals()
    fr = _forward_returns_for(tickers, ts)
    merged = ec._signals_with_forward_returns(signals, fr, horizon="1d")
    # Same number of rows as input signals (no row dropped silently).
    assert len(merged) == len(signals)
    # forward_log_return matched on every row.
    assert merged["forward_log_return"].notna().all()
    # Every group still represented.
    assert (merged.groupby(["set_id", "sentiment_model", "model_type",
                            "panel_mode", "hpo_variant"]).size() > 0).all()


# ---------------------------------------------------------------------------
# economic_group_diagnostics
# ---------------------------------------------------------------------------

def test_economic_group_diagnostics_one_row_per_group_ok():
    signals, tickers, ts = _multi_group_signals()
    fr = _forward_returns_for(tickers, ts)
    diag = ec.economic_group_diagnostics(signals, fr, CFG_DEFAULT, horizon="1d")
    # 6 distinct groups → 6 rows.
    assert len(diag) == 6
    assert (diag["status"] == "ok").all()
    for col in ("horizon", "set_id", "sentiment_model", "model_type",
                "panel_mode", "hpo_variant", "category", "label",
                "n_signal_rows", "n_unique_timestamps", "n_unique_tickers",
                "n_forward_return_rows", "n_joined_rows",
                "n_joined_non_null_forward_returns",
                "n_portfolio_periods", "status", "reason"):
        assert col in diag.columns


def test_economic_group_diagnostics_no_forward_returns():
    signals, _, _ = _multi_group_signals()
    diag = ec.economic_group_diagnostics(
        signals, pd.DataFrame(), CFG_DEFAULT, horizon="1d",
    )
    assert (diag["status"] == "skip_no_forward_returns").all()
    assert (diag["n_joined_non_null_forward_returns"] == 0).all()
    assert len(diag) == 6


def test_economic_group_diagnostics_zero_forward_matches():
    """Signals exist but the FR ticker universe doesn't overlap."""
    signals, _, ts = _multi_group_signals()
    fr = _forward_returns_for(["DOGE", "LTC"], ts)
    diag = ec.economic_group_diagnostics(signals, fr, CFG_DEFAULT, horizon="1d")
    assert (diag["status"] == "skip_zero_forward_matches").all()
    assert (diag["n_joined_non_null_forward_returns"] == 0).all()


def test_economic_group_diagnostics_no_portfolio_periods():
    """A single-ticker group can never form both legs unless allow_single_leg."""
    ts = pd.date_range("2024-01-01", periods=5, tz="UTC", freq="D")
    rows = [{
        "timestamp": t, "ticker": "BTC", "probability": 0.9,
        "prediction": 1, "target": 1,
        "set_id": "B1", "sentiment_model": "-", "horizon": "1d",
        "model_type": "per_asset", "panel_mode": "-", "hpo_variant": "fixed",
    } for t in ts]
    signals = pd.DataFrame(rows)
    fr = pd.DataFrame({
        "ticker": "BTC", "timestamp": ts,
        "forward_log_return": np.linspace(0.001, 0.01, 5),
    })
    diag = ec.economic_group_diagnostics(signals, fr, CFG_DEFAULT, horizon="1d")
    assert len(diag) == 1
    row = diag.iloc[0]
    assert row["status"] == "skip_no_portfolio_periods"
    assert row["n_joined_non_null_forward_returns"] == 5
    assert row["n_portfolio_periods"] == 0


def test_economic_group_diagnostics_missing_required_column():
    df = pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=3, tz="UTC"),
        "ticker":    "BTC",
        # no probability column
        "set_id": "B1", "sentiment_model": "-", "horizon": "1d",
        "model_type": "per_asset", "panel_mode": "-", "hpo_variant": "fixed",
    })
    fr = pd.DataFrame({"ticker": "BTC",
                        "timestamp": pd.date_range("2024-01-01", periods=3, tz="UTC"),
                        "forward_log_return": [0.0, 0.0, 0.0]})
    diag = ec.economic_group_diagnostics(df, fr, CFG_DEFAULT, horizon="1d")
    assert len(diag) == 1
    assert diag.iloc[0]["status"] == "skip_missing_required_columns"
    assert "probability" in diag.iloc[0]["reason"]


# ---------------------------------------------------------------------------
# Threshold backtest with flat zone
# ---------------------------------------------------------------------------

def test_threshold_backtest_skips_when_one_leg_empty():
    # 3 timestamps × 3 assets. Threshold 0.6:
    #   t0: BTC=0.95 (long), ETH=0.55 (flat), XRP=0.05 (short)  → traded
    #   t1: BTC=0.99 (long), ETH=0.50 (flat), XRP=0.50 (flat)   → empty short → skip
    #   t2: BTC=0.99 (long), ETH=0.55 (flat), XRP=0.01 (short)  → traded
    ts = pd.date_range("2024-01-01", periods=3, tz="UTC")
    rows = []
    for i, (pb, pe, px) in enumerate([(0.95, 0.55, 0.05),
                                       (0.99, 0.50, 0.50),
                                       (0.99, 0.55, 0.01)]):
        for tk, p in (("BTC", pb), ("ETH", pe), ("XRP", px)):
            rows.append({
                "timestamp": ts[i], "ticker": tk, "probability": p,
                "forward_log_return": 0.01 if tk == "BTC" else -0.01,
                "prediction": 1 if p > 0.5 else 0, "target": 1,
                "set_id": "S1", "sentiment_model": "vader", "horizon": "1d",
                "category": "sentiment", "label": "test",
            })
    df = pd.DataFrame(rows)
    fr = df[["ticker", "timestamp", "forward_log_return"]]
    cfg = {**CFG_DEFAULT, "thresholds": [0.6], "transaction_cost_bps": [0]}
    out = ec.summarize_high_low_threshold_backtest(df, fr, cfg, horizon="1d")
    assert not out.empty
    row = out.iloc[0]
    # Only 2 of 3 timestamps yielded a portfolio.
    assert int(row["n_periods"]) == 2
    assert row["coverage"] == pytest.approx(2 / 3)
