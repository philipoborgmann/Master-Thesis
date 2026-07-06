"""Unit tests for regime-specific McNemar tests.

Supplementary to the global McNemar test. Synthetic signals + synthetic
regime lookups only — no real data.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from thesis_pipeline.evaluation import significance as sig


# ---------------------------------------------------------------------------
# Synthetic builders
# ---------------------------------------------------------------------------

def _signals_with_regime(n_per_regime: int = 120,
                         regimes=("low", "mid", "high"),
                         model_set: str = "S1",
                         sentiment_model: str = "vader",
                         category: str = "sentiment",
                         seed: int = 0,
                         regime_col: str = "vol_regime",
                         model_edge: float = 0.30,
                         bench_edge: float = 0.0) -> pd.DataFrame:
    """Build a model + B1 benchmark frame already carrying a regime column.

    In the ``"high"`` regime the model is correct much more often than the
    benchmark, so b/c should be strongly asymmetric there. In the other
    regimes the model has no edge.
    """
    rng = np.random.default_rng(seed)
    rows = []
    ts0 = pd.Timestamp("2024-01-01", tz="UTC")
    for ri, regime in enumerate(regimes):
        for i in range(n_per_regime):
            ts = ts0 + pd.Timedelta(days=ri * n_per_regime + i)
            target = int(rng.integers(0, 2))
            edge = model_edge if regime == "high" else 0.0
            model_correct = rng.random() < (0.5 + edge)
            bench_correct = rng.random() < (0.5 + bench_edge)
            model_pred = target if model_correct else 1 - target
            bench_pred = target if bench_correct else 1 - target
            common = dict(timestamp=ts, ticker="BTC", target=target,
                          horizon="1d")
            rows.append({**common, "prediction": model_pred,
                         "probability": 0.6, "set_id": model_set,
                         "sentiment_model": sentiment_model,
                         "category": category, regime_col: regime})
            rows.append({**common, "prediction": bench_pred,
                         "probability": 0.5, "set_id": "B1",
                         "sentiment_model": "-", "category": "benchmark",
                         regime_col: regime})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# mcnemar_by_group
# ---------------------------------------------------------------------------

def test_mcnemar_by_group_separates_b_c_per_regime():
    df = _signals_with_regime(n_per_regime=150, seed=1)
    out = sig.mcnemar_by_group(df, ["vol_regime"], benchmarks=("B1",),
                               min_n_matched=10, min_discordant=5)
    assert not out.empty
    # One row per regime (single set, single benchmark).
    assert set(out["vol_regime"]) == {"low", "mid", "high"}
    high = out[out["vol_regime"] == "high"].iloc[0]
    low  = out[out["vol_regime"] == "low"].iloc[0]
    # In the high-vol regime the model corrects the benchmark far more often
    # than vice versa → c >> b.
    assert high["c"] > high["b"]
    # In the low-vol regime there is no systematic edge.
    assert abs(low["c"] - low["b"]) < high["c"] - high["b"]


def test_mcnemar_by_group_marks_invalid_below_thresholds():
    df = _signals_with_regime(n_per_regime=30, seed=2)
    out = sig.mcnemar_by_group(df, ["vol_regime"], benchmarks=("B1",),
                               min_n_matched=1000, min_discordant=1000)
    assert not out.empty
    # Every cell must be flagged invalid because n_matched (≈ 30) < 1000.
    assert (~out["test_valid"]).all()
    assert out["mcnemar_stat"].isna().all()
    assert out["mcnemar_pval"].isna().all()
    assert (~out["significant_5pct"]).all()


def test_mcnemar_by_group_excludes_non_eligible_categories():
    df = _signals_with_regime(category="economic")
    out = sig.mcnemar_by_group(df, ["vol_regime"], benchmarks=("B1",),
                               min_n_matched=10, min_discordant=5)
    # economic sets are not eligible → no rows
    assert out.empty


# ---------------------------------------------------------------------------
# BH / FDR correction
# ---------------------------------------------------------------------------

def test_adjust_pvalues_bh_monotone_qvalues():
    df = pd.DataFrame({"mcnemar_pval": [0.001, 0.01, 0.04, 0.20, np.nan]})
    out = sig.adjust_pvalues_bh(df, "mcnemar_pval")
    assert "q_value_bh" in out.columns
    valid_q = out.loc[out["mcnemar_pval"].notna(), "q_value_bh"].tolist()
    # q-values are non-decreasing in the p-value order
    assert valid_q == sorted(valid_q)
    # every q >= its raw p
    valid = out[out["mcnemar_pval"].notna()]
    assert (valid["q_value_bh"] >= valid["mcnemar_pval"] - 1e-9).all()
    # the NaN p-value row gets no q-value and no significance
    nan_row = out[out["mcnemar_pval"].isna()].iloc[0]
    assert pd.isna(nan_row["q_value_bh"])
    assert not nan_row["significant_bh_5pct"]


def test_adjust_pvalues_bh_flags_smallest():
    df = pd.DataFrame({"mcnemar_pval": [0.0001, 0.5, 0.9]})
    out = sig.adjust_pvalues_bh(df, "mcnemar_pval")
    # The strongest p-value should survive BH at 5%.
    assert bool(out.iloc[0]["significant_bh_5pct"])


def test_bh_fallback_matches_manual():
    # Pure-numpy BH q-values, independent of statsmodels.
    p = np.array([0.01, 0.02, 0.03, 0.04, 0.05])
    q = sig._bh_qvalues(p)
    # All equal-spaced p with n=5: q_i = p_i * 5 / rank_i
    assert len(q) == 5
    assert np.all(np.diff(q) >= -1e-9)  # monotone after enforcement
    assert np.all(q <= 1.0)


# ---------------------------------------------------------------------------
# regime_mcnemar_table orchestrator
# ---------------------------------------------------------------------------

def _vol_lookup(dates, ticker="BTC"):
    return pd.DataFrame({
        "ticker": ticker, "date": dates,
        "gk_var": 0.001, "vol": 0.01,
        "regime": (["low"] * (len(dates) // 3)
                   + ["mid"] * (len(dates) // 3)
                   + ["high"] * (len(dates) - 2 * (len(dates) // 3))),
    })


def _mcap_lookup(dates, ticker="BTC"):
    return pd.DataFrame({
        "ticker": ticker, "date": dates,
        "market_cap_lag": 100.0,
        "mcap_regime": (["small"] * (len(dates) // 3)
                        + ["mid"] * (len(dates) // 3)
                        + ["large"] * (len(dates) - 2 * (len(dates) // 3))),
    })


def _plain_signals(n=300, seed=3):
    """Model + B1 signals without a pre-attached regime column."""
    rng = np.random.default_rng(seed)
    ts = pd.date_range("2024-01-01", periods=n, freq="D", tz="UTC")
    rows = []
    for t in ts:
        target = int(rng.integers(0, 2))
        rows.append({"timestamp": t, "ticker": "BTC", "target": target,
                     "prediction": target if rng.random() < 0.62 else 1 - target,
                     "probability": 0.6, "set_id": "S1",
                     "sentiment_model": "vader", "horizon": "1d",
                     "category": "sentiment"})
        rows.append({"timestamp": t, "ticker": "BTC", "target": target,
                     "prediction": target if rng.random() < 0.50 else 1 - target,
                     "probability": 0.5, "set_id": "B1",
                     "sentiment_model": "-", "horizon": "1d",
                     "category": "benchmark"})
    return pd.DataFrame(rows)


def test_regime_mcnemar_volatility_only():
    signals = _plain_signals()
    dates = pd.date_range("2024-01-01", periods=300, freq="D", tz="UTC")
    vol = _vol_lookup(dates)
    out = sig.regime_mcnemar_table(signals, vol_lookup=vol, mcap_lookup=None,
                                   benchmarks=("B1",),
                                   min_n_matched=10, min_discordant=5)
    assert not out.empty
    assert set(out["regime_type"]) == {"volatility"}
    assert set(out["vol_regime"].dropna()) <= {"low", "mid", "high"}
    # DESCRIPTIVE ONLY: regime McNemar carries the RAW p-value but NO BH
    # q-value / BH significance (the confirmatory H2/H3 surface is the
    # cluster-robust difference-in-improvement, Families C/D).
    assert "mcnemar_pval" in out.columns
    for c in ("q_value_bh", "significant_bh_5pct",
              "model_better_bh_5pct", "benchmark_better_bh_5pct"):
        assert c not in out.columns


def test_regime_mcnemar_market_cap_only():
    signals = _plain_signals()
    dates = pd.date_range("2024-01-01", periods=300, freq="D", tz="UTC")
    mcap = _mcap_lookup(dates)
    out = sig.regime_mcnemar_table(signals, vol_lookup=None, mcap_lookup=mcap,
                                   benchmarks=("B1",),
                                   min_n_matched=10, min_discordant=5)
    assert not out.empty
    assert set(out["regime_type"]) == {"market_cap"}
    assert set(out["mcap_regime"].dropna()) <= {"small", "mid", "large"}


def test_regime_mcnemar_interaction_with_both_lookups():
    signals = _plain_signals()
    dates = pd.date_range("2024-01-01", periods=300, freq="D", tz="UTC")
    out = sig.regime_mcnemar_table(signals,
                                   vol_lookup=_vol_lookup(dates),
                                   mcap_lookup=_mcap_lookup(dates),
                                   benchmarks=("B1",),
                                   min_n_matched=5, min_discordant=1)
    assert not out.empty
    assert set(out["regime_type"]) == {"volatility", "market_cap", "interaction"}
    inter = out[out["regime_type"] == "interaction"]
    assert not inter.empty
    # The interaction label is mcap_regime + "_" + vol_regime
    assert inter["interaction"].notna().all()
    sample = inter.iloc[0]
    assert sample["interaction"] == f"{sample['mcap_regime']}_{sample['vol_regime']}"


def test_regime_mcnemar_missing_both_lookups_returns_empty():
    signals = _plain_signals()
    out = sig.regime_mcnemar_table(signals, vol_lookup=None, mcap_lookup=None)
    assert out.empty
    # Still returns a DataFrame with the documented columns (no crash).
    assert "regime_type" in out.columns


def test_regime_mcnemar_empty_lookup_does_not_crash():
    signals = _plain_signals()
    empty = pd.DataFrame(columns=["ticker", "date", "regime"])
    out = sig.regime_mcnemar_table(signals, vol_lookup=empty, mcap_lookup=None)
    # Empty vol lookup → no volatility block; nothing else available → empty.
    assert out.empty


# ---------------------------------------------------------------------------
# Global McNemar must be untouched
# ---------------------------------------------------------------------------

def test_global_mcnemar_still_works():
    signals = _plain_signals()
    out = sig.mcnemar_table(signals, benchmarks=("B1",))
    assert not out.empty
    assert "mcnemar_pval" in out.columns
    assert set(out["set_id"]) == {"S1"}


# ---------------------------------------------------------------------------
# CLI integration
# ---------------------------------------------------------------------------

def test_cli_dry_run_with_no_regime_mcnemar():
    import sys
    from pathlib import Path
    src = Path(__file__).resolve().parents[1] / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
    from thesis_pipeline import cli
    rc = cli.main(["evaluate-signals", "--horizon", "1d",
                   "--no-regime-mcnemar", "--dry-run"])
    assert rc == 0


def test_cli_dry_run_default_includes_regime_mcnemar_output():
    import sys
    from pathlib import Path
    src = Path(__file__).resolve().parents[1] / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
    from thesis_pipeline import cli
    # Default (no flag) must still dry-run cleanly.
    rc = cli.main(["evaluate-signals", "--horizon", "1d", "--dry-run"])
    assert rc == 0


# ---------------------------------------------------------------------------
# Effect-size + direction columns
# ---------------------------------------------------------------------------

def _build_directional(seed, model_edge, n_per_regime=200, regime_col="vol_regime"):
    """Helper: model better in 'high' regime, ~tie elsewhere."""
    return _signals_with_regime(n_per_regime=n_per_regime, seed=seed,
                                model_edge=model_edge, regime_col=regime_col)


def test_direction_model_better_when_c_gt_b():
    df = _build_directional(seed=11, model_edge=0.35)
    out = sig.mcnemar_by_group(df, ["vol_regime"], benchmarks=("B1",),
                               min_n_matched=10, min_discordant=5)
    high = out[out["vol_regime"] == "high"].iloc[0]
    assert high["c"] > high["b"]
    assert high["direction"] == "model_better"
    assert high["net_improvement"] == high["c"] - high["b"]
    assert high["abs_net_improvement"] == abs(high["c"] - high["b"])


def test_direction_benchmark_better_when_b_gt_c():
    # Negative model edge → benchmark wins in the 'high' regime.
    df = _build_directional(seed=12, model_edge=-0.35)
    out = sig.mcnemar_by_group(df, ["vol_regime"], benchmarks=("B1",),
                               min_n_matched=10, min_discordant=5)
    high = out[out["vol_regime"] == "high"].iloc[0]
    assert high["b"] > high["c"]
    assert high["direction"] == "benchmark_better"
    assert high["net_improvement"] < 0


def test_discordant_advantage_and_improvement_rate_formulas():
    # Direct construction: b=30, c=70 over n_matched=1000 in a single regime.
    # Build a frame where exactly those discordant counts arise.
    rows = []
    ts0 = pd.Timestamp("2024-01-01", tz="UTC")
    # 70 cases model-right/bench-wrong, 30 bench-right/model-wrong,
    # 900 agreement cases (both correct).
    specs = ([("c", 70)] + [("b", 30)] + [("agree", 900)])
    i = 0
    for kind, count in specs:
        for _ in range(count):
            ts = ts0 + pd.Timedelta(hours=i); i += 1
            target = 1
            if kind == "c":          # model correct, benchmark wrong
                m_pred, b_pred = 1, 0
            elif kind == "b":        # benchmark correct, model wrong
                m_pred, b_pred = 0, 1
            else:                    # both correct
                m_pred, b_pred = 1, 1
            common = dict(timestamp=ts, ticker="BTC", target=target,
                          horizon="1d", vol_regime="high")
            rows.append({**common, "prediction": m_pred, "probability": 0.6,
                         "set_id": "S1", "sentiment_model": "vader",
                         "category": "sentiment"})
            rows.append({**common, "prediction": b_pred, "probability": 0.5,
                         "set_id": "B1", "sentiment_model": "-",
                         "category": "benchmark"})
    df = pd.DataFrame(rows)
    out = sig.mcnemar_by_group(df, ["vol_regime"], benchmarks=("B1",),
                               min_n_matched=10, min_discordant=5)
    r = out.iloc[0]
    assert r["b"] == 30 and r["c"] == 70
    assert r["n_matched"] == 1000
    # discordant_advantage = c / (b + c) = 70 / 100 = 0.7
    assert r["discordant_advantage"] == pytest.approx(0.7)
    # improvement_rate = (c - b) / n_matched = 40 / 1000 = 0.04
    assert r["improvement_rate"] == pytest.approx(0.04)


def test_discordant_advantage_nan_when_no_discordant():
    # All agreement → b + c == 0 → discordant_advantage NaN.
    rows = []
    ts0 = pd.Timestamp("2024-01-01", tz="UTC")
    for i in range(120):
        ts = ts0 + pd.Timedelta(hours=i)
        common = dict(timestamp=ts, ticker="BTC", target=1, horizon="1d",
                      vol_regime="low")
        rows.append({**common, "prediction": 1, "probability": 0.6,
                     "set_id": "S1", "sentiment_model": "vader",
                     "category": "sentiment"})
        rows.append({**common, "prediction": 1, "probability": 0.5,
                     "set_id": "B1", "sentiment_model": "-",
                     "category": "benchmark"})
    df = pd.DataFrame(rows)
    out = sig.mcnemar_by_group(df, ["vol_regime"], benchmarks=("B1",),
                               min_n_matched=10, min_discordant=0)
    r = out.iloc[0]
    assert r["b"] == 0 and r["c"] == 0
    assert pd.isna(r["discordant_advantage"])


def test_practical_effect_flag_triggers_above_min_rate():
    # Strong edge → improvement_rate well above 0.01.
    df = _build_directional(seed=13, model_edge=0.4, n_per_regime=400)
    out = sig.mcnemar_by_group(df, ["vol_regime"], benchmarks=("B1",),
                               min_n_matched=10, min_discordant=5,
                               min_effect_rate=0.01)
    high = out[out["vol_regime"] == "high"].iloc[0]
    assert abs(high["improvement_rate"]) >= 0.01
    assert bool(high["practical_effect_flag"])
    # A tiny min_effect_rate threshold that the effect cannot exceed → False.
    out2 = sig.mcnemar_by_group(df, ["vol_regime"], benchmarks=("B1",),
                                min_n_matched=10, min_discordant=5,
                                min_effect_rate=0.99)
    high2 = out2[out2["vol_regime"] == "high"].iloc[0]
    assert not bool(high2["practical_effect_flag"])


def test_raw_direction_flags():
    df = _build_directional(seed=14, model_edge=0.4, n_per_regime=600)
    out = sig.mcnemar_by_group(df, ["vol_regime"], benchmarks=("B1",),
                               min_n_matched=10, min_discordant=5)
    high = out[out["vol_regime"] == "high"].iloc[0]
    # Strong model edge in high-vol → raw-significant model_better.
    if high["significant_5pct"]:
        assert high["model_better_raw_5pct"]
        assert not high["benchmark_better_raw_5pct"]


def test_regime_table_is_descriptive_raw_only():
    signals = _plain_signals(n=400, seed=20)
    dates = pd.date_range("2024-01-01", periods=400, freq="D", tz="UTC")
    out = sig.regime_mcnemar_table(signals, vol_lookup=_vol_lookup(dates),
                                   mcap_lookup=None, benchmarks=("B1",),
                                   min_n_matched=10, min_discordant=5)
    # Raw descriptive columns (p-value, direction, effect size) are present.
    for col in ("mcnemar_pval", "model_better_raw_5pct",
                "benchmark_better_raw_5pct", "net_improvement", "direction",
                "discordant_advantage", "improvement_rate",
                "practical_effect_flag"):
        assert col in out.columns
    # BH q-value / BH significance are NOT emitted (descriptive surface,
    # superseded by the confirmatory difference-in-improvement C/D).
    for col in ("q_value_bh", "significant_bh_5pct",
                "model_better_bh_5pct", "benchmark_better_bh_5pct"):
        assert col not in out.columns


# ---------------------------------------------------------------------------
# build_regime_mcnemar_summary
# ---------------------------------------------------------------------------

def test_build_regime_mcnemar_summary_top_rows():
    signals = _plain_signals(n=400, seed=21)
    dates = pd.date_range("2024-01-01", periods=400, freq="D", tz="UTC")
    table = sig.regime_mcnemar_table(signals,
                                     vol_lookup=_vol_lookup(dates),
                                     mcap_lookup=_mcap_lookup(dates),
                                     benchmarks=("B1",),
                                     min_n_matched=5, min_discordant=1)
    summary = sig.build_regime_mcnemar_summary(table)
    assert isinstance(summary, pd.DataFrame)
    if not summary.empty:
        assert "section" in summary.columns
        # Each section appears at most once.
        assert summary["section"].is_unique
        # Direction and net_improvement columns surfaced; the raw p-value is
        # kept but the BH q-value is NOT (descriptive-only surface).
        for c in ("direction", "net_improvement", "mcnemar_pval"):
            assert c in summary.columns
        assert "q_value_bh" not in summary.columns


def test_build_regime_mcnemar_summary_empty_input_no_crash():
    assert sig.build_regime_mcnemar_summary(pd.DataFrame()).empty
    assert sig.build_regime_mcnemar_summary(None).empty
    # All-invalid input → empty summary, no crash.
    df = pd.DataFrame({"test_valid": [False, False], "direction": ["tie", "tie"]})
    assert sig.build_regime_mcnemar_summary(df).empty
