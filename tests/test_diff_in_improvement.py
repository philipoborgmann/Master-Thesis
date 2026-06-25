"""Tests for the H2/H3 cluster-robust difference-in-improvement test
(``thesis_pipeline.evaluation.diff_in_improvement``).

Aufgabe 6.4 — H2/H3 are tests on the **difference** of mean improvement
between regimes. A regime-stratified McNemar is NOT a test of "the effect
in regime A differs from the effect in regime B"; the right inference
regresses ``d = 1(aug correct) − 1(ECON correct)`` on a regime dummy
with cluster-robust standard errors, clustering at minimum by ticker.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from thesis_pipeline.evaluation import diff_in_improvement as di
from thesis_pipeline.evaluation.incremental import MATCHED_ECONOMIC_BENCHMARK


# ---------------------------------------------------------------------------
# Observation-level improvement indicator
# ---------------------------------------------------------------------------

def _frame(set_id, sm, preds, target):
    n = len(preds)
    return pd.DataFrame({
        "timestamp":   pd.date_range("2024-01-01", periods=n, freq="D", tz="UTC"),
        "ticker":      ["BTC"] * n,
        "target":      target,
        "prediction":  preds,
        "set_id":      set_id,
        "sentiment_model": sm,
        "horizon":     "1d",
        "model_type":  "per_asset",
        "panel_mode":  "-",
        "hpo_variant": "fixed",
    })


def test_observation_improvement_indicator_values_are_minus_one_zero_plus_one():
    target = [0, 1, 0, 1, 0]
    aug    = _frame("ECON_VAD_F", "vader", [0, 1, 1, 1, 0], target)  # all correct except idx 2
    econ   = _frame("ECON",       "-",     [0, 0, 0, 1, 1], target)  # idx 1,4 wrong
    d = di.observation_improvement_indicator(aug, econ)
    # d_i = aug_correct[i] - econ_correct[i].
    # i=0: 1-1=0, i=1: 1-0=+1, i=2: 0-1=-1, i=3: 1-1=0, i=4: 1-0=+1
    assert d["d"].tolist() == [0, 1, -1, 0, 1]
    assert set(d["d"].unique()).issubset({-1, 0, 1})


def test_observation_improvement_indicator_inner_join_on_ts_ticker():
    """Mismatched ticker / timestamp → row dropped, NOT a NaN-filled join."""
    aug = _frame("ECON_VAD_F", "vader", [1, 1, 0], [1, 0, 0])
    econ = aug.copy()
    econ["ticker"] = "ETH"  # different ticker → no match
    d = di.observation_improvement_indicator(aug, econ)
    assert d.empty


# ---------------------------------------------------------------------------
# Cluster-robust OLS — numerical sanity vs statsmodels
# ---------------------------------------------------------------------------

def _toy_panel(n_clusters=10, n_per_cluster=30, beta=0.1, sigma=1.0, seed=0):
    rng = np.random.default_rng(seed)
    n = n_clusters * n_per_cluster
    x = (np.arange(n) % 2).astype(float)  # alternating treatment dummy
    cluster_id = np.repeat(np.arange(n_clusters), n_per_cluster)
    # Cluster-correlated noise so the cluster-robust SE > naive SE.
    eps_cluster = rng.normal(0, sigma, n_clusters).repeat(n_per_cluster)
    eps_obs     = rng.normal(0, sigma * 0.3, n)
    y = beta * x + eps_cluster + eps_obs
    return y, x, cluster_id


def test_cluster_robust_difference_in_improvement_recovers_treatment_effect():
    y, x, cl = _toy_panel(n_clusters=20, n_per_cluster=40, beta=0.25, seed=1)
    res = di.cluster_robust_difference_in_improvement(y, x, cl)
    # Point estimate close to the true β.
    assert abs(res["beta"] - 0.25) < 0.20
    # Cluster-robust SE is non-degenerate; t-stat finite.
    assert res["se_beta"] > 0
    assert np.isfinite(res["t_beta"])
    assert res["n_clusters"] == 20
    assert res["test_valid"] is True


def test_cluster_robust_difference_in_improvement_matches_statsmodels_numpy_fallback():
    """Numpy fallback must match statsmodels point estimate exactly."""
    y, x, cl = _toy_panel(n_clusters=15, n_per_cluster=50, beta=-0.15, seed=42)
    sm = di.cluster_robust_difference_in_improvement(
        y, x, cl, use_statsmodels=True)
    np_ = di.cluster_robust_difference_in_improvement(
        y, x, cl, use_statsmodels=False)
    assert sm["beta"] == pytest.approx(np_["beta"], rel=1e-6, abs=1e-10)
    # SEs differ slightly (statsmodels uses pivoted dof), but should agree
    # to a few percent.
    assert sm["se_beta"] == pytest.approx(np_["se_beta"], rel=0.10)


def test_cluster_robust_drops_unequal_treatment_values():
    """treatment ∉ {0, 1} (e.g. ``mid`` tercile encoded as NaN) must be
    dropped before the regression."""
    y = np.array([0.0, 1.0, -1.0, 0.0, 1.0, -1.0, 0.0, 1.0, 0.0, 1.0])
    x = np.array([1.0, 1.0, 1.0, np.nan, np.nan, 0.0, 0.0, 0.0, 0.0, 1.0])
    cl = np.array([1, 1, 2, 3, 3, 4, 4, 5, 5, 6])
    res = di.cluster_robust_difference_in_improvement(y, x, cl)
    # Only 8 obs survive (the two NaN-treatment rows are dropped).
    assert res["n_control"] + res["n_treatment"] == 8
    assert res["test_valid"] is True


def test_cluster_robust_handles_degenerate_input():
    """All-treatment or all-control → not testable, returns invalid result."""
    y = np.array([0.0, 1.0, 1.0])
    x = np.array([1.0, 1.0, 1.0])  # all-treatment, no control
    cl = np.array([1, 1, 2])
    res = di.cluster_robust_difference_in_improvement(y, x, cl)
    assert res["test_valid"] is False
    assert np.isnan(res["beta"])


# ---------------------------------------------------------------------------
# End-to-end H2/H3 difference-in-improvement table
# ---------------------------------------------------------------------------

def _multi_ticker_signal_frame(set_id, sm, n_per_ticker=60, base_acc=0.6, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for tk in ("BTC", "ETH", "SOL"):
        ts = pd.date_range("2024-01-01", periods=n_per_ticker, freq="D", tz="UTC")
        target = rng.integers(0, 2, n_per_ticker)
        flip = rng.random(n_per_ticker) > base_acc
        pred = np.where(flip, 1 - target, target).astype(int)
        rows.append(pd.DataFrame({
            "timestamp":  ts,
            "ticker":     tk,
            "target":     target,
            "prediction": pred,
            "probability": np.where(pred == 1, 0.7, 0.3),
            "set_id":     set_id,
            "sentiment_model": sm,
            "horizon":    "1d",
            "model_type": "per_asset",
            "panel_mode": "-",
            "hpo_variant": "fixed",
        }))
    return pd.concat(rows, ignore_index=True)


def _regime_lookup(tickers, dates, regime_for):
    rows = []
    for tk in tickers:
        for d in dates:
            rows.append({"ticker": tk, "date": d, "vol_regime": regime_for(tk, d)})
    return pd.DataFrame(rows)


def test_diff_in_improvement_table_produces_one_row_per_combined_set():
    """A multi-ticker fixture with ECON + ECON_VAD_F must yield one
    diff-in-improvement row, carrying the cluster-robust statistics."""
    aug  = _multi_ticker_signal_frame("ECON_VAD_F", "vader", base_acc=0.65, seed=1)
    econ = _multi_ticker_signal_frame("ECON",       "-",     base_acc=0.50, seed=2)
    # Make sure ECON sees the same target stream so the (ts, ticker) join works.
    econ["target"] = aug["target"].values

    dates = pd.date_range("2024-01-01", periods=60, freq="D", tz="UTC").normalize()
    look = _regime_lookup(
        tickers=("BTC", "ETH", "SOL"),
        dates=dates,
        # Make BTC always high-vol, ETH/SOL always low-vol — so each cluster
        # is fully in one regime, exposing the diff-in-improvement contrast.
        regime_for=lambda tk, d: "high" if tk == "BTC" else "low",
    )

    out = di.difference_in_improvement_table(
        signals=pd.concat([aug, econ], ignore_index=True),
        matched_benchmark=MATCHED_ECONOMIC_BENCHMARK,
        regime_lookup=look,
        regime_col="vol_regime",
        treatment_value="high",
        control_value="low",
        hypothesis_family="H2_volatility",
    )
    assert len(out) == 1
    row = out.iloc[0]
    assert row["set_id"] == "ECON_VAD_F"
    assert row["benchmark_set_id"] == "ECON"
    assert row["hypothesis_family"] == "H2_volatility"
    assert row["regime_col"] == "vol_regime"
    assert row["treatment_value"] == "high"
    assert row["control_value"] == "low"
    assert bool(row["test_valid"])
    assert row["n_clusters"] >= 2
    assert np.isfinite(row["diff_in_improvement"])
    assert np.isfinite(row["se_diff"])
    assert np.isfinite(row["p_value"])


def test_diff_in_improvement_table_drops_middle_tercile_observations():
    """If a regime label is neither treatment nor control, the row is
    excluded from the regression — the ``mid`` tercile must vanish."""
    aug  = _multi_ticker_signal_frame("ECON_VAD_F", "vader", base_acc=0.6, seed=3)
    econ = _multi_ticker_signal_frame("ECON",       "-",     base_acc=0.5, seed=4)
    econ["target"] = aug["target"].values
    dates = pd.date_range("2024-01-01", periods=60, freq="D", tz="UTC").normalize()

    # Half BTC/ETH/SOL days labeled "mid" — they must NOT enter the regression.
    def regime_for(tk, d):
        if (d.day % 3) == 0:
            return "mid"
        return "high" if tk == "BTC" else "low"

    look = _regime_lookup(("BTC", "ETH", "SOL"), dates, regime_for)
    out = di.difference_in_improvement_table(
        signals=pd.concat([aug, econ], ignore_index=True),
        matched_benchmark=MATCHED_ECONOMIC_BENCHMARK,
        regime_lookup=look,
        regime_col="vol_regime",
        treatment_value="high",
        control_value="low",
        hypothesis_family="H2_volatility",
    )
    assert len(out) == 1
    row = out.iloc[0]
    # 60 days × 3 tickers - mid days (~60/3 = ~20 per ticker, but only mid days
    # are dropped per ticker). Either way: row counts < 180.
    assert row["n_control"] + row["n_treatment"] < 180
    assert bool(row["test_valid"])


# ---------------------------------------------------------------------------
# Family-aware BH adjustment
# ---------------------------------------------------------------------------

def test_adjust_pvalues_bh_within_family_partitions_by_family():
    """H1 and H2 p-values must be BH-corrected separately, never pooled."""
    df = pd.DataFrame({
        "hypothesis_family": ["H1"] * 4 + ["H2"] * 4,
        # H1: one tiny p, three large; only the tiny one passes alone.
        # H2: same shape but with smaller p's overall.
        "p_value": [0.001, 0.40, 0.55, 0.80,
                    0.002, 0.005, 0.04, 0.20],
    })
    out = di.adjust_pvalues_bh_within_family(df)
    h1 = out[out["hypothesis_family"] == "H1"]
    h2 = out[out["hypothesis_family"] == "H2"]
    # H1: q-values are corrected within four tests.
    assert h1["q_value_bh"].notna().all()
    # The tiny H1 p survives BH 5 % even within its family.
    assert bool(h1.iloc[0]["significant_bh_5pct"])
    # H2: with three small p's BH 5 % accepts at least the smallest two.
    assert int(h2["significant_bh_5pct"].sum()) >= 2


def test_adjust_pvalues_bh_within_family_ignores_nan_pvalues():
    df = pd.DataFrame({
        "hypothesis_family": ["H1", "H1", "H1"],
        "p_value":           [0.001, np.nan, 0.05],
    })
    out = di.adjust_pvalues_bh_within_family(df)
    assert pd.isna(out.iloc[1]["q_value_bh"])
    assert not bool(out.iloc[1]["significant_bh_5pct"])
    # The two valid tests are corrected (n=2 in the family BH pool).
    assert pd.notna(out.iloc[0]["q_value_bh"])
    assert pd.notna(out.iloc[2]["q_value_bh"])


def test_h_hypothesis_families_constant_documented():
    """The canonical family labels are documented as a module constant."""
    assert di.H_HYPOTHESIS_FAMILIES == ("H1_incremental",
                                         "H2_volatility",
                                         "H3_market_cap")
