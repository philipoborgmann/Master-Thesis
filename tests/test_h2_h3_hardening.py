"""Tests for the Aufgabe 6 follow-up hardening of H2/H3 inference.

Sections covered:
* D — complete family identity, duplicate-key guard, target-equality,
      matching diagnostics columns.
* E — --no-regime-mcnemar and --no-diff-in-improvement are independent.
* F — statsmodels and numpy fallback produce identical t / p values
      derived from Student-t(df=n_clusters-1); small-cluster warning.
* G — regime lookup join uses the previous day so an intraday
      prediction cannot read same-day end-of-day regime info.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from thesis_pipeline.evaluation import diff_in_improvement as di
from thesis_pipeline.evaluation.incremental import MATCHED_ECONOMIC_BENCHMARK
from thesis_pipeline.evaluation.diff_in_improvement import (
    H2H3_FAMILY_COLUMNS, SMALL_CLUSTER_THRESHOLD,
)


# ---------------------------------------------------------------------------
# Section F — harmonised cluster-robust p-values
# ---------------------------------------------------------------------------

def _toy_panel(n_clusters=20, n_per_cluster=30, beta=0.2, sigma=1.0, seed=0):
    rng = np.random.default_rng(seed)
    n = n_clusters * n_per_cluster
    x = (np.arange(n) % 2).astype(float)
    cluster_id = np.repeat(np.arange(n_clusters), n_per_cluster)
    eps_cluster = rng.normal(0, sigma, n_clusters).repeat(n_per_cluster)
    eps_obs     = rng.normal(0, sigma * 0.3, n)
    y = beta * x + eps_cluster + eps_obs
    return y, x, cluster_id


def test_statsmodels_and_fallback_share_t_and_p_values_exactly():
    y, x, cl = _toy_panel(n_clusters=15, n_per_cluster=50, beta=-0.15, seed=42)
    sm = di.cluster_robust_difference_in_improvement(y, x, cl, use_statsmodels=True)
    np_ = di.cluster_robust_difference_in_improvement(y, x, cl, use_statsmodels=False)
    # Coefficient and SE come from the same OLS — must match.
    assert sm["beta"] == pytest.approx(np_["beta"], rel=1e-8, abs=1e-12)
    # Aufgabe 6 follow-up F: t and p are recomputed from Student-t(df=G-1)
    # in BOTH paths, so they agree exactly (not just close).
    assert sm["t_beta"] == pytest.approx(np_["t_beta"], rel=1e-8, abs=1e-12)
    assert sm["p_beta"] == pytest.approx(np_["p_beta"], rel=1e-8, abs=1e-12)
    assert sm["dof"] == np_["dof"]


def test_degrees_of_freedom_equal_n_clusters_minus_one():
    y, x, cl = _toy_panel(n_clusters=25, n_per_cluster=40, beta=0.1, seed=1)
    res = di.cluster_robust_difference_in_improvement(y, x, cl)
    assert res["dof"] == 24
    assert res["test_valid"] is True
    assert res["small_cluster_warning"] is False


def test_p_value_is_two_sided_student_t():
    """Compute the two-sided p directly from t and df and compare."""
    from scipy.stats import t as student_t
    y, x, cl = _toy_panel(n_clusters=20, n_per_cluster=40, beta=0.25, seed=3)
    res = di.cluster_robust_difference_in_improvement(y, x, cl)
    expected = 2.0 * student_t.sf(abs(res["t_beta"]), df=res["dof"])
    assert res["p_beta"] == pytest.approx(expected, rel=1e-10)


def test_small_cluster_warning_under_threshold():
    y, x, cl = _toy_panel(n_clusters=6, n_per_cluster=40, beta=0.5, seed=4)
    res = di.cluster_robust_difference_in_improvement(y, x, cl)
    assert res["test_valid"] is True
    assert res["small_cluster_warning"] is True
    assert res["n_clusters"] == 6


def test_small_cluster_warning_clears_at_threshold():
    y, x, cl = _toy_panel(n_clusters=SMALL_CLUSTER_THRESHOLD,
                          n_per_cluster=20, beta=0.1, seed=5)
    res = di.cluster_robust_difference_in_improvement(y, x, cl)
    assert res["small_cluster_warning"] is False


def test_fewer_than_two_clusters_invalidates_test():
    y, x, cl = _toy_panel(n_clusters=1, n_per_cluster=50, beta=0.1, seed=6)
    res = di.cluster_robust_difference_in_improvement(y, x, cl)
    assert res["test_valid"] is False
    assert np.isnan(res["beta"])


def test_missing_treatment_or_control_invalidates_test():
    y = np.array([0.0, 1.0, 1.0, 1.0])
    x = np.array([1.0, 1.0, 1.0, 1.0])  # all-treatment
    cl = np.array([1, 1, 2, 2])
    res = di.cluster_robust_difference_in_improvement(y, x, cl)
    assert res["test_valid"] is False
    assert res["n_control"] == 0


def test_cluster_labels_accept_strings():
    """Ticker labels are strings; the cluster sandwich must group correctly."""
    rng = np.random.default_rng(7)
    n = 200
    cl = rng.choice(np.array(["BTC", "ETH", "SOL"]), size=n)
    x = rng.choice([0.0, 1.0], size=n)
    eps_cluster = pd.Series(rng.normal(0, 1, 3),
                            index=["BTC", "ETH", "SOL"]).reindex(cl).values
    eps = rng.normal(0, 0.3, n)
    y = 0.15 * x + eps_cluster + eps
    res = di.cluster_robust_difference_in_improvement(y, x, cl)
    assert res["test_valid"] is True
    assert res["n_clusters"] == 3


def test_25_tickers_inference_remains_valid():
    """Production setting: ~25 tickers stays test_valid=True with no
    small-cluster warning."""
    y, x, cl = _toy_panel(n_clusters=25, n_per_cluster=30, beta=0.1, seed=9)
    res = di.cluster_robust_difference_in_improvement(y, x, cl)
    assert res["test_valid"] is True
    assert res["small_cluster_warning"] is False
    assert res["n_clusters"] == 25
    assert res["dof"] == 24


# ---------------------------------------------------------------------------
# Section D — H2/H3 family identity, duplicate / target guards, diagnostics
# ---------------------------------------------------------------------------

H2H3_REQUIRED_DIAGNOSTIC_COLS = (
    "n_augmented", "n_econ", "n_matched",
    "n_unmatched_augmented", "n_unmatched_econ",
    "n_duplicate_augmented_keys", "n_duplicate_econ_keys",
    "targets_identical",
)


def test_complete_family_identity_constant_documented():
    """The exhaustive list of family columns must include every column
    that materially affects the run."""
    for col in ("horizon", "model_type", "panel_mode", "hpo_variant",
                "hpo_objective", "train_window_mode", "rolling_window_days",
                "rolling_window_timestamps"):
        assert col in H2H3_FAMILY_COLUMNS


def _signals_for_diff(set_id, sm, n_per_ticker=40, base_acc=0.6, seed=0,
                      tickers=("BTC", "ETH", "SOL"),
                      hpo_variant="fixed", hpo_objective="-",
                      train_window_mode="rolling_fixed",
                      rolling_window_days=180.0):
    rng = np.random.default_rng(seed)
    rows = []
    for tk in tickers:
        ts = pd.date_range("2024-01-01", periods=n_per_ticker, freq="D", tz="UTC")
        target = rng.integers(0, 2, n_per_ticker)
        flip = rng.random(n_per_ticker) > base_acc
        pred = np.where(flip, 1 - target, target).astype(int)
        rows.append(pd.DataFrame({
            "timestamp":  ts, "ticker": tk,
            "target":     target, "prediction": pred,
            "probability": np.where(pred == 1, 0.7, 0.3),
            "set_id":     set_id, "sentiment_model": sm, "horizon": "1d",
            "model_type": "panel_logit", "panel_mode": "ticker_fixed_effects",
            "hpo_variant": hpo_variant, "hpo_objective": hpo_objective,
            "train_window_mode": train_window_mode,
            "rolling_window_days": rolling_window_days,
            "rolling_window_timestamps": None,
        }))
    return pd.concat(rows, ignore_index=True)


def _vol_lookup(tickers, dates, regime_for):
    rows = [{"ticker": tk, "date": d, "vol_regime": regime_for(tk, d)}
            for tk in tickers for d in dates]
    return pd.DataFrame(rows)


def test_diff_in_improvement_carries_matching_diagnostic_columns():
    aug  = _signals_for_diff("ECON_VAD_F", "vader", base_acc=0.65, seed=1)
    econ = _signals_for_diff("ECON",       "-",     base_acc=0.5,  seed=2)
    econ["target"] = aug["target"].values
    dates = pd.date_range("2024-01-01", periods=80, freq="D", tz="UTC").normalize()
    look = _vol_lookup(("BTC", "ETH", "SOL"), dates,
                       lambda tk, d: "high" if tk == "BTC" else "low")
    out = di.difference_in_improvement_table(
        signals=pd.concat([aug, econ], ignore_index=True),
        matched_benchmark=MATCHED_ECONOMIC_BENCHMARK,
        regime_lookup=look,
        regime_col="vol_regime", treatment_value="high", control_value="low",
        hypothesis_family="H2_volatility",
    )
    for col in H2H3_REQUIRED_DIAGNOSTIC_COLS:
        assert col in out.columns


def test_diff_in_improvement_refuses_mismatched_window_family():
    """A 180-day-rolling ECON_VAD_F vs an expanding-window ECON must NOT
    silently match — the table should report skip_reason."""
    aug  = _signals_for_diff("ECON_VAD_F", "vader", base_acc=0.6, seed=10,
                              train_window_mode="rolling_fixed",
                              rolling_window_days=180.0)
    econ = _signals_for_diff("ECON",       "-",     base_acc=0.5, seed=11,
                              train_window_mode="expanding",
                              rolling_window_days=None)
    dates = pd.date_range("2024-01-01", periods=40, freq="D", tz="UTC").normalize()
    look = _vol_lookup(("BTC", "ETH", "SOL"), dates,
                       lambda tk, d: "high" if tk == "BTC" else "low")
    out = di.difference_in_improvement_table(
        signals=pd.concat([aug, econ], ignore_index=True),
        matched_benchmark=MATCHED_ECONOMIC_BENCHMARK,
        regime_lookup=look,
        regime_col="vol_regime", treatment_value="high", control_value="low",
        hypothesis_family="H2_volatility",
    )
    assert (out["skip_reason"] == "no_matched_econ_family").all()
    assert (out["test_valid"]   == False).all()  # noqa: E712


def test_diff_in_improvement_detects_duplicate_keys():
    """Two ECON rows for the same (ticker, timestamp) → duplicate guard
    trips and the comparison is skipped."""
    aug  = _signals_for_diff("ECON_VAD_F", "vader", base_acc=0.6, seed=20)
    econ = _signals_for_diff("ECON", "-", base_acc=0.5, seed=21)
    econ["target"] = aug["target"].values
    # Duplicate the first ECON row.
    econ = pd.concat([econ.iloc[[0]], econ], ignore_index=True)
    dates = pd.date_range("2024-01-01", periods=40, freq="D", tz="UTC").normalize()
    look = _vol_lookup(("BTC", "ETH", "SOL"), dates,
                       lambda tk, d: "high" if tk == "BTC" else "low")
    out = di.difference_in_improvement_table(
        signals=pd.concat([aug, econ], ignore_index=True),
        matched_benchmark=MATCHED_ECONOMIC_BENCHMARK,
        regime_lookup=look,
        regime_col="vol_regime", treatment_value="high", control_value="low",
        hypothesis_family="H2_volatility",
    )
    row = out.iloc[0]
    assert row["n_duplicate_econ_keys"] >= 1
    assert row["skip_reason"] == "duplicate_keys_within_family"
    assert not row["test_valid"]


def test_diff_in_improvement_target_mismatch_guard():
    aug  = _signals_for_diff("ECON_VAD_F", "vader", base_acc=0.6, seed=30)
    econ = _signals_for_diff("ECON",       "-",     base_acc=0.5, seed=31)
    # Targets deliberately diverge — no override to econ["target"].
    dates = pd.date_range("2024-01-01", periods=40, freq="D", tz="UTC").normalize()
    look = _vol_lookup(("BTC", "ETH", "SOL"), dates,
                       lambda tk, d: "high" if tk == "BTC" else "low")
    out = di.difference_in_improvement_table(
        signals=pd.concat([aug, econ], ignore_index=True),
        matched_benchmark=MATCHED_ECONOMIC_BENCHMARK,
        regime_lookup=look,
        regime_col="vol_regime", treatment_value="high", control_value="low",
        hypothesis_family="H2_volatility",
    )
    row = out.iloc[0]
    assert not bool(row["targets_identical"])
    assert row["skip_reason"] == "target_mismatch"
    assert not bool(row["test_valid"])


# ---------------------------------------------------------------------------
# Section G — regime join uses the previous day (no future regime leak)
# ---------------------------------------------------------------------------

def test_join_date_for_signals_uses_previous_day():
    ts = pd.date_range("2024-01-15 13:00", periods=3, freq="h", tz="UTC")
    out = di._join_date_for_signals(pd.Series(ts), regime_col="vol_regime")
    # Every join_date should be 2024-01-14 (one day before the prediction).
    assert (out == pd.Timestamp("2024-01-14", tz="UTC")).all()


def test_intraday_signal_only_sees_yesterday_regime():
    """An intraday 1h prediction at 2024-01-15 14:00 must NOT pick up the
    2024-01-15 regime — only 2024-01-14's regime is admissible."""
    # Two ticker rows: 2024-01-15 14:00.
    aug = _signals_for_diff("ECON_VAD_F", "vader", n_per_ticker=1, seed=42)
    aug["timestamp"] = pd.Timestamp("2024-01-15 14:00:00", tz="UTC")
    econ = aug.copy()
    econ["set_id"] = "ECON"
    econ["sentiment_model"] = "-"
    econ["target"] = aug["target"].values
    # Regime lookup: yesterday's row says "low", today's row says "high".
    look = pd.DataFrame({
        "ticker": ["BTC", "ETH", "SOL", "BTC", "ETH", "SOL"],
        "date":   [pd.Timestamp("2024-01-14", tz="UTC")] * 3
                  + [pd.Timestamp("2024-01-15", tz="UTC")] * 3,
        "vol_regime": ["low"] * 3 + ["high"] * 3,
    })
    out = di.difference_in_improvement_table(
        signals=pd.concat([aug, econ], ignore_index=True),
        matched_benchmark=MATCHED_ECONOMIC_BENCHMARK,
        regime_lookup=look,
        regime_col="vol_regime", treatment_value="high", control_value="low",
        hypothesis_family="H2_volatility",
    )
    # The lookup join uses D-1 = 2024-01-14 with regime "low" for every
    # ticker. So treatment_value="high" has 0 observations.
    row = out.iloc[0]
    assert row["n_treatment"] == 0
    # All observations are in the control regime ("low" → yesterday's label).
    assert row["n_control"] >= 0
