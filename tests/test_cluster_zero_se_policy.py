"""Tests for the zero / invalid cluster-robust SE policy (commit 3 Section F).

The cluster-robust helper now distinguishes:

* SE NaN / negative / +inf      → ``inference_skip_reason="invalid_cluster_robust_se"``
* SE == 0 AND beta == 0          → exact null, ``t=0, p=1``, ``test_valid=True``
* SE == 0 AND beta != 0          → ``inference_skip_reason="zero_cluster_robust_se_nonzero_effect"``
* otherwise                       → ``t = beta / se``, ``p`` from Student-t(df=G−1)

Invalid rows MUST NOT enter BH correction.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from thesis_pipeline.evaluation import diff_in_improvement as di


# ---------------------------------------------------------------------------
# Helper — call the cluster-robust function directly with hand-built inputs
# ---------------------------------------------------------------------------

def _toy_panel(n_clusters=15, n_per_cluster=20, beta=0.0,
                sigma_cluster=0.0, sigma_obs=0.0, seed=0):
    """Build a deterministic panel where the residual variance is
    controllable down to exactly zero. With ``sigma_cluster == sigma_obs
    == 0`` the regression has zero residuals and the cluster-robust SE
    collapses to zero — the case we want to exercise."""
    rng = np.random.default_rng(seed)
    n = n_clusters * n_per_cluster
    x = (np.arange(n) % 2).astype(float)
    cluster_id = np.repeat(np.arange(n_clusters), n_per_cluster)
    eps_c = rng.normal(0, sigma_cluster, n_clusters).repeat(n_per_cluster) \
            if sigma_cluster > 0 else np.zeros(n)
    eps_o = rng.normal(0, sigma_obs, n) if sigma_obs > 0 else np.zeros(n)
    y = beta * x + eps_c + eps_o
    return y, x, cluster_id


# ---------------------------------------------------------------------------
# Section F — branch coverage
# ---------------------------------------------------------------------------

def test_zero_se_and_zero_beta_yields_exact_null():
    """Both effect and SE exactly zero → t=0, p=1, test_valid=True."""
    y, x, cl = _toy_panel(beta=0.0, sigma_cluster=0.0, sigma_obs=0.0)
    out = di.cluster_robust_difference_in_improvement(y, x, cl)
    assert out["test_valid"] is True
    assert out["t_beta"] == 0.0
    assert out["p_beta"] == 1.0
    assert out["inference_skip_reason"] == ""


def test_zero_se_and_nonzero_beta_is_skipped():
    """Non-zero point estimate paired with zero SE is degenerate — must
    be flagged as invalid rather than reported as p=1."""
    y, x, cl = _toy_panel(beta=0.5, sigma_cluster=0.0, sigma_obs=0.0)
    out = di.cluster_robust_difference_in_improvement(y, x, cl)
    assert out["test_valid"] is False
    assert np.isnan(out["t_beta"])
    assert np.isnan(out["p_beta"])
    assert out["inference_skip_reason"] == "zero_cluster_robust_se_nonzero_effect"


def test_negative_se_is_invalid(monkeypatch):
    """A patched OLS returning a negative SE must be flagged invalid
    rather than silently propagated as a real test."""
    real = di._ols_cluster_robust

    def fake_ols(y, X, cl, small_sample=True):
        res = real(y, X, cl, small_sample=small_sample)
        res["se"] = np.array([res["se"][0], -1.0])
        return res

    monkeypatch.setattr(di, "_ols_cluster_robust", fake_ols)
    y, x, cl = _toy_panel(beta=0.1, sigma_cluster=0.1, sigma_obs=0.1)
    out = di.cluster_robust_difference_in_improvement(
        y, x, cl, use_statsmodels=False,
    )
    assert out["test_valid"] is False
    assert out["inference_skip_reason"] == "invalid_cluster_robust_se"


def test_nan_se_is_invalid(monkeypatch):
    real = di._ols_cluster_robust

    def fake_ols(y, X, cl, small_sample=True):
        res = real(y, X, cl, small_sample=small_sample)
        res["se"] = np.array([res["se"][0], float("nan")])
        return res

    monkeypatch.setattr(di, "_ols_cluster_robust", fake_ols)
    y, x, cl = _toy_panel(beta=0.1, sigma_cluster=0.1, sigma_obs=0.1)
    out = di.cluster_robust_difference_in_improvement(
        y, x, cl, use_statsmodels=False,
    )
    assert out["test_valid"] is False
    assert out["inference_skip_reason"] == "invalid_cluster_robust_se"


def test_infinite_se_is_invalid(monkeypatch):
    real = di._ols_cluster_robust

    def fake_ols(y, X, cl, small_sample=True):
        res = real(y, X, cl, small_sample=small_sample)
        res["se"] = np.array([res["se"][0], float("inf")])
        return res

    monkeypatch.setattr(di, "_ols_cluster_robust", fake_ols)
    y, x, cl = _toy_panel(beta=0.1, sigma_cluster=0.1, sigma_obs=0.1)
    out = di.cluster_robust_difference_in_improvement(
        y, x, cl, use_statsmodels=False,
    )
    assert out["test_valid"] is False
    assert out["inference_skip_reason"] == "invalid_cluster_robust_se"


def test_normal_positive_se_is_valid():
    rng = np.random.default_rng(42)
    n_clusters, n_per = 20, 30
    n = n_clusters * n_per
    x = (np.arange(n) % 2).astype(float)
    cl = np.repeat(np.arange(n_clusters), n_per)
    eps_c = rng.normal(0, 0.3, n_clusters).repeat(n_per)
    eps_o = rng.normal(0, 0.5, n)
    y = 0.2 * x + eps_c + eps_o
    out = di.cluster_robust_difference_in_improvement(y, x, cl)
    assert out["test_valid"] is True
    assert out["inference_skip_reason"] == ""
    assert out["se_beta"] > 0
    assert np.isfinite(out["t_beta"])
    assert np.isfinite(out["p_beta"])


def test_statsmodels_and_numpy_share_validity_policy():
    """Both backends must agree on test_valid and the skip-reason classification
    for the same degenerate input."""
    y, x, cl = _toy_panel(beta=0.5, sigma_cluster=0.0, sigma_obs=0.0)
    sm = di.cluster_robust_difference_in_improvement(y, x, cl, use_statsmodels=True)
    np_ = di.cluster_robust_difference_in_improvement(y, x, cl, use_statsmodels=False)
    assert sm["test_valid"] == np_["test_valid"]
    assert sm["inference_skip_reason"] == np_["inference_skip_reason"]


# ---------------------------------------------------------------------------
# Invalid rows must NOT enter BH correction
# ---------------------------------------------------------------------------

def test_invalid_rows_excluded_from_bh_correction():
    df = pd.DataFrame({
        "hypothesis_family": ["H2_volatility"] * 4,
        "p_value":           [0.001, 0.04, float("nan"), float("nan")],
        "test_valid":        [True, True, False, False],
        "inference_skip_reason": ["", "",
                                   "zero_cluster_robust_se_nonzero_effect",
                                   "invalid_cluster_robust_se"],
    })
    adjusted = di.adjust_pvalues_bh_within_family(df)
    valid = adjusted[adjusted["test_valid"]]
    invalid = adjusted[~adjusted["test_valid"]]
    # Valid rows get q-values; invalid ones don't (BH pool ignores NaN p).
    assert valid["q_value_bh"].notna().all()
    assert invalid["q_value_bh"].isna().all()
    assert not invalid["significant_bh_5pct"].any()


# ---------------------------------------------------------------------------
# End-to-end: inference_skip_reason flows into difference_in_improvement.csv
# ---------------------------------------------------------------------------

def _e2e_signals(n=200, beta=0.0, seed=11):
    rng = np.random.default_rng(seed)
    ts = pd.date_range("2024-01-01", periods=n, freq="D", tz="UTC")
    target = rng.integers(0, 2, n)
    rows = []
    for sid, sm, edge in (("ECON", "-", 0.55),
                          ("ECON_VAD_F", "vader", 0.55 + beta)):
        pred = np.where(rng.random(n) > edge, 1 - target, target)
        rows.append(pd.DataFrame({
            "timestamp": ts, "ticker": "BTC",
            "target": target, "prediction": pred,
            "probability": rng.uniform(0.4, 0.6, n),
            "horizon": "1d", "set_id": sid,
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


def test_inference_skip_reason_present_in_output_columns():
    sig = _e2e_signals()
    dates = pd.date_range("2024-01-01", periods=200, tz="UTC")
    third = len(dates) // 3
    regimes = ["low"] * third + ["mid"] * third + ["high"] * (len(dates) - 2*third)
    lk = pd.DataFrame({
        "ticker": "BTC", "regime_available_at": dates,
        "vol_regime": regimes,
    })
    out = di.difference_in_improvement_table(
        signals=sig,
        matched_benchmark={"ECON_VAD_F": "ECON"},
        regime_lookup=lk,
        regime_col="vol_regime",
        treatment_value="high", control_value="low",
        hypothesis_family="H2_volatility",
    )
    assert "inference_skip_reason" in out.columns
