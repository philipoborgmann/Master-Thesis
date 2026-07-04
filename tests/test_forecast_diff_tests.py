"""Unit tests for the timestamp-level differential forecast tests (Part 2A)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from thesis_pipeline.evaluation import forecast_diff_tests as fdt
from thesis_pipeline.evaluation.preregistration import LOGLOSS_CLIP_EPS


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------

def _pred_frame(tickers, ts, y, p, pred=None):
    rows = []
    for tk in tickers:
        for i, t in enumerate(ts):
            yi = int(y[tk][i]); pi = float(p[tk][i])
            rows.append({
                "ticker": tk, "timestamp": t, "target": yi,
                "probability": pi,
                "prediction": int(pi >= 0.5) if pred is None else int(pred[tk][i]),
            })
    return pd.DataFrame(rows)


def _small_cfg(**over):
    cfg = {"n_boot": 500, "block_length": "auto", "hac_lag": "auto"}
    cfg.update(over)
    return cfg


# ---------------------------------------------------------------------------
# (i) dm_logloss_timestamp_test — known d_t, effect, SE, degenerate
# ---------------------------------------------------------------------------

def test_dm_logloss_effect_positive_when_sentiment_better():
    """Sentiment assigns probability closer to the truth → lower log
    loss → positive ll_effect. Per-timestamp noise keeps d_t non-
    degenerate."""
    T = 60
    ts = pd.date_range("2024-01-01", periods=T, freq="D", tz="UTC")
    rng = np.random.default_rng(0)
    y = {"BTC": rng.integers(0, 2, T)}
    # Sentiment: confident toward truth but with per-timestamp noise.
    noise = rng.uniform(0.0, 0.25, T)
    p_sent  = {"BTC": np.clip(np.where(y["BTC"] == 1, 0.80, 0.20)
                              + np.where(y["BTC"] == 1, -noise, noise), 0.05, 0.95)}
    p_bench = {"BTC": np.clip(0.50 + rng.normal(0, 0.03, T), 0.05, 0.95)}
    sent  = _pred_frame(["BTC"], ts, y, p_sent)
    bench = _pred_frame(["BTC"], ts, y, p_bench)
    out = fdt.dm_logloss_timestamp_test(bench, sent, _small_cfg())
    assert bool(out["test_valid"]) is True
    assert out["ll_effect"] > 0
    assert out["n_timestamps"] == T
    assert np.isfinite(out["ll_p_boot"])
    assert out["ll_p_value_raw"] == out["ll_p_boot"]
    assert out["dm_inference_primary"] == "moving_block_bootstrap"


def test_dm_logloss_degenerate_zero_variance():
    """Identical predictions → d_t ≡ 0 → degenerate path, no division by
    zero."""
    T = 40
    ts = pd.date_range("2024-01-01", periods=T, freq="D", tz="UTC")
    y = {"BTC": np.tile([0, 1], T // 2)}
    p = {"BTC": np.full(T, 0.55)}
    sent  = _pred_frame(["BTC"], ts, y, p)
    bench = _pred_frame(["BTC"], ts, y, p)   # identical → d_t = 0
    out = fdt.dm_logloss_timestamp_test(bench, sent, _small_cfg())
    assert out["test_valid"] is False
    assert out["skip_reason"] == "degenerate_zero_variance"
    assert out["dm_nested_caveat"] is True
    # No NaN blow-up from dividing by ~0.
    assert np.isnan(out["ll_se_hac"]) or out["ll_se_hac"] == 0 or np.isnan(out["ll_p_boot"])


def test_dm_logloss_hac_and_block_auto_values():
    T = 125
    ts = pd.date_range("2024-01-01", periods=T, freq="D", tz="UTC")
    rng = np.random.default_rng(1)
    y = {"BTC": rng.integers(0, 2, T)}
    p_sent  = {"BTC": np.clip(y["BTC"] * 0.6 + 0.2 + rng.normal(0, 0.05, T), 0.01, 0.99)}
    p_bench = {"BTC": np.full(T, 0.5)}
    out = fdt.dm_logloss_timestamp_test(
        _pred_frame(["BTC"], ts, y, p_bench),
        _pred_frame(["BTC"], ts, y, p_sent), _small_cfg())
    # auto lag = floor(4*(T/100)^(2/9)); auto block = round(T^(1/3))
    assert out["hac_lag"] == int(np.floor(4.0 * (T / 100.0) ** (2.0 / 9.0)))
    assert out["block_length"] == int(round(T ** (1.0 / 3.0)))


def test_dm_logloss_no_overlap():
    ts1 = pd.date_range("2024-01-01", periods=10, freq="D", tz="UTC")
    ts2 = pd.date_range("2025-01-01", periods=10, freq="D", tz="UTC")
    y = {"BTC": np.ones(10, dtype=int)}
    p = {"BTC": np.full(10, 0.6)}
    out = fdt.dm_logloss_timestamp_test(
        _pred_frame(["BTC"], ts1, y, p),
        _pred_frame(["BTC"], ts2, y, p), _small_cfg())
    assert out["test_valid"] is False
    assert out["skip_reason"] == "no_overlap"


# ---------------------------------------------------------------------------
# (ii) accuracy_timestamp_effect synthetic
# ---------------------------------------------------------------------------

def test_accuracy_effect_positive_when_sentiment_more_accurate():
    T = 50
    ts = pd.date_range("2024-01-01", periods=T, freq="D", tz="UTC")
    rng = np.random.default_rng(2)
    y = {"BTC": rng.integers(0, 2, T)}
    # Sentiment predicts the truth; benchmark always predicts 0.
    # Sentiment mostly right (per-timestamp variation), benchmark always 0.
    sent_pred = np.where(rng.random(T) < 0.85, y["BTC"], 1 - y["BTC"])
    sent = _pred_frame(["BTC"], ts, y, {"BTC": np.full(T, 0.5)},
                       pred={"BTC": sent_pred})
    bench = _pred_frame(["BTC"], ts, y, {"BTC": np.full(T, 0.5)},
                        pred={"BTC": np.zeros(T, dtype=int)})
    out = fdt.accuracy_timestamp_effect(bench, sent, _small_cfg())
    assert bool(out["test_valid"]) is True
    assert out["acc_effect"] > 0
    assert np.isfinite(out["acc_p_boot"])
    assert out["acc_p_value_raw"] == out["acc_p_boot"]


# ---------------------------------------------------------------------------
# (iii) N_t-weighted sanity: == pooled improvement on the same sample
# ---------------------------------------------------------------------------

def _pooled_logloss(y, p):
    p = np.clip(np.asarray(p, float), LOGLOSS_CLIP_EPS, 1 - LOGLOSS_CLIP_EPS)
    y = np.asarray(y, float)
    return float(np.mean(-(y * np.log(p) + (1 - y) * np.log(1 - p))))


def test_ll_effect_ntweighted_equals_pooled_logloss_improvement():
    """The N_t-weighted timestamp effect must equal the pooled log-loss
    improvement (benchmark − sentiment) on the identical matched sample."""
    T = 40
    tickers = ["BTC", "ETH", "SOL"]
    ts = pd.date_range("2024-01-01", periods=T, freq="D", tz="UTC")
    rng = np.random.default_rng(3)
    y = {tk: rng.integers(0, 2, T) for tk in tickers}
    p_sent  = {tk: np.clip(rng.uniform(0.2, 0.8, T), 0.01, 0.99) for tk in tickers}
    p_bench = {tk: np.clip(rng.uniform(0.2, 0.8, T), 0.01, 0.99) for tk in tickers}
    sent  = _pred_frame(tickers, ts, y, p_sent)
    bench = _pred_frame(tickers, ts, y, p_bench)
    out = fdt.dm_logloss_timestamp_test(bench, sent, _small_cfg())

    # Pooled improvement over the full (identical-coverage) sample.
    all_y = np.concatenate([y[tk] for tk in tickers])
    all_ps = np.concatenate([p_sent[tk] for tk in tickers])
    all_pb = np.concatenate([p_bench[tk] for tk in tickers])
    pooled_improvement = _pooled_logloss(all_y, all_pb) - _pooled_logloss(all_y, all_ps)
    assert out["ll_effect_ntweighted"] == pytest.approx(pooled_improvement, abs=1e-9)


def test_acc_effect_ntweighted_equals_pooled_accuracy_lift():
    T = 40
    tickers = ["BTC", "ETH"]
    ts = pd.date_range("2024-01-01", periods=T, freq="D", tz="UTC")
    rng = np.random.default_rng(4)
    y = {tk: rng.integers(0, 2, T) for tk in tickers}
    sent_pred  = {tk: rng.integers(0, 2, T) for tk in tickers}
    bench_pred = {tk: rng.integers(0, 2, T) for tk in tickers}
    sent  = _pred_frame(tickers, ts, y, {tk: np.full(T, 0.5) for tk in tickers}, pred=sent_pred)
    bench = _pred_frame(tickers, ts, y, {tk: np.full(T, 0.5) for tk in tickers}, pred=bench_pred)
    out = fdt.accuracy_timestamp_effect(bench, sent, _small_cfg())

    all_y = np.concatenate([y[tk] for tk in tickers])
    all_sp = np.concatenate([sent_pred[tk] for tk in tickers])
    all_bp = np.concatenate([bench_pred[tk] for tk in tickers])
    pooled_lift = float(np.mean((all_sp == all_y).astype(int)
                                - (all_bp == all_y).astype(int)))
    assert out["acc_effect_ntweighted"] == pytest.approx(pooled_lift, abs=1e-9)


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------

def test_bootstrap_is_deterministic():
    T = 45
    ts = pd.date_range("2024-01-01", periods=T, freq="D", tz="UTC")
    rng = np.random.default_rng(5)
    y = {"BTC": rng.integers(0, 2, T)}
    p_sent  = {"BTC": np.clip(y["BTC"] * 0.5 + 0.25 + rng.normal(0, 0.05, T), 0.01, 0.99)}
    p_bench = {"BTC": np.clip(0.5 + rng.normal(0, 0.03, T), 0.01, 0.99)}
    a = fdt.dm_logloss_timestamp_test(_pred_frame(["BTC"], ts, y, p_bench),
                                      _pred_frame(["BTC"], ts, y, p_sent), _small_cfg())
    b = fdt.dm_logloss_timestamp_test(_pred_frame(["BTC"], ts, y, p_bench),
                                      _pred_frame(["BTC"], ts, y, p_sent), _small_cfg())
    assert a["ll_se_boot"] == b["ll_se_boot"]
    assert a["ll_p_boot"] == b["ll_p_boot"]
