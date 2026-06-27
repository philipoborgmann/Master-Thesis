"""Tests for --strict-feature-set-ids preserving NAIVE (Aufgabe 6 follow-up C).

NAIVE is NOT in SET_ID_PATTERN (it is an explicit evaluation reference,
never a feature set) but a strict-filter evaluation run must keep
NAIVE-tagged signal rows alongside the registered v4 sets.
"""
from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from thesis_pipeline.evaluation import evaluate_signals as eval_main
from thesis_pipeline.evaluation.incremental import NAIVE_REFERENCE_LABEL
from thesis_pipeline.features.feature_registry import SET_ID_PATTERN


# ---------------------------------------------------------------------------
# Registry invariant — NAIVE is NOT a v4 feature set
# ---------------------------------------------------------------------------

def test_naive_not_in_registry_pattern():
    assert NAIVE_REFERENCE_LABEL not in set(SET_ID_PATTERN)


# ---------------------------------------------------------------------------
# Integration: --strict-feature-set-ids keeps NAIVE alongside the 17 sets
# ---------------------------------------------------------------------------

def _synthetic_signal(set_id, sentiment_model, accuracy_target=0.55,
                      n=120, ticker="BTC", horizon="1d", seed=0):
    rng = np.random.default_rng(seed)
    target = rng.integers(0, 2, size=n).astype(int)
    flip = rng.random(n) > accuracy_target
    pred = np.where(flip, 1 - target, target).astype(int)
    proba = np.where(pred == 1,
                     rng.uniform(0.55, 0.85, size=n),
                     rng.uniform(0.15, 0.45, size=n))
    return pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=n, freq="D", tz="UTC"),
        "ticker":    ticker,
        "target":    target, "prediction": pred,
        "probability": proba.astype(float),
        "set_id": set_id, "sentiment_model": sentiment_model,
        "horizon": horizon,
    })


@pytest.fixture
def strict_env(tmp_path, monkeypatch):
    signals_root = tmp_path / "Outputs" / "Signals" / "1d"
    signals_root.mkdir(parents=True)
    raw_1d = tmp_path / "Data" / "Raw" / "Price" / "1d"
    raw_1d.mkdir(parents=True)

    rng = np.random.default_rng(0)
    sigs = {
        ("ECON", "-"):           _synthetic_signal("ECON", "-",
                                                     accuracy_target=0.55, seed=1),
        ("ECON_VAD_F", "vader"): _synthetic_signal("ECON_VAD_F", "vader",
                                                     accuracy_target=0.62, seed=2),
        # NAIVE is NOT in the v4 registry but MUST survive strict filter.
        ("NAIVE", "-"):          _synthetic_signal("NAIVE", "-",
                                                     accuracy_target=0.50, seed=3),
        # A stale legacy id — must be dropped by strict filter.
        ("ZZ_STALE", "vader"):   _synthetic_signal("ZZ_STALE", "vader",
                                                     accuracy_target=0.5, seed=4),
    }
    for (sid, sm), df in sigs.items():
        name = f"{sid}_{sm}.parquet" if sm and sm != "-" else f"{sid}.parquet"
        df.to_parquet(signals_root / name, index=False)

    # OHLCV stub so the volatility step does not blow up.
    n_days = 200
    rng2 = np.random.default_rng(7)
    pd.DataFrame({
        "timestamp": pd.date_range("2023-12-01", periods=n_days, freq="D"),
        "open":  20000 + rng2.normal(0, 100, n_days),
        "high":  20100 + rng2.normal(0, 100, n_days),
        "low":   19900 + rng2.normal(0, 100, n_days),
        "close": 20000 + rng2.normal(0, 100, n_days),
        "volume": np.abs(rng2.normal(1000, 50, n_days)),
    }).to_parquet(raw_1d / "BTCUSDT_1d.parquet", index=False)

    # Minimal v4 feature_sets workbook.
    feature_sets = pd.DataFrame({
        "set_id":   ["ECON", "ECON_VAD_F"],
        "category": ["benchmark", "combined_vader"],
        "sentiment_model": ["-", "vader"],
        "label":    ["econ", "econ-vad-full"],
        "feature_columns_comma_separated": [
            "log_return_t,cum_log_return_7d",
            "log_return_t,vader_title_score_mean",
        ],
    })
    fs_path = tmp_path / "feature_sets.xlsx"
    with pd.ExcelWriter(fs_path, engine="openpyxl") as w:
        feature_sets.to_excel(w, sheet_name="feature_sets", index=False)

    from thesis_pipeline import config as cfg
    monkeypatch.setattr(cfg, "project_root", lambda: tmp_path)
    cfg.load_config.cache_clear()
    shutil.copytree(Path(__file__).resolve().parents[1] / "configs",
                    tmp_path / "configs")
    return {"root": tmp_path, "feature_sets": fs_path,
            "signals_root": signals_root}


def test_strict_filter_keeps_naive_and_v4_sets_drops_stale(strict_env):
    out_dir = strict_env["root"] / "Outputs" / "Evaluation_strict"
    rc = eval_main.main([
        "--horizon", "1d", "--force",
        "--output-dir", str(out_dir),
        "--feature-config", str(strict_env["feature_sets"]),
        "--strict-feature-set-ids",
        "--no-economic", "--no-market-cap",
    ])
    assert rc == 0
    pooled = pd.read_csv(out_dir / "pooled_metrics.csv")
    set_ids = set(pooled["set_id"].astype(str).unique())
    # Registered v4 sets present.
    assert {"ECON", "ECON_VAD_F"}.issubset(set_ids)
    # NAIVE survives even under --strict-feature-set-ids.
    assert "NAIVE" in set_ids
    # Stale legacy id dropped.
    assert "ZZ_STALE" not in set_ids
