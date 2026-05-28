"""Tests for the evaluation pipeline's separation of model families.

The panel-logit family (``model_type='panel_logit'``, ``panel_mode`` in
{pooled, ticker_fixed_effects}) must never be pooled with — or benchmarked
against — the canonical per-asset family that merely shares (set_id,
sentiment_model). Old per-asset signal files (no model_type / panel_mode
columns) must keep evaluating as ``per_asset`` / ``-``.

Synthetic frames only — no real Data/ or Outputs/.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from thesis_pipeline.evaluation import (
    economic as ec, loading, metrics, significance as sig, thresholds,
)
from thesis_pipeline.evaluation import evaluate_signals as eval_main


# ---------------------------------------------------------------------------
# Synthetic builders
# ---------------------------------------------------------------------------

def _signal_block(set_id, sentiment_model, category, *, model_type, panel_mode,
                  acc=0.6, n=200, seed=0, include_cols=True):
    rng = np.random.default_rng(seed)
    ts = pd.date_range("2024-01-01", periods=n, freq="D", tz="UTC")
    target = rng.integers(0, 2, n)
    flip = rng.random(n) > acc
    pred = np.where(flip, 1 - target, target).astype(int)
    prob = np.where(pred == 1, rng.uniform(0.55, 0.9, n), rng.uniform(0.1, 0.45, n))
    d = {
        "timestamp": ts, "ticker": "BTC", "target": target,
        "prediction": pred, "probability": prob.astype(float),
        "set_id": set_id, "sentiment_model": sentiment_model,
        "horizon": "1d", "category": category,
    }
    if include_cols:
        d["model_type"] = model_type
        d["panel_mode"] = panel_mode
    return pd.DataFrame(d)


# ---------------------------------------------------------------------------
# A. loading defaults
# ---------------------------------------------------------------------------

def test_loading_defaults_for_old_signal_file(tmp_path):
    """A parquet with no model_type / panel_mode → per_asset / '-'."""
    old = _signal_block("S1", "vader", "sentiment", model_type="x",
                        panel_mode="x", include_cols=False)
    p = tmp_path / "S1_vader.parquet"
    old.to_parquet(p, index=False)
    loaded = loading.load_signal_file(p)
    assert loaded is not None
    assert (loaded["model_type"] == "per_asset").all()
    assert (loaded["panel_mode"] == "-").all()


def test_loading_preserves_panel_columns(tmp_path):
    panel = _signal_block("C2", "vader", "combined",
                          model_type="panel_logit", panel_mode="pooled")
    p = tmp_path / "C2_vader_panel_pooled.parquet"
    panel.to_parquet(p, index=False)
    loaded = loading.load_signal_file(p)
    assert (loaded["model_type"] == "panel_logit").all()
    assert (loaded["panel_mode"] == "pooled").all()


def test_loading_blank_values_fall_back_to_defaults(tmp_path):
    df = _signal_block("S1", "vader", "sentiment", model_type="",
                       panel_mode="")
    p = tmp_path / "S1_vader.parquet"
    df.to_parquet(p, index=False)
    loaded = loading.load_signal_file(p)
    assert (loaded["model_type"] == "per_asset").all()
    assert (loaded["panel_mode"] == "-").all()


# ---------------------------------------------------------------------------
# B. metrics: same (set_id, sentiment_model) in two families → two rows
# ---------------------------------------------------------------------------

def test_pooled_metrics_separates_families():
    per_asset = _signal_block("C2", "vader", "combined", model_type="per_asset",
                              panel_mode="-", seed=1)
    panel = _signal_block("C2", "vader", "combined", model_type="panel_logit",
                          panel_mode="pooled", seed=2)
    both = pd.concat([per_asset, panel], ignore_index=True)
    pooled = metrics.pooled_metrics_table(both)
    # Two distinct rows despite identical set_id / sentiment_model.
    assert len(pooled) == 2
    fams = set(zip(pooled["model_type"], pooled["panel_mode"]))
    assert fams == {("per_asset", "-"), ("panel_logit", "pooled")}
    # The model-family columns lead the table.
    assert list(pooled.columns[:3]) == ["horizon", "model_type", "panel_mode"]


def test_ensure_group_columns_defaults_missing():
    df = _signal_block("S1", "vader", "sentiment", model_type="x",
                       panel_mode="x", include_cols=False)
    out = metrics.ensure_group_columns(df)
    assert (out["model_type"] == "per_asset").all()
    assert (out["panel_mode"] == "-").all()


# ---------------------------------------------------------------------------
# C. McNemar matches benchmarks only within the same family
# ---------------------------------------------------------------------------

def _two_family_with_benchmarks(seed=0):
    pa_model = _signal_block("S1", "vader", "sentiment", model_type="per_asset",
                             panel_mode="-", acc=0.62, seed=seed)
    pa_bench = _signal_block("B1", "-", "benchmark", model_type="per_asset",
                             panel_mode="-", acc=0.50, seed=seed + 1)
    pl_model = _signal_block("S1", "vader", "sentiment", model_type="panel_logit",
                             panel_mode="pooled", acc=0.62, seed=seed + 2)
    pl_bench = _signal_block("B1", "-", "benchmark", model_type="panel_logit",
                             panel_mode="pooled", acc=0.50, seed=seed + 3)
    return pd.concat([pa_model, pa_bench, pl_model, pl_bench], ignore_index=True)


def test_mcnemar_matches_within_family():
    df = _two_family_with_benchmarks()
    out = sig.mcnemar_table(df, benchmarks=("B1",))
    assert not out.empty
    # One S1 row per family — never pooled across families.
    fam_rows = out[out["set_id"] == "S1"]
    assert len(fam_rows) == 2
    assert set(zip(fam_rows["model_type"], fam_rows["panel_mode"])) == {
        ("per_asset", "-"), ("panel_logit", "pooled")}


def test_mcnemar_no_cross_family_fallback_by_default():
    """Panel model with no same-family benchmark → no row (default)."""
    pa_model = _signal_block("S1", "vader", "sentiment", model_type="per_asset",
                             panel_mode="-", acc=0.62, seed=0)
    pa_bench = _signal_block("B1", "-", "benchmark", model_type="per_asset",
                             panel_mode="-", acc=0.50, seed=1)
    pl_model = _signal_block("S1", "vader", "sentiment", model_type="panel_logit",
                             panel_mode="pooled", acc=0.62, seed=2)
    # No panel benchmark in the frame.
    df = pd.concat([pa_model, pa_bench, pl_model], ignore_index=True)

    out = sig.mcnemar_table(df, benchmarks=("B1",))
    panel_rows = out[(out["set_id"] == "S1") & (out["model_type"] == "panel_logit")]
    assert panel_rows.empty, "panel model must NOT borrow the per-asset benchmark"

    # With the explicit opt-in, the panel model is allowed to fall back.
    out2 = sig.mcnemar_table(df, benchmarks=("B1",),
                             allow_cross_model_benchmark=True)
    panel_rows2 = out2[(out2["set_id"] == "S1") & (out2["model_type"] == "panel_logit")]
    assert not panel_rows2.empty


# ---------------------------------------------------------------------------
# E. Threshold lift respects the family boundary
# ---------------------------------------------------------------------------

def test_threshold_lift_within_family_only():
    df = _two_family_with_benchmarks()
    out = thresholds.threshold_lift_table(df, benchmarks=("B1",),
                                          thresholds=(0.50, 0.65))
    assert not out.empty
    # Both families represented; each lifted only against its own B1.
    assert {"per_asset", "panel_logit"} <= set(out["model_type"])


def test_threshold_lift_no_cross_family_default():
    pa_model = _signal_block("S1", "vader", "sentiment", model_type="per_asset",
                             panel_mode="-", seed=0)
    pa_bench = _signal_block("B1", "-", "benchmark", model_type="per_asset",
                             panel_mode="-", seed=1)
    pl_model = _signal_block("S1", "vader", "sentiment", model_type="panel_logit",
                             panel_mode="pooled", seed=2)
    df = pd.concat([pa_model, pa_bench, pl_model], ignore_index=True)
    out = thresholds.threshold_lift_table(df, benchmarks=("B1",),
                                          thresholds=(0.50,))
    panel = out[out["model_type"] == "panel_logit"]
    assert panel.empty


# ---------------------------------------------------------------------------
# F. Economic benchmark lift only within the same family; BUY_HOLD global
# ---------------------------------------------------------------------------

def test_attach_benchmark_lifts_within_family():
    base = pd.DataFrame({
        "horizon":   ["1d"] * 4,
        "set_id":    ["B1", "S1", "B1", "S1"],
        "model_type": ["per_asset", "per_asset", "panel_logit", "panel_logit"],
        "panel_mode": ["-", "-", "pooled", "pooled"],
        "cost_bps":  [0.0, 0.0, 0.0, 0.0],
        "sentiment_model": ["-", "vader", "-", "vader"],
        "sharpe":    [0.3, 0.8, 1.0, 1.7],
        "net_cumulative_simple_return": [0.05, 0.20, 0.10, 0.35],
    })
    out = ec.attach_benchmark_lifts(base, benchmark_ids=["B1"])
    idx = out.set_index(["set_id", "model_type"])
    # per_asset S1 lifts against per_asset B1: 0.8 - 0.3 = 0.5
    assert idx.loc[("S1", "per_asset"), "sharpe_lift_vs_b1"] == pytest.approx(0.5)
    # panel S1 lifts against panel B1: 1.7 - 1.0 = 0.7 (NOT 1.7 - 0.3)
    assert idx.loc[("S1", "panel_logit"), "sharpe_lift_vs_b1"] == pytest.approx(0.7)


def test_attach_benchmark_lifts_backward_compatible():
    """A frame without the family columns still gets per_asset lifts."""
    base = pd.DataFrame({
        "horizon":   ["1d", "1d"],
        "set_id":    ["B1", "S1"],
        "cost_bps":  [0.0, 0.0],
        "sharpe":    [0.3, 0.8],
        "net_cumulative_simple_return": [0.05, 0.20],
    })
    out = ec.attach_benchmark_lifts(base, benchmark_ids=["B1"])
    assert out.set_index("set_id").loc["S1", "sharpe_lift_vs_b1"] == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# H. Full-pipeline integration on a mixed-family signals tree
# ---------------------------------------------------------------------------

@pytest.fixture
def mixed_env(tmp_path, monkeypatch):
    import shutil
    from pathlib import Path
    from thesis_pipeline import config as cfg

    signals_root = tmp_path / "Outputs" / "Signals" / "1d"
    signals_root.mkdir(parents=True)
    raw_1d = tmp_path / "Data" / "Raw" / "Price" / "1d"
    raw_1d.mkdir(parents=True)

    # Per-asset (old-style, no family columns) and panel-logit signals that
    # share set_id / sentiment_model.
    _signal_block("B2", "-", "economic", model_type="per_asset", panel_mode="-",
                  include_cols=False, seed=1).to_parquet(
        signals_root / "B2.parquet", index=False)
    _signal_block("C2", "vader", "combined", model_type="per_asset",
                  panel_mode="-", include_cols=False, seed=2).to_parquet(
        signals_root / "C2_vader.parquet", index=False)
    _signal_block("B2", "-", "economic", model_type="panel_logit",
                  panel_mode="pooled", seed=3).to_parquet(
        signals_root / "B2_panel_pooled.parquet", index=False)
    _signal_block("C2", "vader", "combined", model_type="panel_logit",
                  panel_mode="pooled", seed=4).to_parquet(
        signals_root / "C2_vader_panel_pooled.parquet", index=False)

    feature_sets = pd.DataFrame({
        "set_id":   ["B2", "C2"],
        "category": ["economic", "combined"],
        "sentiment_model": ["-", "vader"],
        "label":    ["econ", "combo"],
        "feature_columns_comma_separated": ["log_return_t", "log_return_t,vader_mean"],
    })
    fs_path = tmp_path / "feature_sets.xlsx"
    with pd.ExcelWriter(fs_path, engine="openpyxl") as w:
        feature_sets.to_excel(w, sheet_name="feature_sets", index=False)

    monkeypatch.setattr(cfg, "project_root", lambda: tmp_path)
    cfg.load_config.cache_clear()
    shutil.copytree(Path(__file__).resolve().parents[1] / "configs",
                    tmp_path / "configs")
    return {"root": tmp_path, "feature_sets": fs_path}


def test_full_pipeline_separates_families(mixed_env):
    out_dir = mixed_env["root"] / "Outputs" / "Evaluation"
    rc = eval_main.main([
        "--horizon", "1d", "--force",
        "--output-dir", str(out_dir),
        "--feature-config", str(mixed_env["feature_sets"]),
        "--no-economic", "--no-volatility", "--no-market-cap",
        "--no-regime-mcnemar",
    ])
    assert rc == 0
    pooled = pd.read_csv(out_dir / "pooled_metrics.csv")
    for col in ("model_type", "panel_mode"):
        assert col in pooled.columns
    # C2/vader appears once per family — never pooled together.
    c2 = pooled[pooled["set_id"] == "C2"]
    assert len(c2) == 2
    assert set(zip(c2["model_type"], c2["panel_mode"])) == {
        ("per_asset", "-"), ("panel_logit", "pooled")}
    # Family columns lead the file.
    assert list(pooled.columns[:3]) == ["horizon", "model_type", "panel_mode"]
