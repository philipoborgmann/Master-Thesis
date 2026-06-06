"""Tests for the matched-benchmark comparison (C/CV/M_k vs B_k).

Covers spec items 1-8 for the post-refactor mapping ``X_k → B_k`` across the
combined CryptoBERT (``C*``), combined VADER (``CV*``) and multi (``M*``)
families. Also exercises matched-key joining, lift-sign symmetry, the
missing-benchmark sentinel, and the CSV/Excel output from ``evaluate-signals``.
"""
from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from thesis_pipeline.evaluation import incremental as inc
from thesis_pipeline.evaluation import evaluate_signals as eval_main


# ---------------------------------------------------------------------------
# Items 1-4 — mapping is explicit and exhaustive for C1-C6
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("set_id,expected", [
    *[(f"C{k}", f"B{k}") for k in range(1, 7)],
    *[(f"CV{k}", f"B{k}") for k in range(1, 7)],
    *[(f"M{k}", f"B{k}") for k in range(1, 7)],
])
def test_matched_economic_benchmark_mapping(set_id, expected):
    assert inc.matched_economic_benchmark_for_combined(set_id) == expected


def test_matched_economic_benchmark_unknown_returns_none():
    assert inc.matched_economic_benchmark_for_combined("Z9") is None
    # Pure sentiment / benchmark families never appear on the model side of
    # the incremental comparison, so the mapping yields None.
    assert inc.matched_economic_benchmark_for_combined("S3") is None
    assert inc.matched_economic_benchmark_for_combined("SV3") is None
    assert inc.matched_economic_benchmark_for_combined("B4") is None


# ---------------------------------------------------------------------------
# Synthetic signal frames for items 5-7
# ---------------------------------------------------------------------------

def _signal_frame(*, set_id, sentiment_model, model_type="per_asset",
                  panel_mode="-", hpo_variant="fixed",
                  train_window_mode="expanding",
                  train_window_timestamps=None,
                  rolling_window_days=None,
                  n=120, acc=0.55, seed=0):
    rng = np.random.default_rng(seed)
    ts = pd.date_range("2024-01-01", periods=n, freq="D", tz="UTC")
    target = rng.integers(0, 2, n)
    flip = rng.random(n) > acc
    pred = np.where(flip, 1 - target, target).astype(int)
    prob = np.where(pred == 1, rng.uniform(0.55, 0.85, n),
                                rng.uniform(0.15, 0.45, n))
    return pd.DataFrame({
        "timestamp": ts, "ticker": "BTC",
        "target": target, "prediction": pred, "probability": prob.astype(float),
        "set_id": set_id, "sentiment_model": sentiment_model,
        "horizon": "1d", "category": _category_for(set_id),
        "model_type": model_type, "panel_mode": panel_mode,
        "hpo_variant": hpo_variant, "hpo_enabled": hpo_variant != "fixed",
        "hpo_objective": "brier_score" if hpo_variant != "fixed" else "-",
        "train_window_mode": train_window_mode,
        "train_window_timestamps": train_window_timestamps,
        "rolling_window_days": rolling_window_days,
    })


def _category_for(set_id: str) -> str:
    if set_id.startswith("B"):  return "benchmark"
    if set_id.startswith("SV"): return "sentiment_vader"
    if set_id.startswith("S"):  return "sentiment_cryptobert"
    if set_id.startswith("CV"): return "combined_vader"
    if set_id.startswith("C"):  return "combined_cryptobert"
    if set_id.startswith("M"):  return "multi"
    return "other"


# ---------------------------------------------------------------------------
# Item 5 — comparisons are matched on (horizon, timestamp, ticker)
# ---------------------------------------------------------------------------

def test_comparison_matches_on_timestamp_and_ticker():
    # The combined frame's first 60 timestamps overlap the benchmark; the
    # benchmark covers only the first 60 days. n_matched must be 60, not 120.
    combined = _signal_frame(set_id="C2", sentiment_model="cryptobert", n=120,
                              acc=0.70, seed=11)
    bench    = _signal_frame(set_id="B2", sentiment_model="-",     n=60,
                              acc=0.55, seed=12)
    out = inc.incremental_sentiment_value_table(
        pd.concat([combined, bench], ignore_index=True))
    row = out[(out["set_id"] == "C2") & (out["sentiment_model"] == "cryptobert")].iloc[0]
    assert row["benchmark_set_id"] == "B2"
    assert int(row["n_matched"]) == 60


# ---------------------------------------------------------------------------
# Item 6 — positive accuracy_lift means combined model > benchmark
# ---------------------------------------------------------------------------

def test_accuracy_lift_sign_follows_combined_better():
    # Combined model is constructed to be more accurate than the benchmark.
    combined = _signal_frame(set_id="C1", sentiment_model="cryptobert",
                              acc=0.75, seed=1, n=200)
    bench    = _signal_frame(set_id="B1", sentiment_model="-",
                              acc=0.55, seed=1, n=200)
    # Align target so the matched-subset comparison is fair (both rows share τ).
    bench["target"] = combined["target"].values
    bench["prediction"] = np.where(
        np.random.default_rng(7).random(len(bench)) > 0.55,
        1 - bench["target"], bench["target"]).astype(int)
    out = inc.incremental_sentiment_value_table(
        pd.concat([combined, bench], ignore_index=True))
    row = out.iloc[0]
    assert row["status"] == "ok"
    assert row["model_accuracy"] > row["benchmark_accuracy"]
    assert row["accuracy_lift"] > 0
    # Symmetry of "positive = combined is better" for the other metrics.
    assert row["brier_improvement"] == pytest.approx(
        row["benchmark_brier"] - row["model_brier"])
    assert row["log_loss_improvement"] == pytest.approx(
        row["benchmark_log_loss"] - row["model_log_loss"])
    assert row["f1_lift"] == pytest.approx(row["model_f1"] - row["benchmark_f1"])
    # McNemar c counts cases where the model is correct & benchmark wrong;
    # with a real lift c >> b.
    assert int(row["mcnemar_c"]) > int(row["mcnemar_b"])
    assert row["interpretation_flag"] in {"improved", "improved_significant"}


# ---------------------------------------------------------------------------
# Item 7 — missing benchmark emits sentinel row, no crash
# ---------------------------------------------------------------------------

def test_missing_benchmark_emits_sentinel_row():
    """``C3 → B3``. With no B3 frame in the signals, the comparison yields a
    ``status='missing_benchmark'`` row instead of crashing.
    """
    only_combined = _signal_frame(set_id="C3", sentiment_model="cryptobert",
                                   acc=0.6, seed=3)
    captured = []
    out = inc.incremental_sentiment_value_table(
        only_combined,
        warn_missing=lambda h, sid, sm, b: captured.append((h, sid, sm, b)),
    )
    assert len(out) == 1
    row = out.iloc[0]
    assert row["status"] == "missing_benchmark"
    assert row["benchmark_set_id"] == "B3"
    assert int(row["n_matched"]) == 0
    # Lift values are NaN, not zero, to signal "not computed".
    assert pd.isna(row["accuracy_lift"])
    assert captured == [("1d", "C3", "cryptobert", "B3")]


# ---------------------------------------------------------------------------
# Rolling-window matching — combined and benchmark must agree on window key
# ---------------------------------------------------------------------------

def test_rolling_window_must_match_for_comparison():
    """A combined rolling_fixed run must NOT borrow an expanding benchmark."""
    combined = _signal_frame(set_id="CV2", sentiment_model="vader",
                              train_window_mode="rolling_fixed",
                              train_window_timestamps=30, n=120, seed=4)
    expanding_bench = _signal_frame(set_id="B2", sentiment_model="-",
                                    train_window_mode="expanding",
                                    n=120, seed=5)
    out = inc.incremental_sentiment_value_table(
        pd.concat([combined, expanding_bench], ignore_index=True))
    assert (out["status"] == "missing_benchmark").all()

    # Same rolling configuration → comparison succeeds.
    matched_bench = _signal_frame(set_id="B2", sentiment_model="-",
                                  train_window_mode="rolling_fixed",
                                  train_window_timestamps=30, n=120, seed=6)
    out2 = inc.incremental_sentiment_value_table(
        pd.concat([combined, matched_bench], ignore_index=True))
    assert (out2["status"] == "ok").any()
    ok = out2[out2["status"] == "ok"].iloc[0]
    assert ok["train_window_mode"] == "rolling_fixed"
    assert int(ok["train_window_timestamps"]) == 30


# ---------------------------------------------------------------------------
# Item 8 — evaluate-signals writes the CSV (and the Excel sheet)
# ---------------------------------------------------------------------------

def _build_synth_signal_frame(*, set_id, sentiment_model, acc, n=80, seed=0):
    rng = np.random.default_rng(seed)
    ts = pd.date_range("2024-01-01", periods=n, freq="D", tz="UTC")
    target = rng.integers(0, 2, n)
    flip = rng.random(n) > acc
    pred = np.where(flip, 1 - target, target).astype(int)
    prob = np.where(pred == 1, rng.uniform(0.55, 0.85, n),
                                rng.uniform(0.15, 0.45, n))
    return pd.DataFrame({
        "timestamp": ts, "ticker": "BTC", "target": target,
        "prediction": pred, "probability": prob.astype(float),
        "set_id": set_id, "sentiment_model": sentiment_model, "horizon": "1d",
    })


@pytest.fixture
def signals_env(tmp_path, monkeypatch):
    signals_root = tmp_path / "Outputs" / "Signals"
    (signals_root / "1d").mkdir(parents=True)
    raw_1d = tmp_path / "Data" / "Raw" / "Price" / "1d"
    raw_1d.mkdir(parents=True)
    # Combined CV2 / VADER + matched B2 (per the new C/CV/M_k → B_k map).
    _build_synth_signal_frame(set_id="CV2", sentiment_model="vader",
                               acc=0.70, n=80, seed=1).to_parquet(
        signals_root / "1d" / "CV2_vader.parquet", index=False)
    _build_synth_signal_frame(set_id="B2", sentiment_model="-",
                               acc=0.55, n=80, seed=2).to_parquet(
        signals_root / "1d" / "B2.parquet", index=False)
    # Minimal feature_sets workbook so attach_feature_set_metadata works.
    fs = pd.DataFrame({
        "set_id":   ["B1", "B2", "CV2"],
        "category": ["benchmark", "benchmark", "combined_vader"],
        "sentiment_model": ["-", "-", "vader"],
        "label":    ["b", "lag", "cv2"],
        "feature_columns_comma_separated": [
            "__majority_class__", "log_return_t",
            "log_return_t,vader_title_score_mean,vader_bullishness_ratio",
        ],
    })
    fs_path = tmp_path / "feature_sets.xlsx"
    with pd.ExcelWriter(fs_path, engine="openpyxl") as w:
        fs.to_excel(w, sheet_name="feature_sets", index=False)
    from thesis_pipeline import config as cfg
    monkeypatch.setattr(cfg, "project_root", lambda: tmp_path)
    cfg.load_config.cache_clear()
    shutil.copytree(Path(__file__).resolve().parents[1] / "configs",
                    tmp_path / "configs")
    return {"root": tmp_path, "feature_sets": fs_path}


def test_evaluate_signals_writes_incremental_csv_and_sheet(signals_env):
    out_dir = signals_env["root"] / "Outputs" / "Evaluation"
    rc = eval_main.main([
        "--horizon", "1d", "--force",
        "--output-dir", str(out_dir),
        "--feature-config", str(signals_env["feature_sets"]),
        "--no-volatility", "--no-market-cap", "--no-economic",
        "--no-regime-mcnemar",
    ])
    assert rc == 0
    csv_path = out_dir / "incremental_sentiment_value.csv"
    assert csv_path.exists()
    df = pd.read_csv(csv_path)
    # The combined CV2 / VADER vs B2 comparison must be present and ok.
    row = df[(df["set_id"] == "CV2") & (df["sentiment_model"] == "vader")].iloc[0]
    assert row["benchmark_set_id"] == "B2"
    assert row["status"] == "ok"
    assert int(row["n_matched"]) > 0
    # Required column set from the spec.
    expected_cols = {
        "horizon", "set_id", "sentiment_model", "model_type", "panel_mode",
        "train_window_mode", "train_window_timestamps",
        "hpo_variant", "benchmark_set_id", "benchmark_sentiment_model",
        "n_matched",
        "model_accuracy", "benchmark_accuracy", "accuracy_lift",
        "model_balanced_accuracy", "benchmark_balanced_accuracy",
        "balanced_accuracy_lift",
        "model_brier", "benchmark_brier", "brier_improvement",
        "model_log_loss", "benchmark_log_loss", "log_loss_improvement",
        "model_f1", "benchmark_f1", "f1_lift",
        "mcnemar_b", "mcnemar_c", "mcnemar_stat", "mcnemar_p_value",
        "model_correct", "benchmark_correct", "interpretation_flag",
    }
    assert expected_cols <= set(df.columns)

    # The Excel report carries the new sheet.
    xlsx = out_dir / "signal_evaluation.xlsx"
    assert xlsx.exists()
    sheets = pd.ExcelFile(xlsx).sheet_names
    assert "incremental_sentiment_value" in sheets
