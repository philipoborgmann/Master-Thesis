"""Tests for the v4 nested matched-benchmark comparison (ECON_* vs ECON).

The v4 17-set registry collapses the v3 ``C_k → B_k`` / ``CV_k → B_k`` /
``M_k → B_k`` mapping into a single matched benchmark: every combined set
(``ECON_VAD_*`` / ``ECON_CBT_*``) shares the same ``ECON`` core and is
compared against ``ECON`` directly. This file pins:

* the mapping covers exactly the v4 combined family,
* matching joins on ``(horizon, timestamp, ticker)``,
* lift signs are consistent with "combined model beats ECON",
* a missing-benchmark row falls through to the sentinel status,
* the rolling-window guard still discriminates between expanding and
  ``rolling_fixed`` configurations,
* ``evaluate-signals`` writes the CSV + Excel sheet.

Plus the new ``NAIVE`` separate-reference contract (Aufgabe 6.3) — NAIVE
is NOT a feature set and is never returned by the matched-benchmark
mapper.
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
# Items 1-4 — mapping is explicit and exhaustive for ECON_VAD_* and ECON_CBT_*
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("set_id", [
    "ECON_VAD_L", "ECON_VAD_LD", "ECON_VAD_DA", "ECON_VAD_F",
    "ECON_CBT_L", "ECON_CBT_LD", "ECON_CBT_DA", "ECON_CBT_F",
])
def test_matched_economic_benchmark_mapping_v4(set_id):
    assert inc.matched_economic_benchmark_for_combined(set_id) == "ECON"


def test_matched_economic_benchmark_unknown_returns_none():
    assert inc.matched_economic_benchmark_for_combined("Z9") is None
    # Pure sentiment / benchmark / NAIVE families never appear on the model
    # side of the v4 incremental comparison.
    assert inc.matched_economic_benchmark_for_combined("SENT_CBT_F") is None
    assert inc.matched_economic_benchmark_for_combined("SENT_VAD_F") is None
    assert inc.matched_economic_benchmark_for_combined("ECON") is None
    assert inc.matched_economic_benchmark_for_combined("NAIVE") is None
    # And the removed v3 ids must NOT secretly retain a mapping entry.
    assert inc.matched_economic_benchmark_for_combined("C3") is None
    assert inc.matched_economic_benchmark_for_combined("CV3") is None
    assert inc.matched_economic_benchmark_for_combined("M1") is None
    assert inc.matched_economic_benchmark_for_combined("B1") is None


def test_matched_economic_benchmark_count_is_eight():
    """The v4 mapping must contain exactly 8 entries — 4 VADER + 4 CryptoBERT
    combined sets. Any drift means an ECON_* set was added or removed
    without an explicit decision."""
    assert len(inc.MATCHED_ECONOMIC_BENCHMARK) == 8
    assert set(inc.MATCHED_ECONOMIC_BENCHMARK.values()) == {"ECON"}


# ---------------------------------------------------------------------------
# Synthetic signal frames
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
        "hpo_objective": "log_loss" if hpo_variant != "fixed" else "-",
        "train_window_mode": train_window_mode,
        "train_window_timestamps": train_window_timestamps,
        "rolling_window_days": rolling_window_days,
    })


def _category_for(set_id: str) -> str:
    s = str(set_id)
    if s == "ECON":            return "benchmark"
    if s.startswith("SENT_VAD"):  return "sentiment_vader"
    if s.startswith("SENT_CBT"):  return "sentiment_cryptobert"
    if s.startswith("ECON_VAD"):  return "combined_vader"
    if s.startswith("ECON_CBT"):  return "combined_cryptobert"
    return "other"


# ---------------------------------------------------------------------------
# Item 5 — comparisons are matched on (horizon, timestamp, ticker)
# ---------------------------------------------------------------------------

def test_comparison_matches_on_timestamp_and_ticker():
    # The combined frame's first 60 timestamps overlap the benchmark; the
    # benchmark covers only the first 60 days. n_matched must be 60, not 120.
    combined = _signal_frame(set_id="ECON_CBT_F", sentiment_model="cryptobert",
                              n=120, acc=0.70, seed=11)
    bench    = _signal_frame(set_id="ECON", sentiment_model="-",
                              n=60,  acc=0.55, seed=12)
    out = inc.incremental_sentiment_value_table(
        pd.concat([combined, bench], ignore_index=True))
    row = out[(out["set_id"] == "ECON_CBT_F")
              & (out["sentiment_model"] == "cryptobert")].iloc[0]
    assert row["benchmark_set_id"] == "ECON"
    assert int(row["n_matched"]) == 60


# ---------------------------------------------------------------------------
# Item 6 — positive accuracy_lift means combined model > benchmark
# ---------------------------------------------------------------------------

def test_accuracy_lift_sign_follows_combined_better():
    combined = _signal_frame(set_id="ECON_VAD_L", sentiment_model="vader",
                              acc=0.75, seed=1, n=200)
    bench    = _signal_frame(set_id="ECON", sentiment_model="-",
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
    assert row["brier_improvement"] == pytest.approx(
        row["benchmark_brier"] - row["model_brier"])
    assert row["log_loss_improvement"] == pytest.approx(
        row["benchmark_log_loss"] - row["model_log_loss"])
    assert row["f1_lift"] == pytest.approx(row["model_f1"] - row["benchmark_f1"])
    assert int(row["mcnemar_c"]) > int(row["mcnemar_b"])
    assert row["interpretation_flag"] in {"improved", "improved_significant"}


# ---------------------------------------------------------------------------
# Item 7 — missing benchmark emits sentinel row, no crash
# ---------------------------------------------------------------------------

def test_missing_benchmark_emits_sentinel_row():
    """``ECON_CBT_F → ECON``. With no ECON frame in the signals, the
    comparison yields a ``status='missing_benchmark'`` row instead of
    crashing.
    """
    only_combined = _signal_frame(set_id="ECON_CBT_F",
                                   sentiment_model="cryptobert",
                                   acc=0.6, seed=3)
    captured = []
    out = inc.incremental_sentiment_value_table(
        only_combined,
        warn_missing=lambda h, sid, sm, b: captured.append((h, sid, sm, b)),
    )
    assert len(out) == 1
    row = out.iloc[0]
    assert row["status"] == "missing_benchmark"
    assert row["benchmark_set_id"] == "ECON"
    assert int(row["n_matched"]) == 0
    assert pd.isna(row["accuracy_lift"])
    assert captured == [("1d", "ECON_CBT_F", "cryptobert", "ECON")]


# ---------------------------------------------------------------------------
# Rolling-window matching — combined and benchmark must agree on window key
# ---------------------------------------------------------------------------

def test_rolling_window_must_match_for_comparison():
    """A combined rolling_fixed run must NOT borrow an expanding benchmark."""
    combined = _signal_frame(set_id="ECON_VAD_F", sentiment_model="vader",
                              train_window_mode="rolling_fixed",
                              train_window_timestamps=30, n=120, seed=4)
    expanding_bench = _signal_frame(set_id="ECON", sentiment_model="-",
                                    train_window_mode="expanding",
                                    n=120, seed=5)
    out = inc.incremental_sentiment_value_table(
        pd.concat([combined, expanding_bench], ignore_index=True))
    assert (out["status"] == "missing_benchmark").all()

    # Same rolling configuration → comparison succeeds.
    matched_bench = _signal_frame(set_id="ECON", sentiment_model="-",
                                  train_window_mode="rolling_fixed",
                                  train_window_timestamps=30, n=120, seed=6)
    out2 = inc.incremental_sentiment_value_table(
        pd.concat([combined, matched_bench], ignore_index=True))
    assert (out2["status"] == "ok").any()
    ok = out2[out2["status"] == "ok"].iloc[0]
    assert ok["train_window_mode"] == "rolling_fixed"
    assert int(ok["train_window_timestamps"]) == 30


# ---------------------------------------------------------------------------
# NAIVE separate-reference contract (Aufgabe 6.3)
# ---------------------------------------------------------------------------

def test_naive_label_constant_value():
    assert inc.NAIVE_REFERENCE_LABEL == "NAIVE"


def test_is_naive_signal_row_detects_naive_set_id():
    row = pd.Series({"set_id": "NAIVE"})
    assert inc.is_naive_signal_row(row)


def test_is_naive_signal_row_detects_panel_benchmark_model():
    row = pd.Series({
        "set_id": "ECON",
        "benchmark_model": "ticker_rolling_probability_with_pooled_fallback",
    })
    assert inc.is_naive_signal_row(row)


def test_is_naive_signal_row_rejects_econ_signal():
    row = pd.Series({"set_id": "ECON", "benchmark_model": ""})
    assert not inc.is_naive_signal_row(row)


def test_is_naive_signal_row_rejects_sentiment_signal():
    row = pd.Series({"set_id": "ECON_VAD_F", "benchmark_model": np.nan})
    assert not inc.is_naive_signal_row(row)


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
    # Combined ECON_VAD_F + matched ECON (v4 nested H1 pairing).
    _build_synth_signal_frame(set_id="ECON_VAD_F", sentiment_model="vader",
                               acc=0.70, n=80, seed=1).to_parquet(
        signals_root / "1d" / "ECON_VAD_F_vader.parquet", index=False)
    _build_synth_signal_frame(set_id="ECON", sentiment_model="-",
                               acc=0.55, n=80, seed=2).to_parquet(
        signals_root / "1d" / "ECON.parquet", index=False)
    # Minimal feature_sets workbook so attach_feature_set_metadata works.
    fs = pd.DataFrame({
        "set_id":   ["ECON", "ECON_VAD_F"],
        "category": ["benchmark", "combined_vader"],
        "sentiment_model": ["-", "vader"],
        "label":    ["econ", "econ-vader-full"],
        "feature_columns_comma_separated": [
            "log_return_t,cum_log_return_7d,cum_log_return_14d,cum_log_return_21d,realized_vol_14d,volume_diff,log_market_cap_lag1",
            "log_return_t,cum_log_return_7d,cum_log_return_14d,cum_log_return_21d,realized_vol_14d,volume_diff,log_market_cap_lag1,vader_title_score_mean,vader_title_score_std,vader_bullishness_ratio,log1p_post_count",
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
    row = df[(df["set_id"] == "ECON_VAD_F")
             & (df["sentiment_model"] == "vader")].iloc[0]
    assert row["benchmark_set_id"] == "ECON"
    assert row["status"] == "ok"
    assert int(row["n_matched"]) > 0
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
