"""Tests for HPO output-variant separation.

Covers the second HPO hardening pass:

* tuned runs write variant-suffixed signal files (never overwriting fixed-C),
* old signals without HPO columns load as fixed / per_asset defaults,
* the evaluation never pools fixed vs HPO (or per-asset vs panel),
* benchmark comparisons (McNemar, threshold lift, economic lift) stay within
  the same horizon + model_type + panel_mode + hpo_variant family,
* the programmatic ``run(...)`` wrappers accept the HPO arguments.

Synthetic data only — real Data/ and Outputs/ are never touched (the full
pipeline runs inside ``tmp_path`` via ``monkeypatch.chdir``).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from thesis_pipeline.evaluation import (
    economic as ec, loading, metrics, significance as sig, thresholds,
)
from thesis_pipeline.modeling import run_models as rm
from thesis_pipeline.modeling import panel_logit as pl


# ---------------------------------------------------------------------------
# Synthetic full-pipeline repo
# ---------------------------------------------------------------------------

def _features(n_per_ticker=160, tickers=("BTC", "ETH", "SOL"), seed=0):
    rng = np.random.default_rng(seed)
    frames = []
    ts = pd.date_range("2024-01-01", periods=n_per_ticker, freq="D", tz="UTC")
    for tk in tickers:
        s = rng.normal(0, 1, n_per_ticker)
        e = rng.normal(0, 1, n_per_ticker)
        frames.append(pd.DataFrame({
            "timestamp": ts, "ticker": tk, "horizon": "1d",
            "target": (2.0 * s + 0.3 * e > 0).astype(int),
            "signal_feature": s, "noise_feature": rng.normal(0, 1, n_per_ticker),
        }))
    return pd.concat(frames, ignore_index=True)


@pytest.fixture
def repo(tmp_path, monkeypatch):
    (tmp_path / "Data" / "Final").mkdir(parents=True)
    (tmp_path / "Outputs" / "Signals").mkdir(parents=True)
    _features().to_parquet(tmp_path / "Data" / "Final" / "features_1d.parquet",
                           index=False)
    fs = pd.DataFrame({
        "set_id":   ["B1", "B2", "C3"],
        "category": ["benchmark", "economic", "combined"],
        "sentiment_model": ["-", "-", "cryptobert"],
        "label":    ["bench", "econ", "combo"],
        "Feature Columns (comma-separated)": [
            "__majority_class__", "signal_feature", "signal_feature,noise_feature",
        ],
    })
    with pd.ExcelWriter(tmp_path / "feature_sets.xlsx", engine="openpyxl") as w:
        fs.to_excel(w, sheet_name="feature_sets", index=False)
    monkeypatch.chdir(tmp_path)
    return tmp_path


def _sig_dir(repo):
    return repo / "Outputs" / "Signals" / "1d"


# ---------------------------------------------------------------------------
# 1 + 2. HPO naming differs from fixed and carries the objective suffix
# ---------------------------------------------------------------------------

def test_hpo_run_does_not_overwrite_fixed_filename(repo):
    rm.main(["--horizon", "1d", "--set-id", "C3", "--sentiment-model",
             "cryptobert", "--restart"])
    rm.main(["--horizon", "1d", "--set-id", "C3", "--sentiment-model",
             "cryptobert", "--tune-hyperparams", "--hpo-objective",
             "brier_score", "--restart"])
    fixed = _sig_dir(repo) / "C3_cryptobert.parquet"
    hpo = _sig_dir(repo) / "C3_cryptobert_hpo_brier.parquet"
    assert fixed.exists()
    assert hpo.exists()
    # Distinct files; the fixed one has no HPO objective.
    f = pd.read_parquet(fixed)
    h = pd.read_parquet(hpo)
    assert (f["hpo_variant"] == "fixed").all()
    assert (h["hpo_variant"] == "hpo_brier").all()
    assert (h["hpo_enabled"]).all()
    assert (f["hpo_objective"] == "-").all()
    assert (h["hpo_objective"] == "brier_score").all()


@pytest.mark.parametrize("objective,suffix", [
    ("brier_score", "hpo_brier"),
    ("log_loss",    "hpo_logloss"),
    ("accuracy",    "hpo_accuracy"),
])
def test_hpo_filename_objective_suffix(repo, objective, suffix):
    rm.main(["--horizon", "1d", "--set-id", "C3", "--sentiment-model",
             "cryptobert", "--tune-hyperparams", "--hpo-objective", objective,
             "--restart"])
    assert (_sig_dir(repo) / f"C3_cryptobert_{suffix}.parquet").exists()


# ---------------------------------------------------------------------------
# 9. Panel HPO filenames
# ---------------------------------------------------------------------------

def test_panel_hpo_filenames(repo):
    rm.main(["--horizon", "1d", "--set-id", "C3", "--sentiment-model",
             "cryptobert", "--model-type", "panel_logit", "--panel-mode",
             "pooled", "--tune-hyperparams", "--hpo-objective", "brier_score",
             "--restart"])
    assert (_sig_dir(repo) / "C3_cryptobert_panel_pooled_hpo_brier.parquet").exists()

    rm.main(["--horizon", "1d", "--set-id", "C3", "--sentiment-model",
             "cryptobert", "--model-type", "panel_logit", "--panel-mode",
             "ticker_fixed_effects", "--tune-hyperparams", "--hpo-objective",
             "brier_score", "--restart"])
    assert (_sig_dir(repo) / "C3_cryptobert_panel_ticker_fe_hpo_brier.parquet").exists()


def test_metrics_summary_has_hpo_variant_aggregate(repo):
    rm.main(["--horizon", "1d", "--set-id", "C3", "--sentiment-model",
             "cryptobert", "--tune-hyperparams", "--hpo-objective",
             "brier_score", "--restart"])
    m = pd.read_csv(repo / "Outputs" / "Signals" / "metrics_summary.csv")
    pooled = m[m["ticker"] == "pooled"]
    assert (pooled["hpo_variant"] == "hpo_brier").any()
    assert "best_C_mode" in m.columns and "hpo_status_counts" in m.columns


# ---------------------------------------------------------------------------
# 3. Old signals without HPO columns → fixed / per_asset defaults
# ---------------------------------------------------------------------------

def test_loading_old_signal_defaults_to_fixed(tmp_path):
    old = pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=5, tz="UTC"),
        "ticker": "BTC", "target": [1, 0, 1, 0, 1],
        "prediction": [1, 0, 1, 0, 1], "probability": [0.7, 0.3, 0.6, 0.4, 0.8],
        "set_id": "C3", "sentiment_model": "cryptobert", "horizon": "1d",
    })
    p = tmp_path / "C3_cryptobert.parquet"
    old.to_parquet(p, index=False)
    loaded = loading.load_signal_file(p)
    assert (loaded["hpo_enabled"] == False).all()  # noqa: E712
    assert (loaded["hpo_objective"] == "-").all()
    assert (loaded["hpo_variant"] == "fixed").all()
    assert (loaded["model_type"] == "per_asset").all()
    assert (loaded["panel_mode"] == "-").all()


# ---------------------------------------------------------------------------
# Synthetic signal blocks for evaluation-grouping tests
# ---------------------------------------------------------------------------

def _block(set_id, sentiment_model, category, *, hpo_variant, model_type="per_asset",
           panel_mode="-", acc=0.62, n=200, seed=0):
    rng = np.random.default_rng(seed)
    ts = pd.date_range("2024-01-01", periods=n, freq="D", tz="UTC")
    target = rng.integers(0, 2, n)
    flip = rng.random(n) > acc
    pred = np.where(flip, 1 - target, target).astype(int)
    prob = np.where(pred == 1, rng.uniform(0.55, 0.9, n), rng.uniform(0.1, 0.45, n))
    return pd.DataFrame({
        "timestamp": ts, "ticker": "BTC", "target": target, "prediction": pred,
        "probability": prob.astype(float), "set_id": set_id,
        "sentiment_model": sentiment_model, "horizon": "1d", "category": category,
        "model_type": model_type, "panel_mode": panel_mode,
        "hpo_enabled": hpo_variant != "fixed",
        "hpo_objective": "brier_score" if hpo_variant != "fixed" else "-",
        "hpo_variant": hpo_variant,
    })


# ---------------------------------------------------------------------------
# 4. Evaluation separates fixed vs hpo_brier
# ---------------------------------------------------------------------------

def test_pooled_metrics_separates_fixed_and_hpo():
    both = pd.concat([
        _block("C3", "cryptobert", "combined", hpo_variant="fixed", seed=1),
        _block("C3", "cryptobert", "combined", hpo_variant="hpo_brier", seed=2),
    ], ignore_index=True)
    pooled = metrics.pooled_metrics_table(both)
    assert len(pooled) == 2
    assert set(pooled["hpo_variant"]) == {"fixed", "hpo_brier"}
    # The family columns lead the table.
    assert list(pooled.columns[:4]) == ["horizon", "model_type", "panel_mode",
                                        "hpo_variant"]


# ---------------------------------------------------------------------------
# 5. McNemar fixed-vs-fixed and hpo-vs-hpo only
# ---------------------------------------------------------------------------

def _model_plus_bench(hpo_variant, seed):
    return pd.concat([
        _block("C3", "cryptobert", "combined", hpo_variant=hpo_variant,
               acc=0.64, seed=seed),
        _block("B2", "-", "economic", hpo_variant=hpo_variant, acc=0.50, seed=seed + 1),
    ], ignore_index=True)


def test_mcnemar_matches_within_hpo_variant():
    df = pd.concat([_model_plus_bench("fixed", 10),
                    _model_plus_bench("hpo_brier", 20)], ignore_index=True)
    out = sig.mcnemar_table(df, benchmarks=("B2",))
    c3 = out[out["set_id"] == "C3"]
    # One row per variant — never pooled across fixed/hpo.
    assert set(c3["hpo_variant"]) == {"fixed", "hpo_brier"}
    assert len(c3) == 2


def test_mcnemar_no_fixed_hpo_cross_fallback():
    # hpo_brier model present, but only a FIXED B2 benchmark exists.
    df = pd.concat([
        _block("C3", "cryptobert", "combined", hpo_variant="hpo_brier", seed=1),
        _block("B2", "-", "economic", hpo_variant="fixed", acc=0.5, seed=2),
    ], ignore_index=True)
    out = sig.mcnemar_table(df, benchmarks=("B2",))
    # No same-variant benchmark exists → the comparison is skipped entirely
    # (empty frame), and certainly no hpo_brier row borrows the fixed B2.
    if not out.empty:
        assert out[(out["set_id"] == "C3")
                   & (out["hpo_variant"] == "hpo_brier")].empty
    # Even the cross-model opt-in must not mix fixed and HPO.
    out2 = sig.mcnemar_table(df, benchmarks=("B2",), allow_cross_model_benchmark=True)
    if not out2.empty:
        assert out2[(out2["set_id"] == "C3")
                    & (out2["hpo_variant"] == "hpo_brier")].empty


# ---------------------------------------------------------------------------
# 6. Threshold lift does not mix fixed and HPO
# ---------------------------------------------------------------------------

def test_threshold_lift_within_hpo_variant():
    df = pd.concat([_model_plus_bench("fixed", 30),
                    _model_plus_bench("hpo_brier", 40)], ignore_index=True)
    out = thresholds.threshold_lift_table(df, benchmarks=("B2",), thresholds=(0.50,))
    c3 = out[out["set_id"] == "C3"]
    assert set(c3["hpo_variant"]) == {"fixed", "hpo_brier"}

    # Missing same-variant benchmark → no lift row for that variant.
    df2 = pd.concat([
        _block("C3", "cryptobert", "combined", hpo_variant="hpo_brier", seed=1),
        _block("B2", "-", "economic", hpo_variant="fixed", acc=0.5, seed=2),
    ], ignore_index=True)
    out2 = thresholds.threshold_lift_table(df2, benchmarks=("B2",), thresholds=(0.50,))
    if not out2.empty:
        assert out2[(out2["set_id"] == "C3")
                    & (out2["hpo_variant"] == "hpo_brier")].empty


# ---------------------------------------------------------------------------
# 7. Economic benchmark lift does not mix fixed and HPO
# ---------------------------------------------------------------------------

def test_economic_benchmark_lift_within_hpo_variant():
    base = pd.DataFrame({
        "horizon":   ["1d"] * 4,
        "set_id":    ["B2", "C3", "B2", "C3"],
        "model_type": ["per_asset"] * 4,
        "panel_mode": ["-"] * 4,
        "hpo_variant": ["fixed", "fixed", "hpo_brier", "hpo_brier"],
        "cost_bps":  [0.0, 0.0, 0.0, 0.0],
        "sentiment_model": ["-", "cryptobert", "-", "cryptobert"],
        "sharpe":    [0.3, 0.8, 1.0, 1.9],
        "net_cumulative_simple_return": [0.05, 0.20, 0.10, 0.40],
    })
    out = ec.attach_benchmark_lifts(base, benchmark_ids=["B2"])
    idx = out.set_index(["set_id", "hpo_variant"])
    # fixed C3 lifts vs fixed B2: 0.8 - 0.3 = 0.5
    assert idx.loc[("C3", "fixed"), "sharpe_lift_vs_b2"] == pytest.approx(0.5)
    # hpo_brier C3 lifts vs hpo_brier B2: 1.9 - 1.0 = 0.9 (NOT 1.9 - 0.3)
    assert idx.loc[("C3", "hpo_brier"), "sharpe_lift_vs_b2"] == pytest.approx(0.9)


# ---------------------------------------------------------------------------
# 8. Programmatic run(...) accepts HPO arguments
# ---------------------------------------------------------------------------

def test_run_models_run_wrapper_accepts_hpo_args():
    rc = rm.run(horizon="1d", set_id="C3", sentiment_model="cryptobert",
                tune_hyperparams=True, hpo_objective="brier_score",
                hpo_grid_C=[0.1, 1.0], hpo_class_weight=["none"], dry_run=True)
    assert rc == 0


def test_panel_run_wrapper_accepts_hpo_args():
    rc = pl.run(horizon="1d", set_id="C3", sentiment_model="cryptobert",
                panel_mode="pooled", tune_hyperparams=True,
                hpo_objective="accuracy", dry_run=True)
    assert rc == 0
