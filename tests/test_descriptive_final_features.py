"""Tests for the descriptive-statistics stage on final modelling features."""
from __future__ import annotations

import shutil
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from thesis_pipeline.diagnostics import descriptive_final_features as dff


def _build_tiny_panel(tickers=("BTC", "ETH"), n=80, seed=0):
    rng = np.random.default_rng(seed)
    ts = pd.date_range("2024-01-01", periods=n, freq="D", tz="UTC")
    rows = []
    for tk in tickers:
        rows.append(pd.DataFrame({
            "timestamp":           ts,
            "ticker":              tk,
            "horizon":             "1d",
            "target":              rng.integers(0, 2, n),
            "log_return_t":        rng.normal(0, 0.02, n),
            "realized_vol_14":     np.abs(rng.normal(0.02, 0.005, n)),
            "volume_diff":         rng.normal(0, 0.5, n),
            "market_cap_t":        rng.normal(1e9, 1e7, n),
            "vader_combined_mean": rng.normal(0, 0.1, n),
            "vader_post_count":    rng.integers(0, 5, n),
            "finbert_combined_mean": rng.normal(0, 0.1, n),
            "finbert_post_count":    rng.integers(0, 5, n),
            "cryptobert_bullishness_ratio": rng.uniform(0.3, 0.7, n),
            "cryptobert_post_count":        rng.integers(0, 5, n),
        }))
    return pd.concat(rows, ignore_index=True)


# ---------------------------------------------------------------------------
# Direct table builders (no IO)
# ---------------------------------------------------------------------------

def test_by_ticker_feature_table_has_required_columns():
    df = _build_tiny_panel()
    out = dff.by_ticker_feature_table(df, "1d",
        feature_cols=["log_return_t", "vader_combined_mean"],
        feature_sets={"C2": {"features": ["log_return_t", "{model}_combined_mean"],
                              "category": "combined", "label": "c",
                              "sentiment_model": None}})
    assert not out.empty
    expected = {
        "horizon", "ticker", "feature", "feature_group", "n_rows", "n_non_missing",
        "n_missing", "share_missing", "n_zero", "share_zero", "n_unique",
        "mean", "median", "std", "min", "max", "p01", "p05", "p10", "p25",
        "p75", "p90", "p95", "p99", "skewness", "kurtosis",
        "first_timestamp", "last_timestamp", "date_range_days",
        "duplicate_timestamp_count", "empty_row_count", "share_empty_rows",
        "feature_set_ids_using_feature",
    }
    assert expected <= set(out.columns)
    # The sentiment column lists the set that uses it.
    row = out[(out["ticker"] == "BTC") & (out["feature"] == "vader_combined_mean")]
    assert "C2" in row["feature_set_ids_using_feature"].iat[0].split(";")


def test_correlation_summary_long_format():
    df = _build_tiny_panel()
    out = dff.correlation_summary(df, "1d",
        feature_cols=["log_return_t", "realized_vol_14", "volume_diff"],
        min_pairwise_n=10)
    assert set(out.columns) == {
        "horizon", "feature_1", "feature_2",
        "pearson_corr", "spearman_corr", "n_pairwise",
        "abs_pearson_corr", "abs_spearman_corr",
    }
    # 3 features → 3 unordered pairs.
    assert len(out) == 3
    assert ((out["abs_pearson_corr"] >= 0) & (out["abs_pearson_corr"] <= 1)).all()


def test_correlation_summary_drops_pairs_below_min_n():
    df = _build_tiny_panel(n=20)  # 2 tickers × 20 rows → 40 pairwise obs
    out = dff.correlation_summary(df, "1d",
        ["log_return_t", "realized_vol_14"], min_pairwise_n=100)
    assert out.empty


def test_overview_row_counts_structural_empties_and_duplicates():
    df = _build_tiny_panel(n=40)
    # Force two structurally empty sentiment rows: zero vader_mean + 0 posts.
    df.loc[df["ticker"] == "BTC", "vader_combined_mean"] = 0.0
    df.loc[df["ticker"] == "BTC", "vader_post_count"]    = 0
    # And one duplicate (ticker, timestamp).
    df = pd.concat([df, df.iloc[[0]]], ignore_index=True)
    rec = dff.overview_row(df, "1d", feature_cols=["log_return_t",
                                                    "vader_combined_mean"])
    assert rec["horizon"] == "1d"
    assert rec["n_tickers"] == 2
    assert rec["duplicate_ticker_timestamp_rows"] == 1
    assert rec["total_structural_empty_sentiment_rows"] >= 40


# ---------------------------------------------------------------------------
# Full pipeline on a synthetic repo
# ---------------------------------------------------------------------------

@pytest.fixture
def synth_repo(tmp_path, monkeypatch):
    (tmp_path / "Data" / "Final").mkdir(parents=True)
    df = _build_tiny_panel(n=80)
    df.to_parquet(tmp_path / "Data" / "Final" / "features_1d.parquet", index=False)
    fs = pd.DataFrame({
        "set_id":   ["B1", "B2", "C2"],
        "category": ["benchmark", "economic", "combined"],
        "sentiment_model": ["-", "-", "vader"],
        "label":    ["bench", "econ", "combo"],
        "Feature Columns (comma-separated)": [
            "__majority_class__", "log_return_t", "log_return_t,{model}_combined_mean",
        ],
    })
    with pd.ExcelWriter(tmp_path / "feature_sets.xlsx", engine="openpyxl") as w:
        fs.to_excel(w, sheet_name="feature_sets", index=False)
    monkeypatch.chdir(tmp_path)
    from thesis_pipeline import config as cfg
    monkeypatch.setattr(cfg, "project_root", lambda: tmp_path)
    cfg.load_config.cache_clear()
    shutil.copytree(Path(__file__).resolve().parents[1] / "configs",
                    tmp_path / "configs")
    return tmp_path


def test_main_writes_all_six_csvs(synth_repo):
    rc = dff.main(["--horizon", "1d"])
    assert rc == 0
    out_dir = synth_repo / "Outputs" / "deskriptiv" / "final_feature_sets"
    expected = {
        "final_feature_descriptives_by_ticker_feature.csv",
        "final_feature_descriptives_by_horizon_feature.csv",
        "final_feature_descriptives_by_horizon_ticker.csv",
        "final_feature_descriptives_overview.csv",
        "final_feature_correlation_summary.csv",
        "final_feature_resolution.csv",
    }
    assert expected <= {p.name for p in out_dir.iterdir()}
    by_tf = pd.read_csv(out_dir / "final_feature_descriptives_by_ticker_feature.csv")
    assert not by_tf.empty
    # Final-feature universe includes price + sentiment.
    feats = set(by_tf["feature"])
    assert "log_return_t" in feats
    assert "vader_combined_mean" in feats
    # Resolution report flags the used columns.
    res = pd.read_csv(out_dir / "final_feature_resolution.csv")
    assert "used_in_descriptive_statistics" in res.columns


def test_main_dry_run_short_circuits(synth_repo):
    out_dir = synth_repo / "Outputs" / "deskriptiv" / "final_feature_sets"
    if out_dir.exists():
        shutil.rmtree(out_dir)
    rc = dff.main(["--horizon", "1d", "--dry-run"])
    assert rc == 0
    assert not out_dir.exists()


# ---------------------------------------------------------------------------
# CLI source selection for stationarity (without importing the heavy module)
# ---------------------------------------------------------------------------

def test_cli_stationarity_forwards_source_final(monkeypatch):
    captured = {}

    fake = types.ModuleType("thesis_pipeline.sentiment.stationarity")
    def fake_main(argv=None):
        captured["argv"] = list(argv or [])
        return 0
    fake.main = fake_main
    monkeypatch.setitem(sys.modules, "thesis_pipeline.sentiment.stationarity", fake)

    from thesis_pipeline import cli
    rc = cli.main(["stationarity", "--source", "final", "--horizon", "1d", "--no-plots"])
    assert rc == 0
    assert "--source" in captured["argv"]
    assert captured["argv"][captured["argv"].index("--source") + 1] == "final"
    assert "--horizon" in captured["argv"]
    assert "--no_plots" in captured["argv"]


def test_cli_stationarity_default_is_final(monkeypatch):
    captured = {}

    fake = types.ModuleType("thesis_pipeline.sentiment.stationarity")
    def fake_main(argv=None):
        captured["argv"] = list(argv or [])
        return 0
    fake.main = fake_main
    monkeypatch.setitem(sys.modules, "thesis_pipeline.sentiment.stationarity", fake)

    from thesis_pipeline import cli
    rc = cli.main(["stationarity", "--horizon", "1d"])
    assert rc == 0
    assert captured["argv"][captured["argv"].index("--source") + 1] == "final"


def test_cli_descriptive_final_features_dry_run(synth_repo):
    from thesis_pipeline import cli
    rc = cli.main(["descriptive-final-features", "--horizon", "1d", "--dry-run"])
    assert rc == 0
