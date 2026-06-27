"""Tests for the final-feature stationarity stage after the registry fix.

The stage must:

* Read ``Data/Final/features_{horizon}.parquet`` and resolve the modelling
  feature universe via the corrected
  :func:`thesis_pipeline.features.feature_registry.load_feature_sets`.
* Surface **sentiment** features (e.g. ``vader_title_score_mean``,
  ``*_title_score_weighted_mean``, ``*_bullishness_ratio``,
  ``*_title_score_std``, ``post_count``) — not silently fall back to
  economic-only features.
* Exclude identifiers / target / prediction columns.
* Tag the resolution log with ``used_in_stationarity``, ``feature_group`` and
  ``registry_source``.
* Emit dual percentages in the summary CSV: ``pct_within_horizon`` and
  ``pct_within_feature_group``.

The heavy battery (DF-GLS / PP / KPSS / GPH) requires ``arch``; the
full-pipeline test is gated on that import. The resolution-level checks run
unconditionally.
"""
from __future__ import annotations

import shutil
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from thesis_pipeline.features import feature_registry as fr
from thesis_pipeline.features.final_feature_utils import (
    classify_feature_group, resolve_final_feature_columns,
)


# ---------------------------------------------------------------------------
# Synthetic final-features frame + matching feature_sets.xlsx
# ---------------------------------------------------------------------------

SENTIMENT_COLUMNS = [
    "vader_title_score_mean",
    "cryptobert_title_score_mean",
    "vader_title_score_weighted_mean",
    "cryptobert_title_score_weighted_mean",
    "vader_bullishness_ratio",
    "cryptobert_bullishness_ratio",
    "vader_title_score_std",
    "cryptobert_title_score_std",
    "vader_post_count",
    "cryptobert_post_count",
    "post_count",
    # Diagnostic attention columns: present in the final parquet but
    # NOT in the 17-set modeling registry. The stationarity stage
    # picks them up via DIAGNOSTIC_ATTENTION_FEATURES so the report
    # still surfaces them.
    "log1p_post_count",
    "has_posts",
]
ECONOMIC_COLUMNS = ["log_return_t", "cum_log_return_7", "cum_log_return_14",
                    "realized_vol_14", "volume_diff", "market_cap_t"]
IDENTIFIER_COLUMNS = ["timestamp", "ticker", "horizon", "target", "row_id",
                      "prediction", "probability"]


def _build_final_features(n_per_ticker=80, tickers=("BTC", "ETH"), seed=0):
    rng = np.random.default_rng(seed)
    ts = pd.date_range("2024-01-01", periods=n_per_ticker, freq="D", tz="UTC")
    frames = []
    for tk in tickers:
        row = {
            "timestamp": ts, "ticker": tk, "horizon": "1d",
            "target": rng.integers(0, 2, n_per_ticker),
            "row_id": [f"{tk}-{i}" for i in range(n_per_ticker)],
            "prediction": rng.integers(0, 2, n_per_ticker),
            "probability": rng.uniform(0, 1, n_per_ticker),
        }
        for col in ECONOMIC_COLUMNS:
            row[col] = rng.normal(0, 0.05, n_per_ticker)
        for col in SENTIMENT_COLUMNS:
            row[col] = rng.normal(0.5 if "bullishness" in col else 0.0,
                                  0.1, n_per_ticker)
        frames.append(pd.DataFrame(row))
    return pd.concat(frames, ignore_index=True)


def _write_feature_sets_xlsx(path):
    """Workbook covering economic + sentiment + combined sets with the
    capitalised header layout the corrected registry now handles."""
    headers = ["Set ID", "Category", "Sentiment Model", "Label",
               "# Features", "Feature Columns (comma-separated)", "Description"]
    rows = [
        ["B1", "benchmark", "-", "rolling probability", 0,
         "__majority_class__", "no features"],
        ["B6", "benchmark", "-", "log-return + cum",    3,
         "log_return_t,cum_log_return_7,cum_log_return_14", "price"],
        ["SV1", "sentiment_vader",      "vader",      "vader title score",  1,
         "vader_title_score_mean", "single sentiment"],
        ["S1",  "sentiment_cryptobert", "cryptobert", "cryptobert title score", 1,
         "cryptobert_title_score_mean", "single sentiment"],
        # {model} placeholders are expanded by the loader to cryptobert / vader.
        ["S_W", "sentiment_cryptobert", "-", "weighted mean", 1,
         "{model}_title_score_weighted_mean", "weighted across scorers"],
        ["S_B", "sentiment_cryptobert", "-", "bullishness", 1,
         "{model}_bullishness_ratio", "bullishness across scorers"],
        ["S_S", "sentiment_cryptobert", "-", "title score std", 1,
         "{model}_title_score_std", "std across scorers"],
        ["CV1", "combined_vader",      "vader",      "combined vader", 2,
         "log_return_t,vader_title_score_mean", "combined"],
        ["C1",  "combined_cryptobert", "cryptobert", "combined cryptobert", 4,
         "log_return_t,realized_vol_14,cryptobert_title_score_mean,cryptobert_post_count",
         "full combined set"],
    ]
    pd.DataFrame(rows, columns=headers).to_excel(
        path, sheet_name="feature_sets", index=False)


@pytest.fixture
def stationarity_repo(tmp_path, monkeypatch):
    (tmp_path / "Data" / "Final").mkdir(parents=True)
    _build_final_features().to_parquet(
        tmp_path / "Data" / "Final" / "features_1d.parquet", index=False)
    _write_feature_sets_xlsx(tmp_path / "feature_sets.xlsx")
    monkeypatch.chdir(tmp_path)
    from thesis_pipeline import config as cfg
    monkeypatch.setattr(cfg, "project_root", lambda: tmp_path)
    cfg.load_config.cache_clear()
    shutil.copytree(Path(__file__).resolve().parents[1] / "configs",
                    tmp_path / "configs")
    return tmp_path


# ---------------------------------------------------------------------------
# Resolution-level checks (no heavy deps)
# ---------------------------------------------------------------------------

def test_registry_resolves_sentiment_and_economic_features(stationarity_repo):
    """The corrected loader must yield BOTH price AND sentiment columns —
    the stationarity stage cannot silently fall back to economic-only when the
    sentiment columns exist in the final parquet.
    """
    feature_sets = fr.load_feature_sets() or {}
    assert feature_sets, "registry should not be empty"
    df = pd.read_parquet(stationarity_repo / "Data" / "Final" / "features_1d.parquet")
    feature_cols, resolution = resolve_final_feature_columns(df, feature_sets)

    # Sentiment columns explicitly mentioned in the request must be in the
    # universe. FinBERT was removed from the pipeline; the placeholder
    # expansion now only covers CryptoBERT + VADER.
    required_sentiment = {
        "vader_title_score_mean",
        "cryptobert_title_score_mean",
        "vader_title_score_weighted_mean",
        "cryptobert_title_score_weighted_mean",
        "vader_bullishness_ratio",
        "cryptobert_bullishness_ratio",
        "vader_title_score_std",
        "cryptobert_title_score_std",
    }
    missing = required_sentiment - set(feature_cols)
    assert not missing, f"sentiment features dropped from universe: {sorted(missing)}"
    # FinBERT columns must never be resolved any more.
    assert not any("finbert" in c for c in feature_cols)

    # And economic features stay in.
    assert {"log_return_t", "cum_log_return_7", "realized_vol_14"} <= set(feature_cols)

    # Identifiers / target / prediction columns never appear.
    for forbidden in ("target", "ticker", "timestamp", "horizon", "row_id",
                      "prediction", "probability"):
        assert forbidden not in feature_cols

    # The resolution log records every requested column with a reason when missing.
    flagged_present = {r["resolved_feature"] for r in resolution
                       if r["exists_in_final_data"]}
    assert required_sentiment <= flagged_present


# ---------------------------------------------------------------------------
# End-to-end stationarity pipeline (gated on arch)
# ---------------------------------------------------------------------------

@pytest.fixture
def stationarity_module():
    pytest.importorskip("arch")
    from thesis_pipeline.sentiment import stationarity as _s
    return _s


def test_run_final_records_include_sentiment(stationarity_repo, stationarity_module,
                                              monkeypatch):
    """End-to-end: ``_run_final`` writes records for both economic and
    sentiment features and never collapses to an economic-only universe.

    The heavy unit-root battery is stubbed so the test runs in <1 s; the goal
    is to verify the *plumbing* (resolution → records → CSVs), not the
    statistics. CIPS and fold-stability are skipped via --no-panel.
    """
    monkeypatch.setattr(stationarity_module, "test_series",
                        lambda series, run_long_memory_default=False: {
                            "joint_class": "I(0)", "n_obs": int(series.dropna().size),
                            "adf_c_pval": 0.01, "kpss_c_pval": 0.10,
                            "dfgls_c_pval": 0.02, "pp_c_pval": 0.03,
                        })
    rc = stationarity_module.main(["--source", "final", "--horizon", "1d",
                                    "--no_panel", "--no_plots"])
    assert rc == 0

    out_dir = stationarity_repo / "Data" / "Features" / "stationarity_final"
    records = pd.read_csv(out_dir / "stationarity_final_records.csv")
    feats = set(records["feature"].astype(str))

    # v4 final feature path: no ``*_weighted_mean`` rows survive in the
    # canonical registry. The diagnostic attention columns
    # (``post_count``, ``log1p_post_count``, ``has_posts``) are surfaced
    # by :data:`final_feature_utils.DIAGNOSTIC_ATTENTION_FEATURES` when
    # they exist in the parquet — even though they are NOT modeling
    # features.
    must_have = {
        "log_return_t",
        "vader_title_score_mean", "cryptobert_title_score_mean",
        "vader_bullishness_ratio",
        "vader_title_score_std",
        "post_count",
    }
    # Optional diagnostic-attention columns: required iff the fixture
    # actually carries them.
    fixture_cols = set(
        pd.read_parquet(
            stationarity_repo / "Data" / "Final" / "features_1d.parquet"
        ).columns
    )
    for opt in ("log1p_post_count", "has_posts"):
        if opt in fixture_cols:
            must_have.add(opt)
    missing = must_have - feats
    assert not missing, (
        f"stationarity records lost {sorted(missing)} — the stage fell back "
        f"to economic-only features. Got groups: "
        f"{records['feature_group'].value_counts().to_dict()}"
    )
    # At least one sentiment row per remaining scorer (CryptoBERT + VADER).
    sentiment_rows = records[records["feature_group"] == "sentiment"]
    assert not sentiment_rows.empty
    assert {"vader", "cryptobert"} <= {
        f.split("_", 1)[0] for f in sentiment_rows["feature"].astype(str)
    }
    # And no FinBERT rows.
    assert not any("finbert" in str(f).lower() for f in sentiment_rows["feature"])


def test_run_final_resolution_csv_has_new_columns(stationarity_repo,
                                                   stationarity_module,
                                                   monkeypatch):
    monkeypatch.setattr(stationarity_module, "test_series",
                        lambda series, run_long_memory_default=False: {
                            "joint_class": "I(0)", "n_obs": int(series.dropna().size),
                        })
    rc = stationarity_module.main(["--source", "final", "--horizon", "1d",
                                    "--no_panel", "--no_plots"])
    assert rc == 0

    out_dir = stationarity_repo / "Data" / "Features" / "stationarity_final"
    res = pd.read_csv(out_dir / "stationarity_final_feature_resolution.csv")
    for col in ("horizon", "set_id", "category", "sentiment_model",
                "requested_feature", "resolved_feature", "exists_in_final_data",
                "used_in_stationarity", "feature_group", "registry_source"):
        assert col in res.columns, f"resolution CSV missing column {col!r}"

    # Sentiment features that exist in the parquet must be marked used.
    sentiment_rows = res[(res["feature_group"] == "sentiment")
                         & (res["exists_in_final_data"].astype(str).str.lower() == "true")]
    assert not sentiment_rows.empty
    assert (sentiment_rows["used_in_stationarity"].astype(str).str.lower() == "true").any()
    # And the registry source is the xlsx tag (the workbook lives in tmp_path).
    assert (res["registry_source"] == "feature_sets.xlsx").any()


def test_run_final_summary_has_dual_percentages(stationarity_repo,
                                                stationarity_module,
                                                monkeypatch):
    monkeypatch.setattr(stationarity_module, "test_series",
                        lambda series, run_long_memory_default=False: {
                            "joint_class": "I(0)", "n_obs": int(series.dropna().size),
                        })
    rc = stationarity_module.main(["--source", "final", "--horizon", "1d",
                                    "--no_panel", "--no_plots"])
    assert rc == 0
    out_dir = stationarity_repo / "Data" / "Features" / "stationarity_final"
    summary = pd.read_csv(out_dir / "stationarity_final_summary.csv")
    assert {"pct_within_horizon", "pct_within_feature_group"} <= set(summary.columns)
    # Both pct columns sum to ~100 within their respective scopes.
    horizon_totals = summary.groupby("horizon")["pct_within_horizon"].sum()
    np.testing.assert_allclose(horizon_totals.values, 100.0, atol=0.01)
    group_totals = summary.groupby(["horizon", "feature_group"]) \
                          ["pct_within_feature_group"].sum()
    np.testing.assert_allclose(group_totals.values, 100.0, atol=0.01)


def test_run_final_uses_paths_config(stationarity_repo, stationarity_module,
                                      monkeypatch, tmp_path):
    """Override ``stationarity_final_root`` via paths.yaml and verify the
    stage writes there (not under Data/Features/stationarity_final)."""
    custom = tmp_path / "custom_stationarity_out"
    paths_yaml = stationarity_repo / "configs" / "paths.yaml"
    txt = paths_yaml.read_text(encoding="utf-8")
    txt = txt.replace('stationarity_final_root: "Data/Features/stationarity_final"',
                      f'stationarity_final_root: "{custom.as_posix()}"')
    paths_yaml.write_text(txt, encoding="utf-8")
    from thesis_pipeline import config as cfg
    cfg.load_config.cache_clear()

    monkeypatch.setattr(stationarity_module, "test_series",
                        lambda series, run_long_memory_default=False: {
                            "joint_class": "I(0)", "n_obs": int(series.dropna().size),
                        })
    rc = stationarity_module.main(["--source", "final", "--horizon", "1d",
                                    "--no_panel", "--no_plots"])
    assert rc == 0
    assert (custom / "stationarity_final_records.csv").exists()
    # And the default location is NOT used for this run.
    default = stationarity_repo / "Data" / "Features" / "stationarity_final" \
              / "stationarity_final_records.csv"
    assert not default.exists()
