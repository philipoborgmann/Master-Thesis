"""Tests for the shared final-feature resolution and grouping helpers."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from thesis_pipeline.features import final_feature_utils as ffu


# ---------------------------------------------------------------------------
# classify_feature_group
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("feature,group", [
    ("log_return_t", "price"),
    ("cum_log_return_7", "price"),
    ("cum_log_return_14", "price"),
    ("realized_vol_14", "volatility"),
    ("volume_diff", "volume"),
    ("market_cap_t", "market_cap"),
    ("vader_combined_mean", "sentiment"),
    ("finbert_post_count", "sentiment"),
    ("cryptobert_bullishness_ratio", "sentiment"),
    ("post_count", "sentiment"),
    ("some_other_feature", "other"),
])
def test_classify_feature_group_known_inputs(feature, group):
    assert ffu.classify_feature_group(feature) == group


# ---------------------------------------------------------------------------
# resolve_final_feature_columns
# ---------------------------------------------------------------------------

def _df_with(cols):
    """One-row dataframe carrying ``cols`` (identifiers + features)."""
    data = {"timestamp": pd.Timestamp("2024-01-01", tz="UTC"),
            "ticker": "BTC", "horizon": "1d", "target": 0, "row_id": "x"}
    for c in cols:
        data[c] = 0.0
    return pd.DataFrame([data])


def test_resolve_expands_model_placeholder_across_all_scorers():
    df = _df_with(["vader_combined_mean", "finbert_combined_mean",
                   "cryptobert_combined_mean", "log_return_t"])
    sets = {
        "B1": {"features": [], "category": "benchmark", "label": "b",
                "sentiment_model": None},
        "C2": {"features": ["log_return_t", "{model}_combined_mean"],
                "category": "combined", "label": "c", "sentiment_model": None},
    }
    cols, resolution = ffu.resolve_final_feature_columns(df, sets)
    assert "log_return_t" in cols
    assert set(cols) >= {
        "vader_combined_mean", "finbert_combined_mean", "cryptobert_combined_mean",
        "log_return_t",
    }
    # Resolution log carries one row per expanded feature.
    expanded = {r["resolved_feature"] for r in resolution
                if r["requested_feature"] == "{model}_combined_mean"}
    assert expanded == {"vader_combined_mean", "finbert_combined_mean",
                        "cryptobert_combined_mean"}


def test_resolve_respects_set_specific_sentiment_model():
    df = _df_with(["vader_combined_mean", "finbert_combined_mean"])
    sets = {
        "S1": {"features": ["{model}_combined_mean"], "category": "sentiment",
                "label": "s", "sentiment_model": "vader"},
    }
    cols, _ = ffu.resolve_final_feature_columns(df, sets)
    assert cols == ["vader_combined_mean"]


def test_resolve_excludes_b1_and_identifiers_and_target():
    df = _df_with(["log_return_t"])
    sets = {
        "B1": {"features": [], "category": "benchmark", "label": "b",
                "sentiment_model": None},
        "B2": {"features": ["log_return_t"], "category": "economic",
                "label": "b2", "sentiment_model": None},
    }
    cols, resolution = ffu.resolve_final_feature_columns(df, sets)
    assert "target" not in cols
    assert "ticker" not in cols
    assert "row_id" not in cols
    assert cols == ["log_return_t"]
    # B1 contributes nothing.
    assert all(r["set_id"] != "B1" for r in resolution)


def test_resolve_missing_feature_logs_reason():
    df = _df_with(["log_return_t"])
    sets = {"S2": {"features": ["does_not_exist"], "category": "sentiment",
                    "label": "s", "sentiment_model": None}}
    cols, resolution = ffu.resolve_final_feature_columns(df, sets)
    # When the registry resolves to nothing in df, the helper falls back to the
    # numeric column universe — log_return_t is picked up that way.
    assert "log_return_t" in cols
    # The missing requested feature is still recorded with a reason.
    flags = [r for r in resolution if r["requested_feature"] == "does_not_exist"]
    assert flags and not flags[0]["exists_in_final_data"]
    assert "not in final features" in flags[0]["reason_if_missing"]


def test_resolve_numeric_fallback_when_registry_empty():
    df = _df_with(["log_return_t", "realized_vol_14", "vader_combined_mean"])
    cols, resolution = ffu.resolve_final_feature_columns(df, feature_sets={})
    # Fallback excludes identifiers and target, keeps numeric features.
    assert "target" not in cols
    assert {"log_return_t", "realized_vol_14", "vader_combined_mean"} <= set(cols)
    assert all(r["set_id"] == "__numeric_fallback__" for r in resolution)


# ---------------------------------------------------------------------------
# Structural-empty mask
# ---------------------------------------------------------------------------

def test_structural_empty_sentiment_neutral_plus_zero_count():
    df = pd.DataFrame({
        "vader_combined_mean":  [0.0, 0.4, 0.0, np.nan, 0.0],
        "vader_post_count":     [0,   3,   2,   0,      0],
    })
    mask = ffu.compute_structural_empty_mask(df, "vader_combined_mean")
    # Row 0: neutral 0.0 AND post_count 0 → structural empty.
    # Row 1: non-neutral → not empty.
    # Row 2: neutral 0.0 BUT post_count 2 → real observation, not empty.
    # Row 3: NaN → missing → empty.
    # Row 4: neutral + count 0 → structural empty.
    assert mask.tolist() == [True, False, False, True, True]


def test_structural_empty_bullishness_uses_neutral_half():
    df = pd.DataFrame({
        "cryptobert_bullishness_ratio": [0.5, 0.5, 0.7],
        "cryptobert_post_count":        [0, 5, 0],
    })
    mask = ffu.compute_structural_empty_mask(df, "cryptobert_bullishness_ratio")
    # 0.5 + 0 posts → structural; 0.5 + posts → real; 0.7 → not empty.
    assert mask.tolist() == [True, False, False]


def test_structural_empty_non_sentiment_only_missing():
    df = pd.DataFrame({"log_return_t": [0.0, np.nan, 0.01]})
    mask = ffu.compute_structural_empty_mask(df, "log_return_t")
    # 0.0 is a perfectly real return; only NaN is empty.
    assert mask.tolist() == [False, True, False]


def test_find_matching_post_count_prefers_per_model():
    cols = ["vader_combined_mean", "vader_post_count", "post_count"]
    assert ffu.find_matching_post_count("vader_combined_mean", cols) == "vader_post_count"
    cols = ["vader_combined_mean", "post_count"]
    assert ffu.find_matching_post_count("vader_combined_mean", cols) == "post_count"
    assert ffu.find_matching_post_count("log_return_t", cols) is None
