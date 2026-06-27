"""Tests for the v4 generator boundary schema contracts (commit 8).

Two reusable validators live in
:mod:`thesis_pipeline.diagnostics.feature_schema`:

* :func:`validate_price_feature_schema`     — refuses pre-v4 momentum
  columns and missing market-cap availability.
* :func:`validate_sentiment_feature_schema` — refuses every
  ``*_weighted_mean`` column and the raw engagement set.

The error messages must be actionable — they tell the user the exact
``python -m thesis_pipeline.cli create-*-features --force`` command to
run.
"""
from __future__ import annotations

import pandas as pd
import pytest

from thesis_pipeline.diagnostics.feature_schema import (
    EXPECTED_SENTIMENT_FAMILIES,
    FORBIDDEN_RAW_ENGAGEMENT_COLUMNS,
    LEGACY_MOMENTUM_NAMES,
    REQUIRED_PRICE_COLUMNS,
    StalePriceFeatureSchema,
    StaleSentimentFeatureSchema,
    validate_price_feature_schema,
    validate_sentiment_feature_schema,
)


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------

def _v4_price_frame(n=5):
    ts = pd.date_range("2024-01-01", periods=n, freq="D", tz="UTC")
    return pd.DataFrame({
        "timestamp":              ts,
        "ticker":                 "BTC",
        "log_return_t":           0.0,
        "cum_log_return_7d":      0.0,
        "cum_log_return_14d":     0.0,
        "cum_log_return_21d":     0.0,
        "realized_vol_14d":       0.02,
        "volume_diff":            0.0,
        "log_market_cap_lag1":    20.0,
        "market_cap_available_at": ts - pd.Timedelta(days=1),
    })


def _v4_sentiment_frame(n=5):
    ts = pd.date_range("2024-01-01", periods=n, freq="D", tz="UTC")
    return pd.DataFrame({
        "timestamp": ts, "ticker": "BTC",
        "vader_title_score_mean":      0.0,
        "vader_title_score_std":       0.1,
        "vader_bullishness_ratio":     0.5,
        "cryptobert_title_score_mean": 0.0,
        "cryptobert_title_score_std":  0.1,
        "cryptobert_bullishness_ratio": 0.5,
        "post_count":                  0,
        "log1p_post_count":            0.0,
        "has_posts":                   0,
    })


# ---------------------------------------------------------------------------
# Price schema
# ---------------------------------------------------------------------------

def test_required_price_columns_cover_econ_core():
    """The required set must include every column the ECON registry
    consumes. The active registry's ECON entry is the source of truth."""
    from thesis_pipeline.features.feature_registry import load_feature_sets
    fs = load_feature_sets() or {}
    econ_core = set(fs.get("ECON", {}).get("features", []))
    # Tolerate registry layouts where the canonical key already names
    # the v4 columns; the validator must include every one of them.
    required = set(REQUIRED_PRICE_COLUMNS)
    missing = econ_core - required
    assert not missing, (
        f"ECON registry requests columns not enforced by the validator: {sorted(missing)}"
    )


def test_price_validator_passes_on_v4_frame():
    validate_price_feature_schema(_v4_price_frame())  # no raise


def test_price_validator_rejects_legacy_momentum_only_frame():
    df = _v4_price_frame().drop(columns=[
        "cum_log_return_7d", "cum_log_return_14d", "cum_log_return_21d",
        "realized_vol_14d",
    ])
    df["cum_log_return_7"]  = 0.0
    df["cum_log_return_14"] = 0.0
    df["cum_log_return_21"] = 0.0
    df["realized_vol_14"]   = 0.02
    with pytest.raises(StalePriceFeatureSchema) as exc:
        validate_price_feature_schema(df, horizon="1d",
                                        source="Data/Features/price_features_1d.parquet")
    msg = str(exc.value)
    assert "Detected pre-v4 price features" in msg
    for needed in ("cum_log_return_7d", "cum_log_return_14d",
                    "cum_log_return_21d", "realized_vol_14d"):
        assert needed in msg
    for legacy in LEGACY_MOMENTUM_NAMES:
        assert legacy in msg
    # Actionable regenerate-with hint.
    assert "create-price-features" in msg
    assert "--horizon 1d" in msg
    assert "--force" in msg


def test_price_validator_rejects_missing_market_cap_available_at():
    df = _v4_price_frame().drop(columns=["market_cap_available_at"])
    with pytest.raises(StalePriceFeatureSchema, match="market_cap_available_at"):
        validate_price_feature_schema(df, horizon="1d")


def test_price_validator_rejects_missing_log_market_cap_lag1():
    df = _v4_price_frame().drop(columns=["log_market_cap_lag1"])
    with pytest.raises(StalePriceFeatureSchema, match="log_market_cap_lag1"):
        validate_price_feature_schema(df)


# ---------------------------------------------------------------------------
# Sentiment schema
# ---------------------------------------------------------------------------

def test_sentiment_validator_passes_on_clean_frame():
    df = _v4_sentiment_frame()
    validate_sentiment_feature_schema(df, require_polarity=True)


@pytest.mark.parametrize("col", [
    "vader_title_score_weighted_mean",
    "cryptobert_title_score_weighted_mean",
    "vader_combined_score_weighted_mean",
    "foo_weighted_mean",
])
def test_sentiment_validator_rejects_any_weighted_mean(col):
    df = _v4_sentiment_frame()
    df[col] = 0.0
    with pytest.raises(StaleSentimentFeatureSchema) as exc:
        validate_sentiment_feature_schema(df, horizon="1d")
    assert col in str(exc.value)
    assert "create-sentiment-features" in str(exc.value)


@pytest.mark.parametrize("col", sorted(FORBIDDEN_RAW_ENGAGEMENT_COLUMNS))
def test_sentiment_validator_rejects_raw_engagement(col):
    df = _v4_sentiment_frame()
    df[col] = 1.0
    with pytest.raises(StaleSentimentFeatureSchema) as exc:
        validate_sentiment_feature_schema(df, horizon="1d")
    assert col in str(exc.value)


def test_sentiment_validator_polarity_required_when_flag_set():
    df = _v4_sentiment_frame().drop(columns=list(EXPECTED_SENTIMENT_FAMILIES))
    with pytest.raises(StaleSentimentFeatureSchema, match="polarity"):
        validate_sentiment_feature_schema(df, require_polarity=True)


def test_sentiment_validator_polarity_not_required_by_default():
    df = _v4_sentiment_frame().drop(columns=list(EXPECTED_SENTIMENT_FAMILIES))
    validate_sentiment_feature_schema(df)  # no raise


# ---------------------------------------------------------------------------
# Validators are wired into the merge load path
# ---------------------------------------------------------------------------

def test_merge_load_price_features_runs_schema_validator(tmp_path):
    """The merge step's :func:`load_price_features` must invoke the
    schema validator so a stale v3 parquet never silently feeds the
    final-frame leakage audit."""
    feature_dir = tmp_path
    legacy = _v4_price_frame().drop(columns=[
        "cum_log_return_7d", "cum_log_return_14d", "cum_log_return_21d",
        "realized_vol_14d",
    ])
    legacy["cum_log_return_7"]  = 0.0
    legacy["cum_log_return_14"] = 0.0
    legacy["realized_vol_14"]   = 0.02
    (feature_dir / "price_features_1d.parquet")
    legacy.to_parquet(feature_dir / "price_features_1d.parquet", index=False)
    from thesis_pipeline.features.merge import load_price_features
    with pytest.raises(StalePriceFeatureSchema):
        load_price_features("1d", feature_dir=str(feature_dir))


def test_merge_load_sentiment_features_runs_schema_validator(tmp_path):
    feature_dir = tmp_path
    bad = _v4_sentiment_frame()
    bad["vader_title_score_weighted_mean"] = 0.0
    bad.to_parquet(feature_dir / "sentiment_features_1d.parquet", index=False)
    from thesis_pipeline.features.merge import load_sentiment_features
    with pytest.raises(StaleSentimentFeatureSchema):
        load_sentiment_features("1d", feature_dir=str(feature_dir))
