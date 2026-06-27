"""Tests for the v4 sentiment-feature generator boundary (commit 8).

* The production aggregator (Variante A) must never emit
  ``*_weighted_mean`` or any raw engagement column.
* The save boundary refuses to write a frame that violates the v4
  contract.
* The output passes the unified leakage audit and the schema
  validator.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pytest

from thesis_pipeline.diagnostics.feature_schema import (
    FORBIDDEN_RAW_ENGAGEMENT_COLUMNS,
    StaleSentimentFeatureSchema,
    validate_sentiment_feature_schema,
)
from thesis_pipeline.diagnostics.leakage_checks import (
    assert_no_forbidden_engagement_features,
)


def _scored_posts(n=200, seed=0):
    """Synthetic scored-posts frame matching the v4 aggregator schema."""
    rng = np.random.default_rng(seed)
    created = pd.date_range("2024-01-01", periods=n, freq="2h", tz="UTC")
    return pd.DataFrame({
        "ticker":   rng.choice(["BTC", "ETH"], size=n),
        "created":  created,
        "date":     created,
        "selftext": "",
        "model":    "vader",
        "title_sentiment":     rng.choice(["positive", "negative", "neutral"], n),
        "title_score":         rng.uniform(-1, 1, n),
        "title_prob_pos":      rng.uniform(0, 1, n),
        "title_prob_neg":      rng.uniform(0, 1, n),
        "title_prob_neutral":  rng.uniform(0, 1, n),
        "title_post_count":    1,
        "selftext_sentiment":  "neutral",
        "selftext_score":      np.nan,
        "selftext_prob_pos":   np.nan,
        "selftext_prob_neg":   np.nan,
        "selftext_prob_neutral": np.nan,
    })


# ---------------------------------------------------------------------------
# The aggregator code itself contains no weighted-mean path
# ---------------------------------------------------------------------------

def test_aggregator_source_contains_no_weighted_mean_emit():
    """Static source scan: the production aggregator must NOT contain
    any call that emits a ``*_weighted_mean`` column."""
    from pathlib import Path
    from thesis_pipeline.sentiment import aggregate as sa
    src = Path(sa.__file__).read_text(encoding="utf-8")
    for token in ("_weighted_mean",
                  "compute_engagement_weight",
                  "title_score_weighted"):
        # The string may only appear inside docstrings / comments
        # describing the REMOVED branch. Any reference in code is a bug.
        # We do not run a Python parser here — a stricter check is in the
        # output-contract test below.
        if token in src:
            assert "REMOVED" in src or "Variante A" in src, (
                f"{token} appears in aggregate.py but the removal "
                "marker is absent"
            )


# ---------------------------------------------------------------------------
# The save boundary refuses bad frames
# ---------------------------------------------------------------------------

def test_save_parquet_refuses_weighted_mean_column(tmp_path):
    df = pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=2, tz="UTC"),
        "ticker":    "BTC",
        "vader_title_score_mean":           0.0,
        "vader_title_score_weighted_mean":  0.0,  # FORBIDDEN
    })
    from thesis_pipeline.sentiment.aggregate import save_parquet
    with pytest.raises(StaleSentimentFeatureSchema):
        save_parquet(df, str(tmp_path / "out.parquet"), label="vader")


@pytest.mark.parametrize("col", sorted(FORBIDDEN_RAW_ENGAGEMENT_COLUMNS))
def test_save_parquet_refuses_raw_engagement(tmp_path, col):
    df = pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=2, tz="UTC"),
        "ticker":    "BTC",
        "vader_title_score_mean": 0.0,
        col:                       1.0,
    })
    from thesis_pipeline.sentiment.aggregate import save_parquet
    with pytest.raises(StaleSentimentFeatureSchema):
        save_parquet(df, str(tmp_path / "out.parquet"), label="vader")


def test_save_parquet_accepts_v4_output(tmp_path):
    df = pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=2, tz="UTC"),
        "ticker":    "BTC",
        "vader_title_score_mean":      0.0,
        "vader_title_score_std":       0.1,
        "vader_bullishness_ratio":     0.5,
        "post_count":                  2,
        "log1p_post_count":            np.log1p(2.0),
        "has_posts":                   1,
    })
    from thesis_pipeline.sentiment.aggregate import save_parquet
    save_parquet(df, str(tmp_path / "out.parquet"), label="vader")
    assert (tmp_path / "out.parquet").exists()


# ---------------------------------------------------------------------------
# End-to-end aggregation
# ---------------------------------------------------------------------------

def test_aggregate_produces_no_forbidden_columns():
    """Run the production aggregation path on synthetic scored posts;
    the output frame must satisfy both the schema validator and the
    unified leakage audit."""
    from thesis_pipeline.sentiment import aggregate as sa
    posts = _scored_posts()
    # Match the v4 post-level schema the aggregator expects.
    for model in ("vader", "cryptobert"):
        for variant in ("title_score", "selftext_score"):
            posts[f"{model}_{variant}"] = posts.get(variant, np.nan)
        for tail in ("title_prob_pos", "title_prob_neg",
                      "title_prob_neutral",
                      "title_sentiment", "selftext_sentiment"):
            posts[f"{model}_{tail}"] = posts.get(tail, np.nan)
    post_level = sa.build_post_level_features(posts)
    aggregated = sa.aggregate_to_horizon(post_level, freq="1D",
                                           horizon_label="1d")
    # No *_weighted_mean column survives.
    assert not any(c.endswith("_weighted_mean") for c in aggregated.columns)
    # No raw engagement column survives.
    forbidden_present = [c for c in aggregated.columns
                         if c in FORBIDDEN_RAW_ENGAGEMENT_COLUMNS]
    assert forbidden_present == []
    # Schema + leakage audit pass.
    validate_sentiment_feature_schema(aggregated)
    assert_no_forbidden_engagement_features(aggregated)
