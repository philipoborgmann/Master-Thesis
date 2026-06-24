"""Tests for the Variante A merge fill rules.

* `bullishness_ratio` neutral fill = 0.5 (NOT 0 — 0 means bearish-unanimity).
* `*_score_mean / _median / _std` neutral fill = 0.
* `post_count`, `*_post_count`, `*_directional_post_count` fill = 0.
* `log1p_post_count = np.log1p(post_count)` and `has_posts = (post_count > 0)`
  are derived AFTER the fill so empty slots get 0 / 0.
* `*_weighted_mean` columns are not recognised (Variante A removed them).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from thesis_pipeline.features import merge as mg


# ---------------------------------------------------------------------------
# sentiment_neutral_columns — no _weighted_mean recognised
# ---------------------------------------------------------------------------

def test_sentiment_neutral_columns_drops_weighted_mean():
    cols = [
        "vader_title_score_mean",
        "vader_title_score_median",
        "vader_title_score_std",
        "vader_title_score_weighted_mean",   # Variante A — must NOT be filled
        "vader_bullishness_ratio",
        "post_count",
        "vader_directional_post_count",
    ]
    keep = set(mg.sentiment_neutral_columns(cols))
    assert "vader_title_score_weighted_mean" not in keep
    # The legitimate ones survive:
    for must in ("vader_title_score_mean", "vader_title_score_median",
                  "vader_title_score_std", "vader_bullishness_ratio",
                  "post_count", "vader_directional_post_count"):
        assert must in keep


# ---------------------------------------------------------------------------
# fill_missing_sentiment — neutral values, especially bullishness=0.5
# ---------------------------------------------------------------------------

def test_fill_missing_sentiment_uses_0_5_for_bullishness_ratio():
    df = pd.DataFrame({
        "ticker":    ["BTC", "BTC"],
        "timestamp": pd.date_range("2024-01-01", periods=2, tz="UTC"),
        "vader_bullishness_ratio":       [np.nan, 0.8],
        "vader_title_score_mean":        [np.nan, 0.3],
        "vader_title_score_median":      [np.nan, 0.3],
        "vader_title_score_std":         [np.nan, 0.1],
        "post_count":                    [0, 5],
        "vader_directional_post_count":  [np.nan, 4],
    })
    out = mg.fill_missing_sentiment(
        df, ["vader_bullishness_ratio", "vader_title_score_mean",
              "vader_title_score_median", "vader_title_score_std",
              "post_count", "vader_directional_post_count"],
    )
    # Bullishness ratio neutral = 0.5
    assert out["vader_bullishness_ratio"].iloc[0] == 0.5
    # Score moments → 0
    assert out["vader_title_score_mean"].iloc[0]   == 0.0
    assert out["vader_title_score_median"].iloc[0] == 0.0
    assert out["vader_title_score_std"].iloc[0]    == 0.0
    # Counts → 0
    assert out["vader_directional_post_count"].iloc[0] == 0


def test_fill_missing_sentiment_does_not_fill_with_zero_for_bullishness():
    """Pre-Variante-A bug guard: empty slot's bullishness must not be 0
    (0 means "all directional posts were negative" — a strong bearish
    signal that the slot does NOT actually convey)."""
    df = pd.DataFrame({"vader_bullishness_ratio": [np.nan]})
    out = mg.fill_missing_sentiment(df, ["vader_bullishness_ratio"])
    assert out["vader_bullishness_ratio"].iloc[0] != 0.0
    assert out["vader_bullishness_ratio"].iloc[0] == 0.5


# ---------------------------------------------------------------------------
# derive_post_count_features
# ---------------------------------------------------------------------------

def test_derive_post_count_features_for_empty_slots():
    df = pd.DataFrame({"post_count": [0, 1, 5, 0]})
    out = mg.derive_post_count_features(df.copy())
    assert (out["log1p_post_count"] == np.log1p(df["post_count"])).all()
    assert out["has_posts"].tolist() == [0, 1, 1, 0]
    # log1p(0) == 0 — empty slot gets exactly 0.
    assert out.loc[0, "log1p_post_count"] == 0.0
    assert out.loc[3, "log1p_post_count"] == 0.0


def test_derive_post_count_features_idempotent():
    df = pd.DataFrame({"post_count": [0, 7]})
    once  = mg.derive_post_count_features(df.copy())
    twice = mg.derive_post_count_features(once.copy())
    pd.testing.assert_frame_equal(once, twice)


def test_derive_post_count_features_no_op_when_post_count_missing():
    df = pd.DataFrame({"foo": [1, 2]})
    out = mg.derive_post_count_features(df.copy())
    assert "log1p_post_count" not in out.columns
    assert "has_posts" not in out.columns


# ---------------------------------------------------------------------------
# Full pipeline: empty slot bears neutral fills + derived columns at 0
# ---------------------------------------------------------------------------

def test_empty_slot_after_fill_and_derive_is_fully_zero_neutral():
    """One slot with no posts: bullishness should be 0.5, score stats 0,
    log1p_post_count = 0, has_posts = 0."""
    sent_cols = [
        "post_count", "vader_directional_post_count",
        "vader_bullishness_ratio",
        "vader_title_score_mean", "vader_title_score_std",
    ]
    df = pd.DataFrame({
        "ticker":                       ["BTC"],
        "timestamp":                    pd.date_range("2024-01-01", periods=1, tz="UTC"),
        "post_count":                   [0],
        "vader_directional_post_count": [np.nan],
        "vader_bullishness_ratio":      [np.nan],
        "vader_title_score_mean":       [np.nan],
        "vader_title_score_std":        [np.nan],
    })
    df = mg.fill_missing_sentiment(df, sent_cols)
    df = mg.derive_post_count_features(df)
    row = df.iloc[0]
    assert row["vader_bullishness_ratio"]      == 0.5
    assert row["vader_title_score_mean"]       == 0.0
    assert row["vader_title_score_std"]        == 0.0
    assert row["vader_directional_post_count"] == 0
    assert row["log1p_post_count"]             == 0.0
    assert row["has_posts"]                    == 0
