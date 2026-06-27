"""Tests for Variante A of the sentiment aggregation pipeline.

Designed to eliminate identified engagement-related look-ahead leakage —
no engagement weighting may appear in any output of this module.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from thesis_pipeline.sentiment import aggregate as agg


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _synthetic_posts(n_posts: int = 40, n_tickers: int = 2,
                     n_days: int = 5) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    rows = []
    for i in range(n_posts):
        day = i % n_days
        tk  = ["BTC", "ETH"][i % n_tickers]
        for model in ("vader", "cryptobert"):
            pass
        rows.append({
            "date":     pd.Timestamp("2024-01-01", tz="UTC") + pd.Timedelta(days=day),
            "ticker":   tk,
            "selftext": "" if (i % 3 == 0) else f"body {i}",
            "vader_title_score":     float(rng.uniform(-1, 1)),
            "vader_selftext_score":  float(rng.uniform(-1, 1)),
            "vader_title_sentiment": ["positive", "negative", "neutral"][i % 3],
            "cryptobert_title_score":     float(rng.uniform(-1, 1)),
            "cryptobert_selftext_score":  float(rng.uniform(-1, 1)),
            "cryptobert_title_sentiment": ["positive", "negative", "neutral"][(i + 1) % 3],
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# compute_engagement_weight is gone (Variante A)
# ---------------------------------------------------------------------------

def test_compute_engagement_weight_is_removed():
    """Variante A — no engagement weighting may exist in this module."""
    assert not hasattr(agg, "compute_engagement_weight"), (
        "compute_engagement_weight must be removed (Variante A)."
    )


def test_build_post_level_features_does_not_add_engagement_weight():
    df = _synthetic_posts()
    out = agg.build_post_level_features(df.copy())
    assert "engagement_weight" not in out.columns
    # And — defensively — none of the raw engagement columns leaked into the
    # post-level frame from outside this test (we never put them there).
    for forbidden in ("score", "upvote_ratio", "num_comments",
                       "total_awards_received", "gilded"):
        assert forbidden not in out.columns


def test_build_post_level_features_does_not_require_engagement_columns():
    """Pre-Variante-A the function required score/upvote_ratio/num_comments.
    Variante A removes that hard dependency."""
    df = _synthetic_posts().drop(
        columns=[], errors="ignore",
    )
    # Should not raise even though score/upvote_ratio/num_comments are absent.
    out = agg.build_post_level_features(df.copy())
    assert isinstance(out, pd.DataFrame)


# ---------------------------------------------------------------------------
# aggregate_to_horizon — no *_weighted_mean, directional_post_count present
# ---------------------------------------------------------------------------

def test_aggregate_to_horizon_drops_weighted_mean_columns():
    df = _synthetic_posts()
    df = agg.build_post_level_features(df.copy())
    out = agg.aggregate_to_horizon(df, freq="1D", horizon_label="1d")
    weighted = [c for c in out.columns if c.endswith("_weighted_mean")]
    assert not weighted, f"Variante A leaves no *_weighted_mean cols, found {weighted}"


def test_aggregate_to_horizon_adds_directional_post_count_per_model():
    df = _synthetic_posts()
    df = agg.build_post_level_features(df.copy())
    out = agg.aggregate_to_horizon(df, freq="1D", horizon_label="1d")
    for m in ("vader", "cryptobert"):
        col = f"{m}_directional_post_count"
        assert col in out.columns, f"missing {col}"
        # Must equal #positive + #negative posts in the slot.
        # All values should be >= 0 and <= post_count.
        assert (out[col] >= 0).all()
        assert (out[col] <= out["post_count"]).all()
        # And integer-valued.
        assert pd.api.types.is_integer_dtype(out[col].dropna())


def test_aggregate_to_horizon_keeps_mean_median_std_branches():
    df = _synthetic_posts()
    df = agg.build_post_level_features(df.copy())
    out = agg.aggregate_to_horizon(df, freq="1D", horizon_label="1d")
    for m in ("vader", "cryptobert"):
        for stat in ("mean", "median", "std"):
            col = f"{m}_title_score_{stat}"
            assert col in out.columns, f"missing {col}"


# ---------------------------------------------------------------------------
# Engagement plot helpers are gone
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name", [
    "plot_engagement_weight_distribution",
    "plot_raw_vs_weighted_comparison",
])
def test_engagement_plot_helpers_removed(name):
    assert not hasattr(agg, name), (
        f"{name} must be removed under Variante A — engagement weighting is gone."
    )


# ---------------------------------------------------------------------------
# Bucket only fully-closed slots: floor(date, freq) puts the post into the
# slot it belongs to (its start), never into a future slot.
# ---------------------------------------------------------------------------

def test_bucketing_uses_floor_so_no_partial_future_slot():
    """A post created at 2024-01-01 23:59 UTC belongs in the 2024-01-01 day
    slot, not 2024-01-02. The aggregation must not borrow from the next
    interval — fundamental remaining-assumption check 1."""
    df = pd.DataFrame({
        "date":     [pd.Timestamp("2024-01-01 23:59:00", tz="UTC"),
                     pd.Timestamp("2024-01-02 00:00:00", tz="UTC")],
        "ticker":   ["BTC", "BTC"],
        "selftext": ["", ""],
        "vader_title_score":     [0.5, -0.5],
        "vader_selftext_score":  [np.nan, np.nan],
        "vader_title_sentiment": ["positive", "negative"],
        "cryptobert_title_score":     [0.5, -0.5],
        "cryptobert_selftext_score":  [np.nan, np.nan],
        "cryptobert_title_sentiment": ["positive", "negative"],
    })
    df = agg.build_post_level_features(df)
    out = agg.aggregate_to_horizon(df, freq="1D", horizon_label="1d")
    # Two distinct day slots — the 23:59 post must NOT have been pulled
    # into the 2024-01-02 slot.
    slots = sorted(out["timestamp"].astype(str).tolist())
    assert slots[0].startswith("2024-01-01")
    assert slots[1].startswith("2024-01-02")
    # And the values stay in their original day.
    btc_d1 = out[(out["ticker"] == "BTC") &
                  (out["timestamp"] == pd.Timestamp("2024-01-01", tz="UTC"))]
    assert float(btc_d1["vader_title_score_mean"].iloc[0]) == pytest.approx(0.5)
