"""Tests for the v4 leakage assertions (Aufgabe 7 + Aufgabe 8.D.4/8.D.5/8.E).

Two layers:

* walk-forward sanity (preserved from v3) — :func:`assert_train_precedes_test`,
  :func:`scaler_fit_scope_ok`;
* final-frame leakage —
  :func:`assert_no_forbidden_engagement_features`,
  :func:`assert_market_cap_asof_correct`,
  :func:`assert_posts_respect_cutoff`,
  :func:`assert_no_target_interval_posts_in_predictors`.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from thesis_pipeline.diagnostics import leakage_checks as lc


# ---------------------------------------------------------------------------
# Walk-forward sanity (regression — must remain UNCHANGED)
# ---------------------------------------------------------------------------

def test_assert_train_precedes_test_accepts_separate_ranges():
    lc.assert_train_precedes_test([0, 1, 2], [3, 4, 5])  # no raise


def test_assert_train_precedes_test_rejects_overlap():
    with pytest.raises(AssertionError):
        lc.assert_train_precedes_test([0, 1, 2], [2, 3, 4])


def test_assert_train_precedes_test_empty_inputs_are_ok():
    lc.assert_train_precedes_test([], [3, 4])
    lc.assert_train_precedes_test([1, 2], [])


def test_scaler_fit_scope_ok_accepts_subset():
    assert lc.scaler_fit_scope_ok([0, 1, 2], [0, 1, 2, 3])


def test_scaler_fit_scope_ok_rejects_extra_index():
    assert not lc.scaler_fit_scope_ok([0, 1, 4], [0, 1, 2, 3])


# ---------------------------------------------------------------------------
# Forbidden engagement columns (Section C.1)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("col", [
    "score", "upvote_ratio", "num_comments", "engagement_weight",
])
def test_forbidden_raw_engagement_column_raises(col):
    df = pd.DataFrame({col: [1.0], "log_return_t": [0.0]})
    with pytest.raises(AssertionError, match="forbidden engagement"):
        lc.assert_no_forbidden_engagement_features(df)


def test_forbidden_weighted_mean_variant_raises():
    df = pd.DataFrame({"foo_weighted_mean": [1.0]})
    with pytest.raises(AssertionError, match="weighted-mean"):
        lc.assert_no_forbidden_engagement_features(df)


@pytest.mark.parametrize("col", [
    "score_mean",                # legitimate aggregate of sentiment polarity
    "post_count",
    "log1p_post_count",
    "has_posts",
    "directional_post_count",
    "vader_title_score_mean",
    "log_return_t", "cum_log_return_7d", "realized_vol_14d",
])
def test_allowed_columns_pass(col):
    df = pd.DataFrame({col: [0.0]})
    lc.assert_no_forbidden_engagement_features(df)  # no raise


def test_assert_lists_every_offender():
    df = pd.DataFrame({
        "score": [1.0], "upvote_ratio": [1.0], "x_weighted_mean": [1.0],
        "log_return_t": [0.0],
    })
    with pytest.raises(AssertionError) as exc:
        lc.assert_no_forbidden_engagement_features(df)
    msg = str(exc.value)
    for tok in ("score", "upvote_ratio", "x_weighted_mean"):
        assert tok in msg


# ---------------------------------------------------------------------------
# Market-cap availability assertion (Section C.2 / Aufgabe 8.D.5)
# ---------------------------------------------------------------------------

def _mcap_rows(deltas_days):
    ts0 = pd.Timestamp("2024-01-10", tz="UTC")
    return pd.DataFrame({
        "ticker": "BTC",
        "timestamp":                 [ts0] * len(deltas_days),
        "market_cap_available_at":   [ts0 - pd.Timedelta(days=d)
                                       for d in deltas_days],
    })


def test_market_cap_strictly_before_passes():
    lc.assert_market_cap_asof_correct(_mcap_rows([1, 2, 3]))  # no raise


def test_market_cap_equal_raises():
    df = _mcap_rows([0])  # available_at == prediction_timestamp
    with pytest.raises(AssertionError, match="strictly before"):
        lc.assert_market_cap_asof_correct(df)


def test_market_cap_after_raises():
    df = _mcap_rows([-1])  # available_at > prediction_timestamp
    with pytest.raises(AssertionError):
        lc.assert_market_cap_asof_correct(df)


def test_market_cap_unmatched_rows_skipped():
    """Rows whose market-cap as-of value is NaN (unmatched) must NOT
    fail the assertion."""
    df = pd.DataFrame({
        "ticker": "BTC",
        "timestamp":                [pd.Timestamp("2024-01-10", tz="UTC")],
        "market_cap_available_at":  [pd.NaT],
    })
    lc.assert_market_cap_asof_correct(df)  # no raise


def test_market_cap_error_message_reports_first_offender():
    df = _mcap_rows([0, -1])
    with pytest.raises(AssertionError) as exc:
        lc.assert_market_cap_asof_correct(df)
    msg = str(exc.value)
    assert "BTC" in msg
    assert "2024-01-10" in msg


# ---------------------------------------------------------------------------
# Post-cutoff assertion (Section E.2)
# ---------------------------------------------------------------------------

def test_post_cutoff_excludes_after_strict():
    cutoff = pd.Timestamp("2024-01-10 12:00:00", tz="UTC")
    posts = pd.DataFrame({"created": [
        pd.Timestamp("2024-01-10 11:59:59", tz="UTC"),  # before
        pd.Timestamp("2024-01-10 12:00:00", tz="UTC"),  # equal — included
    ]})
    lc.assert_posts_respect_cutoff(posts, cutoff=cutoff)  # no raise


def test_post_cutoff_raises_when_after():
    cutoff = pd.Timestamp("2024-01-10 12:00:00", tz="UTC")
    posts = pd.DataFrame({"created": [
        pd.Timestamp("2024-01-10 12:00:01", tz="UTC"),
    ]})
    with pytest.raises(AssertionError, match="created after cutoff"):
        lc.assert_posts_respect_cutoff(posts, cutoff=cutoff)


# ---------------------------------------------------------------------------
# Predictor/target interval separation (Section E.3)
# ---------------------------------------------------------------------------

def test_target_interval_post_in_predictors_raises():
    pred_t = pd.Timestamp("2024-01-10 14:00:00", tz="UTC")
    posts = pd.DataFrame({"created": [
        pd.Timestamp("2024-01-10 14:00:00", tz="UTC"),  # equal — allowed
        pd.Timestamp("2024-01-10 14:00:01", tz="UTC"),  # strictly after
    ]})
    with pytest.raises(AssertionError,
                        match="predictor/target separation"):
        lc.assert_no_target_interval_posts_in_predictors(
            posts, prediction_timestamp=pred_t,
        )


def test_target_interval_no_violation_passes():
    pred_t = pd.Timestamp("2024-01-10 14:00:00", tz="UTC")
    posts = pd.DataFrame({"created": [
        pd.Timestamp("2024-01-10 13:59:59", tz="UTC"),
        pd.Timestamp("2024-01-10 14:00:00", tz="UTC"),  # equal — allowed
    ]})
    lc.assert_no_target_interval_posts_in_predictors(
        posts, prediction_timestamp=pred_t,
    )


# ---------------------------------------------------------------------------
# Unified audit
# ---------------------------------------------------------------------------

def test_run_feature_leakage_audit_summary_on_clean_frame():
    df = pd.DataFrame({
        "ticker":    ["BTC"],
        "timestamp": [pd.Timestamp("2024-01-10", tz="UTC")],
        "market_cap_available_at":
                     [pd.Timestamp("2024-01-09", tz="UTC")],
        "log_return_t": [0.01],
        "score_mean":   [0.42],
    })
    summary = lc.run_feature_leakage_audit(df)
    assert summary["forbidden_engagement_check"] == "PASS"
    assert summary["market_cap_asof_check"] == "PASS"
    assert summary["n_rows_audited"] == 1


def test_run_feature_leakage_audit_raises_on_offender():
    df = pd.DataFrame({"score": [1.0]})
    with pytest.raises(AssertionError):
        lc.run_feature_leakage_audit(df)
