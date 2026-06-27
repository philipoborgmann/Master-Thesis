"""Aufgabe 8 Section E — completed-slot + post-cutoff + predictor/target
interval separation.

These tests pin the temporal-leakage rules so a regression cannot land
silently before the production run.
"""
from __future__ import annotations

import pandas as pd
import pytest

from thesis_pipeline.diagnostics.leakage_checks import (
    assert_posts_respect_cutoff,
    assert_no_target_interval_posts_in_predictors,
)
from thesis_pipeline.diagnostics.timing_audit import (
    CompletedSlotAssumption, assert_only_completed_slots,
)


# ---------------------------------------------------------------------------
# Completed-slot rule
# ---------------------------------------------------------------------------

def _aggregated(slot_starts, horizon_seconds):
    return pd.DataFrame({
        "timestamp": pd.to_datetime(slot_starts, utc=True),
        "ticker": "BTC",
        "value": list(range(len(slot_starts))),
    })


def test_completed_slot_passes_when_slot_ends_at_or_before_cutoff():
    """A 6h slot starting 2024-01-10 18:00 ends at 2024-01-11 00:00. With
    cutoff = 2024-01-11 00:00 the slot is at-the-boundary and is kept."""
    df = _aggregated(["2024-01-10 18:00:00"], 6 * 3600)
    out = assert_only_completed_slots(
        df, horizon="6h",
        observation_cutoff=pd.Timestamp("2024-01-11 00:00:00", tz="UTC"),
    )
    assert len(out) == 1


def test_completed_slot_drops_partial_final_slot():
    df = _aggregated(["2024-01-10 18:00:00", "2024-01-11 00:00:00"], 6 * 3600)
    out = assert_only_completed_slots(
        df, horizon="6h",
        observation_cutoff=pd.Timestamp("2024-01-11 02:00:00", tz="UTC"),
    )
    # The second slot ends at 06:00 which is past the 02:00 cutoff → dropped.
    assert len(out) == 1
    assert out["timestamp"].iat[0] == pd.Timestamp("2024-01-10 18:00:00", tz="UTC")


def test_completed_slot_without_cutoff_warns_but_keeps_rows(caplog):
    import logging
    df = _aggregated(["2024-01-10 18:00:00"], 6 * 3600)
    with caplog.at_level(logging.WARNING, logger="thesis_pipeline.completed_slot"):
        out = assert_only_completed_slots(df, horizon="6h",
                                            observation_cutoff=None)
    # No rows dropped, unresolved warning emitted.
    pd.testing.assert_frame_equal(out, df)
    assert any("REMAINING ASSUMPTION" in r.getMessage()
                for r in caplog.records)


# ---------------------------------------------------------------------------
# Post cutoff — boundary semantics: ``created <= cutoff`` allowed
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("delta_seconds,allowed", [
    (-1, True),    # strictly before
    (0, True),     # equality allowed
    (1, False),    # strictly after
])
def test_post_cutoff_boundary(delta_seconds, allowed):
    cutoff = pd.Timestamp("2024-01-10 12:00:00", tz="UTC")
    posts = pd.DataFrame({
        "created": [cutoff + pd.Timedelta(seconds=delta_seconds)],
    })
    if allowed:
        assert_posts_respect_cutoff(posts, cutoff=cutoff)
    else:
        with pytest.raises(AssertionError):
            assert_posts_respect_cutoff(posts, cutoff=cutoff)


# ---------------------------------------------------------------------------
# Predictor/target interval separation — fixture with a post arriving
# just after the prediction instant
# ---------------------------------------------------------------------------

def test_post_just_after_prediction_must_not_enter_predictor_stack():
    pred_t = pd.Timestamp("2024-01-10 14:00:00", tz="UTC")
    posts = pd.DataFrame({"created": [
        pd.Timestamp("2024-01-10 13:30:00", tz="UTC"),
        pred_t,                                # equality allowed
        pred_t + pd.Timedelta(seconds=30),     # belongs to target interval
    ]})
    with pytest.raises(AssertionError) as exc:
        assert_no_target_interval_posts_in_predictors(
            posts, prediction_timestamp=pred_t,
        )
    assert "1 post" in str(exc.value)


def test_naive_floor_post_in_same_calendar_bucket_does_not_leak():
    """A daily bucket nominally covers [D, D+1). A post at D+1 - 1ms still
    belongs to the predictor bucket; a post at D + horizon does not.
    Here we exercise the predictor/target rule with a 1h granularity:
    prediction at D 14:00, post at D 14:00:01 — the 14:00 bucket carries
    NO post arriving after the prediction instant."""
    pred_t = pd.Timestamp("2024-01-10 14:00:00", tz="UTC")
    one_microsec_after = pred_t + pd.Timedelta(microseconds=1)
    with pytest.raises(AssertionError):
        assert_no_target_interval_posts_in_predictors(
            pd.DataFrame({"created": [one_microsec_after]}),
            prediction_timestamp=pred_t,
        )


# ---------------------------------------------------------------------------
# Unresolved-warning text remains discoverable
# ---------------------------------------------------------------------------

def test_completed_slot_unresolved_warning_text_present():
    msg = CompletedSlotAssumption.UNRESOLVED_WARNING
    assert "REMAINING ASSUMPTION" in msg
    assert "observation cutoff" in msg


# ---------------------------------------------------------------------------
# Regression: caplog must capture the unresolved-warning record even when
# ``get_logger()`` has already wired its own stderr handler (Windows
# failure path — propagate=False used to block the propagation to root).
# ---------------------------------------------------------------------------

def test_completed_slot_warning_propagates_to_caplog(caplog):
    import logging
    from thesis_pipeline.logging_utils import get_logger

    # Trigger the production logger configuration up-front.
    get_logger()
    df = _aggregated(["2024-01-10 18:00:00"], 6 * 3600)
    with caplog.at_level(logging.WARNING,
                          logger="thesis_pipeline.completed_slot"):
        out = assert_only_completed_slots(df, horizon="6h",
                                            observation_cutoff=None)
    # Frame unchanged + exactly one captured warning record.
    pd.testing.assert_frame_equal(out, df)
    relevant = [r for r in caplog.records
                if "REMAINING ASSUMPTION" in r.getMessage()]
    assert len(relevant) == 1
    assert relevant[0].levelno == logging.WARNING
