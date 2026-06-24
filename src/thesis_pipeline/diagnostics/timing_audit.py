"""Timing / availability audit utilities (Section F + G + H).

These diagnostics complement — they do NOT replace — the rolling-window /
as-of-merge rules in :mod:`thesis_pipeline.price.features` and
:mod:`thesis_pipeline.sentiment.aggregate`. The intent is to surface
unresolved methodological assumptions, not to silently change behaviour.

Each function returns a plain pandas DataFrame so the caller can write it
to ``Data/Features/<report>.csv``, log it, or assert on it from tests.
"""

from __future__ import annotations

from typing import Iterable, Optional

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Section F — effective market-cap source-lag diagnostic
# ---------------------------------------------------------------------------

def market_cap_lag_audit(features: pd.DataFrame) -> pd.DataFrame:
    """One row per ``(ticker, horizon, timestamp)`` with the effective
    market-cap source lag.

    Required columns on ``features``:

      * ``timestamp``               — the prediction timestamp.
      * ``ticker``                  — the coin.
      * ``horizon``                 — the bar granularity ("1d" / "6h" / "1h").
      * ``market_cap_source_date``  — the calendar date D the matched CMC
        value belongs to.
      * ``market_cap_available_at`` — the assumed availability instant
        (D + MARKET_CAP_AVAILABILITY_LAG; default D + 1 day, 00:00 UTC).

    Output columns: the four required ones plus
    ``effective_source_lag_days``, computed as
    ``(timestamp − market_cap_source_date) / 1 day``. The value is
    always ≥ 1 by construction (the as-of merge uses
    ``allow_exact_matches=False`` and a 1-day availability offset);
    a 1d-bar at 00:00 UTC therefore receives the value from
    *two* source dates ago, which is what makes this audit useful.
    """
    needed = {"timestamp", "ticker", "horizon",
              "market_cap_source_date", "market_cap_available_at"}
    missing = needed - set(features.columns)
    if missing:
        raise ValueError(
            f"market_cap_lag_audit: missing required columns: {sorted(missing)}"
        )
    out = features.loc[:, sorted(needed)].copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True, errors="coerce")
    out["market_cap_source_date"] = pd.to_datetime(
        out["market_cap_source_date"], utc=True, errors="coerce")
    out["market_cap_available_at"] = pd.to_datetime(
        out["market_cap_available_at"], utc=True, errors="coerce")
    delta = (out["timestamp"] - out["market_cap_source_date"])
    out["effective_source_lag_days"] = (
        delta.dt.total_seconds() / 86_400.0
    ).astype(float)
    # Strict-before contract: every matched row must satisfy
    # market_cap_available_at < timestamp.
    out["available_strictly_before"] = (
        out["market_cap_available_at"] < out["timestamp"]
    )
    return out.reset_index(drop=True)


def market_cap_lag_summary(audit: pd.DataFrame) -> dict:
    """Aggregate the audit into a small dict suitable for the
    feature-generation report. Reports per-horizon stats so 1d's typical
    2-day effective lag (vs intraday's ~1 day) is visible.
    """
    if audit.empty:
        return {
            "median_effective_market_cap_lag_days": float("nan"),
            "min_effective_market_cap_lag_days":    float("nan"),
            "max_effective_market_cap_lag_days":    float("nan"),
            "share_effective_lag_gt_1_day":         float("nan"),
            "all_rows_available_strictly_before":   True,
            "n_rows":                                0,
        }
    lag = audit["effective_source_lag_days"]
    strict = bool(audit["available_strictly_before"].all())
    return {
        "median_effective_market_cap_lag_days":
            float(lag.median()),
        "min_effective_market_cap_lag_days":
            float(lag.min()),
        "max_effective_market_cap_lag_days":
            float(lag.max()),
        "share_effective_lag_gt_1_day":
            float((lag > 1.0).mean()),
        "all_rows_available_strictly_before":
            strict,
        "n_rows": int(len(audit)),
    }


# ---------------------------------------------------------------------------
# Section G — bar-grid regularity diagnostic
# ---------------------------------------------------------------------------

# Expected median gap per horizon, in seconds. Matches BARS_PER_DAY in
# thesis_pipeline.price.features.
_EXPECTED_GAP_SECONDS = {
    "1d":     86_400.0,
    "6h":     21_600.0,
    "1h":      3_600.0,
    "15min":     900.0,
    "4h":     14_400.0,
}


def bar_grid_audit(features: pd.DataFrame,
                   horizons: Optional[Iterable[str]] = None) -> pd.DataFrame:
    """Per-(ticker, horizon) diagnostic of bar-grid regularity.

    The calendar-consistent rolling windows in price/features.py assume
    bars arrive at the expected cadence (1 day, 6 h, 1 h, …). Real data
    can have gaps (exchange downtime, missing files). This audit
    surfaces:

      * ``n_observed_bars``         — bars actually present.
      * ``expected_frequency_secs`` — canonical bar period for the horizon.
      * ``median_timestamp_gap_secs``
      * ``maximum_timestamp_gap_secs``
      * ``n_missing_expected_bars``  — gaps wide enough to swallow ≥ 1
        canonical bar.
      * ``missing_bar_rate``         — share of expected bars not observed.

    A missing bar count of 0 → the grid is regular. Non-zero → the
    rolling implementation still produces values, but the
    ``window_bars = days × BARS_PER_DAY[horizon]`` mapping no longer
    corresponds exactly to that calendar period.
    """
    needed = {"timestamp", "ticker", "horizon"}
    missing = needed - set(features.columns)
    if missing:
        raise ValueError(
            f"bar_grid_audit: missing required columns: {sorted(missing)}"
        )
    rows: list[dict] = []
    horizons_iter = horizons if horizons is not None else features["horizon"].dropna().unique()
    for hz in horizons_iter:
        hz_str = str(hz)
        expected = _EXPECTED_GAP_SECONDS.get(hz_str)
        for tk, grp in features[features["horizon"] == hz_str].groupby("ticker"):
            ts = pd.to_datetime(grp["timestamp"], utc=True, errors="coerce").dropna()
            ts = ts.sort_values().reset_index(drop=True)
            n = int(len(ts))
            if n < 2 or expected is None:
                rows.append({
                    "ticker": str(tk), "horizon": hz_str,
                    "n_observed_bars":          n,
                    "expected_frequency_secs":  float(expected) if expected else float("nan"),
                    "median_timestamp_gap_secs": float("nan"),
                    "maximum_timestamp_gap_secs": float("nan"),
                    "n_missing_expected_bars":  0,
                    "missing_bar_rate":         float("nan"),
                })
                continue
            gaps = ts.diff().dropna().dt.total_seconds()
            # Missing bars: count any gap > 1.5 × expected and divide by
            # expected (rounded down) to get the number of fully-missing
            # canonical bars inside the gap.
            tol = expected * 1.5
            missing_bars = int(((gaps - expected) // expected
                                ).where(gaps > tol, 0).clip(lower=0).sum())
            expected_n = int(((ts.iloc[-1] - ts.iloc[0]).total_seconds()
                              / expected) + 1)
            rows.append({
                "ticker": str(tk), "horizon": hz_str,
                "n_observed_bars":          n,
                "expected_frequency_secs":  float(expected),
                "median_timestamp_gap_secs": float(gaps.median()),
                "maximum_timestamp_gap_secs": float(gaps.max()),
                "n_missing_expected_bars":  missing_bars,
                "missing_bar_rate":
                    float(missing_bars / expected_n) if expected_n else float("nan"),
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Section H — completed-slot assumption
# ---------------------------------------------------------------------------

class CompletedSlotAssumption:
    """Document the (unresolved) "only fully-closed slots are emitted"
    assumption from the PRD.

    The Reddit sentiment source does NOT carry an authoritative
    observation cutoff — we have post-level ``created`` and ``retrieved``
    timestamps, but no dataset-level "data complete up to T" marker.
    Using ``retrieved`` as a cutoff is leakage-positive (it postdates
    the predicted market move), and we explicitly DO NOT use it for
    predictive features.

    Concrete consequences:

      * ``aggregate_to_horizon()`` floors each post's ``created`` into a
        slot. A post in the last partially-observed slot will be emitted
        as if the slot were complete.
      * If you DO have a reliable observation cutoff (e.g. the extraction
        timestamp of the upstream dump), pass it to
        :func:`assert_only_completed_slots` so partial slots get dropped
        with a clear warning.
      * Without a cutoff, the assumption is **declared as unresolved**:
        we emit a defensive warning rather than silently dropping the
        final slot.

    See ``docs/refactor_log.md`` for the methodological note.
    """

    UNRESOLVED_WARNING = (
        "REMAINING ASSUMPTION (completed-slot): the sentiment source "
        "has no dataset-level observation cutoff. aggregate_to_horizon() "
        "emits whatever slot the last post belongs to. Pass a cutoff to "
        "assert_only_completed_slots() if you can derive one from your "
        "extraction; otherwise treat the final slot as potentially "
        "partial."
    )

    @staticmethod
    def horizon_to_slot_size_seconds(horizon: str) -> float:
        return _EXPECTED_GAP_SECONDS.get(str(horizon), float("nan"))


def assert_only_completed_slots(aggregated: pd.DataFrame, *,
                                horizon: str,
                                observation_cutoff: Optional[pd.Timestamp]
                                ) -> pd.DataFrame:
    """Return ``aggregated`` with any partially-observed final slot
    filtered out — IF a cutoff is supplied. Otherwise emit a defensive
    warning describing the unresolved assumption and return the frame
    unchanged.

    A slot ``[ts, ts + slot_size_seconds)`` is "completed" iff
    ``ts + slot_size_seconds <= observation_cutoff``.
    """
    import logging
    log = logging.getLogger("thesis_pipeline.completed_slot")

    if observation_cutoff is None:
        log.warning(CompletedSlotAssumption.UNRESOLVED_WARNING)
        return aggregated
    slot_secs = CompletedSlotAssumption.horizon_to_slot_size_seconds(horizon)
    if not np.isfinite(slot_secs):
        log.warning(
            "assert_only_completed_slots: unknown horizon %r; returning input "
            "unchanged.", horizon,
        )
        return aggregated
    cutoff = pd.Timestamp(observation_cutoff, tz="UTC") if pd.Timestamp(observation_cutoff).tzinfo is None else pd.Timestamp(observation_cutoff)
    out = aggregated.copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True, errors="coerce")
    slot_end = out["timestamp"] + pd.Timedelta(seconds=slot_secs)
    completed_mask = slot_end <= cutoff
    n_dropped = int((~completed_mask).sum())
    if n_dropped:
        log.warning(
            "assert_only_completed_slots: dropping %d slot(s) whose end "
            "extends past observation_cutoff=%s for horizon=%s.",
            n_dropped, cutoff.isoformat(), horizon,
        )
    return out[completed_mask].reset_index(drop=True)
