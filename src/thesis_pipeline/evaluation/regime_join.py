"""Shared availability-based regime as-of helper (commit 3 Section E).

Every production regime join — H2/H3 difference-in-improvement,
supplementary regime McNemar, descriptive volatility / market-cap
stratification, interaction stratification — must go through this one
helper so the strict-< availability semantics are guaranteed everywhere.

The implementation is a thin wrapper around ``pd.merge_asof`` with:

* ``left_on  = "timestamp"``
* ``right_on = "regime_available_at"``
* ``by       = "ticker"``
* ``direction = "backward"``
* ``allow_exact_matches = False``  (strict ``<``)

For every matched row the helper post-asserts that
``regime_available_at < timestamp`` so a future refactor cannot silently
relax the invariant.

Returned columns
----------------
* ``regime_col``                — the regime label (verbatim from the lookup);
* ``regime_source_date``        — original source date of the regime
                                   (alias-prefixed when the caller passes
                                   ``column_prefix``);
* ``regime_available_at``       — UTC instant the regime becomes usable;
* ``effective_regime_lag_days`` — fractional days between the matched
                                   regime's availability and the
                                   prediction timestamp.

When the caller passes ``column_prefix="vol"`` (resp. ``"mcap"``) the
helper renames the regime / source-date / available-at / lag columns to
``vol_regime`` / ``vol_regime_source_date`` / ``vol_regime_available_at``
/ ``vol_regime_lag_days``. This lets a downstream consumer attach BOTH
volatility and market-cap regimes side-by-side without collisions.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

from ..logging_utils import get_logger


REGIME_JOIN_STRATEGY = "asof_backward_strict"


def _prepare_lookup(regime_lookup: pd.DataFrame,
                    regime_col: str) -> pd.DataFrame:
    """Standardise the regime lookup to
    ``(ticker, regime_available_at, regime_source_date, regime_col)``.

    Falls back gracefully when the lookup carries only a ``date`` or
    ``timestamp`` column (legacy v3 layout) by deriving
    ``regime_available_at = date`` — the convention the shifted v4
    helpers already use.
    """
    if regime_lookup is None or regime_lookup.empty:
        return pd.DataFrame(columns=["ticker", "regime_available_at",
                                       "regime_source_date", regime_col])
    look = regime_lookup.copy()
    if "regime_available_at" in look.columns:
        avail = pd.to_datetime(look["regime_available_at"], utc=True,
                                errors="coerce")
    elif "date" in look.columns:
        avail = pd.to_datetime(look["date"], utc=True, errors="coerce")
        if avail.isna().any():
            naive = pd.to_datetime(look["date"], errors="coerce")
            avail = naive.dt.tz_localize("UTC", nonexistent="shift_forward")
        avail = avail.dt.normalize()
    elif "timestamp" in look.columns:
        avail = pd.to_datetime(look["timestamp"], utc=True, errors="coerce")
    else:
        raise ValueError(
            "attach_regime_asof: regime_lookup must have a "
            "'regime_available_at', 'date' or 'timestamp' column."
        )
    # Force ns resolution so the merge-asof key dtype matches the signals.
    look["regime_available_at"] = avail.astype("datetime64[ns, UTC]")
    if "regime_source_date" in look.columns:
        look["regime_source_date"] = pd.to_datetime(
            look["regime_source_date"], utc=True, errors="coerce"
        ).astype("datetime64[ns, UTC]")
    else:
        # Pre-availability lookups: assume the source date = available_at
        # − 1 day (the v3-style shifted convention).
        look["regime_source_date"] = (
            look["regime_available_at"] - pd.Timedelta(days=1)
        )
    keep = ["ticker", "regime_available_at", "regime_source_date", regime_col]
    look = look[keep].copy()
    return look.dropna(subset=["regime_available_at", regime_col])


def attach_regime_asof(signals: pd.DataFrame,
                       regime_lookup: pd.DataFrame | None,
                       *,
                       regime_col: str,
                       column_prefix: str | None = None) -> pd.DataFrame:
    """Per-ticker availability-based as-of join (strict ``<``).

    Parameters
    ----------
    signals
        Long-form signal frame carrying ``timestamp`` (datetime, will be
        coerced to UTC) and ``ticker``.
    regime_lookup
        Long-form regime lookup. Accepts either the v4 layout
        (``regime_available_at`` + ``regime_source_date``) or the v3
        legacy layout (``date`` only).
    regime_col
        Name of the regime label column in the lookup (e.g.
        ``"vol_regime"`` or ``"mcap_regime"``).
    column_prefix
        Optional disambiguation prefix for the diagnostics columns so a
        caller attaching both volatility and market-cap regimes does not
        collide on ``regime_source_date`` / ``regime_available_at`` /
        ``effective_regime_lag_days``. The regime label column itself is
        not renamed (callers usually pass distinct ``regime_col`` values).

    Returns
    -------
    DataFrame with the signal columns plus the attached regime label,
    source date, availability instant and effective lag (days).
    """
    if signals is None or signals.empty:
        return signals if signals is not None else pd.DataFrame()

    sig = signals.copy()
    # Force ns resolution on both sides so merge_asof never trips on a
    # us/ms vs ns dtype mismatch — pandas requires identical key dtypes.
    sig["timestamp"] = pd.to_datetime(sig["timestamp"], utc=True,
                                       errors="coerce").astype("datetime64[ns, UTC]")
    sig["ticker"] = sig["ticker"].astype(str).str.upper()

    if regime_lookup is None or regime_lookup.empty:
        # Preserve the contract: emit the regime / lag columns as NaN
        # so downstream code can still inspect them without crashing.
        out = sig.copy()
        out[regime_col] = np.nan
        out["regime_source_date"]   = pd.NaT
        out["regime_available_at"]  = pd.NaT
        out["effective_regime_lag_days"] = np.nan
        return _maybe_prefix(out, column_prefix, regime_col)

    look = _prepare_lookup(regime_lookup, regime_col)
    look["ticker"] = look["ticker"].astype(str).str.upper()
    # ``merge_asof`` requires both sides to be globally sorted on the
    # ``on`` key; the ``by`` key only secondary-groups the comparison.
    look_sorted = look.sort_values("regime_available_at").reset_index(drop=True)
    sig_sorted = sig.sort_values("timestamp").reset_index()
    joined = pd.merge_asof(
        sig_sorted, look_sorted,
        left_on="timestamp", right_on="regime_available_at",
        by="ticker",
        direction="backward",
        allow_exact_matches=False,
    )
    # Restore original signal-row ordering.
    joined = joined.sort_values("index").drop(columns=["index"]).reset_index(drop=True)

    # Strict-< invariant assertion. Raised in tests and asserts in
    # production so the contract cannot regress silently.
    matched = joined[regime_col].notna()
    if matched.any():
        bad = joined.loc[matched, "regime_available_at"] >= joined.loc[matched, "timestamp"]
        if bad.any():
            offenders = joined.loc[matched].loc[bad]
            get_logger().error(
                "attach_regime_asof: strict-< invariant violated on %d rows; "
                "regime_available_at must be < timestamp for every matched row",
                int(len(offenders)),
            )
            raise AssertionError(
                "attach_regime_asof: regime_available_at must be strictly "
                "before prediction timestamp for every matched row"
            )

    lag = (joined["timestamp"]
           - joined["regime_available_at"]).dt.total_seconds() / 86400.0
    joined["effective_regime_lag_days"] = lag.astype(float)
    return _maybe_prefix(joined, column_prefix, regime_col)


def _maybe_prefix(df: pd.DataFrame, prefix: str | None,
                  regime_col: str) -> pd.DataFrame:
    """Rename the diagnostics columns when the caller asks for a prefix.

    The regime label column itself is left alone — callers usually pass
    a uniquely named regime column already (``vol_regime`` /
    ``mcap_regime``).
    """
    if not prefix:
        return df
    mapping = {
        "regime_source_date":          f"{prefix}_regime_source_date",
        "regime_available_at":         f"{prefix}_regime_available_at",
        "effective_regime_lag_days":   f"{prefix}_regime_lag_days",
    }
    return df.rename(columns={k: v for k, v in mapping.items() if k in df.columns})


def regime_lag_summary(joined: pd.DataFrame,
                       *,
                       regime_col: str,
                       lag_col: str = "effective_regime_lag_days") -> dict:
    """Tiny convenience helper used by H2/H3 + the descriptive paths.

    Returns the share-unmatched + median/min/max/share-lag-gt-1-day
    diagnostics in one dict so each consumer reports the same fields.
    """
    n = int(len(joined))
    if n == 0:
        return {
            "share_unmatched_regime":      1.0,
            "median_regime_lag_days":      float("nan"),
            "min_regime_lag_days":         float("nan"),
            "max_regime_lag_days":         float("nan"),
            "share_regime_lag_gt_1_day":   float("nan"),
        }
    matched = joined[regime_col].notna()
    share_unmatched = float(1.0 - matched.sum() / n) if n else 1.0
    if not matched.any() or lag_col not in joined.columns:
        return {
            "share_unmatched_regime":      share_unmatched,
            "median_regime_lag_days":      float("nan"),
            "min_regime_lag_days":         float("nan"),
            "max_regime_lag_days":         float("nan"),
            "share_regime_lag_gt_1_day":   float("nan"),
        }
    lag = joined.loc[matched, lag_col].astype(float).dropna()
    if lag.empty:
        return {
            "share_unmatched_regime":      share_unmatched,
            "median_regime_lag_days":      float("nan"),
            "min_regime_lag_days":         float("nan"),
            "max_regime_lag_days":         float("nan"),
            "share_regime_lag_gt_1_day":   float("nan"),
        }
    return {
        "share_unmatched_regime":      share_unmatched,
        "median_regime_lag_days":      float(lag.median()),
        "min_regime_lag_days":         float(lag.min()),
        "max_regime_lag_days":         float(lag.max()),
        "share_regime_lag_gt_1_day":   float((lag > 1.0).mean()),
    }
