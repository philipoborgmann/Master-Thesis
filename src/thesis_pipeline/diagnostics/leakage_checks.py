"""Leakage sanity checks — Aufgabe 7.

Two layers of assertions:

* **Walk-forward sanity** — :func:`assert_train_precedes_test` and
  :func:`scaler_fit_scope_ok` guarantee the v3-era walk-forward
  contract still holds at the loop level.
* **Final-frame leakage** — :func:`assert_no_forbidden_engagement_features`
  refuses raw engagement columns + weighted-mean variants in the final
  modeling grid; :func:`assert_market_cap_asof_correct` re-runs the
  strict-availability assertion on the post-merge feature frame.

A convenience entry point :func:`run_feature_leakage_audit` calls every
final-frame assertion and returns a machine-readable summary. Failures
NEVER degrade to warnings — every assertion raises ``AssertionError``
when a violation is found.
"""

from __future__ import annotations

from typing import Iterable, Sequence

import pandas as pd


# ---------------------------------------------------------------------------
# Walk-forward sanity (preserved verbatim from the v3-era checks)
# ---------------------------------------------------------------------------

def assert_train_precedes_test(train_idx: Sequence[int],
                                test_idx: Sequence[int]) -> None:
    """Raise if any training index ≥ any test index."""
    if not len(train_idx) or not len(test_idx):
        return
    if max(train_idx) >= min(test_idx):
        raise AssertionError(
            f"Leakage: max(train_idx)={max(train_idx)} >= "
            f"min(test_idx)={min(test_idx)}"
        )


def scaler_fit_scope_ok(scaler_fit_indices: Sequence[int],
                        train_idx: Sequence[int]) -> bool:
    """True iff every scaler-fit index lives inside the training window."""
    return set(scaler_fit_indices).issubset(set(train_idx))


# ---------------------------------------------------------------------------
# Final-frame leakage assertions (Aufgabe 7 — Sections C.1, C.2)
# ---------------------------------------------------------------------------

#: Raw engagement columns that must NOT appear as modeling features per
#: Variante A. ``score_mean`` / ``num_comments_mean`` etc. are aggregates
#: and remain legal — only the raw post-level columns are forbidden.
FORBIDDEN_ENGAGEMENT_RAW_COLUMNS = frozenset({
    "score",
    "upvote_ratio",
    "num_comments",
    "engagement_weight",
})


def _weighted_mean_offenders(columns: Iterable[str]) -> list[str]:
    return sorted(c for c in columns if str(c).endswith("_weighted_mean"))


def assert_no_forbidden_engagement_features(df: pd.DataFrame) -> None:
    """Raise if ``df`` carries raw engagement columns or any
    ``*_weighted_mean`` variant (Variante A).

    The assertion uses EXACT column-name matching for the raw set so a
    legitimate aggregate like ``score_mean`` (sentiment polarity score
    averaged into a slot) is not rejected just because its name contains
    the substring ``score``.
    """
    raw = sorted(c for c in df.columns if c in FORBIDDEN_ENGAGEMENT_RAW_COLUMNS)
    weighted = _weighted_mean_offenders(df.columns)
    offenders = raw + weighted
    if offenders:
        raise AssertionError(
            "leakage_checks: forbidden engagement columns found in the "
            "final feature frame — raw engagement and weighted-mean "
            f"variants are not allowed: {offenders}"
        )


def assert_market_cap_asof_correct(
    df: pd.DataFrame,
    *,
    prediction_col: str = "timestamp",
    availability_col: str = "market_cap_available_at",
    require_availability_column: bool = True,
) -> None:
    """Raise if any row has ``market_cap_available_at >= prediction_timestamp``.

    Rows without a matched market-cap value (NaN ``availability_col``)
    are skipped — the as-of merge legitimately leaves the first
    observations of each ticker without market-cap data.

    A final modeling frame is expected to carry
    ``market_cap_available_at`` even when individual rows are NaN. The
    column is the contract that lets the audit verify the strict-<
    rule; its complete absence is itself a leakage red flag (the merge
    may have silently dropped the diagnostic). With the default
    ``require_availability_column=True`` the helper raises in that
    case. Callers running the assertion on partial frames (e.g. a
    sentiment-only subset) can opt out by passing ``False``.
    """
    if prediction_col not in df.columns:
        if require_availability_column:
            raise AssertionError(
                "leakage_checks: market_cap_asof check requires "
                f"prediction column {prediction_col!r}"
            )
        return
    if availability_col not in df.columns:
        if require_availability_column:
            raise AssertionError(
                "leakage_checks: market_cap_asof check requires "
                f"availability column {availability_col!r} on the final "
                "feature frame — its absence may indicate the as-of "
                "merge silently dropped the diagnostic"
            )
        return
    avail = pd.to_datetime(df[availability_col], utc=True, errors="coerce")
    pred  = pd.to_datetime(df[prediction_col],  utc=True, errors="coerce")
    matched = avail.notna() & pred.notna()
    if not matched.any():
        return
    bad = matched & (avail >= pred)
    n = int(bad.sum())
    if n == 0:
        return
    first = df.loc[bad].iloc[0]
    raise AssertionError(
        "leakage_checks: market_cap_available_at must be strictly before "
        f"prediction_timestamp for every matched row ({n} violation(s)). "
        f"First offender: ticker={first.get('ticker','?')}, "
        f"prediction_timestamp={pred[bad].iloc[0]}, "
        f"market_cap_available_at={avail[bad].iloc[0]}"
    )


# ---------------------------------------------------------------------------
# Optional unified audit
# ---------------------------------------------------------------------------

def run_feature_leakage_audit(df: pd.DataFrame,
                                 *,
                                 require_market_cap_column: bool = True,
                                 ) -> dict:
    """Run every final-frame assertion and return a machine-readable
    summary. Failing assertions are re-raised so the caller cannot
    swallow them — the summary is for the PASS path only.

    Production-frame contract (Aufgabe 7): the market-cap availability
    column must exist; the merge layer guarantees it. Set
    ``require_market_cap_column=False`` only on partial frames where
    the column is intentionally absent.
    """
    assert_no_forbidden_engagement_features(df)
    assert_market_cap_asof_correct(
        df, require_availability_column=require_market_cap_column,
    )
    return {
        "forbidden_engagement_check":     "PASS",
        "market_cap_asof_check":          "PASS",
        "n_rows_audited":                 int(len(df)),
        "columns_audited":                int(len(df.columns)),
    }


# ---------------------------------------------------------------------------
# Aufgabe 8 Section E.2 — post cutoff assertion
# ---------------------------------------------------------------------------

def assert_posts_respect_cutoff(posts: pd.DataFrame,
                                 *,
                                 cutoff: pd.Timestamp,
                                 created_col: str = "created") -> None:
    """Raise if any post in ``posts`` has ``created > cutoff``.

    Equality (``created == cutoff``) is allowed — the convention is
    "posts created up to and including the cutoff may enter the slot".
    """
    if posts is None or len(posts) == 0:
        return
    created = pd.to_datetime(posts[created_col], utc=True, errors="coerce")
    cutoff_ts = pd.Timestamp(cutoff)
    if cutoff_ts.tzinfo is None:
        cutoff_ts = cutoff_ts.tz_localize("UTC")
    bad = created > cutoff_ts
    n = int(bad.sum())
    if n:
        raise AssertionError(
            "leakage_checks: posts_respect_cutoff violated — "
            f"{n} post(s) created after cutoff={cutoff_ts.isoformat()}"
        )


def assert_no_target_interval_posts_in_predictors(
    posts: pd.DataFrame,
    *,
    prediction_timestamp: pd.Timestamp,
    created_col: str = "created",
) -> None:
    """Raise when a post created at-or-after the prediction instant was
    counted as a feature source.

    The predictor/target separation rule says: every feature post
    contributing to a prediction at ``t`` must have ``created <= t``;
    posts arriving in ``(t, t + horizon]`` belong to the TARGET interval
    and cannot enter the feature stack.
    """
    if posts is None or len(posts) == 0:
        return
    pred_ts = pd.Timestamp(prediction_timestamp)
    if pred_ts.tzinfo is None:
        pred_ts = pred_ts.tz_localize("UTC")
    created = pd.to_datetime(posts[created_col], utc=True, errors="coerce")
    bad = created > pred_ts
    n = int(bad.sum())
    if n:
        raise AssertionError(
            "leakage_checks: predictor/target separation violated — "
            f"{n} post(s) created after prediction_timestamp="
            f"{pred_ts.isoformat()} were used as features"
        )
