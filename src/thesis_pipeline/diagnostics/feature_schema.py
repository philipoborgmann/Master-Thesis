"""Schema contracts at the generator boundaries (commit 8 — v4).

Two validators, both designed to fail loudly with an actionable
``regenerate with`` message rather than silently rewrite stale parquets.

* :func:`validate_price_feature_schema` — guarantees the v4 momentum +
  realised-volatility + market-cap-availability schema.
* :func:`validate_sentiment_feature_schema` — refuses any
  ``*_weighted_mean`` column and the raw engagement set
  (``score``, ``upvote_ratio``, ``num_comments``, ``engagement_weight``).

The validators are called by the merge stage immediately after loading
each input parquet so a v4 run can never be poisoned by a leftover v3
artefact. They are also wired into the create-* stages so the
generators reject stale data BEFORE the merge.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd


# ---------------------------------------------------------------------------
# Required v4 price-feature columns
# ---------------------------------------------------------------------------

#: Columns the ECON registry consumes — every price-feature parquet
#: must carry all of them.
REQUIRED_PRICE_COLUMNS: tuple[str, ...] = (
    "timestamp", "ticker",
    "log_return_t",
    "cum_log_return_7d",
    "cum_log_return_14d",
    "cum_log_return_21d",
    "realized_vol_14d",
    "volume_diff",
    "log_market_cap_lag1",
    "market_cap_available_at",
)

#: Legacy momentum names. Their presence WITHOUT the new ``_Nd``
#: variants signals a pre-v4 parquet that must be regenerated.
LEGACY_MOMENTUM_NAMES: tuple[str, ...] = (
    "cum_log_return_7",
    "cum_log_return_14",
    "cum_log_return_21",
    "realized_vol_14",
)


class StalePriceFeatureSchema(AssertionError):
    """Raised when the price-features parquet predates the v4 schema."""


def _missing(required: Iterable[str], columns: Iterable[str]) -> list[str]:
    cols = set(columns)
    return [c for c in required if c not in cols]


def validate_price_feature_schema(df: pd.DataFrame,
                                    *,
                                    horizon: str | None = None,
                                    source: Path | str | None = None,
                                    ) -> None:
    """Raise :class:`StalePriceFeatureSchema` on any v4 violation.

    ``horizon`` and ``source`` are only used to make the actionable
    error message more useful — they have no effect on the check
    itself.
    """
    missing = _missing(REQUIRED_PRICE_COLUMNS, df.columns)
    legacy_found = [c for c in LEGACY_MOMENTUM_NAMES if c in df.columns]

    if not missing:
        return

    where = f"horizon={horizon}" if horizon else ""
    if source:
        where = f"{where} source={source}".strip()
    legacy_note = (f"\nfound legacy momentum names: {legacy_found}"
                   if legacy_found else "")
    raise StalePriceFeatureSchema(
        "Detected pre-v4 price features"
        f"{(' (' + where + ')') if where else ''}.\n"
        f"missing required columns: {missing}"
        f"{legacy_note}\n\nRegenerate with:\n"
        "  python -m thesis_pipeline.cli create-price-features "
        f"--horizon {horizon or '<horizon>'} --force"
    )


# ---------------------------------------------------------------------------
# Sentiment-feature contract
# ---------------------------------------------------------------------------

#: Raw engagement columns must never reach the production parquet.
#: Mirrors :data:`thesis_pipeline.diagnostics.leakage_checks
#: .FORBIDDEN_ENGAGEMENT_RAW_COLUMNS`.
FORBIDDEN_RAW_ENGAGEMENT_COLUMNS: frozenset = frozenset({
    "score", "upvote_ratio", "num_comments", "engagement_weight",
})

#: At least one of these polarity columns is required so the parquet
#: is not silently empty of usable sentiment information.
EXPECTED_SENTIMENT_FAMILIES: tuple[str, ...] = (
    "vader_title_score_mean",
    "cryptobert_title_score_mean",
)


class StaleSentimentFeatureSchema(AssertionError):
    """Raised when the sentiment-features parquet violates the v4
    Variante-A contract (raw engagement / weighted-mean columns)."""


def validate_sentiment_feature_schema(
    df: pd.DataFrame,
    *,
    horizon: str | None = None,
    source: Path | str | None = None,
    require_polarity: bool = False,
) -> None:
    """Refuse parquets that carry ``*_weighted_mean`` or any raw
    engagement column.

    When ``require_polarity`` is True the frame must also expose at
    least one of :data:`EXPECTED_SENTIMENT_FAMILIES` — used as the
    "non-empty sentiment universe" guard.
    """
    weighted = sorted(c for c in df.columns
                       if str(c).endswith("_weighted_mean"))
    raw = sorted(c for c in df.columns
                  if c in FORBIDDEN_RAW_ENGAGEMENT_COLUMNS)
    offenders = weighted + raw

    where = f"horizon={horizon}" if horizon else ""
    if source:
        where = f"{where} source={source}".strip()
    if offenders:
        raise StaleSentimentFeatureSchema(
            "Detected pre-v4 sentiment features"
            f"{(' (' + where + ')') if where else ''}.\n"
            f"forbidden columns found: {offenders}\n\nRegenerate with:\n"
            "  python -m thesis_pipeline.cli create-sentiment-features "
            f"--horizon {horizon or '<horizon>'} --force"
        )

    if require_polarity:
        present = [c for c in EXPECTED_SENTIMENT_FAMILIES if c in df.columns]
        if not present:
            raise StaleSentimentFeatureSchema(
                "Sentiment-feature parquet has no recognised polarity "
                f"column (expected one of {list(EXPECTED_SENTIMENT_FAMILIES)})."
                "\nRegenerate with:\n"
                "  python -m thesis_pipeline.cli create-sentiment-features "
                f"--horizon {horizon or '<horizon>'} --force"
            )


__all__ = (
    "REQUIRED_PRICE_COLUMNS",
    "LEGACY_MOMENTUM_NAMES",
    "FORBIDDEN_RAW_ENGAGEMENT_COLUMNS",
    "EXPECTED_SENTIMENT_FAMILIES",
    "StalePriceFeatureSchema",
    "StaleSentimentFeatureSchema",
    "validate_price_feature_schema",
    "validate_sentiment_feature_schema",
)
