"""Shared helpers for working with the final modelling feature universe.

Both the stationarity stage (``--source final``) and the descriptive-statistics
stage need to (a) load the final merged feature parquets, (b) resolve the
modelling feature universe from ``feature_sets.xlsx`` (with ``{model}``
placeholders for sentiment scorers), (c) classify features into a small set of
groups, and (d) reason about *structurally empty* sentiment rows. These helpers
keep that logic in one place so the two stages can never drift apart.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd

from ..config import resolve_path


# Canonical identifier columns that must never be tested or summarised.
IDENTIFIER_COLUMNS: tuple[str, ...] = (
    "timestamp", "date", "ticker", "horizon", "target", "row_id",
)
# Prefixes for downstream/model output columns that may sneak into the final
# parquet (predictions, probabilities, derived signals); these are never input
# features.
_PREDICTION_PREFIXES: tuple[str, ...] = ("prediction", "probability", "signal_")

# The three sentiment scorers used in the thesis. Used to expand ``{model}``
# placeholders in registered feature names.
SENTIMENT_MODELS: tuple[str, ...] = ("cryptobert", "vader")
SENTIMENT_PREFIXES: tuple[str, ...] = tuple(f"{m}_" for m in SENTIMENT_MODELS)
_SENTIMENT_TOKENS: tuple[str, ...] = ("bullishness", "post_count")

# Group membership for the canonical thesis features.
_PRICE_FEATURES: frozenset = frozenset({
    "log_return_t", "cum_log_return_7", "cum_log_return_14",
})
_VOLATILITY_FEATURES: frozenset = frozenset({"realized_vol_14"})
_VOLUME_FEATURES: frozenset = frozenset({"volume_diff"})
_MARKETCAP_FEATURES: frozenset = frozenset({"market_cap_t"})


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_final_features(horizon: str) -> pd.DataFrame:
    """Read ``Data/Final/features_{horizon}.parquet`` with a UTC timestamp."""
    path = Path(resolve_path("final_features_pattern", horizon=horizon))
    if not path.exists():
        raise FileNotFoundError(f"Final features missing for horizon={horizon}: {path}")
    df = pd.read_parquet(path)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    return df


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------

def classify_feature_group(feature: str) -> str:
    """Coarse group label used by both stationarity and descriptive outputs.

    Returns one of ``price | volatility | volume | market_cap | sentiment |
    other``.
    """
    f = str(feature)
    if f in _PRICE_FEATURES:
        return "price"
    if f in _VOLATILITY_FEATURES:
        return "volatility"
    if f in _VOLUME_FEATURES:
        return "volume"
    if f in _MARKETCAP_FEATURES:
        return "market_cap"
    if any(f.startswith(p) for p in SENTIMENT_PREFIXES):
        return "sentiment"
    if any(tok in f for tok in _SENTIMENT_TOKENS):
        return "sentiment"
    return "other"


def _is_identifier_column(col: str) -> bool:
    if col in IDENTIFIER_COLUMNS:
        return True
    lower = col.lower()
    return any(lower.startswith(p) for p in _PREDICTION_PREFIXES)


# ---------------------------------------------------------------------------
# Registry → final-data column resolution
# ---------------------------------------------------------------------------

def _expand_placeholder(feature: str,
                        sentiment_model: str | None,
                        sentiment_models: Iterable[str]) -> list[str]:
    """Expand ``{model}`` placeholders to one column per sentiment scorer.

    When the feature-set row specifies a single ``sentiment_model`` we expand
    to that model only; otherwise we expand to every scorer in ``sentiment_models``
    so a feature like ``{model}_combined_mean`` resolves to all three columns.
    """
    if "{model}" not in feature:
        return [feature]
    sm = (sentiment_model or "").strip()
    if sm and sm not in ("-", "nan", "none", "None") and sm in tuple(sentiment_models):
        return [feature.replace("{model}", sm)]
    return [feature.replace("{model}", m) for m in sentiment_models]


def resolve_final_feature_columns(
    df: pd.DataFrame,
    feature_sets: dict | None = None,
    *,
    sentiment_models: tuple[str, ...] = SENTIMENT_MODELS,
    exclude_set_ids: tuple[str, ...] = ("B1",),
) -> tuple[list[str], list[dict]]:
    """Return the modelling feature universe + a resolution report.

    The first return value is the deduplicated list of feature *columns*
    (preserving first-seen order) that exist in ``df``. The second value is
    the full registry-resolution log — one row per (set_id × requested feature
    × expanded model) — suitable for writing as
    ``*_feature_resolution.csv``.

    When ``feature_sets`` is ``None``/empty or every requested feature is
    missing from ``df``, we fall back to every numeric column that is not an
    identifier; a sentinel resolution row is emitted for each fallback column.
    """
    columns_in_df = set(df.columns)
    resolved_order: list[str] = []
    seen: set[str] = set()
    resolution: list[dict] = []

    if feature_sets:
        for set_id, spec in feature_sets.items():
            if set_id in exclude_set_ids:
                continue
            category = str(spec.get("category", "") or "")
            label    = str(spec.get("label", "") or "")
            sm       = spec.get("sentiment_model")
            sm_str   = "" if sm in (None, "", "-", "nan", "None") else str(sm)
            for raw in list(spec.get("features", []) or []):
                requested = str(raw).strip()
                if not requested or requested in ("__majority_class__",
                                                  "__rolling_probability__"):
                    continue
                for expanded in _expand_placeholder(requested, sm_str,
                                                    sentiment_models):
                    exists = expanded in columns_in_df
                    if exists and expanded not in seen:
                        seen.add(expanded)
                        resolved_order.append(expanded)
                    resolution.append({
                        "set_id":              set_id,
                        "category":            category,
                        "label":               label,
                        "sentiment_model":     sm_str,
                        "requested_feature":   requested,
                        "resolved_feature":    expanded,
                        "exists_in_final_data": bool(exists),
                        "reason_if_missing":   ("" if exists else
                                                f"column {expanded!r} not in final features"),
                    })

    if not resolved_order:
        # Fallback: every numeric column that is not an identifier.
        numeric_cols = df.select_dtypes(include="number").columns.tolist()
        for col in numeric_cols:
            if _is_identifier_column(col) or col in seen:
                continue
            seen.add(col)
            resolved_order.append(col)
            resolution.append({
                "set_id":              "__numeric_fallback__",
                "category":            "",
                "label":               "numeric fallback (registry unavailable)",
                "sentiment_model":     "",
                "requested_feature":   col,
                "resolved_feature":    col,
                "exists_in_final_data": True,
                "reason_if_missing":   "",
            })

    # Identifier columns are never returned as features even if registered.
    resolved_order = [c for c in resolved_order if not _is_identifier_column(c)]
    return resolved_order, resolution


# ---------------------------------------------------------------------------
# Structural sentiment-empty detection
# ---------------------------------------------------------------------------

def find_matching_post_count(feature: str, columns: Iterable[str]) -> str | None:
    """Return the matching ``post_count`` column for a sentiment feature.

    Prefers the per-model count (``vader_post_count``) when the feature starts
    with that model's prefix, then falls back to a generic ``post_count`` column
    if available. Non-sentiment features return ``None``.
    """
    cols = list(columns)
    lower = feature.lower()
    for model in SENTIMENT_MODELS:
        if lower.startswith(model + "_"):
            for cand in (f"{model}_post_count", "post_count"):
                if cand in cols:
                    return cand
            return None
    if classify_feature_group(feature) == "sentiment" and "post_count" in cols:
        return "post_count"
    return None


def _neutral_value_for(feature: str) -> float | None:
    """Neutral fill used by the sentiment aggregator for the given feature."""
    lower = feature.lower()
    if "bullishness" in lower:
        return 0.5
    if any(tok in lower for tok in ("_mean", "_weighted_mean", "_score")):
        return 0.0
    return None


def compute_structural_empty_mask(df: pd.DataFrame, feature: str) -> pd.Series:
    """Boolean ``Series`` flagging *structurally empty* rows for ``feature``.

    A row is structurally empty when either:

    * the feature value is missing, **or**
    * (sentiment features only) the value equals the neutral fill *and* the
      matching ``post_count`` is zero (no posts → neutral value is a placeholder,
      not real evidence).

    For non-sentiment features the second condition does not apply and the
    mask collapses to ``isna``.
    """
    if feature not in df.columns:
        return pd.Series([False] * len(df), index=df.index, dtype=bool)
    series = df[feature]
    missing = series.isna()
    if classify_feature_group(feature) != "sentiment":
        return missing
    neutral = _neutral_value_for(feature)
    if neutral is None:
        return missing
    pc_col = find_matching_post_count(feature, df.columns)
    if pc_col is None:
        return missing
    pc = pd.to_numeric(df[pc_col], errors="coerce").fillna(0)
    vals = pd.to_numeric(series, errors="coerce")
    structural = (vals == neutral) & (pc == 0)
    return missing | structural
