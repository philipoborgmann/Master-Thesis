"""Extensive descriptive statistics for the **final modelling feature sets**.

For each horizon under ``Data/Final/features_{horizon}.parquet`` this stage
produces six long-form CSV reports under
``Outputs/deskriptiv/final_feature_sets/``:

1. ``final_feature_descriptives_by_ticker_feature.csv`` — per (horizon, ticker,
   feature) row with summary stats, percentiles, structural-empty counts, and
   timestamp coverage.
2. ``final_feature_descriptives_by_horizon_feature.csv`` — same stats
   aggregated across tickers per horizon × feature.
3. ``final_feature_descriptives_by_horizon_ticker.csv`` — one row per
   (horizon, ticker) with completeness diagnostics and (where available)
   per-scorer sentiment-post statistics.
4. ``final_feature_descriptives_overview.csv`` — one row per horizon: totals,
   row counts, missingness, structurally-empty rows, duplicate (ticker,
   timestamp) rows.
5. ``final_feature_correlation_summary.csv`` — long-form pairwise Pearson and
   Spearman correlations among the modelling features per horizon.
6. ``final_feature_resolution.csv`` — registry-to-column resolution log
   (mirrors the stationarity stage so the two reports stay in lock-step).

The feature universe is shared with stationarity via
:mod:`thesis_pipeline.features.final_feature_utils`; the registry is preferred
and a numeric-column fallback kicks in when the registry is unavailable.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from ..config import resolve_path
from ..features.feature_registry import load_feature_sets
from ..features.final_feature_utils import (
    SENTIMENT_MODELS, classify_feature_group, compute_structural_empty_mask,
    find_matching_post_count, load_final_features, resolve_final_feature_columns,
)

logger = logging.getLogger("descriptive_final_features")

HORIZONS = ("1h", "6h", "1d")

_MIN_PAIRWISE_N = 30  # exclude correlation pairs with fewer than 30 overlaps


def _setup_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(level=level, format="%(asctime)s [%(levelname)s] %(message)s",
                        datefmt="%H:%M:%S")


def _output_root(override: str | None = None) -> Path:
    """``Outputs/deskriptiv/final_feature_sets/`` — overridable."""
    if override:
        return Path(override)
    try:
        base = Path(resolve_path("descriptive_final_root"))
    except KeyError:
        base = Path(resolve_path("descriptive_root")) / "final_feature_sets"
    return base


# ---------------------------------------------------------------------------
# Per-(ticker, feature) descriptives
# ---------------------------------------------------------------------------

_PERCENTILES = (0.01, 0.05, 0.10, 0.25, 0.75, 0.90, 0.95, 0.99)
_PERCENTILE_LABELS = ("p01", "p05", "p10", "p25", "p75", "p90", "p95", "p99")


def _percentiles(series: pd.Series) -> dict[str, float]:
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty:
        return {label: float("nan") for label in _PERCENTILE_LABELS}
    qs = s.quantile(list(_PERCENTILES)).tolist()
    return dict(zip(_PERCENTILE_LABELS, qs))


def _feature_sets_using(feature: str, feature_sets: dict) -> str:
    """Return a ``;``-separated list of set_ids that include ``feature``.

    Honours ``{model}`` placeholders so e.g. ``vader_combined_mean`` is reported
    as a member of every C-set that requests ``{model}_combined_mean``.
    """
    members: list[str] = []
    lower = feature.lower()
    for set_id, spec in feature_sets.items():
        for raw in list(spec.get("features", []) or []):
            requested = str(raw).strip()
            if not requested:
                continue
            if "{model}" in requested:
                expanded = [requested.replace("{model}", m) for m in SENTIMENT_MODELS]
            else:
                expanded = [requested]
            if any(feature == e or lower == e.lower() for e in expanded):
                members.append(set_id)
                break
    return ";".join(sorted(set(members)))


def _per_ticker_feature(df: pd.DataFrame, horizon: str, ticker: str,
                        feature: str, feature_sets: dict) -> dict:
    sub = df[df["ticker"] == ticker]
    n_rows = int(len(sub))
    series = sub.get(feature)
    if series is None:
        return {}
    numeric = pd.to_numeric(series, errors="coerce")
    non_missing = numeric.dropna()
    n_non = int(non_missing.size)
    n_missing = int(numeric.isna().sum())
    n_zero = int((numeric == 0).sum())
    empty_mask = compute_structural_empty_mask(sub, feature)
    record = {
        "horizon": horizon,
        "ticker": ticker,
        "feature": feature,
        "feature_group": classify_feature_group(feature),
        "feature_set_ids_using_feature": _feature_sets_using(feature, feature_sets),
        "n_rows": n_rows,
        "n_non_missing": n_non,
        "n_missing": n_missing,
        "share_missing": (n_missing / n_rows) if n_rows else float("nan"),
        "n_zero": n_zero,
        "share_zero": (n_zero / n_rows) if n_rows else float("nan"),
        "n_unique": int(non_missing.nunique()),
        "mean": float(non_missing.mean()) if n_non else float("nan"),
        "median": float(non_missing.median()) if n_non else float("nan"),
        "std": float(non_missing.std()) if n_non > 1 else float("nan"),
        "min": float(non_missing.min()) if n_non else float("nan"),
        "max": float(non_missing.max()) if n_non else float("nan"),
        "skewness": float(non_missing.skew()) if n_non > 2 else float("nan"),
        "kurtosis": float(non_missing.kurt()) if n_non > 3 else float("nan"),
        "empty_row_count": int(empty_mask.sum()),
        "share_empty_rows": (
            float(empty_mask.mean()) if n_rows else float("nan")
        ),
    }
    record.update(_percentiles(non_missing))
    if "timestamp" in sub.columns and n_rows:
        ts = pd.to_datetime(sub["timestamp"], utc=True, errors="coerce")
        if ts.notna().any():
            first = ts.min()
            last = ts.max()
            record["first_timestamp"] = first.isoformat()
            record["last_timestamp"] = last.isoformat()
            record["date_range_days"] = float((last - first).total_seconds() / 86400.0)
            record["duplicate_timestamp_count"] = int(
                sub.duplicated(subset=["timestamp"]).sum()
            )
        else:
            record.update(first_timestamp=None, last_timestamp=None,
                          date_range_days=float("nan"),
                          duplicate_timestamp_count=0)
    else:
        record.update(first_timestamp=None, last_timestamp=None,
                      date_range_days=float("nan"),
                      duplicate_timestamp_count=0)
    return record


def by_ticker_feature_table(df: pd.DataFrame, horizon: str,
                            feature_cols: Iterable[str],
                            feature_sets: dict) -> pd.DataFrame:
    rows: list[dict] = []
    tickers = sorted(df["ticker"].dropna().unique())
    for tk in tickers:
        for feat in feature_cols:
            r = _per_ticker_feature(df, horizon, tk, feat, feature_sets)
            if r:
                rows.append(r)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Aggregated per-(horizon, feature)
# ---------------------------------------------------------------------------

def by_horizon_feature_table(df: pd.DataFrame, horizon: str,
                             feature_cols: Iterable[str],
                             per_tf: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per-ticker rows back up to (horizon × feature)."""
    rows = []
    for feat in feature_cols:
        all_vals = pd.to_numeric(df.get(feat), errors="coerce").dropna()
        n_tickers_any = 0
        n_tickers_all_missing = 0
        if not per_tf.empty:
            sub = per_tf[per_tf["feature"] == feat]
            n_tickers_any = int((sub["n_non_missing"] > 0).sum())
            n_tickers_all_missing = int((sub["n_non_missing"] == 0).sum())
            mean_missing = float(sub["share_missing"].mean()) if not sub.empty else float("nan")
            max_missing  = float(sub["share_missing"].max())  if not sub.empty else float("nan")
            mean_zero    = float(sub["share_zero"].mean())    if not sub.empty else float("nan")
            avg_non_missing = float(sub["n_non_missing"].mean()) if not sub.empty else float("nan")
        else:
            mean_missing = max_missing = mean_zero = avg_non_missing = float("nan")
        record = {
            "horizon": horizon,
            "feature": feat,
            "feature_group": classify_feature_group(feat),
            "n_obs_total": int(all_vals.size),
            "mean": float(all_vals.mean()) if all_vals.size else float("nan"),
            "median": float(all_vals.median()) if all_vals.size else float("nan"),
            "std": float(all_vals.std()) if all_vals.size > 1 else float("nan"),
            "min": float(all_vals.min()) if all_vals.size else float("nan"),
            "max": float(all_vals.max()) if all_vals.size else float("nan"),
            "mean_missing_share_across_tickers": mean_missing,
            "max_missing_share_across_tickers":  max_missing,
            "mean_zero_share_across_tickers":    mean_zero,
            "avg_non_missing_per_ticker":        avg_non_missing,
            "n_tickers_any_valid":               n_tickers_any,
            "n_tickers_all_missing":             n_tickers_all_missing,
        }
        record.update(_percentiles(all_vals))
        rows.append(record)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Per-(horizon, ticker)
# ---------------------------------------------------------------------------

def _sentiment_post_columns(columns: Iterable[str]) -> dict[str, str | None]:
    cols = list(columns)
    out: dict[str, str | None] = {}
    for model in SENTIMENT_MODELS:
        candidate = f"{model}_post_count"
        out[model] = candidate if candidate in cols else None
    return out


def by_horizon_ticker_table(df: pd.DataFrame, horizon: str,
                            feature_cols: Iterable[str]) -> pd.DataFrame:
    rows = []
    feature_cols = list(feature_cols)
    pc_cols = _sentiment_post_columns(df.columns)
    generic_pc = "post_count" if "post_count" in df.columns else None
    tickers = sorted(df["ticker"].dropna().unique())
    for tk in tickers:
        sub = df[df["ticker"] == tk]
        n_rows = int(len(sub))
        feats = sub[feature_cols] if feature_cols else sub.iloc[:, :0]
        missing_share = feats.isna().mean() if not feats.empty else pd.Series(dtype=float)
        n_features_with_missing = int((missing_share > 0).sum())
        mean_missing = float(missing_share.mean()) if not missing_share.empty else float("nan")
        max_missing  = float(missing_share.max())  if not missing_share.empty else float("nan")
        empty_rows_all = int(feats.isna().all(axis=1).sum()) if not feats.empty else 0
        record = {
            "horizon": horizon,
            "ticker":  tk,
            "n_rows":  n_rows,
            "n_features_total": int(len(feature_cols)),
            "n_features_with_any_missing": n_features_with_missing,
            "mean_missing_share_across_features": mean_missing,
            "max_missing_share":                  max_missing,
            "n_empty_rows_all_features":          empty_rows_all,
            "share_empty_rows_all_features":      (empty_rows_all / n_rows) if n_rows else float("nan"),
        }
        # Per-scorer post-count statistics and aggregate.
        per_unit = []
        total_posts = 0.0
        any_post = pd.Series([False] * n_rows, index=sub.index, dtype=bool)
        for model, col in pc_cols.items():
            if col is None:
                record[f"{model}_post_count_mean"] = float("nan")
                continue
            vals = pd.to_numeric(sub[col], errors="coerce").fillna(0)
            record[f"{model}_post_count_mean"] = float(vals.mean()) if n_rows else float("nan")
            per_unit.append(vals)
            total_posts += float(vals.sum())
            any_post = any_post | (vals > 0)
        if per_unit:
            stacked = sum(per_unit) if len(per_unit) == 1 else pd.concat(per_unit, axis=1).sum(axis=1)
            record["average_sentiment_posts_per_time_unit"] = float(stacked.mean()) if n_rows else float("nan")
            record["median_sentiment_posts_per_time_unit"]  = float(stacked.median()) if n_rows else float("nan")
            record["total_sentiment_posts"]                 = float(total_posts)
            record["number_of_time_units_without_any_sentiment"] = int((~any_post).sum())
            record["share_time_units_without_any_sentiment"] = float((~any_post).mean()) if n_rows else float("nan")
        elif generic_pc is not None:
            vals = pd.to_numeric(sub[generic_pc], errors="coerce").fillna(0)
            record["average_sentiment_posts_per_time_unit"] = float(vals.mean()) if n_rows else float("nan")
            record["median_sentiment_posts_per_time_unit"]  = float(vals.median()) if n_rows else float("nan")
            record["total_sentiment_posts"]                 = float(vals.sum())
            record["number_of_time_units_without_any_sentiment"] = int((vals == 0).sum())
            record["share_time_units_without_any_sentiment"] = float((vals == 0).mean()) if n_rows else float("nan")
        else:
            record["average_sentiment_posts_per_time_unit"] = float("nan")
            record["median_sentiment_posts_per_time_unit"]  = float("nan")
            record["total_sentiment_posts"]                 = float("nan")
            record["number_of_time_units_without_any_sentiment"] = 0
            record["share_time_units_without_any_sentiment"] = float("nan")

        if "timestamp" in sub.columns and n_rows:
            ts = pd.to_datetime(sub["timestamp"], utc=True, errors="coerce")
            if ts.notna().any():
                first = ts.min()
                last  = ts.max()
                record["first_timestamp"] = first.isoformat()
                record["last_timestamp"]  = last.isoformat()
                record["date_range_days"] = float((last - first).total_seconds() / 86400.0)
            else:
                record.update(first_timestamp=None, last_timestamp=None,
                              date_range_days=float("nan"))
        else:
            record.update(first_timestamp=None, last_timestamp=None,
                          date_range_days=float("nan"))
        rows.append(record)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Overview per horizon
# ---------------------------------------------------------------------------

def overview_row(df: pd.DataFrame, horizon: str,
                 feature_cols: Iterable[str]) -> dict:
    feature_cols = list(feature_cols)
    n_rows = int(len(df))
    n_tickers = int(df["ticker"].nunique()) if "ticker" in df.columns else 0
    feats = df[feature_cols] if feature_cols else df.iloc[:, :0]
    total_missing = int(feats.isna().sum().sum()) if not feats.empty else 0
    total_cells = int(feats.size) if not feats.empty else 0
    global_missing_share = (total_missing / total_cells) if total_cells else float("nan")
    # Structurally empty sentiment rows: count per-feature mask sums.
    structural_total = 0
    sent_feats = [f for f in feature_cols if classify_feature_group(f) == "sentiment"]
    for feat in sent_feats:
        mask = compute_structural_empty_mask(df, feat)
        structural_total += int(mask.sum())
    # Sentiment posts.
    pc_cols = _sentiment_post_columns(df.columns)
    per_unit = []
    for col in pc_cols.values():
        if col is None:
            continue
        per_unit.append(pd.to_numeric(df[col], errors="coerce").fillna(0))
    if per_unit:
        stacked = sum(per_unit) if len(per_unit) == 1 else pd.concat(per_unit, axis=1).sum(axis=1)
        avg_posts = float(stacked.mean()) if n_rows else float("nan")
        any_post = (stacked > 0)
        share_no_sent = float((~any_post).mean()) if n_rows else float("nan")
    elif "post_count" in df.columns:
        vals = pd.to_numeric(df["post_count"], errors="coerce").fillna(0)
        avg_posts = float(vals.mean()) if n_rows else float("nan")
        share_no_sent = float((vals == 0).mean()) if n_rows else float("nan")
    else:
        avg_posts = float("nan")
        share_no_sent = float("nan")
    duplicates = int(df.duplicated(subset=["ticker", "timestamp"]).sum()) \
        if {"ticker", "timestamp"}.issubset(df.columns) else 0
    if "timestamp" in df.columns and n_rows:
        ts = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
        first = ts.min().isoformat() if ts.notna().any() else None
        last  = ts.max().isoformat() if ts.notna().any() else None
    else:
        first = last = None
    if "ticker" in df.columns and n_tickers:
        per_ticker_rows = df.groupby("ticker").size()
        avg_rows = float(per_ticker_rows.mean())
        min_rows = int(per_ticker_rows.min())
        max_rows = int(per_ticker_rows.max())
    else:
        avg_rows = float("nan"); min_rows = max_rows = 0
    return {
        "horizon": horizon,
        "total_rows": n_rows,
        "n_tickers": n_tickers,
        "n_modelling_features": len(feature_cols),
        "first_timestamp": first,
        "last_timestamp":  last,
        "avg_rows_per_ticker": avg_rows,
        "min_rows_per_ticker": min_rows,
        "max_rows_per_ticker": max_rows,
        "total_missing_values": total_missing,
        "global_missing_share": global_missing_share,
        "total_structural_empty_sentiment_rows": structural_total,
        "average_sentiment_posts_per_time_unit": avg_posts,
        "share_ticker_time_rows_without_sentiment": share_no_sent,
        "duplicate_ticker_timestamp_rows": duplicates,
    }


# ---------------------------------------------------------------------------
# Correlations
# ---------------------------------------------------------------------------

def correlation_summary(df: pd.DataFrame, horizon: str,
                        feature_cols: Iterable[str],
                        min_pairwise_n: int = _MIN_PAIRWISE_N) -> pd.DataFrame:
    feature_cols = list(feature_cols)
    if len(feature_cols) < 2:
        return pd.DataFrame(columns=["horizon", "feature_1", "feature_2",
                                     "pearson_corr", "spearman_corr", "n_pairwise",
                                     "abs_pearson_corr", "abs_spearman_corr"])
    block = df[feature_cols].apply(pd.to_numeric, errors="coerce")
    pearson = block.corr(method="pearson", min_periods=min_pairwise_n)
    spearman = block.corr(method="spearman", min_periods=min_pairwise_n)
    counts = block.notna().astype(int)
    pair_counts = counts.T @ counts  # symmetric n_pairwise matrix
    rows = []
    n = len(feature_cols)
    for i in range(n):
        for j in range(i + 1, n):
            f1, f2 = feature_cols[i], feature_cols[j]
            n_pair = int(pair_counts.iat[i, j])
            if n_pair < min_pairwise_n:
                continue
            p = pearson.iat[i, j]
            s = spearman.iat[i, j]
            rows.append({
                "horizon": horizon, "feature_1": f1, "feature_2": f2,
                "pearson_corr": float(p) if pd.notna(p) else float("nan"),
                "spearman_corr": float(s) if pd.notna(s) else float("nan"),
                "n_pairwise": n_pair,
                "abs_pearson_corr": float(abs(p)) if pd.notna(p) else float("nan"),
                "abs_spearman_corr": float(abs(s)) if pd.notna(s) else float("nan"),
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Pipeline entrypoint
# ---------------------------------------------------------------------------

def _write_csv(df: pd.DataFrame, path: Path, columns: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if df.empty and columns is not None:
        pd.DataFrame(columns=columns).to_csv(path, index=False, encoding="utf-8")
    elif columns:
        for col in columns:
            if col not in df.columns:
                df[col] = pd.NA
        ordered = columns + [c for c in df.columns if c not in columns]
        df[ordered].to_csv(path, index=False, encoding="utf-8")
    else:
        df.to_csv(path, index=False, encoding="utf-8")
    logger.info("wrote %s (%d rows)", path, len(df))


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Extensive descriptive statistics for final modelling feature sets."
    )
    p.add_argument("--horizon", default=None, choices=list(HORIZONS),
                   help="Restrict to a single horizon (default: all).")
    p.add_argument("--output-dir", "--output_dir", dest="output_dir", default=None,
                   help="Override Outputs/deskriptiv/final_feature_sets/.")
    p.add_argument("--min-pairwise-n", "--min_pairwise_n", dest="min_pairwise_n",
                   type=int, default=_MIN_PAIRWISE_N,
                   help="Drop correlation pairs with fewer pairwise observations.")
    p.add_argument("--smoke", action="store_true",
                   help="Smoke mode (defaults to horizon=1d; harmless here, kept for CLI parity).")
    p.add_argument("--dry-run", "--dry_run", dest="dry_run", action="store_true")
    p.add_argument("--force", action="store_true",
                   help="Allow overwriting existing CSV outputs.")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    _setup_logging()
    if args.smoke and not args.horizon:
        args.horizon = "1d"

    out_dir = _output_root(args.output_dir)
    horizons = [args.horizon] if args.horizon else list(HORIZONS)

    if args.dry_run:
        logger.info("DRY-RUN: would write to %s", out_dir)
        for hz in horizons:
            logger.info("DRY-RUN: horizon %s → load Data/Final/features_%s.parquet", hz, hz)
        return 0

    try:
        feature_sets = load_feature_sets() or {}
    except Exception as exc:  # noqa: BLE001
        logger.warning("could not load feature_sets registry (%s) — numeric fallback", exc)
        feature_sets = {}

    by_tf_all:    list[pd.DataFrame] = []
    by_hf_all:    list[pd.DataFrame] = []
    by_ht_all:    list[pd.DataFrame] = []
    overview_all: list[dict] = []
    corr_all:     list[pd.DataFrame] = []
    resolution_all: list[dict] = []

    for hz in horizons:
        logger.info("HORIZON: %s", hz)
        try:
            df = load_final_features(hz)
        except FileNotFoundError as exc:
            logger.error(str(exc)); continue
        feature_cols, resolution = resolve_final_feature_columns(df, feature_sets)
        used = set(feature_cols)
        for row in resolution:
            row["horizon"] = hz
            row["used_in_descriptive_statistics"] = row["resolved_feature"] in used
        resolution_all.extend(resolution)
        if not feature_cols:
            logger.warning("no modelling features resolved for %s — skipping", hz)
            continue

        by_tf = by_ticker_feature_table(df, hz, feature_cols, feature_sets)
        by_tf_all.append(by_tf)

        by_hf = by_horizon_feature_table(df, hz, feature_cols, by_tf)
        by_hf_all.append(by_hf)

        by_ht = by_horizon_ticker_table(df, hz, feature_cols)
        by_ht_all.append(by_ht)

        overview_all.append(overview_row(df, hz, feature_cols))

        corr_all.append(correlation_summary(df, hz, feature_cols,
                                            min_pairwise_n=args.min_pairwise_n))

    by_tf_df    = pd.concat(by_tf_all,    ignore_index=True) if by_tf_all    else pd.DataFrame()
    by_hf_df    = pd.concat(by_hf_all,    ignore_index=True) if by_hf_all    else pd.DataFrame()
    by_ht_df    = pd.concat(by_ht_all,    ignore_index=True) if by_ht_all    else pd.DataFrame()
    corr_df     = pd.concat(corr_all,     ignore_index=True) if corr_all     else pd.DataFrame()
    overview_df = pd.DataFrame(overview_all)
    res_df      = pd.DataFrame(resolution_all)

    _write_csv(by_tf_df, out_dir / "final_feature_descriptives_by_ticker_feature.csv",
               columns=["horizon", "ticker", "feature", "feature_group",
                        "feature_set_ids_using_feature", "n_rows", "n_non_missing",
                        "n_missing", "share_missing", "n_zero", "share_zero",
                        "n_unique", "mean", "median", "std", "min",
                        "p01", "p05", "p10", "p25", "p75", "p90", "p95", "p99",
                        "max", "skewness", "kurtosis",
                        "first_timestamp", "last_timestamp", "date_range_days",
                        "duplicate_timestamp_count", "empty_row_count",
                        "share_empty_rows"])
    _write_csv(by_hf_df, out_dir / "final_feature_descriptives_by_horizon_feature.csv",
               columns=["horizon", "feature", "feature_group", "n_obs_total",
                        "mean", "median", "std", "min",
                        "p01", "p05", "p10", "p25", "p75", "p90", "p95", "p99",
                        "max", "mean_missing_share_across_tickers",
                        "max_missing_share_across_tickers",
                        "mean_zero_share_across_tickers",
                        "avg_non_missing_per_ticker", "n_tickers_any_valid",
                        "n_tickers_all_missing"])
    _write_csv(by_ht_df, out_dir / "final_feature_descriptives_by_horizon_ticker.csv",
               columns=["horizon", "ticker", "n_rows", "n_features_total",
                        "n_features_with_any_missing",
                        "mean_missing_share_across_features", "max_missing_share",
                        "n_empty_rows_all_features", "share_empty_rows_all_features",
                        "vader_post_count_mean", "cryptobert_post_count_mean",
                        "average_sentiment_posts_per_time_unit",
                        "median_sentiment_posts_per_time_unit",
                        "total_sentiment_posts",
                        "number_of_time_units_without_any_sentiment",
                        "share_time_units_without_any_sentiment",
                        "first_timestamp", "last_timestamp", "date_range_days"])
    _write_csv(overview_df, out_dir / "final_feature_descriptives_overview.csv",
               columns=["horizon", "total_rows", "n_tickers", "n_modelling_features",
                        "first_timestamp", "last_timestamp",
                        "avg_rows_per_ticker", "min_rows_per_ticker",
                        "max_rows_per_ticker", "total_missing_values",
                        "global_missing_share",
                        "total_structural_empty_sentiment_rows",
                        "average_sentiment_posts_per_time_unit",
                        "share_ticker_time_rows_without_sentiment",
                        "duplicate_ticker_timestamp_rows"])
    _write_csv(corr_df, out_dir / "final_feature_correlation_summary.csv",
               columns=["horizon", "feature_1", "feature_2",
                        "pearson_corr", "spearman_corr", "n_pairwise",
                        "abs_pearson_corr", "abs_spearman_corr"])
    _write_csv(res_df, out_dir / "final_feature_resolution.csv",
               columns=["horizon", "set_id", "category", "label",
                        "sentiment_model", "requested_feature", "resolved_feature",
                        "exists_in_final_data", "used_in_descriptive_statistics",
                        "reason_if_missing"])

    logger.info("DONE")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
