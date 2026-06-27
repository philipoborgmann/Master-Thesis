"""Merge price + sentiment features per horizon — canonical implementation.

This module owns the merge logic. The legacy ``Merge_Features.py`` script in
the repository root is now a thin redirect to ``main()`` below; the package
CLI calls ``main(argv)`` directly via ``cmd_merge_features``.

No methodology change from the original ``Merge_Features.py``.
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Sequence

import numpy as np
import pandas as pd


# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

FEATURE_DIR  = os.path.join("Data", "Features")
OUTPUT_DIR   = os.path.join("Data", "Final")
COVERAGE_CSV = os.path.join(FEATURE_DIR, "sentiment_coverage.csv")

HORIZONS = ["1h", "6h", "1d"]

# Default minimum sentiment coverage to include a ticker
COVERAGE_THRESHOLD = 85.0

# Sentiment columns that represent scores (filled with 0 for missing slots)
# vs. std columns (left as NaN) vs. count columns (filled with 0)
SCORE_FILL = 0.0     # neutral sentiment
STD_FILL   = 0.0     # no posts → no observed dispersion (neutral fill)
COUNT_FILL = 0       # no posts = 0 count


# ══════════════════════════════════════════════════════════════════════════════
# LOAD AND FILTER
# ══════════════════════════════════════════════════════════════════════════════

def get_included_tickers(coverage_path: str, horizon: str,
                         threshold: float) -> list[str]:
    """Return tickers meeting the coverage threshold for a given horizon."""
    df = pd.read_csv(coverage_path)
    sub = df[(df["horizon"] == horizon) & (df["coverage_pct"] >= threshold)]
    return sorted(sub["ticker"].tolist())


def load_price_features(horizon: str, feature_dir: str = FEATURE_DIR) -> pd.DataFrame:
    """Load price features parquet for a horizon AND validate the v4 schema."""
    path = os.path.join(feature_dir, f"price_features_{horizon}.parquet")
    if not os.path.isfile(path):
        print(f"  [ERROR] Missing: {path}")
        return pd.DataFrame()
    df = pd.read_parquet(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    # Refuse pre-v4 schemas with an actionable regenerate-with message.
    from ..diagnostics.feature_schema import validate_price_feature_schema
    validate_price_feature_schema(df, horizon=horizon, source=path)
    return df


def load_sentiment_features(horizon: str, feature_dir: str = FEATURE_DIR) -> pd.DataFrame:
    """Load sentiment features parquet for a horizon AND validate the v4
    Variante-A contract."""
    path = os.path.join(feature_dir, f"sentiment_features_{horizon}.parquet")
    if not os.path.isfile(path):
        print(f"  [ERROR] Missing: {path}")
        return pd.DataFrame()
    df = pd.read_parquet(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    from ..diagnostics.feature_schema import validate_sentiment_feature_schema
    validate_sentiment_feature_schema(df, horizon=horizon, source=path)
    return df


# ══════════════════════════════════════════════════════════════════════════════
# FILL MISSING SENTIMENT
# ══════════════════════════════════════════════════════════════════════════════

def sentiment_neutral_columns(sentiment_cols: list[str]) -> list[str]:
    """Return the subset of ``sentiment_cols`` that get neutral-filled.

    Scope is intentionally narrow — only sentiment / attention columns:

    * ``*_score_mean`` / ``*_score_median`` / ``*_score_std``   → ``0`` (neutral)
    * ``*_bullishness_ratio``                                   → ``0.5``
      (no directional information — never ``0``, which would imply
      *bearish unanimity*)
    * ``post_count`` / ``*_post_count`` / ``*_directional_post_count``  → ``0``

    Variante A removed ``*_weighted_mean`` upstream; those columns are
    therefore no longer recognised here.

    Non-sentiment columns (price, volatility, volume, market_cap, target,
    timestamp, ticker) are never returned here, regardless of which flag the
    caller passes.
    """
    keep = []
    for col in sentiment_cols:
        # Variante A: *_weighted_mean is forbidden in the feature frame and
        # must NOT be auto-filled here (engagement-weighted columns no longer
        # exist; recognising them would silently restore the dead branch).
        if col.endswith("_weighted_mean"):
            continue
        if col == "post_count" or col.endswith("_post_count"):
            keep.append(col)
        elif "bullishness_ratio" in col:
            keep.append(col)
        elif any(col.endswith(s) for s in ("_mean", "_median", "_std")):
            keep.append(col)
    return keep


def fill_missing_sentiment(df: pd.DataFrame,
                           sentiment_cols: list[str]) -> pd.DataFrame:
    """Fill NaN sentiment / attention columns with their neutral values.

    Variante A fill rules (separate per family — see ``sentiment_neutral_columns``):

    * ``*_score_mean`` / ``*_score_median`` / ``*_score_std`` → ``0`` (neutral score).
    * ``*_bullishness_ratio`` → ``0.5``. **Not ``0``.** ``0`` would mean
      "unanimously bearish", which is a *directional* signal; the slot
      had no directional posts at all, so the neutral fill must be the
      midpoint. This also catches slots in which every post happens to
      be classified ``neutral`` (denominator = 0 → NaN upstream).
    * ``post_count`` / ``*_post_count`` / ``*_directional_post_count`` → ``0``.

    Non-sentiment columns are never touched.
    """
    for col in sentiment_cols:
        # Variante A: refuse to fill engagement-weighted columns. They should
        # not exist in any v4 pipeline; leaving them NaN forces downstream
        # callers to discover and remove them.
        if col.endswith("_weighted_mean"):
            continue
        if col == "post_count" or col.endswith("_post_count"):
            df[col] = df[col].fillna(COUNT_FILL)
        elif "bullishness_ratio" in col:
            df[col] = df[col].fillna(0.5)
        elif any(col.endswith(s) for s in ("_mean", "_median", "_std")):
            df[col] = df[col].fillna(SCORE_FILL)
        else:
            df[col] = df[col].fillna(SCORE_FILL)

    return df


def derive_post_count_features(df: pd.DataFrame) -> pd.DataFrame:
    """Derive ``log1p_post_count`` and ``has_posts`` AFTER neutral-fill.

    These are deterministic functions of ``post_count`` and must be computed
    *after* the fill so that slots with no posts (``post_count == 0``)
    receive ``log1p_post_count = 0`` and ``has_posts = 0``. Idempotent —
    safe to call repeatedly.
    """
    if "post_count" not in df.columns:
        return df
    df["log1p_post_count"] = np.log1p(df["post_count"].astype(float))
    df["has_posts"]        = (df["post_count"] > 0).astype("int8")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# MERGE
# ══════════════════════════════════════════════════════════════════════════════

def merge_horizon(horizon: str, included_tickers: list[str],
                  feature_dir: str = FEATURE_DIR,
                  *,
                  apply_coverage_filter: bool = True,
                  neutral_fill_sentiment: bool = True) -> tuple:
    """Merge price and sentiment features for one horizon.

    Parameters
    ----------
    apply_coverage_filter
        When ``True`` (default — current behaviour) the merged universe is
        restricted to ``included_tickers``. When ``False``, the coverage
        filter is bypassed and every ticker present in **either** the price or
        sentiment file is kept; tickers that exist in price but not sentiment
        end up with NaN sentiment which is only filled when
        ``neutral_fill_sentiment`` is also ``True``.
    neutral_fill_sentiment
        Controls whether :func:`fill_missing_sentiment` runs on the merged
        frame. Default ``True`` to preserve the historical pipeline output;
        set ``False`` to keep NaN sentiment rows (useful when an analyst wants
        to inspect coverage gaps directly).

    Returns ``(merged_df, report_dict)``.
    """
    df_price = load_price_features(horizon, feature_dir)
    df_sent  = load_sentiment_features(horizon, feature_dir)

    if df_price.empty or df_sent.empty:
        return pd.DataFrame(), {"horizon": horizon, "status": "missing_input"}

    price_tickers = set(df_price["ticker"].unique())
    sent_tickers  = set(df_sent["ticker"].unique())
    if apply_coverage_filter:
        valid_tickers = sorted(set(included_tickers) & price_tickers & sent_tickers)
        n_dropped_low_coverage = len(price_tickers) - len(valid_tickers)
    else:
        # Keep every ticker that has price data; sentiment is left-merged
        # afterwards so its absence shows up as NaN (and is only filled when
        # the neutral-fill flag is set).
        valid_tickers = sorted(price_tickers)
        n_dropped_low_coverage = 0

    if not valid_tickers:
        return pd.DataFrame(), {
            "horizon": horizon, "status": "no_valid_tickers",
            "included_by_coverage": len(included_tickers),
            "in_price": len(price_tickers),
            "in_sentiment": len(sent_tickers),
        }

    df_price = df_price[df_price["ticker"].isin(valid_tickers)].copy()
    df_sent  = df_sent[df_sent["ticker"].isin(valid_tickers)].copy()

    merge_keys = ["ticker", "timestamp"]
    sent_only_cols = [c for c in df_sent.columns if c not in merge_keys]

    overlap = [c for c in sent_only_cols if c in df_price.columns]
    if overlap:
        print(f"  [INFO] Dropping overlapping columns from sentiment: {overlap}")
        df_sent = df_sent.drop(columns=overlap)
        sent_only_cols = [c for c in sent_only_cols if c not in overlap]

    n_price = len(df_price)
    merged = df_price.merge(df_sent, on=merge_keys, how="left")

    has_sent = merged[sent_only_cols[0]].notna().sum() if sent_only_cols else 0
    missing_sent = n_price - has_sent

    # Track how many NaNs we actually fill — useful for diagnostic logging.
    fill_targets = sentiment_neutral_columns(sent_only_cols)
    n_sentiment_nans_filled = 0
    columns_filled: list[str] = []
    if neutral_fill_sentiment:
        before_nan = (merged[fill_targets].isna().sum().sum()
                      if fill_targets else 0)
        merged = fill_missing_sentiment(merged, sent_only_cols)
        after_nan = (merged[fill_targets].isna().sum().sum()
                     if fill_targets else 0)
        n_sentiment_nans_filled = int(before_nan - after_nan)
        columns_filled = [c for c in fill_targets if c in merged.columns]

    # Derive log1p_post_count and has_posts AFTER the fill so that no-post
    # slots get the correct values (0.0 and 0).
    merged = derive_post_count_features(merged)

    merged = merged.sort_values(["ticker", "timestamp"]).reset_index(drop=True)

    report = {
        "horizon": horizon,
        "status": "ok",
        "n_tickers": len(valid_tickers),
        "tickers": ", ".join(valid_tickers),
        "n_rows": len(merged),
        "n_price_rows": n_price,
        "n_sent_matched": has_sent,
        "n_sent_filled": missing_sent,
        "sent_fill_pct": round(missing_sent / n_price * 100, 1) if n_price > 0 else 0,
        "n_features": len(merged.columns),
        "apply_coverage_filter":       bool(apply_coverage_filter),
        "neutral_fill_sentiment":      bool(neutral_fill_sentiment),
        "n_tickers_dropped_low_coverage": int(n_dropped_low_coverage),
        "n_sentiment_nans_filled":     int(n_sentiment_nans_filled),
        "neutral_filled_columns":      ";".join(sorted(columns_filled)),
    }

    return merged, report


# ══════════════════════════════════════════════════════════════════════════════
# CLI / argparse + main
# ══════════════════════════════════════════════════════════════════════════════

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Merge price and sentiment features per horizon"
    )
    parser.add_argument("--feature_dir", "--feature-dir",
                        dest="feature_dir", type=str, default=FEATURE_DIR)
    parser.add_argument("--output_dir", "--output-dir",
                        dest="output_dir", type=str, default=OUTPUT_DIR)
    parser.add_argument("--coverage_threshold", "--coverage-threshold",
                        dest="coverage_threshold", type=float,
                        default=COVERAGE_THRESHOLD,
                        help=f"Min sentiment coverage %% (default: {COVERAGE_THRESHOLD})")
    parser.add_argument("--coverage_csv", "--coverage-csv",
                        dest="coverage_csv", type=str, default=COVERAGE_CSV)
    parser.add_argument("--horizon", default=None,
                        help="Restrict to a single horizon. Default: all of 1h/6h/1d.")
    parser.add_argument("--no-sentiment-coverage-filter",
                        "--no_sentiment_coverage_filter",
                        dest="no_sentiment_coverage_filter",
                        action="store_true",
                        help="Skip the per-horizon sentiment-coverage filter so "
                             "low-coverage tickers are kept (useful for panel "
                             "models). Default: filter is applied (legacy behaviour).")
    parser.add_argument("--neutral-fill-missing-sentiment",
                        "--neutral_fill_missing_sentiment",
                        dest="neutral_fill_missing_sentiment",
                        action=argparse.BooleanOptionalAction, default=True,
                        help="Fill NaN sentiment / attention columns with their "
                             "neutral values (default: on — current behaviour). "
                             "Use --no-neutral-fill-missing-sentiment to leave "
                             "NaN values in place. Non-sentiment columns are "
                             "never touched.")
    parser.add_argument("--smoke", action="store_true",
                        help="Smoke mode: horizon=1d if not specified.")
    parser.add_argument("--dry-run", "--dry_run", dest="dry_run", action="store_true",
                        help="Print planned inputs/outputs and exit.")
    parser.add_argument("--force", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    global FEATURE_DIR, OUTPUT_DIR, COVERAGE_CSV
    args = build_parser().parse_args(argv)

    if args.smoke and not args.horizon:
        args.horizon = "1d"

    FEATURE_DIR  = args.feature_dir
    OUTPUT_DIR   = args.output_dir
    COVERAGE_CSV = args.coverage_csv

    # ── Stage header (best-effort) ──────────────────────────────
    try:
        from ..logging_utils import log_stage_header
        horizons = [args.horizon] if args.horizon else HORIZONS
        inputs = [os.path.join(FEATURE_DIR, f"price_features_{h}.parquet") for h in horizons]
        inputs += [os.path.join(FEATURE_DIR, f"sentiment_features_{h}.parquet") for h in horizons]
        inputs.append(COVERAGE_CSV)
        outputs = [os.path.join(OUTPUT_DIR, f"features_{h}.parquet") for h in horizons]
        outputs.append(os.path.join(OUTPUT_DIR, "merge_report.csv"))
        log_stage_header(
            "merge_features",
            mode="dry-run" if args.dry_run else ("smoke" if args.smoke else "full"),
            inputs=inputs, outputs=outputs,
            extra={"horizon": args.horizon or "(all)",
                   "coverage_threshold": args.coverage_threshold},
        )
    except Exception:  # noqa: BLE001
        pass

    if args.dry_run:
        return 0

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if not os.path.isfile(COVERAGE_CSV):
        print(f"[ERROR] Coverage file not found: {COVERAGE_CSV}")
        return 1

    horizons = [args.horizon] if args.horizon else HORIZONS
    reports: list[dict] = []
    apply_filter = not bool(getattr(args, "no_sentiment_coverage_filter", False))
    neutral_fill = bool(getattr(args, "neutral_fill_missing_sentiment", True))

    for hz in horizons:
        print(f"\n{'=' * 60}")
        print(f"  HORIZON: {hz}")
        print(f"  apply_coverage_filter={apply_filter}  "
              f"neutral_fill_sentiment={neutral_fill}")
        print(f"{'=' * 60}")

        included = get_included_tickers(COVERAGE_CSV, hz, args.coverage_threshold)
        if apply_filter:
            print(f"  Tickers with >={args.coverage_threshold}% coverage: {len(included)}")
        else:
            print(f"  Coverage filter DISABLED — retaining every ticker that has "
                  f"price data (was: {len(included)} above {args.coverage_threshold}% "
                  f"coverage).")
        print(f"    {', '.join(included)}")

        merged, report = merge_horizon(
            hz, included, feature_dir=FEATURE_DIR,
            apply_coverage_filter=apply_filter,
            neutral_fill_sentiment=neutral_fill,
        )
        if not merged.empty:
            print(f"  retained tickers: {report['n_tickers']}  "
                  f"(dropped by coverage filter: {report['n_tickers_dropped_low_coverage']}) "
                  f"sentiment NaNs filled: {report['n_sentiment_nans_filled']}")
        reports.append(report)

        if merged.empty:
            print(f"  [WARN] No data for {hz} — skipping")
            continue

        # ── Final-frame leakage assertion (Aufgabe 7) ─────────────
        # The audit refuses raw engagement columns, ``*_weighted_mean``
        # variants and rows where ``market_cap_available_at`` violates
        # the strict-< availability rule. Failures raise — never
        # warned-and-continue.
        from ..diagnostics.leakage_checks import run_feature_leakage_audit
        run_feature_leakage_audit(merged)

        out_path = os.path.join(OUTPUT_DIR, f"features_{hz}.parquet")
        merged.to_parquet(out_path, index=False, engine="pyarrow")
        size_mb = os.path.getsize(out_path) / 1024 / 1024
        print(f"\n  Saved: {os.path.abspath(out_path)}")
        print(f"    {report['n_rows']:,} rows x {report['n_features']} columns "
              f"({size_mb:.1f} MB)")
        print(f"    {report['n_tickers']} tickers")
        print(f"    Sentiment matched: {report['n_sent_matched']:,} / "
              f"{report['n_price_rows']:,} "
              f"({100 - report['sent_fill_pct']:.1f}%)")
        print(f"    Sentiment filled:  {report['n_sent_filled']:,} "
              f"({report['sent_fill_pct']:.1f}%)")

    report_path = os.path.join(OUTPUT_DIR, "merge_report.csv")
    pd.DataFrame(reports).to_csv(report_path, index=False)
    print(f"\n{'=' * 60}")
    print("DONE")
    print(f"{'=' * 60}")
    print(f"  Report: {os.path.abspath(report_path)}")
    print(f"  Output: {os.path.abspath(OUTPUT_DIR)}")
    print()
    return 0


# Back-compat: external callers (notably ``thesis_pipeline.cli``) used to
# call ``run(...)`` with keyword arguments. This converts to argv → main().
def run(*, horizon: str | None = None, smoke: bool = False,
        dry_run: bool = False, force: bool = False,
        coverage_threshold: float | None = None,
        feature_dir: str | None = None,
        output_dir: str | None = None) -> int:
    argv: list[str] = []
    if horizon:
        argv += ["--horizon", horizon]
    if coverage_threshold is not None:
        argv += ["--coverage-threshold", str(coverage_threshold)]
    if feature_dir:
        argv += ["--feature-dir", feature_dir]
    if output_dir:
        argv += ["--output-dir", output_dir]
    if smoke:
        argv.append("--smoke")
    if dry_run:
        argv.append("--dry-run")
    if force:
        argv.append("--force")
    return main(argv)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
