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
STD_FILL   = np.nan  # unknown dispersion — NaN is honest
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
    """Load price features parquet for a horizon."""
    path = os.path.join(feature_dir, f"price_features_{horizon}.parquet")
    if not os.path.isfile(path):
        print(f"  [ERROR] Missing: {path}")
        return pd.DataFrame()
    df = pd.read_parquet(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df


def load_sentiment_features(horizon: str, feature_dir: str = FEATURE_DIR) -> pd.DataFrame:
    """Load sentiment features parquet for a horizon."""
    path = os.path.join(feature_dir, f"sentiment_features_{horizon}.parquet")
    if not os.path.isfile(path):
        print(f"  [ERROR] Missing: {path}")
        return pd.DataFrame()
    df = pd.read_parquet(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df


# ══════════════════════════════════════════════════════════════════════════════
# FILL MISSING SENTIMENT
# ══════════════════════════════════════════════════════════════════════════════

def fill_missing_sentiment(df: pd.DataFrame,
                           sentiment_cols: list[str]) -> pd.DataFrame:
    """Fill NaN sentiment after a left join from price onto sentiment.

    * Score columns (*_mean / *_weighted_mean / *_median) → 0  (neutral signal)
    * Std columns (*_std)                                 → NaN (honest unknown)
    * Bullishness ratio                                   → 0.5 (neutral)
    * post_count                                          → 0
    """
    for col in sentiment_cols:
        if col == "post_count":
            df[col] = df[col].fillna(COUNT_FILL)
        elif col.endswith("_std"):
            pass  # leave NaN — unknown dispersion is honest
        elif "bullishness_ratio" in col:
            df[col] = df[col].fillna(0.5)
        elif any(col.endswith(s) for s in ("_mean", "_weighted_mean", "_median")):
            df[col] = df[col].fillna(SCORE_FILL)
        else:
            df[col] = df[col].fillna(SCORE_FILL)

    return df


# ══════════════════════════════════════════════════════════════════════════════
# MERGE
# ══════════════════════════════════════════════════════════════════════════════

def merge_horizon(horizon: str, included_tickers: list[str],
                  feature_dir: str = FEATURE_DIR) -> tuple:
    """Merge price and sentiment features for one horizon.

    Returns ``(merged_df, report_dict)``.
    """
    df_price = load_price_features(horizon, feature_dir)
    df_sent  = load_sentiment_features(horizon, feature_dir)

    if df_price.empty or df_sent.empty:
        return pd.DataFrame(), {"horizon": horizon, "status": "missing_input"}

    price_tickers = set(df_price["ticker"].unique())
    sent_tickers  = set(df_sent["ticker"].unique())
    valid_tickers = sorted(set(included_tickers) & price_tickers & sent_tickers)

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

    merged = fill_missing_sentiment(merged, sent_only_cols)
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

    for hz in horizons:
        print(f"\n{'=' * 60}")
        print(f"  HORIZON: {hz}")
        print(f"{'=' * 60}")

        included = get_included_tickers(COVERAGE_CSV, hz, args.coverage_threshold)
        print(f"  Tickers with >={args.coverage_threshold}% coverage: {len(included)}")
        print(f"    {', '.join(included)}")

        merged, report = merge_horizon(hz, included, feature_dir=FEATURE_DIR)
        reports.append(report)

        if merged.empty:
            print(f"  [WARN] No data for {hz} — skipping")
            continue

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
