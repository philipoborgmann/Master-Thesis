#!/usr/bin/env python3
"""
Merge_Features.py
==================
Merges price features and sentiment features per horizon into final
analysis-ready parquet files. Only tickers with sufficient sentiment
coverage (default >=85%) are retained per horizon.

Missing sentiment time slots (where no Reddit posts exist) are filled
with neutral values: 0 for score features, NaN for std features,
and forward-fill for post_count.

Input:
    Data/Features/price_features_{horizon}.parquet
    Data/Features/sentiment_features_{horizon}.parquet
    Data/Features/sentiment_coverage.csv

Output:
    Data/Final/features_1h.parquet
    Data/Final/features_6h.parquet
    Data/Final/features_1d.parquet
    Data/Final/merge_report.csv

Usage:
    python Merge_Features.py
    python Merge_Features.py --coverage_threshold 70
"""

import argparse
import os
import sys

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
# vs. std columns (left as NaN) vs. count columns (forward-filled)
SCORE_FILL = 0.0     # neutral sentiment
STD_FILL   = np.nan   # unknown dispersion — NaN is honest
COUNT_FILL = 0         # no posts = 0 count


# ══════════════════════════════════════════════════════════════════════════════
# LOAD AND FILTER
# ══════════════════════════════════════════════════════════════════════════════

def get_included_tickers(coverage_path: str, horizon: str,
                         threshold: float) -> list[str]:
    """Returns tickers meeting the coverage threshold for a given horizon."""
    df = pd.read_csv(coverage_path)
    sub = df[(df["horizon"] == horizon) & (df["coverage_pct"] >= threshold)]
    return sorted(sub["ticker"].tolist())


def load_price_features(horizon: str) -> pd.DataFrame:
    """Loads price features parquet for a horizon."""
    path = os.path.join(FEATURE_DIR, f"price_features_{horizon}.parquet")
    if not os.path.isfile(path):
        print(f"  [ERROR] Missing: {path}")
        return pd.DataFrame()
    df = pd.read_parquet(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df


def load_sentiment_features(horizon: str) -> pd.DataFrame:
    """Loads sentiment features parquet for a horizon."""
    path = os.path.join(FEATURE_DIR, f"sentiment_features_{horizon}.parquet")
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
    """
    After a left join from price onto sentiment, some rows will have NaN
    sentiment (time slots with no Reddit posts). Fill strategy:

      - Score columns (*_mean, *_weighted_mean, *_median): fill with 0
        (neutral sentiment — no posts = no signal, not missing data)
      - Std columns (*_std): leave as NaN (unknown dispersion)
      - Bullishness ratio: fill with 0.5 (neutral ratio)
      - post_count: fill with 0 (no posts)
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

def merge_horizon(horizon: str, included_tickers: list[str]) -> tuple:
    """
    Merges price and sentiment features for one horizon.

    Returns (merged_df, report_dict).
    """
    df_price = load_price_features(horizon)
    df_sent  = load_sentiment_features(horizon)

    if df_price.empty or df_sent.empty:
        return pd.DataFrame(), {"horizon": horizon, "status": "missing_input"}

    # Filter to included tickers (present in BOTH price and sentiment)
    price_tickers = set(df_price["ticker"].unique())
    sent_tickers  = set(df_sent["ticker"].unique())
    valid_tickers = sorted(
        set(included_tickers) & price_tickers & sent_tickers
    )

    if not valid_tickers:
        return pd.DataFrame(), {
            "horizon": horizon, "status": "no_valid_tickers",
            "included_by_coverage": len(included_tickers),
            "in_price": len(price_tickers),
            "in_sentiment": len(sent_tickers),
        }

    df_price = df_price[df_price["ticker"].isin(valid_tickers)].copy()
    df_sent  = df_sent[df_sent["ticker"].isin(valid_tickers)].copy()

    # Identify sentiment-only columns (everything except ticker, timestamp)
    merge_keys = ["ticker", "timestamp"]
    sent_only_cols = [c for c in df_sent.columns if c not in merge_keys]

    # Drop any overlapping non-key columns from sentiment that already
    # exist in price (e.g. both might have 'date' or 'horizon')
    overlap = [c for c in sent_only_cols if c in df_price.columns]
    if overlap:
        print(f"  [INFO] Dropping overlapping columns from sentiment: {overlap}")
        df_sent = df_sent.drop(columns=overlap)
        sent_only_cols = [c for c in sent_only_cols if c not in overlap]

    # Left join: keep all price rows, attach sentiment where available
    n_price = len(df_price)
    merged = df_price.merge(df_sent, on=merge_keys, how="left")

    # Count how many rows got sentiment data
    has_sent = merged[sent_only_cols[0]].notna().sum() if sent_only_cols else 0
    missing_sent = n_price - has_sent

    # Fill missing sentiment
    merged = fill_missing_sentiment(merged, sent_only_cols)

    # Sort cleanly
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
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    global FEATURE_DIR, OUTPUT_DIR, COVERAGE_CSV

    parser = argparse.ArgumentParser(
        description="Merge price and sentiment features per horizon"
    )
    parser.add_argument("--feature_dir", type=str, default=FEATURE_DIR)
    parser.add_argument("--output_dir", type=str, default=OUTPUT_DIR)
    parser.add_argument("--coverage_threshold", type=float, default=COVERAGE_THRESHOLD,
                        help=f"Min sentiment coverage %% (default: {COVERAGE_THRESHOLD})")
    parser.add_argument("--coverage_csv", type=str, default=COVERAGE_CSV)
    args = parser.parse_args()

    FEATURE_DIR  = args.feature_dir
    OUTPUT_DIR   = args.output_dir
    COVERAGE_CSV = args.coverage_csv

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if not os.path.isfile(COVERAGE_CSV):
        print(f"[ERROR] Coverage file not found: {COVERAGE_CSV}")
        sys.exit(1)

    reports = []

    for hz in HORIZONS:
        print(f"\n{'=' * 60}")
        print(f"  HORIZON: {hz}")
        print(f"{'=' * 60}")

        # Determine included tickers
        included = get_included_tickers(COVERAGE_CSV, hz, args.coverage_threshold)
        print(f"  Tickers with >={args.coverage_threshold}% coverage: "
              f"{len(included)}")
        print(f"    {', '.join(included)}")

        # Merge
        merged, report = merge_horizon(hz, included)
        reports.append(report)

        if merged.empty:
            print(f"  [WARN] No data for {hz} — skipping")
            continue

        # Save
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

    # Save merge report
    report_path = os.path.join(OUTPUT_DIR, "merge_report.csv")
    pd.DataFrame(reports).to_csv(report_path, index=False)
    print(f"\n{'=' * 60}")
    print("DONE")
    print(f"{'=' * 60}")
    print(f"  Report: {os.path.abspath(report_path)}")
    print(f"  Output: {os.path.abspath(OUTPUT_DIR)}")
    print()


if __name__ == "__main__":
    main()