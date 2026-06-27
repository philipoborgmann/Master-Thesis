#!/usr/bin/env python3
"""
Sentiment_Feature_Engineering.py — Variante A
==============================================
Reads the scored sentiment CSVs (VADER, CryptoBERT), combines them
(positional assignment with alignment verification), builds title- and
selftext-based aggregates per (ticker, horizon) slot — and is designed to
eliminate identified engagement-related look-ahead leakage.

Variante A: engagement metrics (``score``, ``upvote_ratio``,
``num_comments``) are a snapshot at the ``retrieved`` instant (~13 h
median after a post). Because engagement reacts to the very price move
we want to forecast, weighting by engagement injects look-ahead bias.
This module therefore:

* drops ``compute_engagement_weight`` and every ``*_weighted_mean`` branch,
* never reads ``score`` / ``upvote_ratio`` / ``num_comments`` /
  ``total_awards_received`` / ``gilded`` as a feature input,
* never uses ``retrieved`` as an input to a feature.

Bucketing is unchanged (post ``date`` → ``ticker × timestamp`` slots, all
floored to the slot start), and per-slot moments (mean / median / std),
the per-model bullishness ratio, post_count, and winsorisation are
unchanged. New per-model column ``{m}_directional_post_count`` records
the count of non-neutral posts in the slot for diagnostics.

Input:
    - Data/Transformed/Sentiment_Scored_Vader.csv
    - Data/Transformed/Sentiment_Scored_Cryptobert.csv

Output:
    - Data/Features/sentiment_features_1h.parquet
    - Data/Features/sentiment_features_6h.parquet
    - Data/Features/sentiment_features_1d.parquet
    - Data/Features/sentiment_coverage.csv
    - Data/Features/plots/  (PNG visualizations)

Usage:
    python -m thesis_pipeline.sentiment.aggregate
    python -m thesis_pipeline.sentiment.aggregate --test_days 30
    python -m thesis_pipeline.sentiment.aggregate --no_plots
"""

import argparse
import os
import sys
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

warnings.filterwarnings("ignore", category=FutureWarning)


# ══════════════════════════════════════════════════════════════════════════════
# 1. CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

MODELS = ["vader", "cryptobert"]   # FinBERT removed (see docs/refactor_log.md)

DEFAULT_INPUTS = {
    "vader":      os.path.join("Data", "Transformed", "Sentiment_Scored_Vader.csv"),
    "cryptobert": os.path.join("Data", "Transformed", "Sentiment_Scored_Cryptobert.csv"),
}

OUTPUT_DIR = os.path.join("Data", "Features")
PLOT_DIR   = os.path.join(OUTPUT_DIR, "plots")

EMPTY_SELFTEXT = {"[removed]", "[deleted]", "", "nan", "None"}

TITLE_WEIGHT = 0.7

HORIZONS = {
    "1h": "1h",
    "6h": "6h",
    "1d": "1D",
}


# ══════════════════════════════════════════════════════════════════════════════
# 2. LOAD AND MERGE
# ══════════════════════════════════════════════════════════════════════════════

def load_scored_data(paths: dict) -> pd.DataFrame:
    """
    Loads the three scored CSVs and combines them into a single DataFrame.
    Positional assignment with alignment verification.
    """
    print("[INFO] Loading scored data...")

    score_cols = ["title_sentiment", "title_score", "title_prob_pos",
                  "title_prob_neg", "title_prob_neu",
                  "selftext_sentiment", "selftext_score", "selftext_prob_pos",
                  "selftext_prob_neg", "selftext_prob_neu"]

    verify_cols = ["date", "ticker", "title"]

    first_model = MODELS[0]
    df = pd.read_csv(paths[first_model], parse_dates=["date"])
    print(f"  {first_model}: {len(df):,} rows from {os.path.abspath(paths[first_model])}")

    existing = [c for c in score_cols if c in df.columns]
    rename_map = {c: f"{first_model}_{c}" for c in existing}
    df = df.rename(columns=rename_map)

    for model in MODELS[1:]:
        df_m = pd.read_csv(paths[model], parse_dates=["date"])
        print(f"  {model}: {len(df_m):,} rows from {os.path.abspath(paths[model])}")

        if len(df_m) != len(df):
            raise ValueError(
                f"Row count mismatch: {first_model} has {len(df):,} rows, "
                f"{model} has {len(df_m):,}.")

        check_idx = np.linspace(0, len(df) - 1, 5, dtype=int)
        for vc in verify_cols:
            if vc not in df_m.columns:
                continue
            base_vals = df[vc].iloc[check_idx].astype(str).values
            model_vals = df_m[vc].iloc[check_idx].astype(str).values
            if not np.array_equal(base_vals, model_vals):
                raise ValueError(
                    f"Row alignment check failed on column '{vc}' between "
                    f"{first_model} and {model}.")

        print(f"  [OK] Alignment verified against {first_model}")

        existing_m = [c for c in score_cols if c in df_m.columns]
        for col in existing_m:
            df[f"{model}_{col}"] = df_m[col].values

    print(f"[INFO] Merged DataFrame: {len(df):,} rows, {len(df.columns)} columns")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# 3. BUILD POST-LEVEL FEATURES
# ══════════════════════════════════════════════════════════════════════════════
#
# Variante A: engagement weighting is intentionally removed. The function
# previously named ``compute_engagement_weight`` no longer exists (any caller
# that still imports it will fail loudly — the absence is the contract).

def build_post_level_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build per-post features without any engagement weight.

    ``selftext`` is still used to construct the per-model ``combined_score``
    (a 0.7 / 0.3 blend of title and selftext sentiment) — this stays in the
    output as a robustness/diagnostic column. It is **not** part of the
    primary 17-set feature grid (see ``features/feature_registry.py``).
    """
    print("\n[INFO] Building post-level features (Variante A — no engagement weighting)...")

    required = ["date", "ticker", "selftext"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    selftext_clean = df["selftext"].fillna("").astype(str).str.strip()
    has_usable_selftext = (
        ~selftext_clean.str.lower().isin({s.lower() for s in EMPTY_SELFTEXT})
        & (selftext_clean.str.len() > 0)
    )

    for model in MODELS:
        st_col = f"{model}_selftext_score"
        if st_col not in df.columns:
            df[st_col] = np.nan

    for model in MODELS:
        t_score  = f"{model}_title_score"
        st_score = f"{model}_selftext_score"

        has_st = has_usable_selftext & df[st_score].notna()
        combined = df[t_score].copy()
        combined[has_st] = (
            TITLE_WEIGHT * df.loc[has_st, t_score].values +
            (1 - TITLE_WEIGHT) * df.loc[has_st, st_score].values
        )
        df[f"{model}_combined_score"] = combined

    n_with_st = has_usable_selftext.sum()
    print(f"  Posts with usable selftext: {n_with_st:,} / {len(df):,} "
          f"({n_with_st / len(df) * 100:.1f}%)")

    return df


# ══════════════════════════════════════════════════════════════════════════════
# 5. TIME-BASED AGGREGATION
# ══════════════════════════════════════════════════════════════════════════════

def _bullishness_ratio(sentiments: pd.Series) -> float:
    n_pos = (sentiments == "positive").sum()
    n_neg = (sentiments == "negative").sum()
    denom = n_pos + n_neg
    if denom == 0:
        return np.nan
    return n_pos / denom


def aggregate_to_horizon(df: pd.DataFrame, freq: str,
                         horizon_label: str) -> pd.DataFrame:
    """Bucket posts to ``(ticker, horizon-slot)`` and compute moments.

    Per slot and per model: ``*_mean / _median / _std`` for the three score
    variants (title / selftext / combined), per-model ``bullishness_ratio``
    and ``directional_post_count`` (count of positively- or negatively-
    classified posts in the slot), plus the slot-level ``post_count``.

    No engagement weighting (Variante A) — designed to eliminate identified
    engagement-related look-ahead leakage.
    """
    print(f"\n[INFO] Aggregating to {horizon_label} ({freq})...")

    df = df.copy()
    df["ts"] = df["date"].dt.floor(freq)

    records = []

    for (ticker, ts), grp in df.groupby(["ticker", "ts"]):
        row = {"ticker": ticker, "timestamp": ts, "post_count": len(grp)}

        for model in MODELS:
            sent_col = f"{model}_title_sentiment"
            if sent_col in grp.columns:
                row[f"{model}_bullishness_ratio"] = _bullishness_ratio(grp[sent_col])
                # Per-model directional post count = pos + neg (neutrals excluded).
                row[f"{model}_directional_post_count"] = int(
                    grp[sent_col].isin(["positive", "negative"]).sum()
                )

            for variant in ["title_score", "selftext_score", "combined_score"]:
                col = f"{model}_{variant}"
                if col not in grp.columns:
                    continue

                vals = grp[col].dropna()
                if len(vals) == 0:
                    row[f"{col}_mean"] = np.nan
                    row[f"{col}_median"] = np.nan
                    row[f"{col}_std"] = np.nan
                else:
                    row[f"{col}_mean"] = vals.mean()
                    row[f"{col}_median"] = vals.median()
                    row[f"{col}_std"] = vals.std() if len(vals) > 1 else 0.0

        records.append(row)

    result = pd.DataFrame(records)
    result = result.sort_values(["ticker", "timestamp"]).reset_index(drop=True)

    n_tickers = result["ticker"].nunique()
    n_slots   = len(result)
    print(f"  {n_slots:,} time slots across {n_tickers} tickers")
    print(f"  Date range: {result['timestamp'].min()} to {result['timestamp'].max()}")
    print(f"  Columns: {len(result.columns)}")

    return result


# ══════════════════════════════════════════════════════════════════════════════
# 5b. WINSORIZATION
# ══════════════════════════════════════════════════════════════════════════════

# Percentile bounds for winsorization (0.5% each tail)
WINSOR_LOWER = 0.005
WINSOR_UPPER = 0.995


def winsorize_features(aggregated: dict) -> dict:
    """
    Clips numeric feature columns at the 0.5th and 99.5th percentile to
    reduce the influence of extreme outliers on downstream models.

    Winsorization is applied globally per column (across all tickers and
    time slots within a horizon). Columns excluded from clipping: ticker,
    timestamp, post_count, and any boolean/string columns.

    Returns the same dict with clipped DataFrames.
    """
    # Columns that should never be winsorized. Per-model directional counts
    # are integers and are kept exact.
    skip_cols = {"ticker", "timestamp", "post_count"}
    for hz_label, df_hz in aggregated.items():
        skip_dyn = skip_cols | {c for c in df_hz.columns
                                if c.endswith("_directional_post_count")}
        numeric_cols = [
            c for c in df_hz.select_dtypes(include=[np.number]).columns
            if c not in skip_dyn
        ]

        n_clipped_total = 0
        n_values_total = 0

        for col in numeric_cols:
            vals = df_hz[col].dropna()
            if len(vals) < 20:
                continue

            lo = vals.quantile(WINSOR_LOWER)
            hi = vals.quantile(WINSOR_UPPER)

            n_below = (df_hz[col] < lo).sum()
            n_above = (df_hz[col] > hi).sum()
            n_clipped_total += n_below + n_above
            n_values_total += vals.shape[0]

            df_hz[col] = df_hz[col].clip(lower=lo, upper=hi)

        pct = n_clipped_total / n_values_total * 100 if n_values_total > 0 else 0
        print(f"  {hz_label}: {n_clipped_total:,} values clipped across "
              f"{len(numeric_cols)} columns ({pct:.2f}%)")

        aggregated[hz_label] = df_hz

    return aggregated


# ══════════════════════════════════════════════════════════════════════════════
# 5c. COVERAGE STATISTICS
# ══════════════════════════════════════════════════════════════════════════════

def compute_coverage_stats(aggregated: dict, output_dir: str) -> pd.DataFrame:
    """
    For each horizon × ticker, computes coverage statistics and classifies
    each combination as 'included' or 'excluded' based on a 70% coverage
    threshold (≤30% missing).

    Threshold justification (Schafer & Graham, 2002; Collins et al., 2001):
      - ≤20% missing: results comparable to complete data
      - 20-30% missing: performance weakens but remains acceptable
      - >30% missing: substantial bias risk, especially with MNAR data

    Sentiment missing data is MNAR (Missing Not At Random): time slots are
    empty precisely when no one posts, which correlates with low market
    activity. Imputing these slots with neutral scores systematically biases
    the feature distribution. The 70% coverage threshold is therefore
    conservative but appropriate for MNAR sentiment data.

    Saves results to sentiment_coverage.csv with an 'included' column.
    """
    COVERAGE_THRESHOLD = 70.0  # minimum coverage % to include

    print("\n[INFO] Computing coverage statistics...")
    print(f"  Inclusion threshold: >= {COVERAGE_THRESHOLD:.0f}% coverage "
          f"(<= {100 - COVERAGE_THRESHOLD:.0f}% missing)")

    records = []

    for hz_label, df_hz in aggregated.items():
        date_min = df_hz["timestamp"].min()
        date_max = df_hz["timestamp"].max()

        if hz_label == "1h":
            freq = "1h"
        elif hz_label == "6h":
            freq = "6h"
        elif hz_label == "1d":
            freq = "1D"
        else:
            freq = hz_label

        full_range = pd.date_range(date_min, date_max, freq=freq)
        max_slots = len(full_range)

        for ticker in sorted(df_hz["ticker"].unique()):
            sub = df_hz[df_hz["ticker"] == ticker]
            n_actual = len(sub)
            coverage_pct = n_actual / max_slots * 100 if max_slots > 0 else 0
            missing_pct = 100 - coverage_pct
            median_posts = sub["post_count"].median()
            single_post_pct = (sub["post_count"] == 1).mean() * 100
            included = coverage_pct >= COVERAGE_THRESHOLD

            records.append({
                "horizon": hz_label,
                "ticker": ticker,
                "max_slots": max_slots,
                "actual_slots": n_actual,
                "coverage_pct": round(coverage_pct, 1),
                "missing_pct": round(missing_pct, 1),
                "median_posts_per_slot": round(median_posts, 1),
                "single_post_pct": round(single_post_pct, 1),
                "included": included,
            })

    df_cov = pd.DataFrame(records)

    # Save to CSV
    csv_path = os.path.join(output_dir, "sentiment_coverage.csv")
    df_cov.to_csv(csv_path, index=False)
    print(f"  Saved: {os.path.abspath(csv_path)}")

    # Print summary per horizon
    for hz in aggregated:
        sub = df_cov[df_cov["horizon"] == hz]
        n_total = len(sub)
        n_included = sub["included"].sum()
        n_excluded = n_total - n_included
        avg_cov = sub["coverage_pct"].mean()

        print(f"\n  {hz.upper()} ({sub['max_slots'].iloc[0]:,} max slots):")
        print(f"    Included: {n_included}/{n_total} tickers  "
              f"(avg coverage: {sub.loc[sub['included'], 'coverage_pct'].mean():.1f}%)")
        if n_excluded > 0:
            excluded = sub[~sub["included"]].sort_values("coverage_pct")
            tickers_excl = ", ".join(
                f"{r['ticker']} ({r['coverage_pct']:.0f}%)"
                for _, r in excluded.iterrows()
            )
            print(f"    Excluded: {n_excluded} tickers — {tickers_excl}")
        else:
            print(f"    Excluded: none")

    return df_cov

    return df_cov


# ══════════════════════════════════════════════════════════════════════════════
# 6. VISUALIZATIONS
# ══════════════════════════════════════════════════════════════════════════════

def plot_score_distributions(df, plot_dir):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)
    colors = {"vader": "#2196F3", "cryptobert": "#FF9800"}
    for ax, model in zip(axes, MODELS):
        col = f"{model}_title_score"
        if col not in df.columns: continue
        vals = df[col].dropna()
        ax.hist(vals, bins=80, color=colors[model], alpha=0.85, edgecolor="white", linewidth=0.3)
        ax.set_title(f"{model.upper()}", fontsize=12, fontweight="bold")
        ax.set_xlabel("Title Score")
        ax.axvline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
        stats = f"\u03bc={vals.mean():.3f}\n\u03c3={vals.std():.3f}\nmed={vals.median():.3f}"
        ax.text(0.97, 0.95, stats, transform=ax.transAxes, fontsize=8,
                verticalalignment="top", horizontalalignment="right",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    axes[0].set_ylabel("Count")
    fig.suptitle("Title Sentiment Score Distributions by Model", fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "score_distributions.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: score_distributions.png")

def plot_sentiment_class_comparison(df, plot_dir):
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = {"positive": "#4CAF50", "neutral": "#9E9E9E", "negative": "#F44336"}
    x = np.arange(len(MODELS))
    width = 0.25
    for i, label in enumerate(["positive", "neutral", "negative"]):
        proportions = []
        for model in MODELS:
            col = f"{model}_title_sentiment"
            if col in df.columns:
                total = df[col].notna().sum()
                count = (df[col] == label).sum()
                proportions.append(count / total * 100 if total > 0 else 0)
            else: proportions.append(0)
        ax.bar(x + (i - 1) * width, proportions, width, label=label.capitalize(),
               color=colors[label], edgecolor="white", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels([m.upper() for m in MODELS])
    ax.set_ylabel("Proportion (%)")
    ax.set_title("Sentiment Classification Distribution by Model")
    ax.legend(loc="upper right")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "sentiment_class_comparison.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: sentiment_class_comparison.png")

def plot_model_correlation(df, plot_dir):
    score_cols = [f"{m}_title_score" for m in MODELS]
    available = [c for c in score_cols if c in df.columns]
    if len(available) < 2: return
    subset = df[available].dropna()
    n = len(available)
    fig, axes = plt.subplots(n, n, figsize=(4 * n, 4 * n))
    if n == 1: axes = np.array([[axes]])
    for i in range(n):
        for j in range(n):
            ax = axes[i][j]
            if i == j:
                ax.hist(subset[available[i]], bins=60, color="#607D8B", alpha=0.7)
                ax.set_title(available[i].replace("_title_score", "").upper(), fontsize=10)
            else:
                ax.scatter(subset[available[j]], subset[available[i]], alpha=0.02, s=1, color="#1976D2")
                corr = subset[available[i]].corr(subset[available[j]])
                ax.text(0.05, 0.95, f"r={corr:.3f}", transform=ax.transAxes, fontsize=10,
                        verticalalignment="top", bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8))
            if j == 0: ax.set_ylabel(available[i].replace("_title_score", "").upper(), fontsize=9)
            if i == n - 1: ax.set_xlabel(available[j].replace("_title_score", "").upper(), fontsize=9)
    fig.suptitle("Pairwise Title Score Correlations", fontsize=13, y=1.01)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "model_correlation_matrix.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: model_correlation_matrix.png")

def plot_daily_sentiment_timeseries(df_daily, plot_dir):
    if "ticker" not in df_daily.columns: return
    ticker_counts = df_daily.groupby("ticker").size()
    top_ticker = ticker_counts.idxmax()
    sub = df_daily[df_daily["ticker"] == top_ticker].sort_values("timestamp")
    fig, ax = plt.subplots(figsize=(14, 5))
    colors = {"vader": "#2196F3", "cryptobert": "#FF9800"}
    for model in MODELS:
        col = f"{model}_title_score_mean"
        if col in sub.columns:
            ax.plot(sub["timestamp"], sub[col], label=model.upper(), color=colors[model], alpha=0.8, linewidth=0.8)
    ax.axhline(0, color="black", linewidth=0.5, linestyle="--", alpha=0.4)
    ax.set_title(f"Daily Mean Title Sentiment Score \u2014 {top_ticker}", fontsize=12)
    ax.set_xlabel("Date"); ax.set_ylabel("Mean Score")
    ax.legend(loc="upper right"); ax.grid(alpha=0.2)
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    plt.xticks(rotation=45); plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "daily_sentiment_timeseries.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: daily_sentiment_timeseries.png")

def create_all_plots(df_post, df_daily, plot_dir):
    """Variante A \u2014 no engagement plots; engagement weighting is removed."""
    print("\n[INFO] Creating visualizations...")
    os.makedirs(plot_dir, exist_ok=True)
    plot_score_distributions(df_post, plot_dir)
    plot_sentiment_class_comparison(df_post, plot_dir)
    plot_model_correlation(df_post, plot_dir)
    plot_daily_sentiment_timeseries(df_daily, plot_dir)
    print(f"[INFO] All plots saved to {os.path.abspath(plot_dir)}")


# ══════════════════════════════════════════════════════════════════════════════
# 7. EXPORT
# ══════════════════════════════════════════════════════════════════════════════

def save_parquet(df, path, label):
    abs_path = os.path.abspath(path)
    os.makedirs(os.path.dirname(abs_path) or ".", exist_ok=True)
    df.to_parquet(abs_path, index=False, engine="pyarrow")
    size = os.path.getsize(abs_path)
    print(f"  {label}: {abs_path} ({size / 1024 / 1024:.1f} MB, "
          f"{len(df):,} rows x {len(df.columns)} cols)")


# ══════════════════════════════════════════════════════════════════════════════
# 8. MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Sentiment Feature Engineering - build aggregated features"
    )
    parser.add_argument("--vader_input", type=str, default=DEFAULT_INPUTS["vader"])
    parser.add_argument("--cryptobert_input", type=str, default=DEFAULT_INPUTS["cryptobert"])
    parser.add_argument("--output_dir", type=str, default=OUTPUT_DIR)
    parser.add_argument("--test_days", type=int, default=None)
    parser.add_argument("--no_plots", action="store_true")
    args = parser.parse_args(argv)

    paths = {
        "vader":      args.vader_input,
        "cryptobert": args.cryptobert_input,
    }

    for model, path in paths.items():
        if not os.path.isfile(path):
            print(f"[ERROR] Missing input: {model} -> {path}")
            sys.exit(1)

    # Step 1: Load and merge
    print("\n" + "=" * 60)
    print("STEP 1: Loading and merging scored data")
    print("=" * 60)
    df = load_scored_data(paths)

    if args.test_days is not None:
        cutoff = df["date"].min() + pd.Timedelta(days=args.test_days)
        before = len(df)
        df = df[df["date"] < cutoff].reset_index(drop=True)
        print(f"[INFO] TEST MODE: first {args.test_days} days -> {before:,} -> {len(df):,} posts")

    # Step 2: Build post-level features
    print("\n" + "=" * 60)
    print("STEP 2: Building post-level features")
    print("=" * 60)
    df = build_post_level_features(df)

    # Step 3: Aggregate to all horizons
    print("\n" + "=" * 60)
    print("STEP 3: Aggregating to time horizons")
    print("=" * 60)
    aggregated = {}
    for label, freq in HORIZONS.items():
        aggregated[label] = aggregate_to_horizon(df, freq, label)

    # Step 4: Winsorize outliers
    print("\n" + "=" * 60)
    print("STEP 4: Winsorizing outliers (0.5th / 99.5th percentile)")
    print("=" * 60)
    aggregated = winsorize_features(aggregated)

    # Step 5: Coverage statistics
    print("\n" + "=" * 60)
    print("STEP 5: Coverage statistics")
    print("=" * 60)
    compute_coverage_stats(aggregated, args.output_dir)

    # Step 6: Visualizations
    if not args.no_plots:
        print("\n" + "=" * 60)
        print("STEP 6: Creating visualizations")
        print("=" * 60)
        plot_dir = os.path.join(args.output_dir, "plots")
        create_all_plots(df, aggregated["1d"], plot_dir)
    else:
        print("\n[INFO] Skipping plots (--no_plots)")

    # Step 7: Export parquet files
    print("\n" + "=" * 60)
    print("STEP 7: Exporting parquet files")
    print("=" * 60)
    for label in HORIZONS:
        out_path = os.path.join(args.output_dir, f"sentiment_features_{label}.parquet")
        save_parquet(aggregated[label], out_path, label)

    # Summary
    print(f"\n{'=' * 60}")
    print("DONE - Feature Engineering Summary")
    print(f"{'=' * 60}")
    print(f"  Input posts:         {len(df):,}")
    print(f"  Tickers:             {df['ticker'].nunique()}")
    print(f"  Date range:          {df['date'].min()} -> {df['date'].max()}")
    print(f"  Models:              {', '.join(m.upper() for m in MODELS)}")
    print(f"  Score variants:      raw (Variante A — engagement weighting removed)")
    print(f"  Text sources:        title (primary), selftext (robustness), combined (robustness)")
    for label in HORIZONS:
        agg = aggregated[label]
        n_cols = len([c for c in agg.columns if c not in ("ticker", "timestamp", "post_count")])
        print(f"  {label:6s} features:    {n_cols} columns, {len(agg):,} rows")
    print(f"  Output:              {os.path.abspath(args.output_dir)}")
    print()


if __name__ == "__main__":
    main()
