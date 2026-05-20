#!/usr/bin/env python3
"""
Run_Models.py
==============
Runs Ridge (L2) logistic regression across all feature sets and horizons
defined in feature_sets.xlsx. Uses expanding-window forward validation.

For each (horizon × feature_set × ticker), the script:
  1. Loads the merged feature parquet from Data/Final/
  2. Splits into expanding train / single-step test windows
  3. Fits LogisticRegressionCV with L2 penalty (Ridge) on the training set
  4. Generates out-of-sample predictions and predicted probabilities
  5. Saves signals (predictions) to Outputs/Signals/
  6. Computes first-pass accuracy metrics

The expanding window starts at 50% of the available data and grows by
one observation per step. StandardScaler is re-fitted on each training
fold to prevent information leakage.

Input:
    Data/Final/features_{horizon}.parquet
    feature_sets.xlsx

Output:
    Outputs/Signals/{horizon}/{set_id}_{sentiment_model}.parquet
    Outputs/Signals/metrics_summary.csv

Usage:
    python Run_Models.py
    python Run_Models.py --horizon 1d
    python Run_Models.py --set_id S1 --horizon 1d
"""

import argparse
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegressionCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

FINAL_DIR      = os.path.join("Data", "Final")
FEATURE_CONFIG = "feature_sets.xlsx"
SIGNAL_DIR     = os.path.join("Outputs", "Signals")

HORIZONS       = ["1h", "6h", "1d"]

# Expanding window: initial training fraction
INIT_TRAIN_FRAC = 0.50

# Ridge logistic regression: cross-validation folds for C selection
CV_FOLDS = 5
# Regularisation strength search grid (inverse of lambda)
CS = np.logspace(-4, 4, 20)


# ══════════════════════════════════════════════════════════════════════════════
# LOAD DATA
# ══════════════════════════════════════════════════════════════════════════════

def load_features(horizon: str) -> pd.DataFrame:
    """Loads the merged feature parquet for a horizon."""
    path = os.path.join(FINAL_DIR, f"features_{horizon}.parquet")
    if not os.path.isfile(path):
        print(f"  [ERROR] Missing: {path}")
        return pd.DataFrame()
    df = pd.read_parquet(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values(["ticker", "timestamp"]).reset_index(drop=True)
    return df


def load_feature_sets(path: str) -> pd.DataFrame:
    """Loads the feature set configuration XLSX."""
    df = pd.read_excel(path, sheet_name="feature_sets")
    # Normalize display headers to snake_case keys
    df.columns = (df.columns
                  .str.strip()
                  .str.lower()
                  .str.replace(" ", "_")
                  .str.replace("#", "n")
                  .str.replace("(comma-separated)", "", regex=False)
                  .str.replace("__", "_")
                  .str.strip("_"))
    # Ensure expected columns exist
    rename_map = {
        "feature_columns": "features",
    }
    for old, new in rename_map.items():
        if old in df.columns and new not in df.columns:
            df = df.rename(columns={old: new})

    # Fallback: if 'features' still not found, use the longest column name
    if "features" not in df.columns:
        longest = max(df.columns, key=len)
        print(f"  [INFO] Renaming '{longest}' → 'features'")
        df = df.rename(columns={longest: "features"})

    print(f"  [INFO] Config columns: {list(df.columns)}")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# EXPANDING WINDOW FORWARD VALIDATION
# ══════════════════════════════════════════════════════════════════════════════

def run_expanding_window(df_ticker: pd.DataFrame,
                         feature_cols: list[str],
                         set_id: str) -> pd.DataFrame:
    """
    Expanding-window forward validation for a single ticker.

    Starting from INIT_TRAIN_FRAC of the data, the model is trained on
    all observations up to time t and predicts the direction at t+1.
    StandardScaler is re-fitted on each expanding window.

    Returns a DataFrame with columns:
        timestamp, ticker, target, prediction, probability, set_id
    """
    df = df_ticker.sort_values("timestamp").reset_index(drop=True)
    n = len(df)
    init_train = max(int(n * INIT_TRAIN_FRAC), 30)

    # Validate features exist and have no NaN in the required columns
    available = [c for c in feature_cols if c in df.columns]
    if len(available) < len(feature_cols):
        missing = set(feature_cols) - set(available)
        return pd.DataFrame()

    X = df[feature_cols].values
    y = df["target"].values
    timestamps = df["timestamp"].values
    ticker = df["ticker"].iloc[0]

    predictions = []
    probabilities = []
    targets = []
    ts_out = []

    for t in range(init_train, n):
        X_train = X[:t]
        y_train = y[:t]
        X_test  = X[t:t+1]
        y_test  = y[t]

        # Skip if only one class in training data
        if len(np.unique(y_train)) < 2:
            continue

        # Skip if test has NaN
        if np.any(np.isnan(X_test)) or np.isnan(y_test):
            continue

        # Handle NaN in training: drop those rows
        train_mask = ~(np.isnan(X_train).any(axis=1) | np.isnan(y_train))
        X_tr = X_train[train_mask]
        y_tr = y_train[train_mask]

        if len(y_tr) < 20 or len(np.unique(y_tr)) < 2:
            continue

        # Scale features (fitted on training only)
        scaler = StandardScaler()
        X_tr_scaled = scaler.fit_transform(X_tr)
        X_te_scaled = scaler.transform(X_test)

        # Fit Ridge logistic regression with CV for C selection
        model = LogisticRegressionCV(
            penalty="l2",
            Cs=CS,
            cv=min(CV_FOLDS, len(y_tr) // 5),  # safety for small samples
            scoring="accuracy",
            solver="lbfgs",
            max_iter=1000,
            random_state=42,
        )
        model.fit(X_tr_scaled, y_tr)

        pred = model.predict(X_te_scaled)[0]
        prob = model.predict_proba(X_te_scaled)[0, 1]  # P(class=1)

        predictions.append(int(pred))
        probabilities.append(float(prob))
        targets.append(int(y_test))
        ts_out.append(timestamps[t])

    if not predictions:
        return pd.DataFrame()

    return pd.DataFrame({
        "timestamp":   ts_out,
        "ticker":      ticker,
        "target":      targets,
        "prediction":  predictions,
        "probability": probabilities,
        "set_id":      set_id,
    })


def run_majority_baseline(df_ticker: pd.DataFrame) -> pd.DataFrame:
    """
    B1 benchmark: predicts the majority class from the expanding training
    window. No features needed.
    """
    df = df_ticker.sort_values("timestamp").reset_index(drop=True)
    n = len(df)
    init_train = max(int(n * INIT_TRAIN_FRAC), 30)

    y = df["target"].values
    timestamps = df["timestamp"].values
    ticker = df["ticker"].iloc[0]

    predictions = []
    targets = []
    ts_out = []

    for t in range(init_train, n):
        y_train = y[:t]
        majority = int(np.round(np.mean(y_train)))  # 1 if >50% positive
        predictions.append(majority)
        targets.append(int(y[t]))
        ts_out.append(timestamps[t])

    return pd.DataFrame({
        "timestamp":   ts_out,
        "ticker":      ticker,
        "target":      targets,
        "prediction":  predictions,
        "probability": [np.nan] * len(predictions),
        "set_id":      "B1",
    })


# ══════════════════════════════════════════════════════════════════════════════
# METRICS
# ══════════════════════════════════════════════════════════════════════════════

def compute_metrics(signals: pd.DataFrame) -> dict:
    """Computes first-pass classification metrics from signal DataFrame."""
    y_true = signals["target"].values
    y_pred = signals["prediction"].values

    if len(y_true) == 0 or len(np.unique(y_true)) < 2:
        return {"accuracy": np.nan, "f1": np.nan,
                "precision": np.nan, "recall": np.nan, "n_obs": len(y_true)}

    return {
        "accuracy":  round(accuracy_score(y_true, y_pred), 4),
        "f1":        round(f1_score(y_true, y_pred, zero_division=0), 4),
        "precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "recall":    round(recall_score(y_true, y_pred, zero_division=0), 4),
        "n_obs":     len(y_true),
    }


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Run Ridge logistic regression across all feature sets"
    )
    parser.add_argument("--feature_config", type=str, default=FEATURE_CONFIG)
    parser.add_argument("--horizon", type=str, default=None,
                        choices=HORIZONS, help="Run only this horizon")
    parser.add_argument("--set_id", type=str, default=None,
                        help="Run only this feature set ID")
    parser.add_argument("--ticker", type=str, default=None,
                        help="Run only this ticker")
    args = parser.parse_args()

    os.makedirs(SIGNAL_DIR, exist_ok=True)

    # Load feature set configuration
    config = load_feature_sets(args.feature_config)
    print(f"[INFO] Loaded {len(config)} feature set configurations")

    if args.set_id:
        config = config[config["set_id"] == args.set_id]
        print(f"  Filtered to set_id={args.set_id}: {len(config)} rows")

    horizons = [args.horizon] if args.horizon else HORIZONS
    all_metrics = []

    for hz in horizons:
        print(f"\n{'=' * 70}")
        print(f"  HORIZON: {hz}")
        print(f"{'=' * 70}")

        df_all = load_features(hz)
        if df_all.empty:
            print(f"  [SKIP] No data for {hz}")
            continue

        tickers = sorted(df_all["ticker"].unique())
        if args.ticker:
            tickers = [t for t in tickers if t == args.ticker]
        print(f"  Tickers: {len(tickers)} — {', '.join(tickers)}")

        hz_dir = os.path.join(SIGNAL_DIR, hz)
        os.makedirs(hz_dir, exist_ok=True)

        for _, cfg_row in config.iterrows():
            set_id = cfg_row["set_id"]
            category = cfg_row["category"]
            sent_model = cfg_row["sentiment_model"]
            label = cfg_row["label"]
            features_str = cfg_row["features"]

            # File naming
            if sent_model and sent_model != "-":
                out_name = f"{set_id}_{sent_model}"
            else:
                out_name = set_id

            print(f"\n  ── {out_name} ({label}) ", end="")

            # Parse feature list
            if features_str == "__majority_class__":
                # B1 benchmark — no features
                all_signals = []
                for ticker in tickers:
                    df_t = df_all[df_all["ticker"] == ticker]
                    sig = run_majority_baseline(df_t)
                    if not sig.empty:
                        all_signals.append(sig)

                if all_signals:
                    signals = pd.concat(all_signals, ignore_index=True)
                    out_path = os.path.join(hz_dir, f"{out_name}.parquet")
                    signals.to_parquet(out_path, index=False, engine="pyarrow")

                    metrics = compute_metrics(signals)
                    metrics.update({"horizon": hz, "set_id": set_id,
                                    "sentiment_model": sent_model, "label": label,
                                    "category": category, "n_tickers": len(all_signals)})
                    all_metrics.append(metrics)
                    print(f"→ acc={metrics['accuracy']:.3f}, n={metrics['n_obs']}")
                else:
                    print("→ no signals")
                continue

            feature_cols = [f.strip() for f in features_str.split(",")]

            # Check if features exist in the data
            missing_feats = [f for f in feature_cols if f not in df_all.columns]
            if missing_feats:
                print(f"→ SKIP (missing: {missing_feats[:3]})")
                all_metrics.append({
                    "horizon": hz, "set_id": set_id, "sentiment_model": sent_model,
                    "label": label, "category": category,
                    "accuracy": np.nan, "n_obs": 0, "n_tickers": 0,
                    "status": f"missing_features: {missing_feats[:3]}",
                })
                continue

            # Run per ticker
            t0 = time.time()
            all_signals = []
            for ticker in tickers:
                df_t = df_all[df_all["ticker"] == ticker]
                sig = run_expanding_window(df_t, feature_cols, set_id)
                if not sig.empty:
                    sig["sentiment_model"] = sent_model
                    all_signals.append(sig)

            elapsed = time.time() - t0

            if all_signals:
                signals = pd.concat(all_signals, ignore_index=True)
                out_path = os.path.join(hz_dir, f"{out_name}.parquet")
                signals.to_parquet(out_path, index=False, engine="pyarrow")

                metrics = compute_metrics(signals)
                metrics.update({"horizon": hz, "set_id": set_id,
                                "sentiment_model": sent_model, "label": label,
                                "category": category, "n_tickers": len(all_signals)})
                all_metrics.append(metrics)
                print(f"→ acc={metrics['accuracy']:.3f}, "
                      f"f1={metrics['f1']:.3f}, "
                      f"n={metrics['n_obs']}, "
                      f"{elapsed:.1f}s")
            else:
                print(f"→ no signals ({elapsed:.1f}s)")
                all_metrics.append({
                    "horizon": hz, "set_id": set_id, "sentiment_model": sent_model,
                    "label": label, "category": category,
                    "accuracy": np.nan, "n_obs": 0, "n_tickers": 0,
                })

    # ── Save metrics summary ──────────────────────────────────────
    if all_metrics:
        metrics_df = pd.DataFrame(all_metrics)
        metrics_path = os.path.join(SIGNAL_DIR, "metrics_summary.csv")
        metrics_df.to_csv(metrics_path, index=False)
        print(f"\n{'=' * 70}")
        print("DONE — Metrics Summary")
        print(f"{'=' * 70}")
        print(f"  Total runs: {len(all_metrics)}")
        print(f"  Metrics:    {metrics_path}")
        print(f"  Signals:    {os.path.abspath(SIGNAL_DIR)}")

        # Quick leaderboard
        valid = metrics_df[metrics_df["accuracy"].notna()].copy()
        if not valid.empty:
            print(f"\n  Top 10 by accuracy:")
            top = valid.nlargest(10, "accuracy")
            for _, r in top.iterrows():
                sm = f"/{r['sentiment_model']}" if r.get("sentiment_model", "-") != "-" else ""
                print(f"    {r['set_id']}{sm:12s} {r['horizon']:3s}  "
                      f"acc={r['accuracy']:.4f}  f1={r.get('f1', np.nan):.4f}  "
                      f"n={int(r['n_obs'])}")
        print()


if __name__ == "__main__":
    main()