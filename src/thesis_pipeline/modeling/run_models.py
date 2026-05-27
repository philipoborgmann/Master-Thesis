"""Modeling stage — Ridge logistic regression with expanding-window walk-forward.

This module owns the **canonical** modeling logic. The original root-level
``Run_Models.py`` is now a one-line legacy redirect; ``scripts/run_models.py``
is a thin entry point on the same ``main()``; the package CLI
(``thesis_pipeline.cli``) also calls this ``main()`` directly.

Validation design and methodology are unchanged from the previous root-level
implementation:

* Initial train split = first ``INIT_TRAIN_FRAC`` (50%) of chronologically
  sorted observations per ticker.
* Expanding walk-forward, step size 1; train on ``[0, t-1]``, predict at t.
* ``StandardScaler`` re-fit on every training window only.
* Logistic regression with L2 penalty, ``C = DEFAULT_C`` (1.0),
  ``random_state = 42``.
* Benchmark sentinel ``__majority_class__`` is mapped to
  ``__rolling_probability__`` and uses ``run_rolling_probability``.

CLI is intentionally permissive about argument names so that

    python Run_Models.py        --set_id B1 --ticker BTC
    python scripts/run_models.py --set-id B1 --coins BTC ETH
    python -m thesis_pipeline.cli run-models --set-id B1 --coins BTC ETH

all work.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import warnings
from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, f1_score,
    precision_score, recall_score, log_loss, brier_score_loss,
)

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

# Default Ridge regularisation strength
DEFAULT_C = 1.0


# ══════════════════════════════════════════════════════════════════════════════
# DATA LOADING
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
    """
    Loads the feature set configuration XLSX. Robust to sheet names
    'feature_sets' or 'model_specs'. Normalises column headers to
    snake_case.

    Header preference (this is the historically-fixed behaviour and
    must be preserved):

      * `feature_columns_comma_separated`  →  `features`
      * `feature_columns`                  →  `features`
      * fallback: longest-mean-length column
    """
    xls = pd.ExcelFile(path)
    sheet = None
    for candidate in ["feature_sets", "model_specs"]:
        if candidate in xls.sheet_names:
            sheet = candidate
            break
    if sheet is None:
        sheet = xls.sheet_names[0]

    df = pd.read_excel(xls, sheet_name=sheet)

    # Normalise display headers to snake_case
    df.columns = (df.columns
                  .str.strip()
                  .str.lower()
                  .str.replace(r"[^a-z0-9]+", "_", regex=True)
                  .str.strip("_"))

    # Rename known variants to expected names.
    #
    # Explicit priority for the column that holds the comma-separated feature
    # list:
    #     feature_columns_comma_separated  >  feature_columns  >  features
    #
    # The previous behaviour skipped the rename when ``features`` already
    # existed in the frame — e.g. because the XLSX still contains a
    # ``# Features`` column that normalises to ``features`` after the regex
    # above. In that situation the wrong column silently won. Fix: when a
    # more specific header is present, drop any pre-existing ``features``
    # column first so the rename actually takes effect.
    for preferred in ("feature_columns_comma_separated", "feature_columns"):
        if preferred in df.columns:
            if "features" in df.columns and preferred != "features":
                df = df.drop(columns=["features"])
            df = df.rename(columns={preferred: "features"})
            break

    # Fallback: if 'features' still missing, use the longest column
    if "features" not in df.columns:
        longest = max(df.columns, key=lambda c: df[c].astype(str).str.len().mean())
        df = df.rename(columns={longest: "features"})

    # Unify benchmark sentinel: `__majority_class__` must not be parsed as a
    # missing feature column called "0".
    df["features"] = df["features"].replace("__majority_class__", "__rolling_probability__")

    print(f"  [INFO] Config columns: {list(df.columns)}")
    print(f"  [INFO] {len(df)} feature set configurations loaded from sheet '{sheet}'")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# WALK-FORWARD VALIDATION
# ══════════════════════════════════════════════════════════════════════════════

def run_walk_forward(df_ticker: pd.DataFrame,
                     feature_cols: list[str],
                     C: float = DEFAULT_C) -> pd.DataFrame:
    """
    Expanding-window walk-forward validation for a single ticker.

    For each step t (starting at 50% of the data):
      1. Train on observations [0, t-1]
      2. Fit StandardScaler on training window only
      3. Fit LogisticRegression(L2, C) on scaled training data
      4. Predict observation t
      5. Advance t by 1

    Returns DataFrame with: timestamp, ticker, target, prediction, probability
    """
    df = df_ticker.sort_values("timestamp").reset_index(drop=True)
    n = len(df)
    init_train = max(int(n * INIT_TRAIN_FRAC), 30)

    X_all = df[feature_cols].values.astype(float)
    y_all = df["target"].values.astype(float)
    timestamps = df["timestamp"].values
    ticker = df["ticker"].iloc[0]

    results = []

    for t in range(init_train, n):
        # ── Skip if test point has NaN ────────────────────────
        if np.any(np.isnan(X_all[t])) or np.isnan(y_all[t]):
            continue

        # ── Training window: [0, t-1], drop NaN rows ─────────
        X_train = X_all[:t]
        y_train = y_all[:t]

        valid_mask = ~(np.isnan(X_train).any(axis=1) | np.isnan(y_train))
        X_tr = X_train[valid_mask]
        y_tr = y_train[valid_mask]

        # Need at least 20 observations and both classes
        if len(y_tr) < 20 or len(np.unique(y_tr)) < 2:
            continue

        # ── Scale: fit on training only, transform test ───────
        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_te_s = scaler.transform(X_all[t:t+1])

        # ── Fit Ridge logistic regression ─────────────────────
        model = LogisticRegression(
            penalty="l2",
            C=C,
            solver="lbfgs",
            max_iter=1000,
            random_state=42,
        )
        model.fit(X_tr_s, y_tr)

        pred = int(model.predict(X_te_s)[0])
        prob = float(model.predict_proba(X_te_s)[0, 1])

        results.append({
            "timestamp":   timestamps[t],
            "ticker":      ticker,
            "target":      int(y_all[t]),
            "prediction":  pred,
            "probability": prob,
        })

    if not results:
        return pd.DataFrame()

    return pd.DataFrame(results)


def run_rolling_probability(df_ticker: pd.DataFrame) -> pd.DataFrame:
    """
    Benchmark: rolling probability from the expanding training window.

    For each test point t:
      p_hat = mean(y_train[0:t])
      prediction = 1 if p_hat >= 0.5 else 0
      probability = p_hat
    """
    df = df_ticker.sort_values("timestamp").reset_index(drop=True)
    n = len(df)
    init_train = max(int(n * INIT_TRAIN_FRAC), 30)

    y = df["target"].values.astype(float)
    timestamps = df["timestamp"].values
    ticker = df["ticker"].iloc[0]

    results = []

    for t in range(init_train, n):
        if np.isnan(y[t]):
            continue

        y_train = y[:t]
        valid = y_train[~np.isnan(y_train)]
        if len(valid) == 0:
            continue

        p_hat = float(np.mean(valid))
        pred = 1 if p_hat >= 0.5 else 0

        results.append({
            "timestamp":   timestamps[t],
            "ticker":      ticker,
            "target":      int(y[t]),
            "prediction":  pred,
            "probability": p_hat,
        })

    if not results:
        return pd.DataFrame()

    return pd.DataFrame(results)


# ══════════════════════════════════════════════════════════════════════════════
# METRICS
# ══════════════════════════════════════════════════════════════════════════════

def compute_metrics(signals: pd.DataFrame, ticker_label: str = "pooled") -> dict:
    """
    Computes classification and probabilistic metrics from signal DataFrame.
    Robust to single-class samples.
    """
    y_true = signals["target"].values
    y_pred = signals["prediction"].values
    y_prob = signals["probability"].values

    n = len(y_true)
    n_classes = len(np.unique(y_true))

    out = {
        "ticker":       ticker_label,
        "n_obs":        n,
        "first_test_timestamp": str(signals["timestamp"].min()),
        "last_test_timestamp":  str(signals["timestamp"].max()),
    }

    if n == 0:
        for m in ["accuracy", "balanced_accuracy", "f1", "precision",
                   "recall", "log_loss", "brier_score"]:
            out[m] = np.nan
        return out

    # Classification metrics
    out["accuracy"]          = round(accuracy_score(y_true, y_pred), 6)
    out["balanced_accuracy"] = round(balanced_accuracy_score(y_true, y_pred), 6)

    if n_classes >= 2:
        out["f1"]        = round(f1_score(y_true, y_pred, zero_division=0), 6)
        out["precision"] = round(precision_score(y_true, y_pred, zero_division=0), 6)
        out["recall"]    = round(recall_score(y_true, y_pred, zero_division=0), 6)
    else:
        out["f1"]        = np.nan
        out["precision"] = np.nan
        out["recall"]    = np.nan

    # Probabilistic metrics
    if n_classes >= 2 and not np.any(np.isnan(y_prob)):
        # Clip probabilities to avoid log(0)
        y_prob_safe = np.clip(y_prob, 1e-15, 1 - 1e-15)
        out["log_loss"]    = round(log_loss(y_true, y_prob_safe), 6)
        out["brier_score"] = round(brier_score_loss(y_true, y_prob_safe), 6)
    else:
        out["log_loss"]    = np.nan
        out["brier_score"] = np.nan

    return out


# ══════════════════════════════════════════════════════════════════════════════
# CLI / ARGPARSE
# ══════════════════════════════════════════════════════════════════════════════

def build_parser() -> argparse.ArgumentParser:
    """Argparse parser accepting both old (``--set_id``, ``--ticker``) and
    new (``--set-id``, ``--coins``) argument names."""
    parser = argparse.ArgumentParser(
        description="Ridge logistic regression — walk-forward validation"
    )
    parser.add_argument("--feature_config", "--feature-config",
                        dest="feature_config", default=FEATURE_CONFIG)
    parser.add_argument("--horizon", default=None, choices=HORIZONS)
    parser.add_argument("--set_id", "--set-id", dest="set_id", default=None)
    # Accept --coins (new, nargs+), --coin (singular legacy) and --ticker (old)
    # all bound to the same destination so multi-coin runs work cleanly.
    parser.add_argument(
        "--coins", "--coin", "--ticker",
        dest="coins", nargs="+", default=None,
        help="One or more tickers to model. Aliases: --coin, --ticker.",
    )
    parser.add_argument("--sentiment_model", "--sentiment-model",
                        dest="sentiment_model", default=None)
    parser.add_argument("--C", type=float, default=DEFAULT_C,
                        help=f"Ridge regularisation strength (default: {DEFAULT_C})")
    parser.add_argument("--model-type", "--model_type", dest="model_type",
                        default="per_asset", choices=["per_asset", "panel_logit"],
                        help="per_asset (canonical, default) or panel_logit "
                             "(pooled / ticker-FE panel regression).")
    parser.add_argument("--panel-mode", "--panel_mode", dest="panel_mode",
                        default="pooled", choices=["pooled", "ticker_fixed_effects"],
                        help="Panel-logit variant (only used when "
                             "--model-type panel_logit).")
    parser.add_argument("--smoke", action="store_true",
                        help="Smoke mode: defaults to horizon=1d, coins=[BTC, ETH], set_id=B1.")
    parser.add_argument("--dry-run", "--dry_run", dest="dry_run",
                        action="store_true",
                        help="Print planned inputs/outputs and exit.")
    parser.add_argument("--force", action="store_true",
                        help="Reserved (no overwrite-protection currently); accepted for CLI uniformity.")
    parser.add_argument("--restart", action="store_true",
                        help="Ignore cached signal parquets and rerun every set.")
    return parser


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main(argv: Sequence[str] | None = None) -> int:
    """Entry point used by the legacy ``Run_Models.py``, ``scripts/run_models.py``
    and the package CLI. Returns 0 on success."""
    args = build_parser().parse_args(argv)

    # ── Alternative model family: delegate to the panel-logit module ──
    # The per-asset logic below is unchanged; panel_logit is purely additive.
    if getattr(args, "model_type", "per_asset") == "panel_logit":
        from .panel_logit import _run_panel
        return _run_panel(args)

    # ── Smoke defaults ──────────────────────────────────────────
    if args.smoke:
        if not args.horizon:
            args.horizon = "1d"
        if not args.coins:
            args.coins = ["BTC", "ETH"]
        if not args.set_id:
            args.set_id = "B1"

    # ── Stage header (best-effort; never fails the run) ─────────
    try:
        from ..logging_utils import log_stage_header
        from ..config import resolve_path
        inputs = []
        if args.horizon:
            inputs.append(resolve_path("final_features_pattern", horizon=args.horizon))
        inputs.append(resolve_path("feature_sets_xlsx"))
        outputs: list = []
        if args.horizon and args.set_id:
            outputs.append(resolve_path("signals_pattern",
                                        horizon=args.horizon, set_id=args.set_id))
        outputs.append(resolve_path("signals_metrics"))
        log_stage_header(
            "run_models",
            mode="dry-run" if args.dry_run else ("smoke" if args.smoke else "full"),
            inputs=inputs,
            outputs=outputs,
            extra={
                "horizon":         args.horizon or "(all)",
                "set_id":          args.set_id or "(all)",
                "coins":           list(args.coins) if args.coins else "(all)",
                "sentiment_model": args.sentiment_model or "(per set_id default)",
                "C":               args.C,
                "restart":         args.restart,
                "force":           args.force,
            },
        )
    except Exception:  # noqa: BLE001 — logging is best-effort
        pass

    # ── Dry-run exits before any heavy I/O ──────────────────────
    if args.dry_run:
        return 0

    os.makedirs(SIGNAL_DIR, exist_ok=True)

    # ── Load feature set configuration ──────────────────────────
    config = load_feature_sets(args.feature_config)

    if args.set_id:
        config = config[config["set_id"] == args.set_id]
        print(f"  Filtered to set_id={args.set_id}: {len(config)} rows")

    # Optional filter by sentiment model. Only applied when the column exists
    # in the config — otherwise it would always evaluate to empty.
    if args.sentiment_model and "sentiment_model" in config.columns:
        sm = args.sentiment_model.lower()
        config = config[config["sentiment_model"].astype(str).str.lower() == sm]
        print(f"  Filtered to sentiment_model={args.sentiment_model}: {len(config)} rows")

    if config.empty:
        print("  [WARN] Feature-set config is empty after filtering; nothing to do.")
        return 0

    horizons = [args.horizon] if args.horizon else HORIZONS
    all_metrics: list[dict] = []

    for hz in horizons:
        print(f"\n{'=' * 70}")
        print(f"  HORIZON: {hz}")
        print(f"{'=' * 70}")

        df_all = load_features(hz)
        if df_all.empty:
            print(f"  [SKIP] No data for {hz}")
            continue

        tickers = sorted(df_all["ticker"].unique())
        if args.coins:
            wanted = set(args.coins)
            tickers = [t for t in tickers if t in wanted]
        print(f"  Tickers: {len(tickers)} — {', '.join(tickers)}")
        print(f"  Model: LogisticRegression(L2, C={args.C})")
        print(f"  Initial train split: {INIT_TRAIN_FRAC:.0%}")

        hz_dir = os.path.join(SIGNAL_DIR, hz)
        os.makedirs(hz_dir, exist_ok=True)

        for _, cfg_row in config.iterrows():
            set_id     = cfg_row["set_id"]
            category   = cfg_row.get("category", "")
            sent_model = cfg_row.get("sentiment_model", "-")
            label      = cfg_row.get("label", "")
            feat_str   = cfg_row["features"]

            # File naming
            if sent_model and str(sent_model) != "-" and str(sent_model) != "nan":
                out_name = f"{set_id}_{sent_model}"
            else:
                out_name = str(set_id)
                sent_model = "-"

            out_path = os.path.join(hz_dir, f"{out_name}.parquet")

            # ── Checkpoint: skip if already computed ──────────
            if os.path.isfile(out_path) and not args.restart:
                try:
                    cached = pd.read_parquet(out_path)
                    m = compute_metrics(cached, "pooled")
                    m.update({"horizon": hz, "set_id": set_id,
                              "sentiment_model": sent_model, "label": label,
                              "category": category,
                              "n_tickers": cached["ticker"].nunique()})
                    all_metrics.append(m)
                    for tk, grp in cached.groupby("ticker"):
                        mt = compute_metrics(grp, tk)
                        mt.update({"horizon": hz, "set_id": set_id,
                                   "sentiment_model": sent_model, "label": label,
                                   "category": category, "n_tickers": 1})
                        all_metrics.append(mt)
                    print(f"\n  ── {out_name} ({label}) "
                          f"→ CACHED acc={m['accuracy']:.4f}, n={m['n_obs']}")
                    continue
                except Exception:
                    pass  # corrupted — will re-run

            print(f"\n  ── {out_name} ({label}) ", end="", flush=True)

            # ── Benchmark: rolling probability ────────────────
            if feat_str.strip() in ("__rolling_probability__", "__majority_class__"):
                all_signals = []
                for tk in tickers:
                    df_t = df_all[df_all["ticker"] == tk]
                    sig = run_rolling_probability(df_t)
                    if not sig.empty:
                        all_signals.append(sig)

                if all_signals:
                    signals = pd.concat(all_signals, ignore_index=True)
                    signals["set_id"] = set_id
                    signals["sentiment_model"] = sent_model
                    signals["horizon"] = hz
                    signals.to_parquet(out_path, index=False, engine="pyarrow")

                    m = compute_metrics(signals, "pooled")
                    m.update({"horizon": hz, "set_id": set_id,
                              "sentiment_model": sent_model, "label": label,
                              "category": category,
                              "n_tickers": len(all_signals)})
                    all_metrics.append(m)
                    for tk, grp in signals.groupby("ticker"):
                        mt = compute_metrics(grp, tk)
                        mt.update({"horizon": hz, "set_id": set_id,
                                   "sentiment_model": sent_model, "label": label,
                                   "category": category, "n_tickers": 1})
                        all_metrics.append(mt)
                    print(f"→ acc={m['accuracy']:.4f}, brier={m.get('brier_score', np.nan):.4f}, "
                          f"n={m['n_obs']}")
                else:
                    print("→ no signals")
                continue

            # ── Parse feature list ────────────────────────────
            feature_cols = [f.strip() for f in feat_str.split(",")]

            missing = [f for f in feature_cols if f not in df_all.columns]
            if missing:
                print(f"→ SKIP (missing: {missing[:3]})")
                all_metrics.append({
                    "horizon": hz, "set_id": set_id, "sentiment_model": sent_model,
                    "label": label, "category": category, "ticker": "pooled",
                    "accuracy": np.nan, "n_obs": 0, "n_tickers": 0,
                    "status": f"missing: {missing[:3]}",
                })
                continue

            # ── Run per ticker ────────────────────────────────
            t0 = time.time()
            all_signals = []

            for tk in tickers:
                df_t = df_all[df_all["ticker"] == tk]
                sig = run_walk_forward(df_t, feature_cols, C=args.C)
                if not sig.empty:
                    all_signals.append(sig)

            elapsed = time.time() - t0

            if all_signals:
                signals = pd.concat(all_signals, ignore_index=True)
                signals["set_id"] = set_id
                signals["sentiment_model"] = sent_model
                signals["horizon"] = hz
                signals.to_parquet(out_path, index=False, engine="pyarrow")

                m = compute_metrics(signals, "pooled")
                m.update({"horizon": hz, "set_id": set_id,
                          "sentiment_model": sent_model, "label": label,
                          "category": category,
                          "n_tickers": len(all_signals)})
                all_metrics.append(m)

                for tk, grp in signals.groupby("ticker"):
                    mt = compute_metrics(grp, tk)
                    mt.update({"horizon": hz, "set_id": set_id,
                               "sentiment_model": sent_model, "label": label,
                               "category": category, "n_tickers": 1})
                    all_metrics.append(mt)

                print(f"→ acc={m['accuracy']:.4f}, "
                      f"f1={m['f1']:.4f}, "
                      f"brier={m.get('brier_score', np.nan):.4f}, "
                      f"n={m['n_obs']}, {elapsed:.1f}s")
            else:
                print(f"→ no signals ({elapsed:.1f}s)")
                all_metrics.append({
                    "horizon": hz, "set_id": set_id, "sentiment_model": sent_model,
                    "label": label, "category": category, "ticker": "pooled",
                    "accuracy": np.nan, "n_obs": 0, "n_tickers": 0,
                })

    # ── Save metrics summary ──────────────────────────────────
    if all_metrics:
        metrics_df = pd.DataFrame(all_metrics)
        metrics_path = os.path.join(SIGNAL_DIR, "metrics_summary.csv")
        metrics_df.to_csv(metrics_path, index=False)

        print(f"\n{'=' * 70}")
        print("DONE")
        print(f"{'=' * 70}")
        print(f"  Total metric rows: {len(metrics_df)}")
        print(f"  Metrics: {metrics_path}")
        print(f"  Signals: {os.path.abspath(SIGNAL_DIR)}")

        pooled = metrics_df[metrics_df["ticker"] == "pooled"].copy()
        valid = pooled[pooled["accuracy"].notna()]
        if not valid.empty:
            print(f"\n  Top 10 by accuracy (pooled):")
            top = valid.nlargest(10, "accuracy")
            for _, r in top.iterrows():
                sm = f"/{r['sentiment_model']}" if str(r.get("sentiment_model", "-")) != "-" else ""
                print(f"    {r['set_id']}{sm:15s} {r['horizon']:3s}  "
                      f"acc={r['accuracy']:.4f}  "
                      f"f1={r.get('f1', np.nan):.4f}  "
                      f"brier={r.get('brier_score', np.nan):.4f}  "
                      f"n={int(r['n_obs'])}")
        print()

    return 0


# ══════════════════════════════════════════════════════════════════════════════
# Back-compat shim — previous package callers may import ``run(...)``.
# Kept as a thin keyword-argument wrapper around the new argparse-based main().
# ══════════════════════════════════════════════════════════════════════════════

def run(*, horizon: str | None = None, set_id: str | None = None,
        coins: Sequence[str] | None = None,
        sentiment_model: str | None = None,
        smoke: bool = False, dry_run: bool = False,
        force: bool = False, restart: bool = False,
        C: float | None = None,
        feature_config: str | None = None) -> int:
    """Programmatic entry point. Translates keyword arguments to argv for
    :func:`main`. Prefer calling :func:`main` directly with an argv list."""
    argv: list[str] = []
    if horizon:
        argv += ["--horizon", horizon]
    if set_id:
        argv += ["--set-id", set_id]
    if coins:
        argv += ["--coins", *list(coins)]
    if sentiment_model:
        argv += ["--sentiment-model", sentiment_model]
    if C is not None:
        argv += ["--C", str(C)]
    if feature_config:
        argv += ["--feature-config", feature_config]
    if smoke:
        argv.append("--smoke")
    if dry_run:
        argv.append("--dry-run")
    if force:
        argv.append("--force")
    if restart:
        argv.append("--restart")
    return main(argv)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
