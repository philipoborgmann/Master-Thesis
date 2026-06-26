"""Modeling stage — Ridge logistic regression / panel-logit walk-forward.

This module owns the **canonical** modeling logic. The package CLI
(``thesis_pipeline.cli``) and ``scripts/run_models.py`` are thin entry
points that both invoke ``main()`` below.

v4 canonical defaults (Aufgabe 5)
---------------------------------
A bare ``python -m thesis_pipeline.cli run-models`` runs the intended
production pipeline:

* ``--model-type panel_logit``
* ``--panel-mode ticker_fixed_effects`` (coin dummies, shared slopes)
* ``--train-window rolling_fixed`` with ``--rolling-window-days 180``
  (180 CALENDAR days — identical wall-clock across 1d / 6h / 1h)
* ``--tune-hyperparams`` (BooleanOptionalAction, default ON)
* ``--hpo-objective log_loss``
* Per-ticker / per-chunk checkpointing on (resume from prior runs).

Pass ``--no-tune-hyperparams``, ``--model-type per_asset``, or
``--train-window expanding`` to opt back into the historical behaviour.

Validation invariants (unchanged from earlier versions)
-------------------------------------------------------
* Initial train split = first ``INIT_TRAIN_FRAC`` (50 %) of
  chronologically sorted observations per ticker.
* Walk-forward, step size 1; train on rows with ``timestamp < τ``, predict τ.
* ``StandardScaler`` re-fit on every training window only.
* Logistic regression with L2 penalty, ``DEFAULT_C = 1.0`` as the fixed
  fallback when ``--no-tune-hyperparams`` is set.
* Benchmark sentinel ``__majority_class__`` is mapped to
  ``__rolling_probability__`` and uses ``run_rolling_probability`` —
  this is the NAIVE evaluation reference, not a v4 feature set.

CLI is intentionally permissive about argument names — both dashed and
underscored spellings work, and the singular legacy ``--ticker`` /
``--coin`` aliases route to the same destination as ``--coins``. All
three of the following are equivalent and run the v4 canonical
pipeline against the ECON feature set:

    python -m thesis_pipeline.modeling.run_models --set_id ECON --ticker BTC
    python scripts/run_models.py                  --set-id ECON --coins BTC ETH
    python -m thesis_pipeline.cli run-models      --set-id ECON --coins BTC ETH

The historical ``B1`` set was the rolling-probability benchmark; it is
no longer a feature set in v4. ``B1`` was removed from the registry
(see :data:`thesis_pipeline.features.feature_registry.REMOVED_SET_IDS`)
and any ``--set-id B1`` invocation now raises a clear migration error —
do not try to run ``--set-id B1``; the rolling-probability behaviour
moved to the NAIVE evaluation reference in
:mod:`thesis_pipeline.modeling.benchmarks`.
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
CHECKPOINT_DIR = os.path.join("Outputs", "Checkpoints", "Models")

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
                     C: float = DEFAULT_C,
                     tune_hyperparams: bool = False,
                     hpo_config: dict | None = None) -> pd.DataFrame:
    """
    Expanding-window walk-forward validation for a single ticker.

    For each step t (starting at 50% of the data):
      1. Train on observations [0, t-1]
      2. Fit StandardScaler on training window only
      3. Fit LogisticRegression(L2, C) on scaled training data
      4. Predict observation t
      5. Advance t by 1

    Returns DataFrame with: timestamp, ticker, target, prediction, probability.

    Hyperparameter tuning (``tune_hyperparams=True``)
    -------------------------------------------------
    When enabled, the fixed ``C`` is ignored. Instead, at every step the
    current training window ``[0, t-1]`` is handed to
    :func:`thesis_pipeline.modeling.hyperparameter_tuning.tune_logistic_hyperparams`,
    which splits *that window* chronologically into inner-train / validation,
    grid-searches the regularisation strength (and optional class weight), and
    re-fits the best model on the full window before predicting step ``t``. The
    test point ``t`` is never used for tuning or for fitting — there is no
    lookahead leakage. Tuned rows carry extra provenance columns
    (``hpo_enabled``, ``hpo_objective``, ``best_C``, ``best_class_weight``,
    ``hpo_score``, ``hpo_status``).

    When ``tune_hyperparams=False`` the behaviour is unchanged: the same fixed
    ``C`` is used and no HPO columns are written.
    """
    df = df_ticker.sort_values("timestamp").reset_index(drop=True)
    n = len(df)
    init_train = max(int(n * INIT_TRAIN_FRAC), 30)

    X_all = df[feature_cols].values.astype(float)
    y_all = df["target"].values.astype(float)
    timestamps = df["timestamp"].values
    ticker = df["ticker"].iloc[0]

    if tune_hyperparams:
        from .hyperparameter_tuning import (
            PER_ASSET, hpo_row_columns, predict_proba, tune_logistic_hyperparams,
        )
        hpo_config = hpo_config or {}
        # v4 canonical objective is log_loss. A caller that passes an
        # `hpo_config` with no `objective` key (e.g. an old YAML, or a
        # synthetic dict in a test) inherits the same v4 default as
        # `load_hpo_config()` — never the v3 "brier_score" sentinel.
        objective = hpo_config.get("objective", "log_loss")
        search_space = hpo_config.get("search_space", {})

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

        if tune_hyperparams:
            # Leakage-safe nested tuning on the training window only. The slice
            # df.iloc[:t][valid_mask] is exactly the rows behind (X_tr, y_tr).
            train_df = df.iloc[:t][valid_mask]
            test_df  = df.iloc[t:t + 1]
            res = tune_logistic_hyperparams(
                train_df, feature_cols,
                family=PER_ASSET, objective=objective,
                search_space=search_space, hpo_cfg=hpo_config,
            )
            prob = float(predict_proba(res["artifacts"], test_df, feature_cols,
                                       family=PER_ASSET)[0])
            pred = int(prob >= 0.5)
            row = {
                "timestamp":   timestamps[t],
                "ticker":      ticker,
                "target":      int(y_all[t]),
                "prediction":  pred,
                "probability": prob,
            }
            row.update(hpo_row_columns(objective, res))
            results.append(row)
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
                        default="panel_logit",
                        choices=["per_asset", "panel_logit"],
                        help="Default: panel_logit (v4 canonical pipeline). "
                             "Use per_asset to opt back into the historical "
                             "ticker-by-ticker logistic-ridge.")
    parser.add_argument("--panel-mode", "--panel_mode", dest="panel_mode",
                        default="ticker_fixed_effects",
                        choices=["pooled", "ticker_fixed_effects"],
                        help="Default: ticker_fixed_effects (shared slopes + "
                             "coin dummies). Use pooled for a single shared "
                             "intercept across all coins.")
    parser.add_argument("--smoke", action="store_true",
                        help="Smoke mode: defaults to horizon=1d, coins=[BTC, ETH], set_id=ECON.")
    parser.add_argument("--dry-run", "--dry_run", dest="dry_run",
                        action="store_true",
                        help="Print planned inputs/outputs and exit.")
    parser.add_argument("--force", action="store_true",
                        help="Reserved (no overwrite-protection currently); accepted for CLI uniformity.")
    parser.add_argument("--restart", action="store_true",
                        help="Ignore cached signal parquets and rerun every set.")
    # ── Hyperparameter tuning (conservative, leakage-safe grid search) ──
    parser.add_argument("--tune-hyperparams", "--tune_hyperparams",
                        dest="tune_hyperparams",
                        action=argparse.BooleanOptionalAction, default=True,
                        help="Run nested grid-search HPO inside each "
                             "walk-forward training window (v4 default: ON). "
                             "Use --no-tune-hyperparams to fall back to the "
                             "fixed-C estimator.")
    parser.add_argument("--hpo-objective", "--hpo_objective",
                        dest="hpo_objective", default="log_loss",
                        choices=["brier_score", "log_loss", "accuracy"],
                        help="HPO selection metric (v4 default: log_loss).")
    parser.add_argument("--hpo-config", "--hpo_config", dest="hpo_config",
                        default=None,
                        help="Path to a YAML holding a hyperparameter_tuning "
                             "section (defaults to configs/model_specs.yaml).")
    parser.add_argument("--hpo-grid-C", "--hpo_grid_C", dest="hpo_grid_C",
                        type=float, nargs="+", default=None,
                        help="Override the C search grid, e.g. --hpo-grid-C 0.01 0.1 1 10.")
    parser.add_argument("--hpo-class-weight", "--hpo_class_weight",
                        dest="hpo_class_weight", nargs="+", default=None,
                        help="Override the class_weight grid, e.g. "
                             "--hpo-class-weight none balanced.")
    # ── Checkpointing / resume ──────────────────────────────────
    parser.add_argument("--checkpoint", dest="checkpoint",
                        action=argparse.BooleanOptionalAction, default=True,
                        help="Write per-ticker checkpoints so a crashed run can "
                             "resume (default: on).")
    parser.add_argument("--resume", dest="resume",
                        action=argparse.BooleanOptionalAction, default=True,
                        help="Reuse existing checkpoints (default: on).")
    parser.add_argument("--checkpoint-dir", "--checkpoint_dir",
                        dest="checkpoint_dir",
                        default=CHECKPOINT_DIR,
                        help="Root directory for model checkpoints.")
    parser.add_argument("--checkpoint-chunk-size", "--checkpoint_chunk_size",
                        dest="checkpoint_chunk_size", type=int, default=20,
                        help="Panel test timestamps per checkpoint chunk.")
    parser.add_argument("--clear-checkpoints", "--clear_checkpoints",
                        dest="clear_checkpoints", action="store_true",
                        help="Delete this run's checkpoint directory before start.")
    # Manual rolling-window controls (panel-logit only); structural breaks are
    # diagnostic and never automatically set the rolling window length.
    parser.add_argument("--train-window", "--train_window",
                        dest="train_window", default="rolling_fixed",
                        choices=["expanding", "rolling_fixed"],
                        help="Panel training window (v4 default: "
                             "rolling_fixed). Use expanding to opt back into "
                             "the historical all-of-history training set.")
    parser.add_argument("--rolling-window-timestamps", "--rolling_window_timestamps",
                        dest="rolling_window_timestamps", type=int, default=None,
                        help="Manual number of pre-tau unique timestamps. "
                             "Mutually exclusive with --rolling-window-days.")
    parser.add_argument("--rolling-window-days", "--rolling_window_days",
                        dest="rolling_window_days", type=float, default=180.0,
                        help="Day-distance rolling window in CALENDAR DAYS "
                             "(v4 default: 180). 180 days is identical "
                             "wall-clock across 1d / 6h / 1h horizons — "
                             "180 timestamps would mean ~180 days at 1d "
                             "but only 45 days at 6h and 7.5 days at 1h.")
    # ── NAIVE reference auto-generation (Task 6 follow-up) ─────
    parser.add_argument(
        "--generate-naive-reference", "--generate_naive_reference",
        dest="generate_naive_reference",
        action=argparse.BooleanOptionalAction, default=True,
        help="Generate the NAIVE rolling-probability reference signal once "
             "per (horizon, model_type, panel_mode, training-window) "
             "configuration, independently of the feature-set grid. "
             "Default ON; use --no-generate-naive-reference to suppress. "
             "NAIVE is never tuned and is never written into "
             "feature_sets.xlsx.",
    )
    return parser


# ══════════════════════════════════════════════════════════════════════════════
# PER-ASSET CHECKPOINTING
# ══════════════════════════════════════════════════════════════════════════════

def _ckpt_metric_cols(ckpt_on: bool, n_loaded: int, n_written: int) -> dict:
    """Optional checkpoint-provenance columns for ``metrics_summary.csv``."""
    return {
        "checkpoint_enabled":      bool(ckpt_on),
        "resumed_from_checkpoint": bool(n_loaded > 0),
        "n_checkpoints_loaded":    int(n_loaded),
        "n_checkpoints_written":   int(n_written),
    }


def _attach_per_asset_meta(sig: pd.DataFrame, *, set_id, sentiment_model,
                           horizon, tune_on: bool) -> pd.DataFrame:
    """Attach the per-asset identity columns to a single-ticker signal frame.

    Mirrors the post-concat metadata so each ticker checkpoint is
    self-describing and a final file rebuilt purely from checkpoints is
    identical to a normal run. For tuned runs the per-row HPO columns are
    already present (from ``run_walk_forward``); for fixed-C runs we stamp the
    ``hpo_*`` sentinels here.
    """
    sig = sig.copy()
    sig["set_id"] = set_id
    sig["sentiment_model"] = sentiment_model
    sig["horizon"] = horizon
    if not tune_on:
        sig["hpo_enabled"] = False
        sig["hpo_objective"] = "-"
        sig["hpo_variant"] = "fixed"
    return sig


def _checkpointed_ticker_loop(tickers, df_all, compute_fn, *,
                              root, ckpt_on: bool, resume: bool,
                              set_id, sentiment_model, horizon, tune_on: bool,
                              manifest_base: dict | None = None):
    """Run a per-ticker compute loop with optional resume-able checkpointing.

    ``compute_fn(df_t)`` returns the raw signal frame for one ticker (e.g.
    ``run_walk_forward`` or ``run_rolling_probability``). Returns
    ``(all_signals, n_loaded, n_written)``.
    """
    from . import checkpointing as ckpt
    all_signals: list[pd.DataFrame] = []
    n_loaded = n_written = 0

    manifest = None
    if ckpt_on:
        base = dict(manifest_base or {})
        base["total_tickers"] = len(tickers)
        manifest = ckpt.init_manifest(root, base=base)

    for tk in tickers:
        cp_path = ckpt.ticker_checkpoint_path(root, tk) if ckpt_on else None
        if ckpt_on and resume and cp_path.exists():
            cached = ckpt.load_checkpoint(cp_path)
            if cached is not None and not cached.empty:
                all_signals.append(cached)
                n_loaded += 1
                print(f"     {tk} → CACHED CHECKPOINT")
                continue
            # corrupt / empty → fall through and recompute

        df_t = df_all[df_all["ticker"] == tk]
        sig = compute_fn(df_t)
        if sig is not None and not sig.empty:
            sig = _attach_per_asset_meta(
                sig, set_id=set_id, sentiment_model=sentiment_model,
                horizon=horizon, tune_on=tune_on)
            all_signals.append(sig)
            if ckpt_on:
                ckpt.save_checkpoint_atomic(sig, cp_path)
                n_written += 1
                manifest["completed_tickers"] = ckpt.list_completed_tickers(root)
                ckpt.write_manifest(root, manifest)
                print(f"     {tk} → computed + checkpointed")
        else:
            print(f"     {tk} → no signals")

    return all_signals, n_loaded, n_written


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def apply_smoke_defaults(args: argparse.Namespace) -> argparse.Namespace:
    """Resolve smoke-mode defaults in place — once, immediately after parse.

    Aufgabe 6 follow-up B: this MUST run before any downstream resolution
    (HPO config, NAIVE generation, dry-run logging, panel/per-asset
    dispatch, filename construction, checkpoint construction). A bare
    ``--smoke`` invocation must never iterate the full coin universe or
    all horizons. Explicit user values always win over the smoke
    defaults — only fields the user left empty are filled in.
    """
    if not getattr(args, "smoke", False):
        return args
    if not getattr(args, "horizon", None):
        args.horizon = "1d"
    if not getattr(args, "coins", None):
        args.coins = ["BTC", "ETH"]
    if not getattr(args, "set_id", None):
        args.set_id = "ECON"
    return args


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point used by the legacy ``Run_Models.py``, ``scripts/run_models.py``
    and the package CLI. Returns 0 on success."""
    args = build_parser().parse_args(argv)

    # ── Smoke defaults FIRST (Aufgabe 6 follow-up B) ─────────────────
    # Resolve --smoke before any HPO / NAIVE / dispatch / dry-run logic
    # so a smoke run never leaks the full coin universe into NAIVE,
    # checkpoint directories, or stage-header output.
    args = apply_smoke_defaults(args)

    # ── Legacy-ID guard (v4 17-set registry refactor) ────────────────
    # The v4 registry replaces the old B/E/S/SV/C/CV/M/SC/SF families with
    # the explicit 17-set grid ECON / SENT_{VAD,CBT}_{L,LD,DA,F} /
    # ECON_{VAD,CBT}_{L,LD,DA,F}. Production code MUST reject any removed
    # ID up front, regardless of what the user's feature_sets.xlsx
    # contains — otherwise a stale legacy fixture can silently reactivate
    # a deprecated information set under the v4 registry. The error
    # message carries the migration explanation from
    # :data:`thesis_pipeline.features.feature_registry.REMOVED_SET_IDS`.
    if args.set_id:
        from ..features.feature_registry import REMOVED_SET_IDS as _REMOVED
        if args.set_id in _REMOVED:
            raise SystemExit(
                f"[run-models] set_id {args.set_id!r} was removed in the v4 "
                f"17-set registry refactor. {_REMOVED[args.set_id]} "
                f"See docs/refactor_log.md for the full migration table."
            )

    # ── Resolve hyperparameter-tuning config (shared by both families) ──
    from .hyperparameter_tuning import hpo_variant_label, load_hpo_config
    hpo_cfg = load_hpo_config(
        # ``--tune-hyperparams`` is now BooleanOptionalAction with default
        # True (v4). Forward the explicit bool so ``--no-tune-hyperparams``
        # really disables HPO regardless of the YAML default.
        enabled_override=bool(getattr(args, "tune_hyperparams", True)),
        objective_override=getattr(args, "hpo_objective", None),
        c_grid=getattr(args, "hpo_grid_C", None),
        class_weight_grid=getattr(args, "hpo_class_weight", None),
        config_path=getattr(args, "hpo_config", None),
    )
    tune_on = bool(hpo_cfg["enabled"])
    hpo_variant = hpo_variant_label(tune_on, hpo_cfg["objective"])

    # ── Alternative model family: delegate to the panel-logit module ──
    # The per-asset logic below is unchanged; panel_logit is purely additive.
    # ── NAIVE auto-generation (Aufgabe 6 follow-up A) ─────────────
    # NAIVE is never tuned and never lives in feature_sets.xlsx. Generate
    # it once per (horizon × family × window) before the per-set loop so
    # the evaluation layer can compare each ECON_* model to it as a
    # separate absolute reference.
    if (getattr(args, "generate_naive_reference", True)
            and not getattr(args, "dry_run", False)):
        from .naive_reference import generate_naive_reference
        from ..logging_utils import get_logger
        log = get_logger()
        _horizons_to_naive = ([args.horizon] if args.horizon else list(HORIZONS))
        for _hz in _horizons_to_naive:
            try:
                _written = generate_naive_reference(
                    horizon=_hz,
                    model_type=getattr(args, "model_type", "panel_logit") or "panel_logit",
                    panel_mode=getattr(args, "panel_mode", "ticker_fixed_effects") or "ticker_fixed_effects",
                    train_window_mode=getattr(args, "train_window", "rolling_fixed") or "rolling_fixed",
                    rolling_window_timestamps=getattr(args, "rolling_window_timestamps", None),
                    rolling_window_days=getattr(args, "rolling_window_days", 180.0),
                    coins=getattr(args, "coins", None),
                    output_dir=SIGNAL_DIR,
                    resume=bool(getattr(args, "resume", True)),
                    restart=bool(getattr(args, "restart", False)),
                )
                if _written is None:
                    log.info("NAIVE reference cached or skipped for horizon=%s", _hz)
                else:
                    log.info("NAIVE reference written: %s", _written)
            except Exception as exc:  # noqa: BLE001 — never break the main run
                log.warning("NAIVE generation failed for horizon=%s: %s", _hz, exc)

    if getattr(args, "model_type", "per_asset") == "panel_logit":
        from .panel_logit import _run_panel
        return _run_panel(args, hpo_cfg=hpo_cfg)

    # Smoke defaults are already resolved by apply_smoke_defaults at the
    # top of main(); no further per-asset-only smoke handling needed.

    # ── Checkpointing config ────────────────────────────────────
    ckpt_on   = bool(getattr(args, "checkpoint", True))
    resume    = bool(getattr(args, "resume", True))
    ckpt_dir  = getattr(args, "checkpoint_dir", CHECKPOINT_DIR) or CHECKPOINT_DIR
    clear_ckpt = bool(getattr(args, "clear_checkpoints", False))

    # ── Stage header (best-effort; never fails the run) ─────────
    try:
        from ..logging_utils import log_stage_header
        from ..config import resolve_path
        from . import checkpointing as _ckpt
        inputs = []
        if args.horizon:
            inputs.append(resolve_path("final_features_pattern", horizon=args.horizon))
        inputs.append(resolve_path("feature_sets_xlsx"))
        outputs: list = []
        if args.horizon and args.set_id:
            outputs.append(resolve_path("signals_pattern",
                                        horizon=args.horizon, set_id=args.set_id))
        outputs.append(resolve_path("signals_metrics"))
        # Example signal filename so the dry-run shows the (HPO-suffixed) name.
        out_name_example = "(per set_id default)"
        if args.set_id:
            _sm = args.sentiment_model or "-"
            _base = (f"{args.set_id}_{_sm}"
                     if _sm and str(_sm) != "-" else str(args.set_id))
            if tune_on:
                _base = f"{_base}_{hpo_variant}"
            out_name_example = f"Outputs/Signals/{args.horizon or '<horizon>'}/{_base}.parquet"
        # Universe identity preview (commit 3 Section H). Resolve the
        # requested universe up-front so the dry-run reports the same
        # hash + expected NAIVE filename the real run would produce.
        from .naive_reference import (
            normalize_coin_universe as _nu,
            coin_universe_hash as _nu_hash,
            naive_output_name as _nu_name,
            CACHE_SCHEMA_VERSION as _nu_cache_v,
        )
        _req_universe = _nu(args.coins) if args.coins else tuple()
        _req_hash = _nu_hash(_req_universe) if _req_universe else "(resolved at run-time)"
        _naive_name = _nu_name(
            model_type=getattr(args, "model_type", "panel_logit") or "panel_logit",
            panel_mode=getattr(args, "panel_mode", "ticker_fixed_effects") or "ticker_fixed_effects",
            train_window_mode=getattr(args, "train_window", "rolling_fixed") or "rolling_fixed",
            rolling_window_timestamps=getattr(args, "rolling_window_timestamps", None),
            rolling_window_days=getattr(args, "rolling_window_days", 180.0),
            coin_universe=_req_universe or None,
        )

        log_stage_header(
            "run_models",
            mode="dry-run" if args.dry_run else ("smoke" if args.smoke else "full"),
            inputs=inputs,
            outputs=outputs,
            extra={
                "horizon":         args.horizon or "(all)",
                "set_id":          args.set_id or "(all)",
                "coins":           list(args.coins) if args.coins else "(all)",
                "requested_tickers":            list(_req_universe) or "(all from features)",
                "requested_coin_universe_hash": _req_hash,
                "expected_naive_filename":      f"{_naive_name}.parquet",
                "naive_cache_schema_version":   _nu_cache_v,
                "sentiment_model": args.sentiment_model or "(per set_id default)",
                "C":               args.C,
                "restart":         args.restart,
                "force":           args.force,
                "tune_hyperparams": tune_on,
                "hpo_variant":     hpo_variant,
                "output_name":     out_name_example,
                "hpo_objective":   hpo_cfg["objective"] if tune_on else "(off)",
                "hpo_C_grid":      hpo_cfg["search_space"].get("C") if tune_on else "(off)",
                "hpo_class_weight_grid": (
                    [("none" if c is None else c)
                     for c in hpo_cfg["search_space"].get("class_weight", [])]
                    if tune_on else "(off)"),
                "hpo_validation_fraction": (
                    hpo_cfg["validation_fraction"] if tune_on else "(off)"),
                "checkpoint_enabled":     ckpt_on,
                "resume":                 resume,
                "checkpoint_dir":         ckpt_dir,
                "checkpoint_chunk_size":  getattr(args, "checkpoint_chunk_size", 20),
                "clear_checkpoints":      clear_ckpt,
                "run_checkpoint_path": (
                    str(_ckpt.checkpoint_root(
                        ckpt_dir, args.horizon or "<horizon>",
                        (out_name_example.split("/")[-1].replace(".parquet", "")
                         if args.set_id else "<out_name>")))
                    if ckpt_on else "(off)"),
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
        if tune_on:
            print(f"  Model: LogisticRegression(L2) + grid-search HPO "
                  f"(objective={hpo_cfg['objective']}, "
                  f"C grid={hpo_cfg['search_space'].get('C')})")
        else:
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

            # The rolling-probability benchmark has no hyperparameters, so it is
            # never tuned (and never suffixed/overwritten under a tuning run).
            is_benchmark = feat_str.strip() in ("__rolling_probability__",
                                                "__majority_class__")

            # File naming. Tuned (non-benchmark) runs get a variant suffix so
            # they never overwrite the fixed-C parquet (and caching/restart
            # keys on the variant-specific path).
            if sent_model and str(sent_model) != "-" and str(sent_model) != "nan":
                out_name = f"{set_id}_{sent_model}"
            else:
                out_name = str(set_id)
                sent_model = "-"
            if tune_on and not is_benchmark:
                out_name = f"{out_name}_{hpo_variant}"

            out_path = os.path.join(hz_dir, f"{out_name}.parquet")

            # ── Checkpoint: skip if already computed ──────────
            if os.path.isfile(out_path) and not args.restart:
                try:
                    from .hyperparameter_tuning import summarize_hpo_columns
                    cached = pd.read_parquet(out_path)
                    m = compute_metrics(cached, "pooled")
                    m.update({"horizon": hz, "set_id": set_id,
                              "sentiment_model": sent_model, "label": label,
                              "category": category,
                              "n_tickers": cached["ticker"].nunique()})
                    m.update(summarize_hpo_columns(cached))
                    all_metrics.append(m)
                    for tk, grp in cached.groupby("ticker"):
                        mt = compute_metrics(grp, tk)
                        mt.update({"horizon": hz, "set_id": set_id,
                                   "sentiment_model": sent_model, "label": label,
                                   "category": category, "n_tickers": 1})
                        mt.update(summarize_hpo_columns(grp))
                        all_metrics.append(mt)
                    print(f"\n  ── {out_name} ({label}) "
                          f"→ CACHED acc={m['accuracy']:.4f}, n={m['n_obs']}")
                    continue
                except Exception:
                    pass  # corrupted — will re-run

            print(f"\n  ── {out_name} ({label}) ", end="", flush=True)

            # ── Benchmark: rolling probability ────────────────
            if is_benchmark:
                # Rolling probability is not a tuned model; under --tune-hyperparams
                # skip it so the fixed-C benchmark parquet (the NAIVE evaluation
                # reference produced by run_rolling_probability) is never
                # overwritten or mislabelled. The tuned families are compared
                # against the panel-logit ECON baseline in the v4 evaluation
                # pipeline; the old B1/B2 sentinel comparisons no longer exist.
                if tune_on:
                    print("→ SKIP (rolling-probability benchmark is not tuned)")
                    continue
                print()
                from . import checkpointing as ckpt
                root = ckpt.checkpoint_root(ckpt_dir, hz, out_name)
                if ckpt_on and clear_ckpt:
                    ckpt.clear_run_checkpoints(root)
                manifest_base = {
                    "horizon": hz, "set_id": set_id, "sentiment_model": sent_model,
                    "model_type": "per_asset", "panel_mode": "-",
                    "hpo_variant": "fixed", "hpo_objective": "-",
                    "feature_cols": ["__rolling_probability__"],
                }
                all_signals, n_loaded, n_written = _checkpointed_ticker_loop(
                    tickers, df_all, run_rolling_probability,
                    root=root, ckpt_on=ckpt_on, resume=resume,
                    set_id=set_id, sentiment_model=sent_model, horizon=hz,
                    tune_on=False, manifest_base=manifest_base,
                )

                if all_signals:
                    signals = pd.concat(all_signals, ignore_index=True)
                    # Universe identity stamp (commit 3 Section B).
                    from .naive_reference import stamp_universe_metadata
                    signals = stamp_universe_metadata(
                        signals, requested_universe=tickers,
                    )
                    signals.to_parquet(out_path, index=False, engine="pyarrow")
                    if ckpt_on:
                        mf = ckpt.load_manifest(root)
                        mf["status"] = "complete"
                        ckpt.write_manifest(root, mf)

                    m = compute_metrics(signals, "pooled")
                    m.update({"horizon": hz, "set_id": set_id,
                              "sentiment_model": sent_model, "label": label,
                              "category": category,
                              "n_tickers": signals["ticker"].nunique()})
                    m.update(_ckpt_metric_cols(ckpt_on, n_loaded, n_written))
                    all_metrics.append(m)
                    for tk, grp in signals.groupby("ticker"):
                        mt = compute_metrics(grp, tk)
                        mt.update({"horizon": hz, "set_id": set_id,
                                   "sentiment_model": sent_model, "label": label,
                                   "category": category, "n_tickers": 1})
                        all_metrics.append(mt)
                    print(f"  → acc={m['accuracy']:.4f}, brier={m.get('brier_score', np.nan):.4f}, "
                          f"n={m['n_obs']}")
                else:
                    print("  → no signals")
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

            # ── Run per ticker (with optional checkpointing) ──
            print()
            t0 = time.time()
            from . import checkpointing as ckpt
            root = ckpt.checkpoint_root(ckpt_dir, hz, out_name)
            if ckpt_on and clear_ckpt:
                ckpt.clear_run_checkpoints(root)
            manifest_base = {
                "horizon": hz, "set_id": set_id, "sentiment_model": sent_model,
                "model_type": "per_asset", "panel_mode": "-",
                "hpo_variant": hpo_variant if tune_on else "fixed",
                "hpo_objective": hpo_cfg["objective"] if tune_on else "-",
                "feature_cols": feature_cols,
            }
            all_signals, n_loaded, n_written = _checkpointed_ticker_loop(
                tickers, df_all,
                lambda df_t: run_walk_forward(df_t, feature_cols, C=args.C,
                                              tune_hyperparams=tune_on,
                                              hpo_config=hpo_cfg),
                root=root, ckpt_on=ckpt_on, resume=resume,
                set_id=set_id, sentiment_model=sent_model, horizon=hz,
                tune_on=tune_on, manifest_base=manifest_base,
            )
            elapsed = time.time() - t0

            if all_signals:
                from .hyperparameter_tuning import summarize_hpo_columns
                # Per-ticker metadata is already attached by the checkpoint loop,
                # so the concat (whether freshly computed or rebuilt purely from
                # checkpoints) is the complete final signal frame.
                signals = pd.concat(all_signals, ignore_index=True)
                # Universe identity stamp (commit 3 Section B).
                from .naive_reference import stamp_universe_metadata
                signals = stamp_universe_metadata(
                    signals, requested_universe=tickers,
                )
                signals.to_parquet(out_path, index=False, engine="pyarrow")
                if ckpt_on:
                    mf = ckpt.load_manifest(root)
                    mf["status"] = "complete"
                    ckpt.write_manifest(root, mf)

                hpo_summary = summarize_hpo_columns(signals)
                m = compute_metrics(signals, "pooled")
                m.update({"horizon": hz, "set_id": set_id,
                          "sentiment_model": sent_model, "label": label,
                          "category": category,
                          "n_tickers": signals["ticker"].nunique()})
                m.update(hpo_summary)
                m.update(_ckpt_metric_cols(ckpt_on, n_loaded, n_written))
                all_metrics.append(m)

                for tk, grp in signals.groupby("ticker"):
                    mt = compute_metrics(grp, tk)
                    mt.update({"horizon": hz, "set_id": set_id,
                               "sentiment_model": sent_model, "label": label,
                               "category": category, "n_tickers": 1})
                    mt.update(summarize_hpo_columns(grp))
                    all_metrics.append(mt)

                print(f"  → acc={m['accuracy']:.4f}, "
                      f"f1={m['f1']:.4f}, "
                      f"brier={m.get('brier_score', np.nan):.4f}, "
                      f"n={m['n_obs']}, {elapsed:.1f}s "
                      f"(ckpt loaded={n_loaded}, written={n_written})")
            else:
                print(f"  → no signals ({elapsed:.1f}s)")
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
        feature_config: str | None = None,
        # v4 canonical defaults — match build_parser():
        model_type: str = "panel_logit",
        panel_mode: str = "ticker_fixed_effects",
        tune_hyperparams: bool = True,
        hpo_objective: str = "log_loss",
        hpo_config: str | None = None,
        hpo_grid_C: Sequence[float] | None = None,
        hpo_class_weight: Sequence[str] | None = None,
        train_window: str = "rolling_fixed",
        rolling_window_days: float | None = 180.0,
        rolling_window_timestamps: int | None = None,
        checkpoint: bool = True,
        resume: bool = True,
        checkpoint_dir: str | None = None,
        checkpoint_chunk_size: int | None = None,
        clear_checkpoints: bool = False,
        generate_naive_reference: bool = True) -> int:
    """Programmatic entry point. Translates keyword arguments to argv for
    :func:`main`. Prefer calling :func:`main` directly with an argv list.

    The defaults match the v4 CLI: panel-logit / ticker fixed effects /
    rolling 180-calendar-day window / HPO on / log-loss objective.
    Pass ``model_type="per_asset"``, ``tune_hyperparams=False``, or
    ``train_window="expanding"`` to opt back into the historical
    behaviour.
    """
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
    if model_type:
        argv += ["--model-type", model_type]
    if panel_mode:
        argv += ["--panel-mode", panel_mode]
    # ``--tune-hyperparams`` is BooleanOptionalAction in v4: emit the
    # explicit negative flag when the caller has turned it off.
    argv += ["--tune-hyperparams"] if tune_hyperparams else ["--no-tune-hyperparams"]
    # ``--generate-naive-reference`` is also BooleanOptionalAction.
    argv += (["--generate-naive-reference"] if generate_naive_reference
             else ["--no-generate-naive-reference"])
    if hpo_objective:
        argv += ["--hpo-objective", hpo_objective]
    if train_window:
        argv += ["--train-window", train_window]
    if rolling_window_timestamps is not None:
        argv += ["--rolling-window-timestamps", str(rolling_window_timestamps)]
    if rolling_window_days is not None and rolling_window_timestamps is None:
        argv += ["--rolling-window-days", str(rolling_window_days)]
    if hpo_config:
        argv += ["--hpo-config", hpo_config]
    if hpo_grid_C:
        argv += ["--hpo-grid-C", *(str(c) for c in hpo_grid_C)]
    if hpo_class_weight:
        argv += ["--hpo-class-weight", *(str(c) for c in hpo_class_weight)]
    if not checkpoint:
        argv.append("--no-checkpoint")
    if not resume:
        argv.append("--no-resume")
    if checkpoint_dir:
        argv += ["--checkpoint-dir", checkpoint_dir]
    if checkpoint_chunk_size is not None:
        argv += ["--checkpoint-chunk-size", str(checkpoint_chunk_size)]
    if clear_checkpoints:
        argv.append("--clear-checkpoints")
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
