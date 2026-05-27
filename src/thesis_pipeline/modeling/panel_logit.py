"""Panel-logit alternative model family.

This is an *additive* alternative to the canonical per-asset walk-forward in
:mod:`thesis_pipeline.modeling.run_models`. The per-asset logic there is not
modified — this module provides a second model family for comparison.

Methodology — pooled panel logistic regression in a *forecasting* setting
(not a classical inference panel model):

* For each unique test timestamp ``τ`` (taken in chronological order):
    - **train** = every coin observation with ``timestamp < τ``
    - **test**  = every coin observation with ``timestamp == τ``
  so a single logistic regression is estimated over the whole panel and used
  to predict all coins at ``τ``. Training never sees ``τ`` or later — no
  lookahead leakage.
* Initial training window: the first 50 % of the *unique timestamps* form the
  initial train set (not 50 % of observations), mirroring the per-asset
  ``INIT_TRAIN_FRAC`` but applied on the time axis of the panel.
* ``StandardScaler`` is fit on the training rows only and applied to test.
* ``LogisticRegression(penalty="l2", C, solver="lbfgs", random_state=42)`` —
  identical estimator settings to the per-asset model.

Two panel modes:

* ``pooled``                → ``y_it ~ X_it`` (shared coefficients, no FE).
* ``ticker_fixed_effects``  → ``y_it ~ X_it + ticker dummies`` — approximates
  coin-specific intercepts. Dummies are fit on the *training* tickers only
  (one reference dropped to avoid the dummy trap given the model intercept);
  test dummies are aligned to the training columns, and tickers unseen in
  training collapse to the reference category (all-zero dummy row).

There are **no time fixed effects**: a dummy for the test timestamp would be
unidentified out-of-sample (it never appears in training), so time FE are
intentionally excluded.

B1 (``__rolling_probability__`` / ``__majority_class__``) is not a panel-logit
model — for ``model_type=panel_logit`` such sets are skipped with a warning
rather than crashing.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from .run_models import (
    DEFAULT_C, FEATURE_CONFIG, HORIZONS, INIT_TRAIN_FRAC, SIGNAL_DIR,
    compute_metrics, load_features, load_feature_sets,
)

MODEL_TYPE = "panel_logit"
PANEL_MODES = ("pooled", "ticker_fixed_effects")
_BENCHMARK_SENTINELS = ("__rolling_probability__", "__majority_class__")

# Minimum number of unique timestamps that must precede the first test point.
MIN_INIT_TIMESTAMPS = 30
# Minimum training rows / classes per step (mirrors the per-asset guards).
MIN_TRAIN_OBS = 20


# ══════════════════════════════════════════════════════════════════════════════
# DESIGN MATRIX
# ══════════════════════════════════════════════════════════════════════════════

def add_ticker_fixed_effects(train_tickers: pd.Series,
                             test_tickers: pd.Series) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return aligned (train_dummies, test_dummies) for ticker fixed effects.

    One training ticker is dropped as the reference category (avoids perfect
    collinearity with the logistic-regression intercept). Test tickers are
    reindexed to the training dummy columns; any ticker unseen in training
    becomes an all-zero row (i.e. the reference category).
    """
    tr = pd.get_dummies(pd.Series(train_tickers).astype(str), prefix="tkr")
    if tr.shape[1] == 0:
        return (pd.DataFrame(index=range(len(train_tickers))),
                pd.DataFrame(index=range(len(test_tickers))))
    # Drop the first column as the reference category.
    ref = tr.columns[0]
    tr = tr.drop(columns=[ref])
    te = pd.get_dummies(pd.Series(test_tickers).astype(str), prefix="tkr")
    te = te.reindex(columns=tr.columns, fill_value=0)
    return tr.reset_index(drop=True), te.reset_index(drop=True)


def build_panel_design_matrix(train_df: pd.DataFrame,
                              test_df: pd.DataFrame,
                              feature_cols: list[str],
                              panel_mode: str = "pooled"
                              ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build (X_train, y_train, X_test).

    Continuous features are standardised with a scaler fit on the training
    rows only. For ``ticker_fixed_effects`` the (unscaled 0/1) ticker dummies
    are concatenated after the scaled continuous block.
    """
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(train_df[feature_cols].values.astype(float))
    X_te = scaler.transform(test_df[feature_cols].values.astype(float))

    if panel_mode == "ticker_fixed_effects":
        tr_d, te_d = add_ticker_fixed_effects(train_df["ticker"], test_df["ticker"])
        if tr_d.shape[1] > 0:
            X_tr = np.hstack([X_tr, tr_d.values.astype(float)])
            X_te = np.hstack([X_te, te_d.values.astype(float)])

    y_tr = train_df["target"].values.astype(int)
    return X_tr, y_tr, X_te


# ══════════════════════════════════════════════════════════════════════════════
# PANEL WALK-FORWARD
# ══════════════════════════════════════════════════════════════════════════════

def run_panel_walk_forward(df: pd.DataFrame,
                           feature_cols: list[str],
                           C: float = DEFAULT_C,
                           panel_mode: str = "pooled",
                           init_train_frac: float = INIT_TRAIN_FRAC,
                           min_init_timestamps: int = MIN_INIT_TIMESTAMPS,
                           min_train_obs: int = MIN_TRAIN_OBS) -> pd.DataFrame:
    """Expanding-window pooled panel logit over all coins.

    Returns a signal frame with columns:
    ``timestamp, ticker, target, prediction, probability``.
    """
    if panel_mode not in PANEL_MODES:
        raise ValueError(f"Unknown panel_mode {panel_mode!r}; expected one of {PANEL_MODES}")

    df = df.sort_values(["timestamp", "ticker"]).reset_index(drop=True)
    unique_ts = np.sort(df["timestamp"].unique())
    n_ts = len(unique_ts)
    if n_ts == 0:
        return pd.DataFrame()

    init_idx = max(int(n_ts * init_train_frac), min_init_timestamps)
    results: list[dict] = []

    for i in range(init_idx, n_ts):
        tau = unique_ts[i]
        train_df = df[df["timestamp"] < tau]
        test_df  = df[df["timestamp"] == tau]

        # Drop rows with NaN features / target.
        train_df = train_df.dropna(subset=feature_cols + ["target"])
        test_df  = test_df.dropna(subset=feature_cols + ["target"])
        if (len(train_df) < min_train_obs
                or train_df["target"].nunique() < 2
                or test_df.empty):
            continue

        X_tr, y_tr, X_te = build_panel_design_matrix(
            train_df, test_df, feature_cols, panel_mode,
        )
        model = LogisticRegression(
            penalty="l2", C=C, solver="lbfgs", max_iter=1000, random_state=42,
        )
        model.fit(X_tr, y_tr)
        preds = model.predict(X_te).astype(int)
        proba = model.predict_proba(X_te)[:, 1]

        test_df = test_df.reset_index(drop=True)
        for j in range(len(test_df)):
            results.append({
                "timestamp":   test_df.loc[j, "timestamp"],
                "ticker":      test_df.loc[j, "ticker"],
                "target":      int(test_df.loc[j, "target"]),
                "prediction":  int(preds[j]),
                "probability": float(proba[j]),
            })

    if not results:
        return pd.DataFrame()
    return pd.DataFrame(results)


def run_panel_model_for_feature_set(df_all: pd.DataFrame,
                                    feature_cols: list[str],
                                    tickers: Sequence[str] | None,
                                    C: float = DEFAULT_C,
                                    panel_mode: str = "pooled") -> pd.DataFrame:
    """Filter to ``tickers`` (if given) and run the panel walk-forward."""
    df = df_all
    if tickers:
        df = df[df["ticker"].isin(set(tickers))]
    if df.empty:
        return pd.DataFrame()
    return run_panel_walk_forward(df, feature_cols, C=C, panel_mode=panel_mode)


# ══════════════════════════════════════════════════════════════════════════════
# OUTPUT NAMING
# ══════════════════════════════════════════════════════════════════════════════

def _mode_suffix(panel_mode: str) -> str:
    return "panel_pooled" if panel_mode == "pooled" else "panel_ticker_fe"


def panel_output_name(set_id: str, sentiment_model: str, panel_mode: str) -> str:
    """``{set_id}[_{sentiment_model}]_{panel_pooled|panel_ticker_fe}``."""
    suffix = _mode_suffix(panel_mode)
    if sentiment_model and str(sentiment_model) not in ("-", "nan"):
        return f"{set_id}_{sentiment_model}_{suffix}"
    return f"{set_id}_{suffix}"


# ══════════════════════════════════════════════════════════════════════════════
# CLI / MAIN
# ══════════════════════════════════════════════════════════════════════════════

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Panel-logit alternative — pooled / ticker-fixed-effects "
                    "logistic regression over the coin panel."
    )
    parser.add_argument("--feature_config", "--feature-config",
                        dest="feature_config", default=FEATURE_CONFIG)
    parser.add_argument("--horizon", default=None, choices=HORIZONS)
    parser.add_argument("--set_id", "--set-id", dest="set_id", default=None)
    parser.add_argument("--coins", "--coin", "--ticker",
                        dest="coins", nargs="+", default=None)
    parser.add_argument("--sentiment_model", "--sentiment-model",
                        dest="sentiment_model", default=None)
    parser.add_argument("--panel-mode", "--panel_mode", dest="panel_mode",
                        default="pooled", choices=list(PANEL_MODES))
    parser.add_argument("--C", type=float, default=DEFAULT_C)
    parser.add_argument("--dry-run", "--dry_run", dest="dry_run", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--restart", action="store_true")
    return parser


def _run_panel(args: argparse.Namespace) -> int:
    """Shared body used by both :func:`main` and the run-models delegation."""
    panel_mode = getattr(args, "panel_mode", "pooled") or "pooled"

    if getattr(args, "dry_run", False):
        try:
            from ..logging_utils import log_stage_header
            from ..config import resolve_path
            inputs = []
            if args.horizon:
                inputs.append(resolve_path("final_features_pattern", horizon=args.horizon))
            inputs.append(resolve_path("feature_sets_xlsx"))
            log_stage_header(
                "run_models", mode="dry-run", inputs=inputs, outputs=[],
                extra={
                    "model_type": MODEL_TYPE,
                    "panel_mode": panel_mode,
                    "horizon":    args.horizon or "(all)",
                    "set_id":     args.set_id or "(all)",
                    "coins":      list(args.coins) if args.coins else "(all)",
                    "C":          args.C,
                },
            )
        except Exception:  # noqa: BLE001
            pass
        return 0

    os.makedirs(SIGNAL_DIR, exist_ok=True)
    config = load_feature_sets(args.feature_config)
    if args.set_id:
        config = config[config["set_id"] == args.set_id]
        print(f"  Filtered to set_id={args.set_id}: {len(config)} rows")
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
        print(f"  HORIZON: {hz}   model=panel_logit/{panel_mode}")
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

        hz_dir = os.path.join(SIGNAL_DIR, hz)
        os.makedirs(hz_dir, exist_ok=True)

        for _, cfg_row in config.iterrows():
            set_id     = cfg_row["set_id"]
            category   = cfg_row.get("category", "")
            sent_model = cfg_row.get("sentiment_model", "-")
            label      = cfg_row.get("label", "")
            feat_str   = cfg_row["features"]
            if not (sent_model and str(sent_model) not in ("-", "nan")):
                sent_model = "-"

            out_name = panel_output_name(set_id, sent_model, panel_mode)
            out_path = os.path.join(hz_dir, f"{out_name}.parquet")

            # Benchmark sentinel is not a panel-logit model.
            if feat_str.strip() in _BENCHMARK_SENTINELS:
                print(f"\n  ── {out_name} ({label}) → SKIP "
                      f"(benchmark sentinel not supported for panel_logit)")
                continue

            if os.path.isfile(out_path) and not args.restart:
                try:
                    cached = pd.read_parquet(out_path)
                    m = compute_metrics(cached, "pooled")
                    m.update({"horizon": hz, "set_id": set_id,
                              "sentiment_model": sent_model, "label": label,
                              "category": category,
                              "n_tickers": cached["ticker"].nunique(),
                              "model_type": MODEL_TYPE, "panel_mode": panel_mode})
                    all_metrics.append(m)
                    print(f"\n  ── {out_name} ({label}) → CACHED "
                          f"acc={m['accuracy']:.4f}, n={m['n_obs']}")
                    continue
                except Exception:
                    pass

            print(f"\n  ── {out_name} ({label}) ", end="", flush=True)

            feature_cols = [f.strip() for f in feat_str.split(",")]
            missing = [f for f in feature_cols if f not in df_all.columns]
            if missing:
                print(f"→ SKIP (missing: {missing[:3]})")
                all_metrics.append({
                    "horizon": hz, "set_id": set_id, "sentiment_model": sent_model,
                    "label": label, "category": category, "ticker": "pooled",
                    "accuracy": np.nan, "n_obs": 0, "n_tickers": 0,
                    "model_type": MODEL_TYPE, "panel_mode": panel_mode,
                    "status": f"missing: {missing[:3]}",
                })
                continue

            t0 = time.time()
            signals = run_panel_model_for_feature_set(
                df_all, feature_cols, tickers, C=args.C, panel_mode=panel_mode,
            )
            elapsed = time.time() - t0

            if signals.empty:
                print(f"→ no signals ({elapsed:.1f}s)")
                all_metrics.append({
                    "horizon": hz, "set_id": set_id, "sentiment_model": sent_model,
                    "label": label, "category": category, "ticker": "pooled",
                    "accuracy": np.nan, "n_obs": 0, "n_tickers": 0,
                    "model_type": MODEL_TYPE, "panel_mode": panel_mode,
                })
                continue

            signals["set_id"]          = set_id
            signals["sentiment_model"] = sent_model
            signals["horizon"]         = hz
            signals["model_type"]      = MODEL_TYPE
            signals["panel_mode"]      = panel_mode
            signals.to_parquet(out_path, index=False, engine="pyarrow")

            m = compute_metrics(signals, "pooled")
            m.update({"horizon": hz, "set_id": set_id,
                      "sentiment_model": sent_model, "label": label,
                      "category": category,
                      "n_tickers": signals["ticker"].nunique(),
                      "model_type": MODEL_TYPE, "panel_mode": panel_mode})
            all_metrics.append(m)
            for tk, grp in signals.groupby("ticker"):
                mt = compute_metrics(grp, tk)
                mt.update({"horizon": hz, "set_id": set_id,
                           "sentiment_model": sent_model, "label": label,
                           "category": category, "n_tickers": 1,
                           "model_type": MODEL_TYPE, "panel_mode": panel_mode})
                all_metrics.append(mt)
            print(f"→ acc={m['accuracy']:.4f}, f1={m['f1']:.4f}, "
                  f"brier={m.get('brier_score', np.nan):.4f}, "
                  f"n={m['n_obs']}, {elapsed:.1f}s")

    if all_metrics:
        metrics_path = os.path.join(SIGNAL_DIR, "metrics_summary.csv")
        new = pd.DataFrame(all_metrics)
        # Append to the existing summary without clobbering per-asset rows.
        if os.path.isfile(metrics_path):
            try:
                existing = pd.read_csv(metrics_path)
                combined = pd.concat([existing, new], ignore_index=True)
            except Exception:  # noqa: BLE001
                combined = new
        else:
            combined = new
        combined.to_csv(metrics_path, index=False)
        print(f"\n  Metrics appended: {metrics_path}  (+{len(new)} panel rows)")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return _run_panel(args)


def run(*, horizon: str | None = None, set_id: str | None = None,
        coins: Sequence[str] | None = None,
        sentiment_model: str | None = None,
        panel_mode: str = "pooled",
        C: float | None = None,
        dry_run: bool = False, force: bool = False,
        restart: bool = False,
        feature_config: str | None = None) -> int:
    """Programmatic entry point mirroring :func:`main`."""
    argv: list[str] = []
    if horizon:
        argv += ["--horizon", horizon]
    if set_id:
        argv += ["--set-id", set_id]
    if coins:
        argv += ["--coins", *list(coins)]
    if sentiment_model:
        argv += ["--sentiment-model", sentiment_model]
    if panel_mode:
        argv += ["--panel-mode", panel_mode]
    if C is not None:
        argv += ["--C", str(C)]
    if feature_config:
        argv += ["--feature-config", feature_config]
    if dry_run:
        argv.append("--dry-run")
    if force:
        argv.append("--force")
    if restart:
        argv.append("--restart")
    return main(argv)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
