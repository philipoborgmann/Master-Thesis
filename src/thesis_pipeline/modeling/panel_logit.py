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

The NAIVE rolling-probability path
(``__rolling_probability__`` / ``__majority_class__``) is not a panel-logit
model — under ``model_type=panel_logit`` a feature-set entry that resolves to
the sentinel is skipped with a warning rather than crashing. Production v4
runs generate NAIVE separately via
:mod:`thesis_pipeline.modeling.naive_reference`, independent of the
feature-set grid.
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
    CHECKPOINT_DIR, DEFAULT_C, FEATURE_CONFIG, HORIZONS, INIT_TRAIN_FRAC,
    SIGNAL_DIR, compute_metrics, load_features, load_feature_sets,
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

    Preprocessing is leakage-safe: continuous features are first winsorised
    with training-window thresholds fit on ``train_df`` ONLY (ticker-specific
    inside the window, pooled fallback), then standardised with a scaler also
    fit on the (winsorised) training rows only. The same frozen thresholds and
    scaler are applied to ``test_df``. For ``ticker_fixed_effects`` the
    (unscaled 0/1) ticker dummies are concatenated after the scaled continuous
    block.
    """
    from .preprocessing import TrainingWindowWinsorizer
    winsorizer = TrainingWindowWinsorizer(feature_cols).fit(train_df)
    train_w = winsorizer.transform(train_df)
    test_w = winsorizer.transform(test_df)
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(train_w[feature_cols].values.astype(float))
    X_te = scaler.transform(test_w[feature_cols].values.astype(float))

    if panel_mode == "ticker_fixed_effects":
        tr_d, te_d = add_ticker_fixed_effects(train_df["ticker"], test_df["ticker"])
        if tr_d.shape[1] > 0:
            X_tr = np.hstack([X_tr, tr_d.values.astype(float)])
            X_te = np.hstack([X_te, te_d.values.astype(float)])

    y_tr = train_df["target"].values.astype(int)
    return X_tr, y_tr, X_te


# ══════════════════════════════════════════════════════════════════════════════
# PARALLELISATION — top-level, picklable worker helpers
# ══════════════════════════════════════════════════════════════════════════════
#
# Chunk-level process parallelism for the panel walk-forward. A "chunk" is a
# STORAGE/COMPUTE partition of consecutive test timestamps — it does NOT change
# the training window or the forecasting methodology: every test timestamp τ
# still selects its own training window (timestamp < τ), fits its own
# preprocessing on that window only, runs its own HPO when enabled, and predicts
# only τ. Parallelising chunks is therefore methodologically identical to the
# sequential path; the only observable difference is wall-clock time and the
# order in which chunk checkpoints are produced (the final frame is always
# re-sorted by (timestamp, ticker), so output is deterministic).
#
# The worker functions are module-level and take only picklable arguments so the
# implementation works under both fork (Linux) and spawn (Windows / some
# Codespaces) multiprocessing. The large feature frame is written ONCE to a
# temporary parquet by the main process and loaded ONCE per worker process
# (cached in ``_WORKER_DF_CACHE``), avoiding a per-task DataFrame pickle/copy.

from dataclasses import dataclass


@dataclass
class _PanelParams:
    """Picklable bundle of the per-run panel parameters shared by every
    timestamp/chunk (never the DataFrame itself)."""
    feature_cols: list
    C: float
    panel_mode: str
    min_train_obs: int
    tune_hyperparams: bool
    hpo_config: dict
    train_window_mode: str
    rolling_window_timestamps: int | None
    rolling_window_days: float | None


def _resolve_n_jobs(value: int) -> int:
    """Resolve the requested worker count.

    ``1`` (default) → sequential; a positive integer → that many workers;
    ``-1`` → all CPUs available to the process/container (via joblib's
    quota-aware :func:`joblib.cpu_count`, not the raw host core count).
    ``0`` and values below ``-1`` raise ``ValueError``.
    """
    v = int(value)
    if v == -1:
        from joblib import cpu_count
        return max(int(cpu_count()), 1)
    if v >= 1:
        return v
    raise ValueError(
        f"n_jobs must be a positive integer or -1 (all CPUs); got {value!r}")


def _predict_panel_timestamp(df: pd.DataFrame, unique_ts: np.ndarray, i: int,
                             p: _PanelParams) -> list[dict]:
    """Predictions for the single test timestamp ``unique_ts[i]``.

    Top-level (picklable) equivalent of the historical nested helper — the SOLE
    source of truth for a single-timestamp panel prediction, used by both the
    sequential loop and the parallel workers so the two paths are identical.
    """
    from .windowing import select_panel_train_window
    tau = unique_ts[i]
    train_df, test_df, window_meta = select_panel_train_window(
        df, tau,
        train_window_mode=p.train_window_mode,
        rolling_window_timestamps=p.rolling_window_timestamps,
        rolling_window_days=p.rolling_window_days,
    )
    train_df = train_df.dropna(subset=p.feature_cols + ["target"])
    test_df = test_df.dropna(subset=p.feature_cols + ["target"])
    if (len(train_df) < p.min_train_obs
            or train_df["target"].nunique() < 2
            or test_df.empty):
        return []
    test_df = test_df.reset_index(drop=True)

    window_cols = {
        "train_window_mode":       window_meta["train_window_mode"],
        "train_window_timestamps": window_meta["train_window_timestamps"],
        "train_start_timestamp":   window_meta["train_start_timestamp"],
        "train_end_timestamp":     window_meta["train_end_timestamp"],
    }

    if p.tune_hyperparams:
        from .hyperparameter_tuning import (
            PANEL, hpo_row_columns, predict_proba, tune_logistic_hyperparams,
        )
        hpo_config = p.hpo_config or {}
        objective = hpo_config.get("objective", "brier_score")
        search_space = hpo_config.get("search_space", {})
        res = tune_logistic_hyperparams(
            train_df, p.feature_cols,
            family=PANEL, objective=objective,
            search_space=search_space, hpo_cfg=hpo_config,
            panel_mode=p.panel_mode,
        )
        proba = predict_proba(res["artifacts"], test_df, p.feature_cols,
                              family=PANEL, panel_mode=p.panel_mode)
        preds = (proba >= 0.5).astype(int)
        hpo_cols = hpo_row_columns(objective, res)
        rows = []
        for j in range(len(test_df)):
            row = {
                "timestamp":   test_df.loc[j, "timestamp"],
                "ticker":      test_df.loc[j, "ticker"],
                "target":      int(test_df.loc[j, "target"]),
                "prediction":  int(preds[j]),
                "probability": float(proba[j]),
            }
            row.update(hpo_cols)
            row.update(window_cols)
            rows.append(row)
        return rows

    X_tr, y_tr, X_te = build_panel_design_matrix(
        train_df, test_df, p.feature_cols, p.panel_mode,
    )
    model = LogisticRegression(
        penalty="l2", C=p.C, solver="lbfgs", max_iter=1000, random_state=42,
    )
    model.fit(X_tr, y_tr)
    preds = model.predict(X_te).astype(int)
    proba = model.predict_proba(X_te)[:, 1]
    return [{
        "timestamp":   test_df.loc[j, "timestamp"],
        "ticker":      test_df.loc[j, "ticker"],
        "target":      int(test_df.loc[j, "target"]),
        "prediction":  int(preds[j]),
        "probability": float(proba[j]),
        **window_cols,
    } for j in range(len(test_df))]


#: Per-worker cache of the input frame, keyed by (path, mtime). loky reuses a
#: worker process across many chunks, so the (potentially large) feature frame
#: is read from the shared temp parquet only ONCE per worker, not per task.
_WORKER_DF_CACHE: dict = {}


def _load_worker_df(path: str, mtime: float) -> pd.DataFrame:
    key = (str(path), float(mtime))
    df = _WORKER_DF_CACHE.get(key)
    if df is None:
        df = (pd.read_parquet(path)
              .sort_values(["timestamp", "ticker"]).reset_index(drop=True))
        _WORKER_DF_CACHE.clear()      # only ever keep the current run's frame
        _WORKER_DF_CACHE[key] = df
    return df


def _compute_panel_chunk_from_path(chunk_id: int, idx_group: list[int],
                                   input_path: str, input_mtime: float,
                                   params: _PanelParams
                                   ) -> tuple[int, list[dict]]:
    """Worker entry point: compute one whole chunk of test timestamps.

    Loads the shared feature frame (cached per worker process) and returns
    ``(chunk_id, rows)``. It does NOT touch the checkpoint files or the
    manifest — only the main process persists results, so there is no
    cross-process write race.

    Every worker caps its BLAS/OpenMP threadpools to ONE thread for the
    duration of the numerical work (``threadpoolctl.threadpool_limits``) — the
    explicit, tested anti-oversubscription mechanism, complementing joblib's
    ``inner_max_num_threads``. This is scoped to the worker call only and never
    mutates the parent process's environment.
    """
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=1):
        df = _load_worker_df(input_path, input_mtime)
        unique_ts = np.sort(df["timestamp"].unique())
        rows: list[dict] = []
        for i in idx_group:
            rows.extend(_predict_panel_timestamp(df, unique_ts, i, params))
    return chunk_id, rows


def _run_panel_chunks_parallel(*, df: pd.DataFrame,
                               pending: list[tuple[int, list[int]]],
                               params: _PanelParams,
                               root,
                               resolved_n_jobs: int,
                               persist) -> None:
    """Compute ``pending`` chunks in a loky process pool and persist each
    result from the MAIN process as it completes.

    * The feature frame is written ONCE to a temp parquet under ``root`` and
      loaded once per worker (cached), so it is not re-pickled per task.
    * ``inner_max_num_threads=1`` caps each worker's BLAS/OpenMP threads at one
      to avoid ``n_jobs × BLAS`` oversubscription — without permanently
      mutating the parent process environment.
    * Results stream back UNORDERED; only the main process writes checkpoints
      and the manifest (no cross-process write race). A worker exception is
      re-raised here (aborting the run) so the command fails loudly instead of
      returning partial results; checkpoints already written stay reusable.
    """
    import tempfile
    from pathlib import Path

    from joblib import Parallel, delayed

    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(suffix=".parquet", prefix="_worker_input_",
                               dir=str(root))
    os.close(fd)
    try:
        df.to_parquet(tmp, index=False)
        mtime = os.path.getmtime(tmp)
        tasks = (delayed(_compute_panel_chunk_from_path)(
                    chunk_id, idx_group, tmp, mtime, params)
                 for chunk_id, idx_group in pending)
        # ``generator_unordered`` yields each chunk as soon as a worker finishes
        # so the main process can persist it immediately (bounded memory, live
        # progress). Worker exceptions surface when the generator is consumed.
        results = Parallel(
            n_jobs=resolved_n_jobs, backend="loky",
            inner_max_num_threads=1, return_as="generator_unordered",
        )(tasks)
        for chunk_id, rows in results:
            persist(chunk_id, rows)
            print(f"     chunk_{chunk_id:04d} → computed + checkpointed (worker)")
    finally:
        try:
            os.remove(tmp)
        except OSError:
            pass


# ══════════════════════════════════════════════════════════════════════════════
# PANEL WALK-FORWARD
# ══════════════════════════════════════════════════════════════════════════════

def run_panel_walk_forward(df: pd.DataFrame,
                           feature_cols: list[str],
                           C: float = DEFAULT_C,
                           panel_mode: str = "pooled",
                           init_train_frac: float = INIT_TRAIN_FRAC,
                           min_init_timestamps: int = MIN_INIT_TIMESTAMPS,
                           min_train_obs: int = MIN_TRAIN_OBS,
                           tune_hyperparams: bool = False,
                           hpo_config: dict | None = None,
                           checkpoint_context: dict | None = None,
                           train_window_mode: str = "expanding",
                           rolling_window_timestamps: int | None = None,
                           rolling_window_days: float | None = None,
                           n_jobs: int = 1) -> pd.DataFrame:
    """Expanding-window pooled panel logit over all coins.

    Returns a signal frame with columns:
    ``timestamp, ticker, target, prediction, probability``.

    Hyperparameter tuning (``tune_hyperparams=True``)
    -------------------------------------------------
    When enabled, the fixed ``C`` is ignored. For each test timestamp ``τ`` the
    training panel ``timestamp < τ`` is split chronologically along its
    **unique timestamps** into inner-train / validation (never along raw
    observation rows), the regularisation strength / class weight are grid
    searched, and the best model is re-fit on the full training panel before
    predicting every coin at ``τ``. For ``ticker_fixed_effects`` the dummies
    are rebuilt leakage-safely for inner-train, validation and the final fit
    exactly as the untuned panel design matrix does. ``τ`` is never used for
    tuning or fitting. Tuned rows carry ``hpo_*`` provenance columns.

    Checkpointing (``checkpoint_context`` not ``None`` and ``enabled``)
    ------------------------------------------------------------------
    The ordered test timestamps are split into **storage chunks** of
    ``chunk_size`` consecutive ``τ``. Each chunk is computed, persisted
    atomically (``chunks/chunk_NNNN.parquet``) and, on resume, reloaded instead
    of recomputed. A chunk is purely a *storage* partition: every ``τ`` still
    trains on all rows with ``timestamp < τ``, so chunking changes nothing about
    the predictions and introduces no leakage.
    """
    if panel_mode not in PANEL_MODES:
        raise ValueError(f"Unknown panel_mode {panel_mode!r}; expected one of {PANEL_MODES}")

    df = df.sort_values(["timestamp", "ticker"]).reset_index(drop=True)
    unique_ts = np.sort(df["timestamp"].unique())
    n_ts = len(unique_ts)
    if n_ts == 0:
        return pd.DataFrame()

    init_idx = max(int(n_ts * init_train_frac), min_init_timestamps)

    # Single source of truth for the per-timestamp/per-chunk parameters —
    # shared verbatim by the sequential loop and the parallel workers.
    params = _PanelParams(
        feature_cols=list(feature_cols),
        C=C,
        panel_mode=panel_mode,
        min_train_obs=min_train_obs,
        tune_hyperparams=bool(tune_hyperparams),
        hpo_config=dict(hpo_config or {}),
        train_window_mode=train_window_mode,
        rolling_window_timestamps=rolling_window_timestamps,
        rolling_window_days=rolling_window_days,
    )

    def _predict_one_timestamp(i: int) -> list[dict]:
        """Sequential-path wrapper around the top-level per-timestamp helper."""
        return _predict_panel_timestamp(df, unique_ts, i, params)

    test_indices = list(range(init_idx, n_ts))
    ckpt_on = bool(checkpoint_context and checkpoint_context.get("enabled"))

    # ── Plain (non-checkpointed) path: behaviour unchanged ──────
    if not ckpt_on:
        results: list[dict] = []
        for i in test_indices:
            results.extend(_predict_one_timestamp(i))
        return pd.DataFrame(results) if results else pd.DataFrame()

    # ── Checkpointed path: persist one parquet per timestamp chunk ──
    from . import checkpointing as ckpt
    root      = checkpoint_context["root"]
    resume    = bool(checkpoint_context.get("resume", True))
    chunk_size = int(checkpoint_context.get("chunk_size", ckpt.DEFAULT_CHUNK_SIZE))
    resolved_n_jobs = _resolve_n_jobs(
        checkpoint_context.get("n_jobs", n_jobs))
    chunks = ckpt.chunk_indices(test_indices, chunk_size)

    manifest = ckpt.load_manifest(root)
    manifest["total_chunks"] = len(chunks)
    manifest["status"] = "running"
    manifest["n_jobs"] = resolved_n_jobs
    ckpt.write_manifest(root, manifest)

    # Partition into already-cached vs pending BEFORE dispatch (main process
    # only). Cached chunks are reused; only pending chunks are (re)computed.
    frames: list[pd.DataFrame] = []
    pending: list[tuple[int, list[int]]] = []
    n_cached = 0
    for chunk_id, idx_group in enumerate(chunks):
        cp_path = ckpt.chunk_checkpoint_path(root, chunk_id)
        if resume and cp_path.exists():
            cached = ckpt.load_checkpoint(cp_path)
            if cached is not None:
                if not cached.empty:
                    frames.append(cached)
                n_cached += 1
                print(f"     chunk_{chunk_id:04d} → CACHED CHECKPOINT")
                continue
        pending.append((chunk_id, idx_group))

    def _persist_chunk(chunk_id: int, rows: list[dict]) -> None:
        """MAIN-process-only checkpoint write + manifest update (no race)."""
        chunk_df = (pd.DataFrame(rows) if rows
                    else pd.DataFrame(columns=list(ckpt.CORE_COLUMNS)))
        ckpt.save_checkpoint_atomic(
            chunk_df, ckpt.chunk_checkpoint_path(root, chunk_id))
        if not chunk_df.empty:
            frames.append(chunk_df)
        manifest["completed_chunks"] = ckpt.list_completed_chunks(root)
        ckpt.write_manifest(root, manifest)

    t0 = time.time()
    if resolved_n_jobs == 1 or not pending:
        # ── Sequential (unchanged behaviour) ───────────────────────
        for chunk_id, idx_group in pending:
            rows: list[dict] = []
            for i in idx_group:
                rows.extend(_predict_one_timestamp(i))
            _persist_chunk(chunk_id, rows)
            print(f"     chunk_{chunk_id:04d} → computed + checkpointed")
    else:
        # ── Parallel: process pool over PENDING chunks (loky) ──────
        _run_panel_chunks_parallel(
            df=df, pending=pending, params=params, root=root,
            resolved_n_jobs=resolved_n_jobs, persist=_persist_chunk)

    elapsed = time.time() - t0
    if resolved_n_jobs > 1 and pending:
        print(f"     [parallel] n_jobs={resolved_n_jobs} "
              f"total_chunks={len(chunks)} cached={n_cached} "
              f"submitted={len(pending)} elapsed={elapsed:.1f}s "
              f"avg_chunk={elapsed / max(len(pending), 1):.2f}s")

    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    return out.sort_values(["timestamp", "ticker"]).reset_index(drop=True)


def run_panel_rolling_probability(df: pd.DataFrame,
                                  *,
                                  panel_mode: str = "pooled",
                                  init_train_frac: float = INIT_TRAIN_FRAC,
                                  min_init_timestamps: int = MIN_INIT_TIMESTAMPS,
                                  min_train_obs: int = MIN_TRAIN_OBS,
                                  train_window_mode: str = "expanding",
                                  rolling_window_timestamps: int | None = None,
                                  rolling_window_days: float | None = None
                                  ) -> pd.DataFrame:
    """Panel-compatible benchmark: ticker-rolling probability with pooled fallback.

    For each test timestamp ``τ``:

    * The training slice is chosen by the manual window selector — the same
      one the logistic models use.
    * The benchmark probability for ticker ``i`` is the empirical mean of
      ``target`` for that ticker inside the training window. If the ticker has
      no usable observations in that window, fall back to the **pooled** mean
      across all tickers in the same window.
    * ``prediction = 1`` if ``p_hat >= 0.5`` else ``0``.

    Output schema matches the logistic panel signals; an additional
    ``benchmark_model`` column tags each row.
    """
    from .windowing import select_panel_train_window

    df = df.sort_values(["timestamp", "ticker"]).reset_index(drop=True)
    unique_ts = np.sort(df["timestamp"].unique())
    n_ts = len(unique_ts)
    if n_ts == 0:
        return pd.DataFrame()
    init_idx = max(int(n_ts * init_train_frac), min_init_timestamps)

    rows: list[dict] = []
    for i in range(init_idx, n_ts):
        tau = unique_ts[i]
        train_df, test_df, window_meta = select_panel_train_window(
            df, tau,
            train_window_mode=train_window_mode,
            rolling_window_timestamps=rolling_window_timestamps,
            rolling_window_days=rolling_window_days,
        )
        train_df = train_df.dropna(subset=["target"])
        test_df  = test_df.dropna(subset=["target"])
        if len(train_df) < min_train_obs or test_df.empty:
            continue
        pooled = float(train_df["target"].astype(float).mean())
        per_ticker = (train_df.groupby("ticker")["target"].mean()
                              .astype(float).to_dict())
        for _, r in test_df.iterrows():
            tk = r["ticker"]
            p_hat = per_ticker.get(tk)
            if p_hat is None or not np.isfinite(p_hat):
                p_hat = pooled
            rows.append({
                "timestamp":         r["timestamp"],
                "ticker":            tk,
                "target":            int(r["target"]),
                "prediction":        int(p_hat >= 0.5),
                "probability":       float(p_hat),
                "benchmark_model":   "ticker_rolling_probability_with_pooled_fallback",
                "train_window_mode": window_meta["train_window_mode"],
                "train_window_timestamps": window_meta["train_window_timestamps"],
                "train_start_timestamp":   window_meta["train_start_timestamp"],
                "train_end_timestamp":     window_meta["train_end_timestamp"],
            })
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(["timestamp", "ticker"]).reset_index(drop=True)


def run_panel_model_for_feature_set(df_all: pd.DataFrame,
                                    feature_cols: list[str],
                                    tickers: Sequence[str] | None,
                                    C: float = DEFAULT_C,
                                    panel_mode: str = "pooled",
                                    tune_hyperparams: bool = False,
                                    hpo_config: dict | None = None,
                                    checkpoint_context: dict | None = None,
                                    train_window_mode: str = "expanding",
                                    rolling_window_timestamps: int | None = None,
                                    rolling_window_days: float | None = None,
                                    n_jobs: int = 1
                                    ) -> pd.DataFrame:
    """Filter to ``tickers`` (if given) and run the panel walk-forward."""
    df = df_all
    if tickers:
        df = df[df["ticker"].isin(set(tickers))]
    if df.empty:
        return pd.DataFrame()
    return run_panel_walk_forward(df, feature_cols, C=C, panel_mode=panel_mode,
                                  tune_hyperparams=tune_hyperparams,
                                  hpo_config=hpo_config,
                                  checkpoint_context=checkpoint_context,
                                  train_window_mode=train_window_mode,
                                  rolling_window_timestamps=rolling_window_timestamps,
                                  rolling_window_days=rolling_window_days,
                                  n_jobs=n_jobs)


# ══════════════════════════════════════════════════════════════════════════════
# OUTPUT NAMING
# ══════════════════════════════════════════════════════════════════════════════

def _mode_suffix(panel_mode: str) -> str:
    return "panel_pooled" if panel_mode == "pooled" else "panel_ticker_fe"


#: Re-exported from :mod:`thesis_pipeline.price.features` so the
#: rolling-window metadata stamp uses the SAME canonical bar-per-day
#: mapping as the feature generator. The horizon → days-to-bars
#: conversion is the only place ``rolling_window_timestamps`` is
#: derived from ``rolling_window_days``; without this single source
#: of truth the two columns could quietly drift apart.
from ..price.features import BARS_PER_DAY as _BARS_PER_DAY


#: Columns the rolling-window output contract requires. Mirrors the
#: set the smoke validator enforces and the v4 evaluation layer reads.
ROLLING_WINDOW_METADATA_COLUMNS: tuple[str, ...] = (
    "rolling_window_days",
    "rolling_window_timestamps",
    "train_window_mode",
    "train_start_timestamp",
    "train_end_timestamp",
    "train_window_timestamps",
)


def _stamp_rolling_window_metadata(signals: pd.DataFrame,
                                     *,
                                     horizon: str,
                                     train_window_mode: str,
                                     rolling_window_days: float | None,
                                     rolling_window_timestamps: int | None,
                                     ) -> pd.DataFrame:
    """Stamp the run's rolling-window configuration on every signal row.

    ``rolling_window_days`` is the canonical configuration variable —
    the value flows in from the CLI / ``run_models`` entry point and
    is NEVER derived from a filename or from
    ``train_window_timestamps`` (which is horizon-specific).

    For ``rolling_fixed`` mode, ``rolling_window_timestamps`` is
    deterministically derived from
    ``rolling_window_days * BARS_PER_DAY[horizon]`` so the column is
    horizon-correct on 6h / 1h runs (where the per-bar count is 4×
    and 24× the day count). Expanding runs leave both columns at
    ``NaN`` — they are conceptually undefined.
    """
    out = signals.copy()
    out["train_window_mode"] = train_window_mode
    if str(train_window_mode) == "rolling_fixed":
        # Days from config (authoritative).
        if rolling_window_days is None:
            out["rolling_window_days"] = np.nan
        else:
            out["rolling_window_days"] = float(rolling_window_days)
        # Timestamps from horizon × days, falling back to the
        # explicitly-passed rolling_window_timestamps when the user
        # invoked rolling-by-bar-count instead of rolling-by-day.
        if rolling_window_timestamps is not None:
            out["rolling_window_timestamps"] = int(rolling_window_timestamps)
        elif rolling_window_days is not None and str(horizon) in _BARS_PER_DAY:
            out["rolling_window_timestamps"] = int(
                float(rolling_window_days) * _BARS_PER_DAY[str(horizon)]
            )
        else:
            out["rolling_window_timestamps"] = np.nan
    else:
        # Expanding / other modes — explicit NaN keeps the schema
        # uniform without misrepresenting the run.
        out["rolling_window_days"] = np.nan
        if "rolling_window_timestamps" not in out.columns:
            out["rolling_window_timestamps"] = np.nan
    return out


def _assert_training_window_contract(signals: pd.DataFrame,
                                       *,
                                       horizon: str,
                                       train_window_mode: str,
                                       ) -> None:
    """Refuse to write a rolling-window parquet that violates the v4
    canonical training-window output schema.

    Two production-supported window flavours under ``rolling_fixed``:

    * **rolling-by-days** (``--rolling-window-days N``, the canonical
      v4 path): ``rolling_window_days`` is the source of truth and
      ``rolling_window_timestamps`` is derived as
      ``days × BARS_PER_DAY[horizon]``.
    * **rolling-by-timestamps** (``--rolling-window-timestamps N``,
      legacy / debugging path): ``rolling_window_timestamps`` is the
      source of truth and ``rolling_window_days`` is genuinely
      undefined.

    The assertion requires the schema columns to exist on both paths;
    numeric / positive / constant invariants apply per column only when
    that side of the configuration was actually supplied.
    """
    if str(train_window_mode) != "rolling_fixed":
        return
    required = ("rolling_window_days", "rolling_window_timestamps",
                 "train_window_mode", "train_start_timestamp",
                 "train_end_timestamp", "train_window_timestamps")
    missing = [c for c in required if c not in signals.columns]
    if missing:
        raise AssertionError(
            "panel_logit output is missing required training-window "
            f"metadata columns: {missing}"
        )
    if signals["train_window_mode"].nunique() != 1:
        raise AssertionError(
            "panel_logit output 'train_window_mode' is not constant"
        )
    days_s = signals["rolling_window_days"]
    ts_s   = signals["rolling_window_timestamps"]

    def _check_col(name: str, s: pd.Series) -> bool:
        """Return True when the column is fully populated, > 0 and
        constant. Raise when ANY non-null value is non-positive or the
        column has more than one distinct value (rules out tampered
        per-row drift). A fully-NaN column is permitted on the
        rolling-by-timestamps legacy path."""
        non_null = s.dropna()
        if non_null.empty:
            return False
        if s.isna().any():
            raise AssertionError(
                f"panel_logit output {name!r} is partially NaN; the "
                "column must be either fully populated or fully NaN "
                "within one parquet"
            )
        if not (non_null > 0).all():
            raise AssertionError(
                f"panel_logit output {name!r} contains non-positive values"
            )
        if s.nunique() != 1:
            raise AssertionError(
                f"panel_logit output {name!r} is not constant within the file "
                f"(got {sorted(s.unique().tolist())[:3]}…)"
            )
        return True

    days_ok = _check_col("rolling_window_days",       days_s)
    ts_ok   = _check_col("rolling_window_timestamps", ts_s)
    if not (days_ok or ts_ok):
        raise AssertionError(
            "panel_logit output: rolling_fixed run carries neither a "
            "valid rolling_window_days nor a valid rolling_window_timestamps "
            "column (numeric, > 0, constant)"
        )
    # Horizon-specific consistency rule applies only when BOTH sides
    # of the configuration are present.
    bpd = _BARS_PER_DAY.get(str(horizon))
    if bpd is not None and days_ok and ts_ok:
        days = float(days_s.iat[0])
        ts   = int(ts_s.iat[0])
        if ts != int(days * bpd):
            raise AssertionError(
                f"panel_logit output: rolling_window_timestamps={ts} "
                f"does not equal rolling_window_days={days} * "
                f"BARS_PER_DAY[{horizon!r}]={bpd}"
            )


def panel_output_name(set_id: str, sentiment_model: str, panel_mode: str,
                      hpo_variant: str = "fixed",
                      window_suffix: str = "",
                      coin_universe: "Iterable[str] | None" = None) -> str:
    """``{set_id}[_{sentiment_model}]_{panel_pooled|panel_ticker_fe}[_{hpo_variant}][_rw{N}][_u_{hash}]``.

    A tuned run appends the HPO variant (e.g. ``..._panel_pooled_hpo_brier``)
    so tuned and fixed-C panel signals never share a filename. A rolling-window
    run appends ``_rw{N}`` (or ``_rw{D}d``) — see
    :func:`thesis_pipeline.modeling.windowing.window_suffix`. Expanding runs
    keep the legacy unsuffixed window section.

    When ``coin_universe`` is provided the requested-universe hash is
    appended as ``_u_{8-hex}`` (commit 4 Section A.2). The hash is
    produced by :func:`thesis_pipeline.modeling.naive_reference
    .coin_universe_hash` — the exact same helper NAIVE uses, so a model
    and its matched NAIVE share the suffix on disk.
    """
    suffix = _mode_suffix(panel_mode)
    if sentiment_model and str(sentiment_model) not in ("-", "nan"):
        base = f"{set_id}_{sentiment_model}_{suffix}"
    else:
        base = f"{set_id}_{suffix}"
    if hpo_variant and hpo_variant not in ("fixed", "-"):
        base = f"{base}_{hpo_variant}"
    if window_suffix:
        base = f"{base}{window_suffix}"
    if coin_universe is not None:
        from .naive_reference import coin_universe_hash
        h = coin_universe_hash(coin_universe)
        if h:
            base = f"{base}_u_{h}"
    return base


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
    parser.add_argument("--tune-hyperparams", "--tune_hyperparams",
                        dest="tune_hyperparams", action="store_true",
                        help="Enable nested grid-search HPO inside each "
                             "panel training window.")
    parser.add_argument("--hpo-objective", "--hpo_objective",
                        dest="hpo_objective", default=None,
                        choices=["brier_score", "log_loss", "accuracy"])
    parser.add_argument("--hpo-config", "--hpo_config", dest="hpo_config",
                        default=None)
    parser.add_argument("--hpo-grid-C", "--hpo_grid_C", dest="hpo_grid_C",
                        type=float, nargs="+", default=None)
    parser.add_argument("--hpo-class-weight", "--hpo_class_weight",
                        dest="hpo_class_weight", nargs="+", default=None)
    parser.add_argument("--checkpoint", dest="checkpoint",
                        action=argparse.BooleanOptionalAction, default=True,
                        help="Write per-chunk checkpoints so a crashed panel run "
                             "can resume (default: on).")
    parser.add_argument("--resume", dest="resume",
                        action=argparse.BooleanOptionalAction, default=True,
                        help="Reuse existing chunk checkpoints (default: on).")
    parser.add_argument("--checkpoint-dir", "--checkpoint_dir",
                        dest="checkpoint_dir", default=CHECKPOINT_DIR,
                        help="Root directory for model checkpoints.")
    parser.add_argument("--checkpoint-chunk-size", "--checkpoint_chunk_size",
                        dest="checkpoint_chunk_size", type=int, default=20,
                        help="Panel test timestamps per checkpoint chunk.")
    parser.add_argument("--clear-checkpoints", "--clear_checkpoints",
                        dest="clear_checkpoints", action="store_true",
                        help="Delete this run's checkpoint directory before start.")
    parser.add_argument("--train-window", "--train_window",
                        dest="train_window", default="expanding",
                        choices=["expanding", "rolling_fixed"],
                        help="Panel training window (default: expanding).")
    parser.add_argument("--rolling-window-timestamps", "--rolling_window_timestamps",
                        dest="rolling_window_timestamps", type=int, default=None,
                        help="Manual number of pre-tau unique timestamps for "
                             "rolling_fixed (no automatic default).")
    parser.add_argument("--rolling-window-days", "--rolling_window_days",
                        dest="rolling_window_days", type=float, default=None,
                        help="Manual day-distance window for rolling_fixed.")
    parser.add_argument("--dry-run", "--dry_run", dest="dry_run", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--restart", action="store_true")
    return parser


def _run_panel(args: argparse.Namespace, hpo_cfg: dict | None = None) -> int:
    """Shared body used by both :func:`main` and the run-models delegation.

    ``hpo_cfg`` is the resolved hyperparameter-tuning config. When ``None``
    (standalone ``python -m ...panel_logit`` invocation) it is resolved here
    from the CLI flags + ``configs/model_specs.yaml``.
    """
    panel_mode = getattr(args, "panel_mode", "ticker_fixed_effects") \
                 or "ticker_fixed_effects"
    from .windowing import window_suffix as _window_suffix
    train_window_mode = getattr(args, "train_window", "rolling_fixed") \
                        or "rolling_fixed"
    rolling_window_timestamps = getattr(args, "rolling_window_timestamps", None)
    rolling_window_days       = getattr(args, "rolling_window_days", None)
    # The v4 default is rolling_window_days=180. A user who explicitly
    # asks for a timestamp-count window must take priority — pass only
    # the one they set so select_panel_train_window's mutex check passes.
    if rolling_window_timestamps is not None and rolling_window_days is not None:
        rolling_window_days = None
    if train_window_mode == "rolling_fixed" and (
        rolling_window_timestamps is None and rolling_window_days is None
    ):
        # Fail fast with a clear message — there is no automatic break-informed
        # default window. Structural breaks are diagnostic only.
        print("  [ERROR] --train-window rolling_fixed requires "
              "--rolling-window-timestamps or --rolling-window-days "
              "(v4 default is 180 calendar days; pass --rolling-window-days 0 "
              "to disable explicitly is NOT supported — pass "
              "--train-window expanding instead).")
        return 2
    win_suffix = _window_suffix(train_window_mode, rolling_window_timestamps,
                                rolling_window_days)

    from .hyperparameter_tuning import hpo_variant_label, load_hpo_config
    if hpo_cfg is None:
        hpo_cfg = load_hpo_config(
            # v4: --tune-hyperparams defaults to True via BooleanOptionalAction.
            # Forward the explicit bool so --no-tune-hyperparams really disables.
            enabled_override=bool(getattr(args, "tune_hyperparams", True)),
            objective_override=getattr(args, "hpo_objective", None),
            c_grid=getattr(args, "hpo_grid_C", None),
            class_weight_grid=getattr(args, "hpo_class_weight", None),
            config_path=getattr(args, "hpo_config", None),
        )
    tune_on = bool(hpo_cfg["enabled"])
    hpo_variant = hpo_variant_label(tune_on, hpo_cfg["objective"])

    # ── Parallel worker count (chunk-level process parallelism) ──────
    # Resolve + validate up front so a bad --n-jobs fails clearly before any
    # heavy work, and the resolved count is visible in the stage header/logs.
    # Parallelism operates over checkpoint chunks, so it only applies when
    # checkpointing is enabled; with --no-checkpoint the panel walk-forward
    # follows its non-checkpointed SEQUENTIAL path regardless of --n-jobs.
    n_jobs_resolved = _resolve_n_jobs(getattr(args, "n_jobs", 1))
    _checkpoint_enabled = bool(getattr(args, "checkpoint", True))
    if n_jobs_resolved > 1 and not _checkpoint_enabled:
        print("  [WARN] --n-jobs is ignored when checkpointing is disabled; "
              "the non-checkpointed panel path runs sequentially.")
    elif n_jobs_resolved > 1:
        print(f"  [INFO] panel walk-forward: {n_jobs_resolved} worker "
              f"process(es) for checkpoint chunks (1 BLAS thread each).")

    # ── Preprocessing signature (Section 4 — cache/checkpoint invalidation) ──
    # Covers winsorisation on/off + quantile bounds + grouping rule + feature
    # allowlist (from modeling.preprocessing config) plus the model training
    # window and the HPO objective. Old checkpoints / signal files produced
    # under a different preprocessing methodology are detected and rejected.
    from .preprocessing import preprocessing_signature as _preprocessing_signature
    from .forecast_sample import (
        load_forecast_sample_config, restrict_to_forecast_sample,
    )
    _fs_cfg = load_forecast_sample_config()
    _preproc_window = (f"{train_window_mode}|days={rolling_window_days}"
                       f"|ts={rolling_window_timestamps}")
    preproc_sig = _preprocessing_signature(
        model_window=_preproc_window,
        hpo_objective=(hpo_cfg["objective"] if tune_on else "-"),
        forecast_sample=_fs_cfg,
    )

    # ── Checkpointing config ────────────────────────────────────
    from . import checkpointing as ckpt
    ckpt_on    = bool(getattr(args, "checkpoint", True))
    resume     = bool(getattr(args, "resume", True))
    ckpt_dir   = getattr(args, "checkpoint_dir", CHECKPOINT_DIR) or CHECKPOINT_DIR
    chunk_size = int(getattr(args, "checkpoint_chunk_size", 20) or 20)
    clear_ckpt = bool(getattr(args, "clear_checkpoints", False))

    if getattr(args, "dry_run", False):
        try:
            from ..logging_utils import log_stage_header
            from ..config import resolve_path
            inputs = []
            if args.horizon:
                inputs.append(resolve_path("final_features_pattern", horizon=args.horizon))
            inputs.append(resolve_path("feature_sets_xlsx"))
            out_name_example = "(per set_id default)"
            if args.set_id:
                out_name_example = (
                    f"Outputs/Signals/{args.horizon or '<horizon>'}/"
                    f"{panel_output_name(args.set_id, args.sentiment_model or '-', panel_mode, hpo_variant, win_suffix)}.parquet"
                )
            log_stage_header(
                "run_models", mode="dry-run", inputs=inputs, outputs=[],
                extra={
                    "model_type": MODEL_TYPE,
                    "panel_mode": panel_mode,
                    "horizon":    args.horizon or "(all)",
                    "set_id":     args.set_id or "(all)",
                    "coins":      list(args.coins) if args.coins else "(all)",
                    "C":          args.C,
                    "n_jobs":     n_jobs_resolved,
                    "hpo_variant": hpo_variant,
                    "output_name": out_name_example,
                    "tune_hyperparams": tune_on,
                    "hpo_objective":   hpo_cfg["objective"] if tune_on else "(off)",
                    "hpo_C_grid":      hpo_cfg["search_space"].get("C") if tune_on else "(off)",
                    "hpo_class_weight_grid": (
                        [("none" if c is None else c)
                         for c in hpo_cfg["search_space"].get("class_weight", [])]
                        if tune_on else "(off)"),
                    "hpo_validation_fraction": (
                        hpo_cfg["validation_fraction"] if tune_on else "(off)"),
                    "train_window_mode":         train_window_mode,
                    "rolling_window_timestamps": rolling_window_timestamps,
                    "rolling_window_days":       rolling_window_days,
                    "checkpoint_enabled":    ckpt_on,
                    "resume":                resume,
                    "checkpoint_dir":        ckpt_dir,
                    "checkpoint_chunk_size": chunk_size,
                    "clear_checkpoints":     clear_ckpt,
                    "run_checkpoint_path": (
                        str(ckpt.checkpoint_root(
                            ckpt_dir, args.horizon or "<horizon>",
                            panel_output_name(args.set_id or "<set_id>",
                                              args.sentiment_model or "-",
                                              panel_mode, hpo_variant)))
                        if ckpt_on else "(off)"),
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
        # Universe resolution — REQUESTED is immutable per-run, AVAILABLE is
        # the subset present in the feature frame (commit 4 Section A.1).
        from .naive_reference import resolve_universes
        uni = resolve_universes(args.coins, tickers)
        requested_tickers = list(uni["requested"])
        available_tickers = list(uni["available"])
        # Estimation runs on the AVAILABLE subset.
        tickers = available_tickers
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

            # Output filename embeds the REQUESTED-universe hash so smoke
            # and full-grid runs never share a parquet (commit 4 A.2).
            out_name = panel_output_name(set_id, sent_model, panel_mode,
                                         hpo_variant, win_suffix,
                                         coin_universe=requested_tickers)
            out_path = os.path.join(hz_dir, f"{out_name}.parquet")

            # Panel-compatible benchmark: ticker-rolling probability with pooled
            # fallback. Honours the same training-window mode as the logistic
            # models so the comparison stays apples-to-apples; the per-row
            # ``benchmark_model`` column tags the rule used.
            is_benchmark = feat_str.strip() in _BENCHMARK_SENTINELS
            if is_benchmark and tune_on:
                print(f"\n  ── {out_name} ({label}) → SKIP "
                      f"(rolling-probability benchmark is not tuned)")
                continue

            if os.path.isfile(out_path) and not args.restart:
                try:
                    from .hyperparameter_tuning import summarize_hpo_columns
                    cached = pd.read_parquet(out_path)
                    # Reject a cached signal file produced under a different
                    # preprocessing methodology (Section 4.4) — a pre-winsoriser
                    # signal must never be mistaken for the corrected final run.
                    cached_sig = (str(cached.get("preprocessing_signature",
                                                 pd.Series([""])).iat[0])
                                  if not cached.empty else "")
                    if cached_sig != preproc_sig:
                        print(f"\n  ── {out_name} ({label}) → cache preprocessing "
                              f"signature mismatch (cached={cached_sig or 'none'} "
                              f"!= {preproc_sig}); recomputing")
                        raise ValueError("preprocessing_signature_mismatch")
                    m = compute_metrics(cached, "pooled")
                    m.update({"horizon": hz, "set_id": set_id,
                              "sentiment_model": sent_model, "label": label,
                              "category": category,
                              "n_tickers": cached["ticker"].nunique(),
                              "model_type": MODEL_TYPE, "panel_mode": panel_mode})
                    m.update(summarize_hpo_columns(cached))
                    all_metrics.append(m)
                    print(f"\n  ── {out_name} ({label}) → CACHED "
                          f"acc={m['accuracy']:.4f}, n={m['n_obs']}")
                    continue
                except Exception:
                    pass

            print(f"\n  ── {out_name} ({label}) ")

            if is_benchmark:
                feature_cols = []  # rolling probability uses no features
            else:
                feature_cols = [f.strip() for f in feat_str.split(",")]
                missing = [f for f in feature_cols if f not in df_all.columns]
                if missing:
                    print(f"  → SKIP (missing: {missing[:3]})")
                    all_metrics.append({
                        "horizon": hz, "set_id": set_id, "sentiment_model": sent_model,
                        "label": label, "category": category, "ticker": "pooled",
                        "accuracy": np.nan, "n_obs": 0, "n_tickers": 0,
                        "model_type": MODEL_TYPE, "panel_mode": panel_mode,
                        "status": f"missing: {missing[:3]}",
                    })
                    continue

            # ── Checkpoint setup (per-run chunk directory) ────
            root = ckpt.checkpoint_root(ckpt_dir, hz, out_name)
            checkpoint_context = None
            manifest_base = {
                "horizon": hz, "set_id": set_id, "sentiment_model": sent_model,
                "model_type": MODEL_TYPE, "panel_mode": panel_mode,
                "hpo_variant": hpo_variant if tune_on else "fixed",
                "hpo_objective": hpo_cfg["objective"] if tune_on else "-",
                "feature_cols": feature_cols,
                "train_window_mode":         train_window_mode,
                "rolling_window_timestamps": rolling_window_timestamps,
                "rolling_window_days":       rolling_window_days,
                "benchmark":                 bool(is_benchmark),
                # Preprocessing methodology signature (Section 4).
                "preprocessing_signature":   preproc_sig,
                # Universe identity (commit 4 A.4). Checkpoints created
                # for a smaller universe must NOT resume a larger run.
                "requested_tickers":            list(requested_tickers),
                "requested_coin_universe_hash": uni["requested_hash"],
                "n_requested_tickers":          len(requested_tickers),
            }
            if ckpt_on:
                # Refuse to reuse checkpoints when the run's window mode,
                # rolling-window size, feature list, HPO objective, panel mode,
                # benchmark identity, OR requested coin universe differs.
                existing = ckpt.load_manifest(root)
                guard_keys = ("train_window_mode", "rolling_window_timestamps",
                              "rolling_window_days", "feature_cols",
                              "hpo_objective", "panel_mode", "benchmark",
                              "preprocessing_signature",
                              "requested_coin_universe_hash",
                              "requested_tickers")
                if existing and any(existing.get(k) != manifest_base.get(k)
                                    for k in guard_keys):
                    print("  [INFO] Existing checkpoint manifest is incompatible "
                          "(window/feature/hpo/panel-mode/universe changed) — clearing.")
                    ckpt.clear_run_checkpoints(root)
                if clear_ckpt:
                    ckpt.clear_run_checkpoints(root)
                ckpt.init_manifest(root, base=manifest_base)
                checkpoint_context = {"enabled": True, "resume": resume,
                                      "root": root, "chunk_size": chunk_size,
                                      "n_jobs": n_jobs_resolved}

            t0 = time.time()
            if is_benchmark:
                df_used = df_all[df_all["ticker"].isin(set(tickers))] if tickers else df_all
                signals = run_panel_rolling_probability(
                    df_used, panel_mode=panel_mode,
                    train_window_mode=train_window_mode,
                    rolling_window_timestamps=rolling_window_timestamps,
                    rolling_window_days=rolling_window_days,
                )
            else:
                signals = run_panel_model_for_feature_set(
                    df_all, feature_cols, tickers, C=args.C, panel_mode=panel_mode,
                    tune_hyperparams=tune_on, hpo_config=hpo_cfg,
                    checkpoint_context=checkpoint_context,
                    train_window_mode=train_window_mode,
                    rolling_window_timestamps=rolling_window_timestamps,
                    rolling_window_days=rolling_window_days,
                    n_jobs=n_jobs_resolved,
                )
            elapsed = time.time() - t0
            if ckpt_on:
                mf = ckpt.load_manifest(root)
                mf["status"] = "complete"
                ckpt.write_manifest(root, mf)

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
            # HPO identity. Tuned rows already carry hpo_enabled/hpo_objective/
            # hpo_variant per row; untuned panel runs get the fixed sentinels.
            if not tune_on:
                signals["hpo_enabled"]   = False
                signals["hpo_objective"] = "-"
                signals["hpo_variant"]   = "fixed"
            # ── Rolling-window metadata (commit 10) ────────────────
            # Stamp the run's training-window configuration on every
            # row at the canonical assembly site so freshly computed
            # AND checkpoint-resumed paths both carry the contract.
            # ``rolling_window_days`` comes straight from the run
            # config — never derived from the filename or from
            # ``train_window_timestamps`` (the two only coincide at
            # the 1d horizon). The per-row ``train_window_*`` columns
            # produced by select_panel_train_window remain untouched.
            from .panel_logit import _stamp_rolling_window_metadata
            signals = _stamp_rolling_window_metadata(
                signals,
                horizon=hz,
                train_window_mode=train_window_mode,
                rolling_window_days=rolling_window_days,
                rolling_window_timestamps=rolling_window_timestamps,
            )
            # Universe identity (commit 3 Section B + commit 4 A.5).
            # Stamp the REQUESTED universe (immutable) alongside the
            # AVAILABLE subset and the REALIZED ticker set so the
            # absolute_vs_naive evaluation matches by requested-hash
            # without inferring from realized tickers later.
            from .naive_reference import stamp_universe_metadata
            signals = stamp_universe_metadata(
                signals,
                requested_universe=requested_tickers,
                available_universe=available_tickers,
            )
            # Preprocessing-methodology stamp (Section 4.4): so an old signal
            # file can never be mistaken for the corrected final run.
            signals["preprocessing_signature"] = preproc_sig
            # Canonical output contract: stamp forecast_origin (= timestamp + h)
            # and keep only rows whose forecast origin is in the configured 2022
            # sample window. Covers the panel logit AND the panel NAIVE
            # benchmark (both reach this single write site).
            signals = restrict_to_forecast_sample(signals, hz, cfg=_fs_cfg)
            # ── Output-schema assertion (commit 10) ───────────────
            # Refuse to write a rolling-window parquet that is missing
            # any of the canonical training-window columns. This is
            # the smoke validator's required set; failing here gives
            # the user a clear stack trace instead of a downstream
            # validator complaint after the fact.
            _assert_training_window_contract(signals, horizon=hz,
                                              train_window_mode=train_window_mode)
            signals.to_parquet(out_path, index=False, engine="pyarrow")

            from .hyperparameter_tuning import summarize_hpo_columns
            hpo_summary = summarize_hpo_columns(signals)
            m = compute_metrics(signals, "pooled")
            m.update({"horizon": hz, "set_id": set_id,
                      "sentiment_model": sent_model, "label": label,
                      "category": category,
                      "n_tickers": signals["ticker"].nunique(),
                      "model_type": MODEL_TYPE, "panel_mode": panel_mode})
            m.update(hpo_summary)
            if ckpt_on:
                n_chunks = len(ckpt.list_completed_chunks(root))
                m.update({"checkpoint_enabled": True,
                          "n_checkpoints_written": n_chunks})
            else:
                m["checkpoint_enabled"] = False
            all_metrics.append(m)
            for tk, grp in signals.groupby("ticker"):
                mt = compute_metrics(grp, tk)
                mt.update({"horizon": hz, "set_id": set_id,
                           "sentiment_model": sent_model, "label": label,
                           "category": category, "n_tickers": 1,
                           "model_type": MODEL_TYPE, "panel_mode": panel_mode})
                mt.update(summarize_hpo_columns(grp))
                all_metrics.append(mt)
            print(f"  → acc={m['accuracy']:.4f}, f1={m['f1']:.4f}, "
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
        feature_config: str | None = None,
        tune_hyperparams: bool = False,
        hpo_objective: str | None = None,
        hpo_config: str | None = None,
        hpo_grid_C: Sequence[float] | None = None,
        hpo_class_weight: Sequence[str] | None = None,
        checkpoint: bool = True,
        resume: bool = True,
        checkpoint_dir: str | None = None,
        checkpoint_chunk_size: int | None = None,
        clear_checkpoints: bool = False,
        train_window: str = "expanding",
        rolling_window_timestamps: int | None = None,
        rolling_window_days: float | None = None) -> int:
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
    if tune_hyperparams:
        argv.append("--tune-hyperparams")
    if hpo_objective:
        argv += ["--hpo-objective", hpo_objective]
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
    if train_window:
        argv += ["--train-window", train_window]
    if rolling_window_timestamps is not None:
        argv += ["--rolling-window-timestamps", str(rolling_window_timestamps)]
    if rolling_window_days is not None:
        argv += ["--rolling-window-days", str(rolling_window_days)]
    if dry_run:
        argv.append("--dry-run")
    if force:
        argv.append("--force")
    if restart:
        argv.append("--restart")
    return main(argv)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
