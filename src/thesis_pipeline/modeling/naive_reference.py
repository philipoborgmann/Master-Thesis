"""NAIVE reference generation (Task 6 follow-up, Section A).

NAIVE is the historical-majority (rolling-probability) reference. It is
**not** a feature set in :data:`thesis_pipeline.features.feature_registry
.SET_ID_PATTERN` and never appears in ``feature_sets.xlsx``. The v4
evaluation distinguishes two questions:

* **Absolute model quality** — model vs NAIVE.
* **Incremental sentiment value** — ``ECON_*`` vs ``ECON`` (the primary
  H1 test) — see :func:`thesis_pipeline.evaluation.incremental
  .incremental_sentiment_value_table`.

This module owns the modeling side of the absolute reference: one NAIVE
signal per ``(horizon, model_type, panel_mode, training-window
configuration, coin universe)``. It is generated independently of the
17-set feature grid and runs whether HPO is on or off — NAIVE is never
tuned.

Output naming
-------------
* ``NAIVE.parquet``                                  (per-asset / no window)
* ``NAIVE_panel_pooled.parquet``                     (panel pooled / expanding)
* ``NAIVE_panel_ticker_fe.parquet``                  (panel ticker FE / expanding)
* ``NAIVE_panel_ticker_fe_rw180d.parquet``           (panel ticker FE / rolling 180d)
* ``NAIVE_panel_pooled_rw30.parquet``                (panel pooled / rolling 30 ts)

No HPO suffix is ever appended — NAIVE is by definition untuned.

Metadata on every emitted row
-----------------------------
``set_id = "NAIVE"``,
``sentiment_model = "-"``,
``hpo_enabled = False``,
``hpo_objective = "-"``,
``hpo_variant = "naive"``,
``benchmark_model = "ticker_rolling_probability_with_pooled_fallback"``
(per-asset rows carry ``benchmark_model =
"per_asset_rolling_probability"``).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from .panel_logit import (
    MODEL_TYPE as PANEL_MODEL_TYPE,
    run_panel_rolling_probability,
)
from .run_models import (
    SIGNAL_DIR,
    load_features,
    run_rolling_probability,
)
from .windowing import window_suffix as _window_suffix


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

NAIVE_SET_ID = "NAIVE"
NAIVE_SENTIMENT_MODEL = "-"
NAIVE_HPO_VARIANT = "naive"
NAIVE_BENCHMARK_PANEL = "ticker_rolling_probability_with_pooled_fallback"
NAIVE_BENCHMARK_PER_ASSET = "per_asset_rolling_probability"


def _mode_suffix(panel_mode: str) -> str:
    """Mirror :func:`panel_logit._mode_suffix` without importing the private name."""
    if str(panel_mode) == "ticker_fixed_effects":
        return "panel_ticker_fe"
    if str(panel_mode) == "pooled":
        return "panel_pooled"
    return ""


def naive_output_name(*,
                      model_type: str,
                      panel_mode: str = "-",
                      train_window_mode: str = "expanding",
                      rolling_window_timestamps: int | None = None,
                      rolling_window_days: float | None = None) -> str:
    """Compute the canonical NAIVE output file stem (no `.parquet` suffix).

    Examples
    --------
    ``model_type="panel_logit"`` / ``panel_mode="ticker_fixed_effects"`` /
    ``train_window_mode="rolling_fixed"`` / ``rolling_window_days=180`` ->
    ``"NAIVE_panel_ticker_fe_rw180d"``.

    No HPO suffix is appended.
    """
    parts = [NAIVE_SET_ID]
    if str(model_type) == "panel_logit":
        ms = _mode_suffix(panel_mode)
        if ms:
            parts.append(ms)
    win_suffix = _window_suffix(train_window_mode,
                                 rolling_window_timestamps,
                                 rolling_window_days)
    base = "_".join(parts)
    if win_suffix:
        base = f"{base}{win_suffix}"
    return base


def _attach_naive_metadata(sig: pd.DataFrame,
                           *,
                           horizon: str,
                           model_type: str,
                           panel_mode: str,
                           train_window_mode: str,
                           rolling_window_timestamps: int | None,
                           rolling_window_days: float | None) -> pd.DataFrame:
    """Stamp the canonical NAIVE metadata columns onto a signal frame."""
    out = sig.copy()
    out["set_id"]          = NAIVE_SET_ID
    out["sentiment_model"] = NAIVE_SENTIMENT_MODEL
    out["horizon"]         = horizon
    out["model_type"]      = model_type
    out["panel_mode"]      = panel_mode if str(model_type) == "panel_logit" else "-"
    out["hpo_enabled"]     = False
    out["hpo_objective"]   = "-"
    out["hpo_variant"]     = NAIVE_HPO_VARIANT
    out["train_window_mode"]         = train_window_mode
    out["train_window_timestamps"]   = rolling_window_timestamps
    out["rolling_window_days"]       = rolling_window_days
    # benchmark_model is set inside the generators; default to the panel
    # label so a per-asset row that forgot to set it still parses as NAIVE.
    if "benchmark_model" not in out.columns:
        out["benchmark_model"] = NAIVE_BENCHMARK_PANEL
    return out


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------

def generate_naive_reference(*,
                             horizon: str,
                             model_type: str = "panel_logit",
                             panel_mode: str = "ticker_fixed_effects",
                             train_window_mode: str = "rolling_fixed",
                             rolling_window_timestamps: int | None = None,
                             rolling_window_days: float | None = 180.0,
                             coins: Iterable[str] | None = None,
                             features_df: pd.DataFrame | None = None,
                             output_dir: str | Path = SIGNAL_DIR,
                             resume: bool = True,
                             restart: bool = False) -> Path | None:
    """Generate the NAIVE rolling-probability reference signal once.

    Reads the merged feature parquet for ``horizon`` (or uses ``features_df``
    when supplied for unit tests), runs the rolling-probability path
    matching the given model_type + window configuration, attaches the
    canonical metadata, and writes
    ``output_dir/<horizon>/<naive_output_name(...)>.parquet``.

    Returns the path actually written, or ``None`` if the output already
    exists and ``resume=True``/``restart=False`` (cache hit) — the caller
    can log this without re-running.
    """
    # ── Resolve the output path ────────────────────────────────
    out_dir = Path(output_dir) / str(horizon)
    out_dir.mkdir(parents=True, exist_ok=True)
    name = naive_output_name(
        model_type=model_type, panel_mode=panel_mode,
        train_window_mode=train_window_mode,
        rolling_window_timestamps=rolling_window_timestamps,
        rolling_window_days=rolling_window_days,
    )
    out_path = out_dir / f"{name}.parquet"

    if out_path.exists() and resume and not restart:
        return None  # cache hit; caller decides what to log

    # ── Load features ──────────────────────────────────────────
    df_all = features_df if features_df is not None else load_features(horizon)
    if df_all is None or df_all.empty:
        return None
    if coins is not None:
        wanted = set(t.upper() for t in coins)
        df_all = df_all[df_all["ticker"].astype(str).str.upper().isin(wanted)]
    if df_all.empty:
        return None

    # ── Run the appropriate rolling-probability path ───────────
    if str(model_type) == "panel_logit":
        signals = run_panel_rolling_probability(
            df_all, panel_mode=panel_mode,
            train_window_mode=train_window_mode,
            rolling_window_timestamps=rolling_window_timestamps,
            rolling_window_days=rolling_window_days,
        )
    else:  # per-asset
        rows: list[pd.DataFrame] = []
        for tk, grp in df_all.groupby("ticker"):
            r = run_rolling_probability(grp)
            if not r.empty:
                r["benchmark_model"] = NAIVE_BENCHMARK_PER_ASSET
                rows.append(r)
        signals = (pd.concat(rows, ignore_index=True)
                   if rows else pd.DataFrame())
        # Per-asset path doesn't propagate window metadata — stamp it.
        if not signals.empty:
            signals["train_window_mode"]       = train_window_mode
            signals["train_window_timestamps"] = rolling_window_timestamps

    if signals is None or signals.empty:
        return None

    out = _attach_naive_metadata(
        signals, horizon=horizon, model_type=model_type, panel_mode=panel_mode,
        train_window_mode=train_window_mode,
        rolling_window_timestamps=rolling_window_timestamps,
        rolling_window_days=rolling_window_days,
    )
    out.to_parquet(out_path, index=False)
    return out_path
