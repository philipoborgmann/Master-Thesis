"""Matched economic-benchmark comparison for combined feature sets.

The thesis asks whether sentiment adds **incremental** predictive value beyond
the corresponding economic backbone. The simple B1 benchmark only answers
"better than a lag-feature baseline". For every combined feature set C1–C6 we
also need the stricter "did sentiment help **on top of** the economic-only
predictors it shares with this combined set" comparison.

This module is a **pure evaluation layer** — no model is retrained. We re-use
the signal frames that ``run-models`` already produced for the economic
benchmark sets E2 / E4 and inner-join them with the combined-set signal frames
on ``(horizon, timestamp, ticker)`` within the same model family (same
``model_type``, ``panel_mode``, training-window configuration and HPO
variant).

The mapping is intentionally explicit (Option A in the spec): the C-set design
is fixed and transparent, and a static dict makes the comparison auditable
without an additional XLSX round-trip.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, brier_score_loss, f1_score, log_loss,
)

from .significance import SIGNIFICANCE_ALPHA, mcnemar_continuity_corrected


# ---------------------------------------------------------------------------
# Mapping
# ---------------------------------------------------------------------------

#: Matched economic-only benchmark for every combined / multi feature set.
#:
#: The combined family pairs sentiment LEVEL with the historically-used
#: benchmark (preserved across the 2026 refactor):
#:
#: * Level 1 (Title Mean)            → B4 Momentum + Volatility
#: * Level 2 (Title + Bullishness)   → B6 Full Economic
#: * Level 3 (Rich Sentiment)        → B6 Full Economic
#:
#: The multi-source family follows the same level→benchmark mapping.
MATCHED_ECONOMIC_BENCHMARK: dict[str, str] = {
    "C1": "B4",
    "C2": "B6",
    "C3": "B6",
    "M1": "B4",
    "M2": "B6",
    "M3": "B6",
}


def matched_economic_benchmark_for_combined(set_id: str) -> str | None:
    """Return the matched economic-only benchmark set_id for a combined set.

    Returns ``None`` when ``set_id`` is not one of C1–C6 so the caller can
    skip the row silently rather than crash.
    """
    return MATCHED_ECONOMIC_BENCHMARK.get(str(set_id))


# ---------------------------------------------------------------------------
# Window-key helper — two runs with identical training-window configuration
# produce identical per-row train_window_timestamps, so the (mode, max) tuple
# uniquely identifies the configuration for matching purposes.
# ---------------------------------------------------------------------------

def _ensure_window_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "train_window_mode" not in out.columns:
        out["train_window_mode"] = "expanding"
    else:
        out["train_window_mode"] = (
            out["train_window_mode"].fillna("expanding").astype(str)
        )
    if "train_window_timestamps" not in out.columns:
        out["train_window_timestamps"] = np.nan
    if "rolling_window_days" not in out.columns:
        out["rolling_window_days"] = np.nan
    return out


def _window_key(frame: pd.DataFrame) -> tuple:
    """Identifier tuple for a frame's configured training window.

    ``(mode, max_train_window_timestamps, rolling_window_days)`` — equal for
    two runs configured the same way, regardless of run order or coin set.
    Expanding runs collapse to ``("expanding", 0, None)`` so any other
    expanding run from the same family matches.
    """
    mode_series = frame["train_window_mode"].dropna().astype(str)
    mode = mode_series.mode().iat[0] if not mode_series.empty else "expanding"
    if mode == "expanding":
        return ("expanding", 0, None)
    sizes = pd.to_numeric(frame["train_window_timestamps"], errors="coerce").dropna()
    max_ts = int(sizes.max()) if not sizes.empty else 0
    days = pd.to_numeric(frame["rolling_window_days"], errors="coerce").dropna()
    days_val = float(days.max()) if not days.empty else None
    return (mode, max_ts, days_val)


# ---------------------------------------------------------------------------
# Metrics on the matched subset
# ---------------------------------------------------------------------------

def _safe(fn: Callable, *args, **kwargs) -> float:
    try:
        return float(fn(*args, **kwargs))
    except Exception:  # noqa: BLE001 — degrade to NaN, never crash the report
        return float("nan")


def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                     y_prob: np.ndarray) -> dict[str, float]:
    if len(y_true) == 0:
        return {k: float("nan") for k in
                ("accuracy", "balanced_accuracy", "brier", "log_loss", "f1")}
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    prob = np.clip(np.asarray(y_prob, dtype=float), 1e-15, 1 - 1e-15)
    two_class = len(np.unique(y_true)) >= 2
    return {
        "accuracy":          _safe(accuracy_score, y_true, y_pred),
        "balanced_accuracy": (_safe(balanced_accuracy_score, y_true, y_pred)
                              if two_class else float("nan")),
        "brier":             _safe(brier_score_loss, y_true, prob),
        "log_loss":          _safe(log_loss, y_true, prob, labels=[0, 1]),
        "f1":                (_safe(f1_score, y_true, y_pred, zero_division=0)
                              if two_class else float("nan")),
    }


def _classify(accuracy_lift: float, mcnemar_p: float) -> str:
    """Short, readable interpretation flag aligned with the lift sign."""
    if not np.isfinite(accuracy_lift):
        return "n/a"
    significant = np.isfinite(mcnemar_p) and mcnemar_p < SIGNIFICANCE_ALPHA
    if accuracy_lift > 0:
        return "improved_significant" if significant else "improved"
    if accuracy_lift < 0:
        return "degraded_significant" if significant else "degraded"
    return "no_change"


# ---------------------------------------------------------------------------
# Identity row (used both for ok rows and missing-benchmark sentinels)
# ---------------------------------------------------------------------------

_OUTPUT_COLUMNS = [
    "horizon", "set_id", "sentiment_model", "model_type", "panel_mode",
    "train_window_mode", "train_window_timestamps", "rolling_window_days",
    "hpo_variant", "benchmark_set_id", "benchmark_sentiment_model",
    "benchmark_hpo_variant", "n_matched",
    "model_accuracy", "benchmark_accuracy", "accuracy_lift",
    "model_balanced_accuracy", "benchmark_balanced_accuracy",
    "balanced_accuracy_lift",
    "model_brier", "benchmark_brier", "brier_improvement",
    "model_log_loss", "benchmark_log_loss", "log_loss_improvement",
    "model_f1", "benchmark_f1", "f1_lift",
    "mcnemar_b", "mcnemar_c", "mcnemar_stat", "mcnemar_p_value",
    "model_correct", "benchmark_correct",
    "interpretation_flag", "status",
]


def _identity_row(*, horizon, set_id, sentiment_model, model_type, panel_mode,
                  hpo_variant, benchmark_set_id, benchmark_hpo_variant,
                  window_mode, window_ts, window_days, status: str) -> dict:
    row = {col: float("nan") for col in _OUTPUT_COLUMNS}
    row.update({
        "horizon": horizon, "set_id": set_id,
        "sentiment_model": sentiment_model,
        "model_type": model_type, "panel_mode": panel_mode,
        "train_window_mode":       window_mode,
        "train_window_timestamps": window_ts,
        "rolling_window_days":     window_days,
        "hpo_variant": hpo_variant,
        "benchmark_set_id": benchmark_set_id,
        "benchmark_sentiment_model": "-",
        "benchmark_hpo_variant": benchmark_hpo_variant,
        "n_matched": 0,
        "mcnemar_b": 0, "mcnemar_c": 0,
        "model_correct": 0, "benchmark_correct": 0,
        "interpretation_flag": "n/a",
        "status": status,
    })
    return row


# ---------------------------------------------------------------------------
# Public table builder
# ---------------------------------------------------------------------------

def incremental_sentiment_value_table(
    signals: pd.DataFrame,
    benchmark_map: dict[str, str] | None = None,
    *,
    warn_missing: Callable[[str, str, str, str], None] | None = None,
) -> pd.DataFrame:
    """One row per (combined model × family) compared to its matched benchmark.

    ``signals`` is the long-form signal frame produced by the evaluation
    loader; it must carry the ``set_id``, ``sentiment_model``, ``model_type``,
    ``panel_mode``, ``hpo_variant``, ``target``, ``prediction`` and
    ``probability`` columns. ``train_window_mode`` /
    ``train_window_timestamps`` / ``rolling_window_days`` default to the
    expanding-window sentinel when absent (e.g. legacy per-asset signals).

    The function never raises on missing benchmark frames — it emits a row
    with ``status='missing_benchmark'`` and, if supplied, calls
    ``warn_missing(horizon, set_id, sentiment_model, benchmark_set_id)`` so
    the orchestrator can log the gap once per family.
    """
    if signals is None or signals.empty:
        return pd.DataFrame(columns=_OUTPUT_COLUMNS)
    mapping = dict(benchmark_map or MATCHED_ECONOMIC_BENCHMARK)
    df = _ensure_window_columns(signals)
    combined_mask = df["set_id"].astype(str).isin(mapping.keys())
    if not combined_mask.any():
        return pd.DataFrame(columns=_OUTPUT_COLUMNS)

    group_cols = ["horizon", "set_id", "sentiment_model", "model_type",
                  "panel_mode", "hpo_variant"]
    rows: list[dict] = []

    for keys, model_grp in df[combined_mask].groupby(group_cols, dropna=False):
        horizon, set_id, sm, model_type, panel_mode, hpo_variant = keys
        bench_set_id = mapping.get(str(set_id))
        if bench_set_id is None:
            continue

        window_key_model = _window_key(model_grp)
        window_mode_label = window_key_model[0]
        window_ts_label   = (None if window_key_model[1] == 0
                             else int(window_key_model[1]))
        window_days_label = window_key_model[2]

        # Same horizon / model_type / panel_mode + benchmark set_id + sentiment_model "-"
        cand = df[
            (df["horizon"] == horizon)
            & (df["model_type"] == model_type)
            & (df["panel_mode"] == panel_mode)
            & (df["set_id"] == bench_set_id)
            & (df["sentiment_model"].astype(str) == "-")
        ]
        # Filter to the same training-window configuration.
        if not cand.empty:
            matching_groups = [g for _, g in cand.groupby("hpo_variant", dropna=False)
                               if _window_key(g) == window_key_model]
            cand = (pd.concat(matching_groups, ignore_index=False)
                    if matching_groups else cand.iloc[0:0])

        if cand.empty:
            row = _identity_row(
                horizon=horizon, set_id=set_id, sentiment_model=sm,
                model_type=model_type, panel_mode=panel_mode,
                hpo_variant=hpo_variant, benchmark_set_id=bench_set_id,
                benchmark_hpo_variant=None,
                window_mode=window_mode_label,
                window_ts=window_ts_label, window_days=window_days_label,
                status="missing_benchmark")
            rows.append(row)
            if warn_missing is not None:
                warn_missing(str(horizon), str(set_id), str(sm), str(bench_set_id))
            continue

        # Choose benchmark hpo_variant: same variant > "fixed" > anything else.
        variants = list(cand["hpo_variant"].astype(str).unique())
        if str(hpo_variant) in variants:
            chosen_hpo = str(hpo_variant)
        elif "fixed" in variants:
            chosen_hpo = "fixed"
        else:
            chosen_hpo = variants[0]
        bench_grp = cand[cand["hpo_variant"].astype(str) == chosen_hpo]

        keys_join = ["horizon", "timestamp", "ticker"]
        m = model_grp[keys_join + ["target", "prediction", "probability"]].copy()
        b = bench_grp[keys_join + ["target", "prediction", "probability"]].rename(
            columns={"prediction": "prediction_b", "probability": "probability_b"})
        # The benchmark target is by construction identical to the model target
        # at the same (horizon, timestamp, ticker), so the merge picks one copy.
        b = b.drop(columns=["target"])
        merged = m.merge(b, on=keys_join, how="inner")
        n_matched = int(len(merged))

        if n_matched == 0:
            rows.append(_identity_row(
                horizon=horizon, set_id=set_id, sentiment_model=sm,
                model_type=model_type, panel_mode=panel_mode,
                hpo_variant=hpo_variant, benchmark_set_id=bench_set_id,
                benchmark_hpo_variant=chosen_hpo,
                window_mode=window_mode_label,
                window_ts=window_ts_label, window_days=window_days_label,
                status="no_overlap"))
            continue

        m_metrics = _compute_metrics(
            merged["target"].values, merged["prediction"].values,
            merged["probability"].values,
        )
        b_metrics = _compute_metrics(
            merged["target"].values, merged["prediction_b"].values,
            merged["probability_b"].values,
        )
        correct_m = (merged["prediction"].astype(int).values
                     == merged["target"].astype(int).values).astype(int)
        correct_b = (merged["prediction_b"].astype(int).values
                     == merged["target"].astype(int).values).astype(int)
        b_count = int(((correct_b == 1) & (correct_m == 0)).sum())
        c_count = int(((correct_b == 0) & (correct_m == 1)).sum())
        mcn_stat, mcn_p = mcnemar_continuity_corrected(b_count, c_count)
        acc_lift = m_metrics["accuracy"] - b_metrics["accuracy"]

        rows.append({
            "horizon": horizon, "set_id": set_id, "sentiment_model": sm,
            "model_type": model_type, "panel_mode": panel_mode,
            "train_window_mode":       window_mode_label,
            "train_window_timestamps": window_ts_label,
            "rolling_window_days":     window_days_label,
            "hpo_variant": hpo_variant,
            "benchmark_set_id":      bench_set_id,
            "benchmark_sentiment_model": "-",
            "benchmark_hpo_variant": chosen_hpo,
            "n_matched": n_matched,
            "model_accuracy":            m_metrics["accuracy"],
            "benchmark_accuracy":        b_metrics["accuracy"],
            "accuracy_lift":             acc_lift,
            "model_balanced_accuracy":   m_metrics["balanced_accuracy"],
            "benchmark_balanced_accuracy": b_metrics["balanced_accuracy"],
            "balanced_accuracy_lift":    (m_metrics["balanced_accuracy"]
                                          - b_metrics["balanced_accuracy"]),
            "model_brier":               m_metrics["brier"],
            "benchmark_brier":           b_metrics["brier"],
            "brier_improvement":         b_metrics["brier"] - m_metrics["brier"],
            "model_log_loss":            m_metrics["log_loss"],
            "benchmark_log_loss":        b_metrics["log_loss"],
            "log_loss_improvement":      b_metrics["log_loss"] - m_metrics["log_loss"],
            "model_f1":                  m_metrics["f1"],
            "benchmark_f1":              b_metrics["f1"],
            "f1_lift":                   m_metrics["f1"] - b_metrics["f1"],
            "mcnemar_b": b_count, "mcnemar_c": c_count,
            "mcnemar_stat": float(mcn_stat), "mcnemar_p_value": float(mcn_p),
            "model_correct": int(correct_m.sum()),
            "benchmark_correct": int(correct_b.sum()),
            "interpretation_flag": _classify(acc_lift, mcn_p),
            "status": "ok",
        })

    out = pd.DataFrame(rows)
    if out.empty:
        return pd.DataFrame(columns=_OUTPUT_COLUMNS)
    # Stable column order — write the schema columns first, anything extra last.
    extras = [c for c in out.columns if c not in _OUTPUT_COLUMNS]
    return out[_OUTPUT_COLUMNS + extras].reset_index(drop=True)
