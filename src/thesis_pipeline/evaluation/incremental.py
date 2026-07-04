"""Matched economic-benchmark comparison for combined feature sets.

The thesis asks whether sentiment adds **incremental** predictive value beyond
the corresponding economic backbone. The v4 17-set registry holds exactly one
economic-only benchmark — ``ECON`` — and every combined set
(``ECON_VAD_*`` / ``ECON_CBT_*``) shares that backbone identically. The
nested comparison is therefore "did the sentiment block help **on top of**
the ECON core?" and the matched benchmark is always ``ECON``.

This module is a **pure evaluation layer** — no model is retrained. We re-use
the signal frames that ``run-models`` already produced for ``ECON`` and the
combined sets, inner-join them on ``(horizon, timestamp, ticker)`` within
the same model family (same ``model_type`` / ``panel_mode`` / training-
window configuration / HPO variant), and compute the McNemar test plus
accuracy / Brier / log-loss lifts.

This is the **primary H1 path** in v4 (Aufgabe 6.1). The legacy "vs B1 /
vs B2" comparison was removed — ``B1`` is now the NAIVE evaluation reference
(see :mod:`thesis_pipeline.modeling.benchmarks`), not a feature set.
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

#: Matched economic-only benchmark for every combined feature set in the
#: v4 registry. Every ``ECON_{VAD,CBT}_{L,LD,DA,F}`` shares the same
#: ``ECON`` core, so the matched benchmark is identical for the whole
#: combined family — making the incremental-value comparison transparent
#: ("did the sentiment block add information beyond ECON?").
MATCHED_ECONOMIC_BENCHMARK: dict[str, str] = {
    **{f"ECON_VAD_{block}": "ECON" for block in ("L", "LD", "DA", "F")},
    **{f"ECON_CBT_{block}": "ECON" for block in ("L", "LD", "DA", "F")},
}


def matched_economic_benchmark_for_combined(set_id: str) -> str | None:
    """Return the matched economic-only benchmark set_id for a combined set.

    Returns ``None`` when ``set_id`` is not one of the v4 ``ECON_*`` sets,
    so the caller can skip the row silently rather than crash.
    """
    return MATCHED_ECONOMIC_BENCHMARK.get(str(set_id))


# ---------------------------------------------------------------------------
# NAIVE — separate evaluation reference (Aufgabe 6.3)
# ---------------------------------------------------------------------------

#: Canonical label for the historical-majority rolling-probability benchmark.
#: NAIVE is a separate **evaluation reference** in v4 — never a feature set
#: in :data:`~thesis_pipeline.features.feature_registry.SET_ID_PATTERN`. It
#: answers a different question ("does the model beat the no-information
#: rule?") from the H1 nested test ("does sentiment beat ECON?"). The two
#: are reported side by side, never pooled.
NAIVE_REFERENCE_LABEL = "NAIVE"


def is_naive_signal_row(row: pd.Series) -> bool:
    """True iff ``row`` is part of a rolling-probability NAIVE signal frame.

    Two markers count:

    * ``set_id`` is the canonical ``NAIVE`` label, or
    * a ``benchmark_model`` column is present with a
      ``"rolling_probability"`` substring (panel rolling-prob writes
      ``"ticker_rolling_probability_with_pooled_fallback"``; per-asset
      rolling-prob currently writes no such column but is identifiable
      from its set_id label upstream).
    """
    if row is None:
        return False
    if str(row.get("set_id", "")).upper() == NAIVE_REFERENCE_LABEL:
        return True
    bm = str(row.get("benchmark_model", "") or "")
    return "rolling_probability" in bm


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
    "benchmark_hpo_variant",
    "test_role", "hypothesis_family",
    "n_matched",
    "model_accuracy", "benchmark_accuracy", "accuracy_lift",
    "model_balanced_accuracy", "benchmark_balanced_accuracy",
    "balanced_accuracy_lift",
    "model_brier", "benchmark_brier", "brier_improvement",
    "model_log_loss", "benchmark_log_loss", "log_loss_improvement",
    "model_f1", "benchmark_f1", "f1_lift",
    "mcnemar_b", "mcnemar_c", "mcnemar_stat", "mcnemar_p_value",
    "model_correct", "benchmark_correct",
    "interpretation_flag",
    # v4 BH (Aufgabe 6 follow-up B) — populated by the table builder
    # after every row is gathered. Always present (NaN/False for
    # rows that did not enter the BH pool).
    "q_value_bh", "significant_raw_5pct",
    "significant_bh_5pct",
    "interpretation_bh",
    "status",
]

# Test-role tags (Aufgabe 6 follow-up C). The primary H1 incremental-value
# test is ECON_* (the v4 combined sets that share the ECON core) compared
# to ECON. The BH-corrected H1 family contains only these rows. A
# sentiment-only set (SENT_*) compared to ECON would be a *different*
# information-set comparison and does not enter H1.
TEST_ROLE_PRIMARY = "primary_H1_nested"
TEST_ROLE_SECONDARY = "secondary_non_nested"

#: Documented scope of the BH-corrected H1 family. The thesis multiplicity
#: plan groups ALL primary-nested ECON_* vs ECON comparisons across the
#: three horizons into one family, so the BH pool size is 8 sets × 3
#: horizons = 24 tests when every horizon has the full panel. The constant
#: is exposed so tests can assert on it and so external configs can
#: override it (e.g. "per_horizon") with an explicit decision.
H1_BH_SCOPE = "all_primary_h1_tests"


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
        "test_role": TEST_ROLE_PRIMARY,  # only primary rows are emitted here
        "hypothesis_family": "H1_incremental",
        "n_matched": 0,
        "mcnemar_b": 0, "mcnemar_c": 0,
        "model_correct": 0, "benchmark_correct": 0,
        "interpretation_flag": "n/a",
        "significant_raw_5pct": False,
        "significant_bh_5pct":  False,
        "interpretation_bh":    "n/a",
        "status": status,
    })
    return row


# ---------------------------------------------------------------------------
# Shared benchmark pairing (used by the table builder AND the timestamp-level
# Diebold-Mariano forecast-diff layer, so both run on IDENTICAL observations)
# ---------------------------------------------------------------------------

_PAIR_JOIN_KEYS = ["horizon", "timestamp", "ticker"]


def _select_benchmark_group(df: pd.DataFrame, *, horizon, model_type,
                            panel_mode, hpo_variant, bench_set_id,
                            window_key_model):
    """Return ``(bench_grp, chosen_hpo)`` for one combined-set identity.

    Mirrors the historical matching: same horizon / model_type / panel_mode,
    the canonical no-sentiment key, the same training-window configuration,
    and hpo preference ``same > "fixed" > first``. ``bench_grp`` is empty
    (and ``chosen_hpo`` ``None``) when no benchmark matches.
    """
    from .loading import canonical_sentiment_model as _canon_sm
    cand = df[
        (df["horizon"] == horizon)
        & (df["model_type"] == model_type)
        & (df["panel_mode"] == panel_mode)
        & (df["set_id"] == bench_set_id)
        & (df["sentiment_model"].map(_canon_sm) == "-")
    ]
    if not cand.empty:
        matching_groups = [g for _, g in cand.groupby("hpo_variant", dropna=False)
                           if _window_key(g) == window_key_model]
        cand = (pd.concat(matching_groups, ignore_index=False)
                if matching_groups else cand.iloc[0:0])
    if cand.empty:
        return cand.iloc[0:0], None
    variants = list(cand["hpo_variant"].astype(str).unique())
    if str(hpo_variant) in variants:
        chosen_hpo = str(hpo_variant)
    elif "fixed" in variants:
        chosen_hpo = "fixed"
    else:
        chosen_hpo = variants[0]
    return cand[cand["hpo_variant"].astype(str) == chosen_hpo], chosen_hpo


def _merge_prediction_pair(model_grp: pd.DataFrame,
                           bench_grp: pd.DataFrame) -> pd.DataFrame:
    """Strict coverage intersection of model vs benchmark predictions on
    ``(horizon, timestamp, ticker)`` — each side deduped first so the
    nested comparison runs on the candidate's true matched coverage."""
    m = model_grp[_PAIR_JOIN_KEYS + ["target", "prediction", "probability"]].copy()
    b = bench_grp[_PAIR_JOIN_KEYS + ["target", "prediction", "probability"]].rename(
        columns={"prediction": "prediction_b", "probability": "probability_b"})
    b = b.drop(columns=["target"])
    m = m.drop_duplicates(subset=_PAIR_JOIN_KEYS, keep="first")
    b = b.drop_duplicates(subset=_PAIR_JOIN_KEYS, keep="first")
    return m.merge(b, on=_PAIR_JOIN_KEYS, how="inner")


def iter_incremental_prediction_pairs(signals: pd.DataFrame,
                                      benchmark_map: dict[str, str] | None = None):
    """Yield ``(identity, merged)`` for every combined set that matches a
    benchmark, where ``merged`` carries the aligned ``target``,
    ``prediction`` / ``probability`` and ``prediction_b`` / ``probability_b``
    columns on the shared observation intersection.

    This is the single source of truth for the paired predictions, so the
    timestamp-level forecast-diff tests (Part 2A) operate on EXACTLY the
    observations the incremental metrics summarise — guaranteeing e.g. the
    Nt-weighted log-loss effect equals ``log_loss_improvement``.
    """
    if signals is None or signals.empty:
        return
    mapping = dict(benchmark_map or MATCHED_ECONOMIC_BENCHMARK)
    df = _ensure_window_columns(signals)
    combined_mask = df["set_id"].astype(str).isin(mapping.keys())
    if not combined_mask.any():
        return
    group_cols = ["horizon", "set_id", "sentiment_model", "model_type",
                  "panel_mode", "hpo_variant"]
    for keys, model_grp in df[combined_mask].groupby(group_cols, dropna=False):
        horizon, set_id, sm, model_type, panel_mode, hpo_variant = keys
        bench_set_id = mapping.get(str(set_id))
        if bench_set_id is None:
            continue
        bench_grp, chosen_hpo = _select_benchmark_group(
            df, horizon=horizon, model_type=model_type, panel_mode=panel_mode,
            hpo_variant=hpo_variant, bench_set_id=bench_set_id,
            window_key_model=_window_key(model_grp))
        if bench_grp.empty:
            continue
        merged = _merge_prediction_pair(model_grp, bench_grp)
        if merged.empty:
            continue
        yield ({
            "horizon": horizon, "set_id": set_id, "sentiment_model": sm,
            "model_type": model_type, "panel_mode": panel_mode,
            "hpo_variant": hpo_variant, "benchmark_set_id": bench_set_id,
            "benchmark_hpo_variant": chosen_hpo,
        }, merged)


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

        # Same horizon / model_type / panel_mode + benchmark set_id +
        # canonical no-sentiment key + matching training-window config, with
        # hpo preference (same > "fixed" > first). Shared with the
        # forecast-diff layer via ``_select_benchmark_group`` so both run on
        # identical benchmark selection.
        bench_grp, chosen_hpo = _select_benchmark_group(
            df, horizon=horizon, model_type=model_type, panel_mode=panel_mode,
            hpo_variant=hpo_variant, bench_set_id=bench_set_id,
            window_key_model=window_key_model)

        if bench_grp.empty:
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

        # Strict coverage intersection (commit 12 Task B) via the shared
        # merge helper — dedupe each side on the observation key so the
        # nested H1 comparison runs on the candidate's true matched coverage.
        merged = _merge_prediction_pair(model_grp, bench_grp)
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
            # v4 (Aufgabe 6 follow-up C): every row this table emits is by
            # construction a primary H1 nested comparison — only ECON_*
            # sets enter MATCHED_ECONOMIC_BENCHMARK.
            "test_role":         TEST_ROLE_PRIMARY,
            "hypothesis_family": "H1_incremental",
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
            "interpretation_flag":   _classify(acc_lift, mcn_p),
            "significant_raw_5pct":  bool(np.isfinite(mcn_p)
                                          and mcn_p < SIGNIFICANCE_ALPHA),
            "status": "ok",
        })

    out = pd.DataFrame(rows)
    if out.empty:
        return pd.DataFrame(columns=_OUTPUT_COLUMNS)

    # ── Family-aware BH correction (Aufgabe 6 follow-up B) ────────
    # H1_BH_SCOPE = "all_primary_h1_tests" — every valid ok row from every
    # horizon enters one BH pool. Invalid / missing_benchmark / no_overlap
    # rows are excluded.
    out = _apply_bh_within_family(out)

    # Stable column order — write the schema columns first, anything extra last.
    extras = [c for c in out.columns if c not in _OUTPUT_COLUMNS]
    return out[_OUTPUT_COLUMNS + extras].reset_index(drop=True)


def _apply_bh_within_family(df: pd.DataFrame) -> pd.DataFrame:
    """Add ``q_value_bh`` / ``significant_bh_*`` / ``interpretation_bh``.

    Uses :func:`thesis_pipeline.evaluation.diff_in_improvement
    .adjust_pvalues_bh_within_family` so H1 reuses the same family-aware
    helper as H2/H3. Only valid ``status='ok'`` rows enter the BH pool.
    """
    out = df.copy()
    # Default columns to NaN/False/"n/a" for rows that do not enter BH.
    out["q_value_bh"]           = np.nan
    out["significant_bh_5pct"]  = False
    out["interpretation_bh"]    = "n/a"

    valid_mask = (out["status"].astype(str) == "ok") \
                 & out["mcnemar_p_value"].notna()
    if not valid_mask.any():
        return out

    # Delegate to the family-aware utility for the actual BH math.
    from .diff_in_improvement import adjust_pvalues_bh_within_family
    sub = out.loc[valid_mask].copy()
    sub = adjust_pvalues_bh_within_family(
        sub,
        family_col="hypothesis_family",
        p_col="mcnemar_p_value",
        q_col="q_value_bh",
        sig5_col="significant_bh_5pct",
    )
    out.loc[valid_mask, "q_value_bh"]           = sub["q_value_bh"].values
    out.loc[valid_mask, "significant_bh_5pct"]  = sub["significant_bh_5pct"].values
    # BH interpretation mirrors the raw direction but is gated on the BH flag.
    direction = np.sign(out.loc[valid_mask, "accuracy_lift"].fillna(0.0).values)
    bh5 = out.loc[valid_mask, "significant_bh_5pct"].values
    interp = np.where(
        direction > 0,
        np.where(bh5, "improved_bh_significant", "improved_bh_ns"),
        np.where(direction < 0,
                 np.where(bh5, "degraded_bh_significant", "degraded_bh_ns"),
                 "no_change"),
    )
    out.loc[valid_mask, "interpretation_bh"] = interp
    return out
