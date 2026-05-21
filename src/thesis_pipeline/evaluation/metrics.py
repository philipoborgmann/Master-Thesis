"""Pooled + per-ticker metrics + confusion-matrix diagnostics.

The base classification + probabilistic metrics are reused from
:func:`thesis_pipeline.modeling.run_models.compute_metrics` so the
evaluation stage never reimplements them.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..modeling.run_models import compute_metrics

# Group key used everywhere in this stage.
GROUP_KEYS = ("horizon", "set_id", "sentiment_model")


def confusion_diagnostics(signals: pd.DataFrame) -> dict:
    """Return ``TP, TN, FP, FN`` plus positive/negative accuracy.

    Positive accuracy = TP / (TP + FP) — share of correctly-called longs.
    Negative accuracy = TN / (TN + FN) — share of correctly-called shorts.
    NaN where the denominator is zero.
    """
    y_true = signals["target"].astype(int).values
    y_pred = signals["prediction"].astype(int).values
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    pos_den = tp + fp
    neg_den = tn + fn
    return {
        "TP": tp, "TN": tn, "FP": fp, "FN": fn,
        "positive_accuracy": (tp / pos_den) if pos_den > 0 else np.nan,
        "negative_accuracy": (tn / neg_den) if neg_den > 0 else np.nan,
    }


def _group_meta(group: pd.DataFrame) -> dict:
    """Return constant metadata columns from a group (category/label)."""
    out = {}
    for col in ("category", "label"):
        if col in group.columns:
            val = group[col].dropna().astype(str)
            out[col] = val.iloc[0] if not val.empty else ""
        else:
            out[col] = ""
    return out


def pooled_metrics_table(signals: pd.DataFrame) -> pd.DataFrame:
    """One row per (horizon × set_id × sentiment_model).

    Includes classification metrics, probabilistic metrics, n_obs, n_tickers,
    and confusion-matrix diagnostics.
    """
    if signals.empty:
        return pd.DataFrame()
    rows = []
    for keys, grp in signals.groupby(list(GROUP_KEYS), dropna=False):
        if len(grp) == 0:
            continue
        m = compute_metrics(grp, ticker_label="pooled")
        m.update(dict(zip(GROUP_KEYS, keys)))
        m.update(_group_meta(grp))
        m["n_tickers"] = int(grp["ticker"].nunique())
        m.update(confusion_diagnostics(grp))
        rows.append(m)
    out = pd.DataFrame(rows)
    # Standard column order for downstream sorting/sheets.
    front = ["horizon", "set_id", "category", "sentiment_model", "label"]
    rest = [c for c in out.columns if c not in front]
    return out[front + rest].reset_index(drop=True)


def per_ticker_metrics_table(signals: pd.DataFrame) -> pd.DataFrame:
    """One row per (horizon × set_id × sentiment_model × ticker).

    For single-class slices ``precision``, ``recall``, ``f1``, ``log_loss``
    are returned as NaN (per PRD §7C); this is the contract of
    :func:`compute_metrics`.
    """
    if signals.empty:
        return pd.DataFrame()
    rows = []
    keys = list(GROUP_KEYS) + ["ticker"]
    for k, grp in signals.groupby(keys, dropna=False):
        if len(grp) == 0:
            continue
        m = compute_metrics(grp, ticker_label=k[-1])
        m.update(dict(zip(GROUP_KEYS, k[:-1])))
        m["ticker"] = k[-1]
        m.update(_group_meta(grp))
        m.update(confusion_diagnostics(grp))
        rows.append(m)
    out = pd.DataFrame(rows)
    front = ["horizon", "set_id", "category", "sentiment_model", "label", "ticker"]
    rest = [c for c in out.columns if c not in front]
    return out[front + rest].reset_index(drop=True)
