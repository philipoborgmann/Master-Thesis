"""High-conviction / threshold analysis.

For each threshold ``t`` ∈ {0.50, 0.55, 0.60, 0.65}:

* a *long*  trade is taken when ``p > t``
* a *short* trade is taken when ``p < 1 - t``
* the middle band is *no trade*

For ``t = 0.50`` there is no neutral band (every observation is traded) so
coverage is always 1.0.

Metrics (computed on the traded subset only):

* accuracy, precision, recall, f1, brier_score
* coverage = n_trades / n_obs
* n_trades
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score, brier_score_loss,
)

from .metrics import GROUP_KEYS, _group_meta

DEFAULT_THRESHOLDS: tuple[float, ...] = (0.50, 0.55, 0.60, 0.65)


def _trade_mask(prob: np.ndarray, threshold: float) -> np.ndarray:
    """Boolean mask of traded observations at a given threshold."""
    if threshold <= 0.5 + 1e-12:
        # Every observation is traded; the default sklearn rule applies
        # (``predict 1`` when p >= 0.5, else 0).
        return np.ones_like(prob, dtype=bool)
    return (prob > threshold) | (prob < (1.0 - threshold))


def _predictions_at_threshold(prob: np.ndarray, threshold: float) -> np.ndarray:
    if threshold <= 0.5 + 1e-12:
        return (prob >= 0.5).astype(int)
    return (prob > threshold).astype(int)  # only meaningful where traded


def threshold_metrics_for_group(group: pd.DataFrame,
                                thresholds: tuple[float, ...] = DEFAULT_THRESHOLDS
                                ) -> pd.DataFrame:
    """One row per threshold for a single (horizon, set_id, sentiment_model) slice."""
    if group.empty:
        return pd.DataFrame()
    prob = np.clip(group["probability"].astype(float).values, 1e-15, 1 - 1e-15)
    y    = group["target"].astype(int).values
    n_obs = int(len(group))
    rows = []
    for t in thresholds:
        mask = _trade_mask(prob, t)
        n_trades = int(mask.sum())
        coverage = float(n_trades / n_obs) if n_obs else 0.0
        if n_trades == 0:
            rows.append({
                "threshold": t, "n_obs": n_obs, "n_trades": 0,
                "coverage": coverage,
                "accuracy": np.nan, "precision": np.nan, "recall": np.nan,
                "f1": np.nan, "brier_score": np.nan,
            })
            continue
        y_tr  = y[mask]
        p_tr  = prob[mask]
        pred  = _predictions_at_threshold(prob, t)[mask]
        acc   = accuracy_score(y_tr, pred)
        # precision/recall/f1 require both classes in y_tr for non-trivial values;
        # zero_division=0 keeps the call defensive.
        if len(np.unique(y_tr)) < 2:
            prec = rec = f1v = np.nan
        else:
            prec = precision_score(y_tr, pred, zero_division=0)
            rec  = recall_score(y_tr, pred, zero_division=0)
            f1v  = f1_score(y_tr, pred, zero_division=0)
        # Brier is computed against the predicted probability of the long class.
        brier = float(brier_score_loss(y_tr, p_tr))
        rows.append({
            "threshold": t, "n_obs": n_obs, "n_trades": n_trades,
            "coverage": round(coverage, 6),
            "accuracy":  round(acc, 6),
            "precision": round(prec, 6) if not np.isnan(prec) else np.nan,
            "recall":    round(rec, 6)  if not np.isnan(rec)  else np.nan,
            "f1":        round(f1v, 6)  if not np.isnan(f1v)  else np.nan,
            "brier_score": round(brier, 6),
        })
    return pd.DataFrame(rows)


def threshold_analysis_table(signals: pd.DataFrame,
                             thresholds: tuple[float, ...] = DEFAULT_THRESHOLDS
                             ) -> pd.DataFrame:
    """Stack threshold metrics across every (horizon, set_id, sentiment_model)."""
    if signals.empty:
        return pd.DataFrame()
    rows = []
    for keys, grp in signals.groupby(list(GROUP_KEYS), dropna=False):
        tdf = threshold_metrics_for_group(grp, thresholds)
        if tdf.empty:
            continue
        for c, v in zip(GROUP_KEYS, keys):
            tdf[c] = v
        meta = _group_meta(grp)
        for c, v in meta.items():
            tdf[c] = v
        rows.append(tdf)
    if not rows:
        return pd.DataFrame()
    out = pd.concat(rows, ignore_index=True)
    front = ["horizon", "set_id", "category", "sentiment_model", "label", "threshold"]
    rest = [c for c in out.columns if c not in front]
    return out[front + rest].reset_index(drop=True)
