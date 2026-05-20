"""Metrics — identical to Run_Models.compute_metrics."""

from __future__ import annotations

import numpy as np
import pandas as pd


def compute_metrics(y_true: pd.Series | np.ndarray,
                    y_pred: pd.Series | np.ndarray,
                    y_proba: pd.Series | np.ndarray) -> dict:
    from sklearn.metrics import (
        accuracy_score, balanced_accuracy_score, f1_score,
        precision_score, recall_score, log_loss, brier_score_loss,
    )
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    y_proba = np.clip(np.asarray(y_proba), 1e-15, 1 - 1e-15)
    return {
        "accuracy":          accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "f1":                f1_score(y_true, y_pred, zero_division=0),
        "precision":         precision_score(y_true, y_pred, zero_division=0),
        "recall":            recall_score(y_true, y_pred, zero_division=0),
        "log_loss":          log_loss(y_true, y_proba, labels=[0, 1]),
        "brier_score":       brier_score_loss(y_true, y_proba),
    }
