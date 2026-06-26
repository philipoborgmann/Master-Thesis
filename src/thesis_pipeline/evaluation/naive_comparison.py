"""Absolute vs NAIVE evaluation (v4 cleanup commit 2 Section F).

This module produces the ``absolute_vs_naive`` evaluation layer: for
every model run we compare its observation-level predictions against
the matched NAIVE reference produced by
:mod:`thesis_pipeline.modeling.naive_reference`. The comparison is
**absolute** ("does the model beat the no-information rolling-
probability rule?"), independent of and complementary to the H1
nested-incremental test (``ECON_*`` vs ``ECON``).

Matching is on the COMPLETE NAIVE identity:

  (horizon, model_type, panel_mode,
   train_window_mode, rolling_window_days, rolling_window_timestamps,
   coin_universe_hash)

This ensures a panel-logit ticker-FE rolling-180d model run on the
``{BTC, ETH}`` universe is only compared against the NAIVE built for
that same combination — never against an expanding NAIVE or a NAIVE
built for a different ticker set. The
:func:`thesis_pipeline.modeling.naive_reference.coin_universe_hash`
helper is the source of truth for the hash.

Output schema
-------------
* CSV  : ``Outputs/Evaluation/absolute_vs_naive.csv``
* Excel: workbook sheet ``absolute_vs_naive``

Each row is one (model run × NAIVE match) pair. Metric conventions
(applied uniformly so the sign reads "model is better when the column
is positive"):

* ``accuracy_lift        = model − NAIVE``
* ``brier_improvement    = NAIVE − model``
* ``log_loss_improvement = NAIVE − model``

The frame additionally carries the hypothesis-family tag
``"absolute_vs_naive"`` so downstream BH correction (if ever applied)
NEVER pools it with the H1 / H2 / H3 families.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

from ..logging_utils import get_logger
from ..modeling.naive_reference import (
    NAIVE_SET_ID,
    coin_universe_hash,
)
from .significance import mcnemar_continuity_corrected, SIGNIFICANCE_ALPHA


HYPOTHESIS_FAMILY = "absolute_vs_naive"

#: COMPLETE NAIVE identity tuple — every column must match between the
#: model row and the NAIVE row. Mirrors
#: :data:`thesis_pipeline.evaluation.diff_in_improvement
#: .H2H3_FAMILY_COLUMNS` minus HPO (NAIVE is by construction untuned).
NAIVE_IDENTITY_COLUMNS = (
    "horizon",
    "model_type",
    "panel_mode",
    "train_window_mode",
    "rolling_window_days",
    "rolling_window_timestamps",
    "coin_universe_hash",
)


OUTPUT_COLUMNS = [
    "hypothesis_family",
    "horizon", "set_id", "sentiment_model",
    "model_type", "panel_mode", "hpo_variant", "hpo_objective",
    "train_window_mode", "rolling_window_days", "rolling_window_timestamps",
    "coin_universe_hash",
    # NAIVE side identity
    "naive_set_id", "naive_coin_universe_hash",
    # Match diagnostics
    "n_model", "n_naive", "n_matched",
    "n_duplicate_model_keys", "n_duplicate_naive_keys",
    "targets_identical",
    # Metrics
    "model_accuracy", "naive_accuracy", "accuracy_lift",
    "model_brier", "naive_brier", "brier_improvement",
    "model_log_loss", "naive_log_loss", "log_loss_improvement",
    # McNemar vs NAIVE
    "mcnemar_b", "mcnemar_c",
    "mcnemar_stat", "mcnemar_p_value",
    "significant_raw_5pct",
    "skip_reason",
    "status",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe(fn, *args, **kwargs) -> float:
    try:
        return float(fn(*args, **kwargs))
    except Exception:  # noqa: BLE001
        return float("nan")


def _compute_basic_metrics(y_true, y_pred, y_prob) -> dict[str, float]:
    from sklearn.metrics import accuracy_score, brier_score_loss, log_loss
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    prob = np.clip(np.asarray(y_prob, dtype=float), 1e-15, 1 - 1e-15)
    two_class = len(np.unique(y_true)) >= 2
    return {
        "accuracy": _safe(accuracy_score, y_true, y_pred),
        "brier":    _safe(brier_score_loss, y_true, prob) if two_class else float("nan"),
        "log_loss": _safe(log_loss, y_true, prob, labels=[0, 1]) if two_class else float("nan"),
    }


def _ensure_identity_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in NAIVE_IDENTITY_COLUMNS:
        if col not in out.columns:
            out[col] = np.nan
    return out


def _coin_universe_hash_from_signals(grp: pd.DataFrame) -> str | None:
    """Resolve the canonical coin-universe hash for one model group.

    Preference order:

    1. A non-empty ``coin_universe_hash`` column on the group (the
       NAIVE generator stamps this, so panel runs that re-used NAIVE
       coins keep the hash).
    2. The set of ``ticker`` values seen on the group, hashed via the
       canonical helper.

    Returns ``None`` when neither is resolvable.
    """
    if "coin_universe_hash" in grp.columns:
        vals = grp["coin_universe_hash"].dropna().astype(str)
        vals = vals[vals.str.len() > 0]
        if not vals.empty:
            return vals.iat[0]
    tickers = grp["ticker"].dropna().astype(str).unique() if "ticker" in grp.columns else []
    if len(tickers) == 0:
        return None
    return coin_universe_hash(list(tickers))


def _empty_output() -> pd.DataFrame:
    return pd.DataFrame(columns=OUTPUT_COLUMNS)


# ---------------------------------------------------------------------------
# Table builder
# ---------------------------------------------------------------------------

def absolute_vs_naive_table(signals: pd.DataFrame) -> pd.DataFrame:
    """One row per (model run × matched NAIVE) pair.

    The function is a pure evaluation layer: it consumes the long-form
    signal frame already loaded by :mod:`thesis_pipeline.evaluation
    .evaluate_signals` and produces a tidy frame for CSV + Excel output.

    Behaviour
    ---------
    * Model rows are every group whose ``set_id != NAIVE`` AND
      ``hpo_variant != "naive"``. NAIVE rows are not compared against
      themselves (that would be the identity comparison).
    * Each model group is identified by the v4 group key
      ``(horizon, set_id, sentiment_model, model_type, panel_mode,
        hpo_variant)``. Its NAIVE counterpart must share the full
      NAIVE identity (excluding HPO/sentiment/set_id, which never apply
      to NAIVE).
    * When several NAIVE candidates remain after identity filtering the
      function picks the one with the largest matched-row count.
    * The model-vs-NAIVE inner join is on ``(ticker, timestamp)``; the
      function verifies target equality on the matched subset and
      records ``n_duplicate_*_keys`` to surface any (ticker, timestamp)
      duplicates within either side.
    """
    if signals is None or signals.empty:
        return _empty_output()

    df = _ensure_identity_columns(signals)
    if "set_id" not in df.columns:
        return _empty_output()

    naive_mask = df["set_id"].astype(str) == NAIVE_SET_ID
    if not naive_mask.any():
        get_logger().info(
            "absolute_vs_naive: no NAIVE rows present — skipping "
            "absolute_vs_naive table (NAIVE generation may be disabled "
            "for this run)."
        )
        return _empty_output()

    naive_df = df[naive_mask].copy()
    model_df = df[~naive_mask].copy()
    # NAIVE never participates in itself; HPO models that are tagged
    # ``hpo_variant=naive`` are by construction NAIVE rows so they were
    # already filtered above, but we guard belt-and-braces.
    if "hpo_variant" in model_df.columns:
        model_df = model_df[
            model_df["hpo_variant"].astype(str).str.lower() != "naive"
        ]
    if model_df.empty:
        return _empty_output()

    model_group_cols = [
        "horizon", "set_id", "sentiment_model",
        "model_type", "panel_mode", "hpo_variant",
    ]
    rows: list[dict] = []
    for keys, m_grp in model_df.groupby(model_group_cols, dropna=False):
        horizon, set_id, sm, model_type, panel_mode, hpo_variant = keys

        # Resolve the model's universe hash + window identity tuple.
        m_hash = _coin_universe_hash_from_signals(m_grp)
        ident_values = {
            "horizon":                  horizon,
            "model_type":               model_type,
            "panel_mode":               panel_mode,
            "train_window_mode":        _first_or_nan(m_grp, "train_window_mode"),
            "rolling_window_days":      _first_or_nan(m_grp, "rolling_window_days"),
            "rolling_window_timestamps": _first_or_nan(m_grp, "rolling_window_timestamps"),
            "coin_universe_hash":       m_hash,
        }

        base = _identity_row(
            horizon=horizon, set_id=set_id, sentiment_model=sm,
            model_type=model_type, panel_mode=panel_mode,
            hpo_variant=hpo_variant,
            hpo_objective=_first_or_nan(m_grp, "hpo_objective"),
            train_window_mode=ident_values["train_window_mode"],
            rolling_window_days=ident_values["rolling_window_days"],
            rolling_window_timestamps=ident_values["rolling_window_timestamps"],
            coin_universe_hash=m_hash,
            n_model=int(len(m_grp)),
        )

        # ── NAIVE candidate filter on full identity ───────────────
        cand_mask = pd.Series(True, index=naive_df.index)
        for col in NAIVE_IDENTITY_COLUMNS:
            cand_mask &= _series_eq_nan_safe(naive_df[col], ident_values.get(col))
        cand = naive_df[cand_mask]
        base["n_naive"] = int(len(cand))
        if cand.empty:
            base["status"] = "missing_naive"
            base["skip_reason"] = "no_matched_naive_identity"
            rows.append(base)
            continue

        # ── Duplicate-key guard ──────────────────────────────────
        dup_model = int(m_grp.duplicated(subset=["ticker", "timestamp"]).sum())
        base["n_duplicate_model_keys"] = dup_model

        # If multiple NAIVE candidates remain (e.g. the file was
        # produced by several runs sharing the identity), pick the one
        # with the most matched observations to maximise statistical
        # power. Duplicate-key counts are recorded per chosen NAIVE.
        best: dict | None = None
        for naive_hash, naive_subset in cand.groupby("coin_universe_hash",
                                                       dropna=False):
            dup_naive = int(naive_subset.duplicated(
                subset=["ticker", "timestamp"]).sum())
            joined = _inner_join_on_keys(m_grp, naive_subset)
            if joined.empty:
                if best is None:
                    best = {"naive_hash": naive_hash, "joined": joined,
                            "dup_naive": dup_naive, "n_naive": int(len(naive_subset))}
                continue
            if best is None or len(joined) > len(best["joined"]):
                best = {"naive_hash": naive_hash, "joined": joined,
                        "dup_naive": dup_naive, "n_naive": int(len(naive_subset))}

        if best is None:
            base["status"] = "missing_naive"
            base["skip_reason"] = "no_matched_naive_identity"
            rows.append(base)
            continue

        joined = best["joined"]
        base["naive_coin_universe_hash"] = best["naive_hash"]
        base["n_duplicate_naive_keys"]   = best["dup_naive"]
        base["n_naive"]                   = best["n_naive"]
        base["n_matched"]                 = int(len(joined))

        if dup_model or best["dup_naive"]:
            base["skip_reason"] = "duplicate_keys_within_identity"
            base["status"]      = "duplicate_keys"
            rows.append(base)
            continue
        if joined.empty:
            base["skip_reason"] = "no_overlap"
            base["status"]      = "no_overlap"
            rows.append(base)
            continue

        # ── Target-equality verification ─────────────────────────
        targets_identical = bool(
            (joined["target_m"].astype(int).values
             == joined["target_n"].astype(int).values).all()
        )
        base["targets_identical"] = targets_identical
        if not targets_identical:
            base["skip_reason"] = "target_mismatch"
            base["status"]      = "target_mismatch"
            rows.append(base)
            continue

        # ── Metrics + McNemar vs NAIVE ───────────────────────────
        y      = joined["target_m"].astype(int).values
        m_pred = joined["prediction_m"].astype(int).values
        n_pred = joined["prediction_n"].astype(int).values
        m_prob = joined["probability_m"].astype(float).values
        n_prob = joined["probability_n"].astype(float).values

        m_metrics = _compute_basic_metrics(y, m_pred, m_prob)
        n_metrics = _compute_basic_metrics(y, n_pred, n_prob)

        correct_m = (m_pred == y).astype(int)
        correct_n = (n_pred == y).astype(int)
        b = int(((correct_n == 1) & (correct_m == 0)).sum())
        c = int(((correct_n == 0) & (correct_m == 1)).sum())
        mcn_stat, mcn_p = mcnemar_continuity_corrected(b, c)

        base.update({
            "model_accuracy":       m_metrics["accuracy"],
            "naive_accuracy":       n_metrics["accuracy"],
            "accuracy_lift":        m_metrics["accuracy"] - n_metrics["accuracy"],
            "model_brier":          m_metrics["brier"],
            "naive_brier":          n_metrics["brier"],
            "brier_improvement":    n_metrics["brier"] - m_metrics["brier"],
            "model_log_loss":       m_metrics["log_loss"],
            "naive_log_loss":       n_metrics["log_loss"],
            "log_loss_improvement": n_metrics["log_loss"] - m_metrics["log_loss"],
            "mcnemar_b":            b,
            "mcnemar_c":            c,
            "mcnemar_stat":         float(mcn_stat),
            "mcnemar_p_value":      float(mcn_p),
            "significant_raw_5pct": bool(np.isfinite(mcn_p)
                                          and mcn_p < SIGNIFICANCE_ALPHA),
            "status":               "ok",
            "skip_reason":          "",
        })
        rows.append(base)

    if not rows:
        return _empty_output()
    out = pd.DataFrame(rows)
    extras = [c for c in out.columns if c not in OUTPUT_COLUMNS]
    return out[OUTPUT_COLUMNS + extras].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------

def _inner_join_on_keys(model_grp: pd.DataFrame,
                        naive_grp: pd.DataFrame) -> pd.DataFrame:
    """Inner-join model vs NAIVE on ``(ticker, timestamp)``.

    Keeps the two sides' ``target`` and ``prediction`` / ``probability``
    columns under disambiguated suffixes so the caller can verify
    target equality and compute metrics in one pass.
    """
    base_cols = ["ticker", "timestamp", "target", "prediction", "probability"]
    m = model_grp[base_cols].copy()
    n = naive_grp[base_cols].copy()
    m["timestamp"] = pd.to_datetime(m["timestamp"], utc=True, errors="coerce")
    n["timestamp"] = pd.to_datetime(n["timestamp"], utc=True, errors="coerce")
    m["ticker"] = m["ticker"].astype(str).str.upper()
    n["ticker"] = n["ticker"].astype(str).str.upper()
    return m.merge(
        n, on=["ticker", "timestamp"], how="inner",
        suffixes=("_m", "_n"),
    )


def _identity_row(*,
                  horizon, set_id, sentiment_model,
                  model_type, panel_mode, hpo_variant,
                  hpo_objective,
                  train_window_mode, rolling_window_days,
                  rolling_window_timestamps,
                  coin_universe_hash,
                  n_model: int) -> dict:
    row = {col: None for col in OUTPUT_COLUMNS}
    for k in ("accuracy_lift", "brier_improvement", "log_loss_improvement",
              "model_accuracy", "naive_accuracy",
              "model_brier", "naive_brier",
              "model_log_loss", "naive_log_loss",
              "mcnemar_stat", "mcnemar_p_value"):
        row[k] = float("nan")
    row["hypothesis_family"]  = HYPOTHESIS_FAMILY
    row["horizon"]            = horizon
    row["set_id"]             = set_id
    row["sentiment_model"]    = sentiment_model
    row["model_type"]         = model_type
    row["panel_mode"]         = panel_mode
    row["hpo_variant"]        = hpo_variant
    row["hpo_objective"]      = hpo_objective
    row["train_window_mode"]  = train_window_mode
    row["rolling_window_days"] = rolling_window_days
    row["rolling_window_timestamps"] = rolling_window_timestamps
    row["coin_universe_hash"] = coin_universe_hash
    row["naive_set_id"]       = NAIVE_SET_ID
    row["naive_coin_universe_hash"] = None
    row["n_model"]            = n_model
    row["n_naive"]            = 0
    row["n_matched"]          = 0
    row["n_duplicate_model_keys"] = 0
    row["n_duplicate_naive_keys"] = 0
    row["targets_identical"]  = False
    row["mcnemar_b"]          = 0
    row["mcnemar_c"]          = 0
    row["significant_raw_5pct"] = False
    row["status"]             = "missing_naive"
    row["skip_reason"]        = ""
    return row


def _first_or_nan(grp: pd.DataFrame, col: str):
    if col not in grp.columns:
        return float("nan")
    vals = grp[col].dropna()
    if vals.empty:
        return float("nan")
    return vals.iat[0]


def _series_eq_nan_safe(s: pd.Series, value) -> pd.Series:
    """Elementwise equality treating NaN==NaN as match."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return s.isna() | (s.astype(str).str.lower() == "nan")
    try:
        return s == value
    except Exception:  # noqa: BLE001
        return s.astype(str) == str(value)
