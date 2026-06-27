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


#: COMPLETE model identity used to group signals before comparing to
#: NAIVE (commit 3 Section C). Every column listed MUST be constant
#: within a model group; otherwise the row is flagged
#: ``invalid_model_identity`` and ``_first_or_nan`` is never used to
#: silently collapse divergent values.
MODEL_GROUP_COLUMNS = [
    "horizon",
    "set_id",
    "sentiment_model",
    "model_type",
    "panel_mode",
    "hpo_variant",
    "hpo_objective",
    "train_window_mode",
    "rolling_window_days",
    "rolling_window_timestamps",
    "coin_universe_hash",
]


OUTPUT_COLUMNS = [
    "hypothesis_family",
    "benchmark_role",
    "benchmark_set_id",
    "horizon", "set_id", "sentiment_model",
    "model_type", "panel_mode", "hpo_variant", "hpo_objective",
    "train_window_mode", "rolling_window_days", "rolling_window_timestamps",
    "coin_universe_hash",
    # NAIVE side identity
    "naive_set_id", "naive_coin_universe_hash",
    # Universe identity provenance (commit 3 Section B / C).
    "universe_identity_source",
    # Match diagnostics
    "n_model", "n_naive", "n_matched",
    "n_unmatched_model", "n_unmatched_naive",
    "n_duplicate_model_keys", "n_duplicate_naive_keys",
    "targets_identical",
    # Metrics — sign conventions: positive ⇒ model is better than NAIVE.
    "model_accuracy", "naive_accuracy", "accuracy_lift",
    "model_balanced_accuracy", "naive_balanced_accuracy",
    "balanced_accuracy_lift",
    "model_brier", "naive_brier", "brier_improvement",
    "model_log_loss", "naive_log_loss", "log_loss_improvement",
    # McNemar vs NAIVE
    "mcnemar_b", "mcnemar_c",
    "mcnemar_stat", "mcnemar_p_value",
    "significant_raw_5pct",
    # Diagnostic columns reporting why a row was skipped.
    "non_constant_columns",
    "skip_reason",
    "status",
]


#: Identifier written into ``benchmark_role`` on every row so the column
#: reads as self-documenting without consulting a separate dictionary.
ABSOLUTE_BENCHMARK_ROLE = "absolute_reference"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe(fn, *args, **kwargs) -> float:
    try:
        return float(fn(*args, **kwargs))
    except Exception:  # noqa: BLE001
        return float("nan")


def _compute_basic_metrics(y_true, y_pred, y_prob) -> dict[str, float]:
    from sklearn.metrics import (
        accuracy_score, balanced_accuracy_score,
        brier_score_loss, log_loss,
    )
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    prob = np.clip(np.asarray(y_prob, dtype=float), 1e-15, 1 - 1e-15)
    two_class = len(np.unique(y_true)) >= 2
    return {
        "accuracy":          _safe(accuracy_score, y_true, y_pred),
        "balanced_accuracy": (_safe(balanced_accuracy_score, y_true, y_pred)
                              if two_class else float("nan")),
        "brier":             _safe(brier_score_loss, y_true, prob) if two_class else float("nan"),
        "log_loss":          _safe(log_loss, y_true, prob, labels=[0, 1]) if two_class else float("nan"),
    }


def _ensure_identity_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in NAIVE_IDENTITY_COLUMNS:
        if col not in out.columns:
            out[col] = np.nan
    return out


def _coin_universe_hash_from_signals(grp: pd.DataFrame) -> tuple[str | None, str]:
    """Resolve the canonical coin-universe hash for one model group AND
    record where it came from (commit 3 Section B / C).

    Preference order:

    1. A non-empty ``coin_universe_hash`` column on the group (the v4
       generator stamps this, so production runs carry the requested-
       universe hash). ``universe_identity_source = "requested_metadata"``.
    2. The set of ``ticker`` values seen on the group, hashed via the
       canonical helper. ``universe_identity_source =
       "legacy_realized_tickers_fallback"``.

    Returns ``(hash, source)`` — ``(None, source)`` when neither is
    resolvable.
    """
    if "coin_universe_hash" in grp.columns:
        vals = grp["coin_universe_hash"].dropna().astype(str)
        vals = vals[vals.str.len() > 0]
        if not vals.empty:
            return vals.iat[0], "requested_metadata"
    tickers = grp["ticker"].dropna().astype(str).unique() if "ticker" in grp.columns else []
    if len(tickers) == 0:
        return None, "legacy_realized_tickers_fallback"
    return coin_universe_hash(list(tickers)), "legacy_realized_tickers_fallback"


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

    Behaviour (commit 3 Sections C + D)
    -----------------------------------
    * Model rows are every group whose ``set_id != NAIVE`` AND
      ``hpo_variant != "naive"``. NAIVE rows are not compared against
      themselves.
    * Each model group is identified by
      :data:`MODEL_GROUP_COLUMNS` — the COMPLETE run identity. This
      includes window mode + window size, HPO objective, and the
      requested-universe hash. ``_first_or_nan`` is NEVER used to
      silently collapse divergent values inside one group: every
      identity column must be constant within a group, otherwise the
      row is emitted with ``status="invalid_model_identity"`` and
      ``skip_reason="non_constant_identity_columns"`` (listing the
      offending columns).
    * NAIVE candidates must match EXACTLY on every column in
      :data:`NAIVE_IDENTITY_COLUMNS`. If multiple NAIVE groups remain
      after this filter the row is flagged ``ambiguous_naive_identity``
      rather than picked silently by max-matched-rows.
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
    if "hpo_variant" in model_df.columns:
        model_df = model_df[
            model_df["hpo_variant"].astype(str).str.lower() != "naive"
        ]
    if model_df.empty:
        return _empty_output()

    # Group DIRECTLY by the complete identity (commit 4 Section B.1).
    # Two valid runs with the same set_id but different rolling windows
    # / HPO objectives / requested universes are now KEPT APART
    # naturally — no short-key collapse, no consistency-flagging of
    # legitimate parallel runs.
    rows: list[dict] = []
    # Ensure the grouping columns exist (NaN is a valid value).
    for col in MODEL_GROUP_COLUMNS:
        if col not in model_df.columns:
            model_df[col] = np.nan

    # Diagnostic-only metadata: must be constant within one complete-
    # identity group. If it varies the row is flagged invalid (B.2).
    DIAGNOSTIC_META_COLUMNS = [
        "requested_tickers", "n_requested_tickers", "universe_identity_source",
        "available_coin_universe_hash",
    ]

    for keys, m_grp in model_df.groupby(MODEL_GROUP_COLUMNS, dropna=False):
        (horizon, set_id, sm, model_type, panel_mode,
         hpo_variant, hpo_objective,
         train_window_mode, rolling_window_days, rolling_window_timestamps,
         m_hash) = keys

        # Resolve universe-identity provenance from any row in the group
        # (NAIVE-generator runs stamp "requested_metadata"; legacy frames
        # without a coin_universe_hash fall back to realised tickers).
        _, universe_src = _coin_universe_hash_from_signals(m_grp)

        base = _identity_row(
            horizon=horizon, set_id=set_id, sentiment_model=sm,
            model_type=model_type, panel_mode=panel_mode,
            hpo_variant=hpo_variant,
            hpo_objective=hpo_objective,
            train_window_mode=train_window_mode,
            rolling_window_days=rolling_window_days,
            rolling_window_timestamps=rolling_window_timestamps,
            coin_universe_hash=m_hash if m_hash is not None and not (
                isinstance(m_hash, float) and np.isnan(m_hash)) else None,
            universe_identity_source=universe_src,
            n_model=int(len(m_grp)),
        )

        ident_values = {
            "horizon":                   horizon,
            "model_type":                model_type,
            "panel_mode":                panel_mode,
            "train_window_mode":         train_window_mode,
            "rolling_window_days":       rolling_window_days,
            "rolling_window_timestamps": rolling_window_timestamps,
            "coin_universe_hash":        m_hash,
        }

        # ── Identity-metadata consistency guard (Section B.2) ─────
        # Non-grouping metadata must be constant inside the group;
        # otherwise the row is flagged invalid.
        nonconst = _non_constant_columns(m_grp, DIAGNOSTIC_META_COLUMNS)
        if nonconst:
            base["status"]       = "invalid_model_identity"
            base["skip_reason"]  = "non_constant_identity_metadata"
            base["non_constant_columns"] = ",".join(nonconst)
            rows.append(base)
            continue

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

        # ── NAIVE ambiguity guard (Section C) ─────────────────────
        # The NAIVE side must collapse to exactly ONE identity group;
        # multiple distinct groups under the same complete identity
        # would force an arbitrary choice, so we refuse to pick.
        naive_groups = list(cand.groupby(
            list(NAIVE_IDENTITY_COLUMNS), dropna=False,
        ))
        if len(naive_groups) > 1:
            base["status"]      = "ambiguous_naive_identity"
            base["skip_reason"] = "multiple_naive_groups_for_identity"
            rows.append(base)
            continue
        (_naive_key, naive_grp), = naive_groups
        base["naive_coin_universe_hash"] = m_hash  # by identity-match

        # ── Duplicate-key guard ──────────────────────────────────
        dup_model = int(m_grp.duplicated(subset=["ticker", "timestamp"]).sum())
        dup_naive = int(naive_grp.duplicated(subset=["ticker", "timestamp"]).sum())
        base["n_duplicate_model_keys"] = dup_model
        base["n_duplicate_naive_keys"] = dup_naive
        if dup_model or dup_naive:
            base["status"]      = "duplicate_keys"
            base["skip_reason"] = "duplicate_keys_within_identity"
            rows.append(base)
            continue

        joined = _inner_join_on_keys(m_grp, naive_grp)
        n_matched = int(len(joined))
        # ``n_unmatched_*`` after duplicate verification.
        n_model_unique = int(m_grp.drop_duplicates(subset=["ticker", "timestamp"]).shape[0])
        n_naive_unique = int(naive_grp.drop_duplicates(subset=["ticker", "timestamp"]).shape[0])
        base["n_model"]            = n_model_unique
        base["n_naive"]            = n_naive_unique
        base["n_matched"]          = n_matched
        base["n_unmatched_model"]  = max(n_model_unique - n_matched, 0)
        base["n_unmatched_naive"]  = max(n_naive_unique - n_matched, 0)
        if joined.empty:
            base["status"]      = "no_overlap"
            base["skip_reason"] = "no_overlap"
            rows.append(base)
            continue

        # ── Target-equality verification ─────────────────────────
        targets_identical = bool(
            (joined["target_m"].astype(int).values
             == joined["target_n"].astype(int).values).all()
        )
        base["targets_identical"] = targets_identical
        if not targets_identical:
            base["status"]      = "target_mismatch"
            base["skip_reason"] = "target_mismatch"
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
            "model_accuracy":          m_metrics["accuracy"],
            "naive_accuracy":          n_metrics["accuracy"],
            "accuracy_lift":           m_metrics["accuracy"] - n_metrics["accuracy"],
            "model_balanced_accuracy": m_metrics["balanced_accuracy"],
            "naive_balanced_accuracy": n_metrics["balanced_accuracy"],
            "balanced_accuracy_lift":  (m_metrics["balanced_accuracy"]
                                         - n_metrics["balanced_accuracy"]),
            "model_brier":             m_metrics["brier"],
            "naive_brier":             n_metrics["brier"],
            "brier_improvement":       n_metrics["brier"] - m_metrics["brier"],
            "model_log_loss":          m_metrics["log_loss"],
            "naive_log_loss":          n_metrics["log_loss"],
            "log_loss_improvement":    n_metrics["log_loss"] - m_metrics["log_loss"],
            "mcnemar_b":               b,
            "mcnemar_c":               c,
            "mcnemar_stat":            float(mcn_stat),
            "mcnemar_p_value":         float(mcn_p),
            "significant_raw_5pct":    bool(np.isfinite(mcn_p)
                                             and mcn_p < SIGNIFICANCE_ALPHA),
            "status":                  "ok",
            "skip_reason":             "",
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
                  universe_identity_source: str,
                  n_model: int) -> dict:
    row = {col: None for col in OUTPUT_COLUMNS}
    for k in ("accuracy_lift", "balanced_accuracy_lift",
              "brier_improvement", "log_loss_improvement",
              "model_accuracy", "naive_accuracy",
              "model_balanced_accuracy", "naive_balanced_accuracy",
              "model_brier", "naive_brier",
              "model_log_loss", "naive_log_loss",
              "mcnemar_stat", "mcnemar_p_value"):
        row[k] = float("nan")
    row["hypothesis_family"]  = HYPOTHESIS_FAMILY
    row["benchmark_role"]     = ABSOLUTE_BENCHMARK_ROLE
    row["benchmark_set_id"]   = NAIVE_SET_ID
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
    row["universe_identity_source"] = universe_identity_source
    row["n_model"]            = n_model
    row["n_naive"]            = 0
    row["n_matched"]          = 0
    row["n_unmatched_model"]  = 0
    row["n_unmatched_naive"]  = 0
    row["n_duplicate_model_keys"] = 0
    row["n_duplicate_naive_keys"] = 0
    row["targets_identical"]  = False
    row["mcnemar_b"]          = 0
    row["mcnemar_c"]          = 0
    row["significant_raw_5pct"] = False
    row["non_constant_columns"] = ""
    row["status"]             = "missing_naive"
    row["skip_reason"]        = ""
    return row


def _first_constant(grp: pd.DataFrame, col: str):
    """Return the unique non-NaN value in ``col`` for the group.

    The caller is expected to have already verified internal constancy
    via :func:`_non_constant_columns` — this helper is only the
    "read the constant value back out" half of the split. Returns NaN
    when the column is absent or empty.
    """
    if col not in grp.columns:
        return float("nan")
    vals = grp[col].dropna()
    if vals.empty:
        return float("nan")
    return vals.iat[0]


def _non_constant_columns(grp: pd.DataFrame,
                          columns: Iterable[str]) -> list[str]:
    """Return the subset of ``columns`` that contain more than one
    distinct value inside the group (NaN treated as a value)."""
    offenders: list[str] = []
    for col in columns:
        if col not in grp.columns:
            continue
        # Treat NaN as a value so an unset + set pair counts as non-constant.
        seen = grp[col].astype(object).where(grp[col].notna(), "__NaN__").unique()
        if len(seen) > 1:
            offenders.append(col)
    return offenders


def _series_eq_nan_safe(s: pd.Series, value) -> pd.Series:
    """Elementwise equality treating NaN==NaN as match."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return s.isna() | (s.astype(str).str.lower() == "nan")
    try:
        return s == value
    except Exception:  # noqa: BLE001
        return s.astype(str) == str(value)
