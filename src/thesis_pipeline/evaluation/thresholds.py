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

from ..logging_utils import get_logger
from .metrics import GROUP_KEYS, FRONT_META, _front_order, _group_meta, ensure_group_columns
from .significance import FAMILY_COLS, _family_benchmark_frames

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
    signals = ensure_group_columns(signals)
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
    return _front_order(out, FRONT_META + ["threshold"])


# ---------------------------------------------------------------------------
# Threshold lift vs benchmark (conditional outperformance)
# ---------------------------------------------------------------------------
#
# At high thresholds the model trades only on its most confident calls. The
# thesis-relevant question is: *on the observations where the model trades,
# how do we compare to the benchmark?*  We measure this with a matched-
# observation lift:
#
#   1. For each (model_set × threshold), pick the rows the MODEL would trade.
#   2. Join those rows against the benchmark on (timestamp, ticker).
#   3. Compute model accuracy and benchmark accuracy on the matched subset.
#   4. lift_accuracy = accuracy_model − accuracy_benchmark.
#   5. Compute a continuity-corrected McNemar on the matched subset to test
#      whether the lift is statistically significant.
#
# This gives a much richer thesis story than the unconditional pooled
# accuracy — sentiment can perfectly well be useless on average yet useful
# at high model confidence (or vice versa).

def _matched_lift(model_grp: pd.DataFrame,
                  bench_grp: pd.DataFrame,
                  threshold: float) -> dict:
    """Lift of model accuracy vs benchmark on the model's traded subset."""
    if model_grp.empty:
        return {"n_traded": 0, "n_matched": 0,
                "accuracy_model": np.nan,
                "accuracy_benchmark": np.nan,
                "lift_accuracy": np.nan,
                "mcnemar_stat": np.nan,
                "mcnemar_pval": np.nan,
                "significant_vs_benchmark": False}

    prob = np.clip(model_grp["probability"].astype(float).values, 1e-15, 1 - 1e-15)
    mask = _trade_mask(prob, threshold)
    traded = model_grp.loc[mask].copy()
    n_traded = int(len(traded))
    if n_traded == 0 or bench_grp.empty:
        return {"n_traded": n_traded, "n_matched": 0,
                "accuracy_model": np.nan,
                "accuracy_benchmark": np.nan,
                "lift_accuracy": np.nan,
                "mcnemar_stat": np.nan,
                "mcnemar_pval": np.nan,
                "significant_vs_benchmark": False}

    traded["__pred_model__"] = _predictions_at_threshold(prob, threshold)[mask]
    matched = traded.merge(
        bench_grp[["timestamp", "ticker", "target", "prediction"]]
            .rename(columns={"target": "target_bench",
                             "prediction": "__pred_bench__"}),
        on=["timestamp", "ticker"], how="inner",
    )
    if matched.empty:
        return {"n_traded": n_traded, "n_matched": 0,
                "accuracy_model": np.nan,
                "accuracy_benchmark": np.nan,
                "lift_accuracy": np.nan,
                "mcnemar_stat": np.nan,
                "mcnemar_pval": np.nan,
                "significant_vs_benchmark": False}

    y = matched["target"].astype(int).values
    correct_m = (matched["__pred_model__"].astype(int).values == y).astype(int)
    correct_b = (matched["__pred_bench__"].astype(int).values == y).astype(int)
    acc_m = float(correct_m.mean())
    acc_b = float(correct_b.mean())

    # Continuity-corrected McNemar on the matched subset.
    from .significance import mcnemar_continuity_corrected, SIGNIFICANCE_ALPHA
    b = int(((correct_b == 1) & (correct_m == 0)).sum())
    c = int(((correct_b == 0) & (correct_m == 1)).sum())
    stat, pval = mcnemar_continuity_corrected(b, c)
    return {
        "n_traded": n_traded, "n_matched": int(len(matched)),
        "accuracy_model": round(acc_m, 6),
        "accuracy_benchmark": round(acc_b, 6),
        "lift_accuracy": round(acc_m - acc_b, 6),
        "mcnemar_stat": stat,
        "mcnemar_pval": pval,
        "significant_vs_benchmark": bool(pval < SIGNIFICANCE_ALPHA),
    }


def threshold_lift_table(signals: pd.DataFrame,
                         benchmarks: tuple[str, ...] = ("ECON",),
                         thresholds: tuple[float, ...] = DEFAULT_THRESHOLDS,
                         allow_cross_model_benchmark: bool = False,
                         ) -> pd.DataFrame:
    """Per (horizon × model_type × panel_mode × set × threshold × benchmark):
    matched-observation lift.

    Benchmark matching is restricted to the **same model family** — a
    ``panel_logit`` / ``pooled`` model is only ever lifted against a
    ``panel_logit`` / ``pooled`` benchmark, never against a per-asset one.
    With ``allow_cross_model_benchmark=True`` a missing same-family benchmark
    falls back to the per-asset benchmark of the same horizon; the default
    (``False``) leaves the lift columns NaN and logs a warning instead.

    Rows for the benchmark set itself are omitted. Eligibility follows the
    McNemar contract — sentiment and combined sets only.
    """
    if signals.empty or "category" not in signals.columns:
        return pd.DataFrame()
    signals = ensure_group_columns(signals)
    family_cols = list(FAMILY_COLS)
    rows: list[dict] = []
    for fam_keys, fam_grp in signals.groupby(family_cols, dropna=False):
        bench_frames = _family_benchmark_frames(
            fam_grp, signals, fam_keys, benchmarks, allow_cross_model_benchmark,
            context="threshold-lift",
        )
        for keys, grp in fam_grp.groupby(list(GROUP_KEYS), dropna=False):
            set_id = keys[1]
            if set_id in benchmarks:
                continue
            cats = grp["category"].dropna().astype(str)
            category = cats.iloc[0] if not cats.empty else ""
            # v4 categories: sentiment_vader / sentiment_cryptobert /
            # combined_vader / combined_cryptobert / multi (and the legacy
            # "sentiment" / "combined" literals from pre-v4 fixtures).
            from .significance import ELIGIBLE_CATEGORIES
            if category.lower() not in ELIGIBLE_CATEGORIES:
                continue
            n_obs = int(len(grp))
            for t in thresholds:
                for bid in benchmarks:
                    bench = bench_frames.get(bid, pd.DataFrame())
                    if bench.empty:
                        continue
                    res = _matched_lift(grp, bench, t)
                    res.update(dict(zip(GROUP_KEYS, keys)))
                    res["benchmark"] = bid
                    res["category"]  = category
                    res["threshold"] = t
                    res["coverage_model"] = round(res["n_traded"] / n_obs, 6) \
                        if n_obs else 0.0
                    rows.append(res)
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    front = FRONT_META + ["benchmark", "threshold", "n_traded", "coverage_model",
                          "n_matched", "accuracy_model", "accuracy_benchmark",
                          "lift_accuracy", "mcnemar_stat", "mcnemar_pval",
                          "significant_vs_benchmark"]
    return _front_order(out, front)
