"""Continuity-corrected McNemar test against the B1 benchmark.

For every non-benchmark signal in categories ``{sentiment, combined}`` we
match observations against B1 by ``(timestamp, ticker)`` and compute:

* ``b`` = benchmark correct, model wrong
* ``c`` = model correct, benchmark wrong
* statistic = (|b − c| − 1)² / (b + c)
* p-value   = χ² survival function with 1 df
* ``b + c == 0`` → ``p_value = 1.0``

Tested against ``statsmodels.stats.contingency_tables.mcnemar(exact=False,
correction=True)`` as a fallback when statsmodels is available; the explicit
implementation guarantees the b+c=0 contract from the PRD.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import chi2

from .metrics import GROUP_KEYS

BENCHMARK_SET_ID = "B1"
ELIGIBLE_CATEGORIES = ("sentiment", "combined")
SIGNIFICANCE_ALPHA = 0.05


def mcnemar_continuity_corrected(b: int, c: int) -> tuple[float, float]:
    """Return ``(statistic, p_value)`` for the continuity-corrected McNemar test."""
    b = int(b)
    c = int(c)
    if b + c == 0:
        return 0.0, 1.0
    stat = (abs(b - c) - 1) ** 2 / (b + c)
    pval = float(chi2.sf(stat, df=1))
    return float(stat), pval


def _correctness(df: pd.DataFrame) -> pd.Series:
    return (df["prediction"].astype(int) == df["target"].astype(int)).astype(int)


def _match_against_benchmark(model_df: pd.DataFrame,
                             benchmark_df: pd.DataFrame) -> pd.DataFrame:
    """Inner-join on ``(timestamp, ticker)`` and return matched correctness."""
    m = model_df[["timestamp", "ticker", "target", "prediction"]].copy()
    b = benchmark_df[["timestamp", "ticker", "target", "prediction"]].copy()
    m["correct_model"]     = _correctness(m)
    b["correct_benchmark"] = _correctness(b)
    j = m.merge(
        b[["timestamp", "ticker", "correct_benchmark"]],
        on=["timestamp", "ticker"], how="inner",
    )
    return j


def mcnemar_for_pair(model_df: pd.DataFrame,
                     benchmark_df: pd.DataFrame) -> dict:
    """Compute b, c, statistic, p-value and significance flag."""
    if model_df.empty or benchmark_df.empty:
        return {"b": 0, "c": 0, "mcnemar_stat": np.nan,
                "mcnemar_pval": np.nan, "significant_vs_benchmark": False}
    j = _match_against_benchmark(model_df, benchmark_df)
    if j.empty:
        return {"b": 0, "c": 0, "mcnemar_stat": np.nan,
                "mcnemar_pval": np.nan, "significant_vs_benchmark": False}
    b = int(((j["correct_benchmark"] == 1) & (j["correct_model"] == 0)).sum())
    c = int(((j["correct_benchmark"] == 0) & (j["correct_model"] == 1)).sum())
    stat, pval = mcnemar_continuity_corrected(b, c)
    return {
        "b": b, "c": c,
        "mcnemar_stat": stat,
        "mcnemar_pval": pval,
        "significant_vs_benchmark": bool(pval < SIGNIFICANCE_ALPHA),
    }


def mcnemar_table(signals: pd.DataFrame) -> pd.DataFrame:
    """Per (horizon × set_id × sentiment_model), test against B1@horizon.

    Returns columns: horizon, set_id, sentiment_model, b, c, mcnemar_stat,
    mcnemar_pval, significant_vs_benchmark. Rows for B1 itself and for sets
    that are not in ELIGIBLE_CATEGORIES are omitted.
    """
    if signals.empty or "category" not in signals.columns:
        return pd.DataFrame()
    rows = []
    for horizon, hz_grp in signals.groupby("horizon", dropna=False):
        bench = hz_grp[hz_grp["set_id"] == BENCHMARK_SET_ID]
        if bench.empty:
            continue
        for keys, grp in hz_grp.groupby(list(GROUP_KEYS), dropna=False):
            set_id = keys[1]
            if set_id == BENCHMARK_SET_ID:
                continue
            category = str(grp["category"].dropna().astype(str).iloc[0]) if "category" in grp else ""
            if category.lower() not in ELIGIBLE_CATEGORIES:
                continue
            res = mcnemar_for_pair(grp, bench)
            res.update(dict(zip(GROUP_KEYS, keys)))
            rows.append(res)
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    front = ["horizon", "set_id", "sentiment_model",
             "mcnemar_stat", "mcnemar_pval", "significant_vs_benchmark", "b", "c"]
    rest = [c for c in out.columns if c not in front]
    return out[front + rest].reset_index(drop=True)
