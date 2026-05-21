"""Continuity-corrected McNemar test against one or more benchmarks.

For every non-benchmark signal in categories ``{sentiment, combined}`` we
match observations against the benchmark by ``(timestamp, ticker)`` and
compute:

* ``b`` = benchmark correct, model wrong
* ``c`` = model correct, benchmark wrong
* statistic = (|b − c| − 1)² / (b + c)
* p-value   = χ² survival function with 1 df
* ``b + c == 0`` → ``p_value = 1.0``

The thesis compares each candidate against both **B1** (rolling-probability
benchmark) and **B2** (logistic on ``log_return_t`` only) to disambiguate
"beats naive" from "beats minimal logistic".
"""

from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from scipy.stats import chi2

from .metrics import GROUP_KEYS

DEFAULT_BENCHMARKS: tuple[str, ...] = ("B1", "B2")
ELIGIBLE_CATEGORIES = ("sentiment", "combined")
SIGNIFICANCE_ALPHA = 0.05

# Back-compat: some external callers still reference this name.
BENCHMARK_SET_ID = "B1"


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
    return m.merge(
        b[["timestamp", "ticker", "correct_benchmark"]],
        on=["timestamp", "ticker"], how="inner",
    )


def mcnemar_for_pair(model_df: pd.DataFrame,
                     benchmark_df: pd.DataFrame) -> dict:
    """Compute b, c, statistic, p-value and significance flag."""
    empty = {"b": 0, "c": 0, "mcnemar_stat": np.nan,
             "mcnemar_pval": np.nan, "significant_vs_benchmark": False,
             "n_matched": 0}
    if model_df.empty or benchmark_df.empty:
        return empty
    j = _match_against_benchmark(model_df, benchmark_df)
    if j.empty:
        return empty
    b = int(((j["correct_benchmark"] == 1) & (j["correct_model"] == 0)).sum())
    c = int(((j["correct_benchmark"] == 0) & (j["correct_model"] == 1)).sum())
    stat, pval = mcnemar_continuity_corrected(b, c)
    return {
        "b": b, "c": c,
        "mcnemar_stat": stat,
        "mcnemar_pval": pval,
        "significant_vs_benchmark": bool(pval < SIGNIFICANCE_ALPHA),
        "n_matched": int(len(j)),
    }


def mcnemar_table(signals: pd.DataFrame,
                  benchmarks: Sequence[str] = DEFAULT_BENCHMARKS) -> pd.DataFrame:
    """Long-form McNemar table: one row per (horizon × set × benchmark).

    The benchmark itself is excluded from the rows it benchmarks (you don't
    test B1 against B1). Rows whose ``category`` is not in
    :data:`ELIGIBLE_CATEGORIES` are skipped — i.e. economic-only sets do not
    appear because the thesis question is about sentiment-augmented sets.
    """
    if signals.empty or "category" not in signals.columns:
        return pd.DataFrame()
    rows: list[dict] = []
    for horizon, hz_grp in signals.groupby("horizon", dropna=False):
        bench_frames: dict[str, pd.DataFrame] = {
            bid: hz_grp[hz_grp["set_id"] == bid] for bid in benchmarks
        }
        for keys, grp in hz_grp.groupby(list(GROUP_KEYS), dropna=False):
            set_id = keys[1]
            if set_id in benchmarks:
                continue
            category = ""
            if "category" in grp.columns:
                cats = grp["category"].dropna().astype(str)
                category = cats.iloc[0] if not cats.empty else ""
            if category.lower() not in ELIGIBLE_CATEGORIES:
                continue
            for bid in benchmarks:
                bench = bench_frames.get(bid, pd.DataFrame())
                if bench.empty:
                    continue
                res = mcnemar_for_pair(grp, bench)
                res.update(dict(zip(GROUP_KEYS, keys)))
                res["benchmark"] = bid
                res["category"]  = category
                rows.append(res)
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    front = ["horizon", "set_id", "category", "sentiment_model", "benchmark",
             "mcnemar_stat", "mcnemar_pval", "significant_vs_benchmark",
             "b", "c", "n_matched"]
    rest = [col for col in out.columns if col not in front]
    return out[front + rest].reset_index(drop=True)


def mcnemar_wide(mcnemar_long: pd.DataFrame,
                 benchmarks: Sequence[str] = DEFAULT_BENCHMARKS) -> pd.DataFrame:
    """Pivot the long-form McNemar table into one row per (horizon × set).

    Each benchmark contributes its own ``_vs_<bid>`` columns so the result can
    be merged onto the pooled metrics table without colliding.
    """
    if mcnemar_long is None or mcnemar_long.empty:
        return pd.DataFrame()
    pieces = []
    for bid in benchmarks:
        sub = mcnemar_long[mcnemar_long["benchmark"] == bid].copy()
        if sub.empty:
            continue
        suffix = f"_vs_{bid.lower()}"
        keep = ["horizon", "set_id", "sentiment_model",
                "mcnemar_stat", "mcnemar_pval",
                "significant_vs_benchmark", "b", "c", "n_matched"]
        sub = sub[keep].rename(columns={
            "mcnemar_stat":              f"mcnemar_stat{suffix}",
            "mcnemar_pval":              f"mcnemar_pval{suffix}",
            "significant_vs_benchmark":  f"significant{suffix}",
            "b":                         f"b{suffix}",
            "c":                         f"c{suffix}",
            "n_matched":                 f"n_matched{suffix}",
        })
        pieces.append(sub)
    if not pieces:
        return pd.DataFrame()
    out = pieces[0]
    for nxt in pieces[1:]:
        out = out.merge(nxt, on=["horizon", "set_id", "sentiment_model"], how="outer")
    return out.reset_index(drop=True)
