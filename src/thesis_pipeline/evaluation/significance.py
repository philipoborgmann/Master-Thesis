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

from ..logging_utils import get_logger
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


# ══════════════════════════════════════════════════════════════════════════════
# Regime-specific McNemar tests (exploratory / supplementary)
# ══════════════════════════════════════════════════════════════════════════════
#
# These tests are *supplementary* to the global McNemar tests above. The
# global test remains the headline result. Here we slice the matched
# (model, benchmark) observations by market regime — volatility tercile,
# market-cap tercile, or their interaction — and ask whether a
# sentiment / combined model systematically corrects benchmark errors
# *within* a regime.
#
# Because this multiplies the number of hypotheses, we report a
# Benjamini-Hochberg FDR correction (q-values) alongside the raw p-values.
# Regime cells with too few matched observations or too few discordant
# pairs are flagged ``test_valid = False`` and excluded from the FDR pool —
# interpret a regime result only when both ``n_matched`` and
# ``discordant_n`` are sufficiently large.

MIN_N_MATCHED_DEFAULT = 50
MIN_DISCORDANT_DEFAULT = 20

_REGIME_FRONT_COLS = [
    "regime_type", "horizon", "set_id", "category", "sentiment_model",
    "benchmark", "vol_regime", "mcap_regime", "interaction",
    "b", "c", "n_matched", "discordant_n",
    "mcnemar_stat", "mcnemar_pval", "q_value_bh",
    "significant_5pct", "significant_10pct",
    "significant_bh_5pct", "significant_bh_10pct", "test_valid",
]


def _match_against_benchmark_grouped(model_df: pd.DataFrame,
                                     benchmark_df: pd.DataFrame,
                                     group_cols: Sequence[str]) -> pd.DataFrame:
    """Inner-join model vs benchmark on (timestamp, ticker), keeping the
    model-side ``group_cols`` (regime labels are a property of the
    observation, so model and benchmark share them)."""
    base = ["timestamp", "ticker", "target", "prediction"]
    m = model_df[base + list(group_cols)].copy()
    b = benchmark_df[base].copy()
    m["correct_model"]     = _correctness(m)
    b["correct_benchmark"] = _correctness(b)
    return m.merge(
        b[["timestamp", "ticker", "correct_benchmark"]],
        on=["timestamp", "ticker"], how="inner",
    )


def mcnemar_by_group(signals: pd.DataFrame,
                     group_cols: list[str],
                     benchmarks: Sequence[str] = DEFAULT_BENCHMARKS,
                     min_n_matched: int = MIN_N_MATCHED_DEFAULT,
                     min_discordant: int = MIN_DISCORDANT_DEFAULT) -> pd.DataFrame:
    """Continuity-corrected McNemar per
    (horizon × set_id × sentiment_model × benchmark × group_cols).

    Only sentiment / combined sets are tested; benchmark sets are excluded
    from the model side. A cell is ``test_valid`` only when
    ``n_matched >= min_n_matched`` **and** ``discordant_n >= min_discordant``;
    invalid cells carry NaN statistics and ``False`` significance flags.
    """
    if signals.empty or "category" not in signals.columns:
        return pd.DataFrame()
    missing = [c for c in group_cols if c not in signals.columns]
    if missing:
        get_logger().warning("regime-mcnemar: missing group columns %s — skipping",
                             missing)
        return pd.DataFrame()

    rows: list[dict] = []
    for _, hz_grp in signals.groupby("horizon", dropna=False):
        bench_frames = {bid: hz_grp[hz_grp["set_id"] == bid] for bid in benchmarks}
        for keys, grp in hz_grp.groupby(list(GROUP_KEYS), dropna=False):
            set_id = keys[1]
            if set_id in benchmarks:
                continue
            cats = grp["category"].dropna().astype(str)
            category = cats.iloc[0] if not cats.empty else ""
            if category.lower() not in ELIGIBLE_CATEGORIES:
                continue
            for bid in benchmarks:
                bench = bench_frames.get(bid, pd.DataFrame())
                if bench.empty:
                    continue
                matched = _match_against_benchmark_grouped(grp, bench, group_cols)
                if matched.empty:
                    continue
                for gvals, gdf in matched.groupby(group_cols, dropna=True):
                    if not isinstance(gvals, tuple):
                        gvals = (gvals,)
                    b = int(((gdf["correct_benchmark"] == 1) &
                             (gdf["correct_model"] == 0)).sum())
                    c = int(((gdf["correct_benchmark"] == 0) &
                             (gdf["correct_model"] == 1)).sum())
                    n_matched = int(len(gdf))
                    discordant = b + c
                    test_valid = (n_matched >= min_n_matched
                                  and discordant >= min_discordant)
                    if test_valid:
                        stat, pval = mcnemar_continuity_corrected(b, c)
                        sig5  = bool(pval < 0.05)
                        sig10 = bool(pval < 0.10)
                    else:
                        stat, pval = np.nan, np.nan
                        sig5 = sig10 = False
                    row = {
                        "horizon":          keys[0],
                        "set_id":           keys[1],
                        "sentiment_model":  keys[2],
                        "category":         category,
                        "benchmark":        bid,
                        "b":                b,
                        "c":                c,
                        "n_matched":        n_matched,
                        "discordant_n":     discordant,
                        "mcnemar_stat":     stat,
                        "mcnemar_pval":     pval,
                        "significant_5pct":  sig5,
                        "significant_10pct": sig10,
                        "test_valid":       test_valid,
                    }
                    for col, val in zip(group_cols, gvals):
                        row[col] = val
                    rows.append(row)
    return pd.DataFrame(rows)


def _bh_qvalues(pvals: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg q-values (monotone), pure-numpy fallback."""
    p = np.asarray(pvals, dtype=float)
    n = len(p)
    if n == 0:
        return p
    order = np.argsort(p)
    ranked = p[order]
    q = ranked * n / np.arange(1, n + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    q = np.clip(q, 0.0, 1.0)
    out = np.empty(n, dtype=float)
    out[order] = q
    return out


def adjust_pvalues_bh(df: pd.DataFrame, p_col: str = "mcnemar_pval") -> pd.DataFrame:
    """Append BH/FDR columns: ``q_value_bh``, ``significant_bh_5pct``,
    ``significant_bh_10pct``.

    Only valid tests (non-NaN p-value) enter the FDR pool. Uses
    ``statsmodels.stats.multitest.multipletests`` when available, otherwise
    a pure-numpy BH fallback.
    """
    out = df.copy()
    out["q_value_bh"] = np.nan
    out["significant_bh_5pct"] = False
    out["significant_bh_10pct"] = False
    if out.empty or p_col not in out.columns:
        return out

    mask = out[p_col].notna()
    pvals = out.loc[mask, p_col].to_numpy(dtype=float)
    if len(pvals) == 0:
        return out

    q5 = q10 = rej5 = rej10 = None
    try:
        from statsmodels.stats.multitest import multipletests
        rej5,  q5,  _, _ = multipletests(pvals, alpha=0.05, method="fdr_bh")
        rej10, _,   _, _ = multipletests(pvals, alpha=0.10, method="fdr_bh")
    except Exception:  # noqa: BLE001 — fall back to the numpy BH
        q5 = _bh_qvalues(pvals)
        rej5  = q5 < 0.05
        rej10 = q5 < 0.10

    out.loc[mask, "q_value_bh"]          = q5
    out.loc[mask, "significant_bh_5pct"] = rej5
    out.loc[mask, "significant_bh_10pct"] = rej10
    return out


def regime_mcnemar_table(signals: pd.DataFrame,
                         vol_lookup: pd.DataFrame | None = None,
                         mcap_lookup: pd.DataFrame | None = None,
                         benchmarks: Sequence[str] = DEFAULT_BENCHMARKS,
                         min_n_matched: int = MIN_N_MATCHED_DEFAULT,
                         min_discordant: int = MIN_DISCORDANT_DEFAULT) -> pd.DataFrame:
    """Regime-stratified McNemar tests (volatility, market-cap, interaction).

    Supplementary to the global test. Blocks are produced only when the
    matching lookup is available:

    * **volatility**  — needs ``vol_lookup``  (group: ``vol_regime``)
    * **market_cap**  — needs ``mcap_lookup`` (group: ``mcap_regime``)
    * **interaction** — needs both           (group: ``mcap_regime`` × ``vol_regime``)

    BH/FDR q-values are computed over *all* valid regime tests pooled
    together (across the three blocks), since they form one exploratory
    family of hypotheses.
    """
    from .market_cap import attach_market_cap_regimes
    from .volatility import attach_regimes

    logger = get_logger()
    has_vol  = vol_lookup  is not None and not vol_lookup.empty
    has_mcap = mcap_lookup is not None and not mcap_lookup.empty

    if signals.empty:
        return pd.DataFrame(columns=_REGIME_FRONT_COLS)
    if not has_vol and not has_mcap:
        logger.warning("regime-mcnemar: no volatility or market-cap lookup — "
                       "skipping all regime McNemar tests")
        return pd.DataFrame(columns=_REGIME_FRONT_COLS)

    blocks: list[pd.DataFrame] = []

    # ── Volatility regimes ──────────────────────────────────────
    if has_vol:
        enr = attach_regimes(signals, vol_lookup)
        if "regime" in enr.columns:
            enr = enr.rename(columns={"regime": "vol_regime"})
        enr = enr[enr.get("vol_regime").notna()] if "vol_regime" in enr.columns else enr.iloc[0:0]
        if not enr.empty:
            tbl = mcnemar_by_group(enr, ["vol_regime"], benchmarks,
                                   min_n_matched, min_discordant)
            if not tbl.empty:
                tbl["regime_type"] = "volatility"
                tbl["mcap_regime"] = np.nan
                tbl["interaction"] = np.nan
                blocks.append(tbl)
    else:
        logger.warning("regime-mcnemar: volatility lookup unavailable — "
                       "skipping volatility regime block")

    # ── Market-cap regimes ──────────────────────────────────────
    if has_mcap:
        enr = attach_market_cap_regimes(signals, mcap_lookup)
        enr = enr[enr.get("mcap_regime").notna()] if "mcap_regime" in enr.columns else enr.iloc[0:0]
        if not enr.empty:
            tbl = mcnemar_by_group(enr, ["mcap_regime"], benchmarks,
                                   min_n_matched, min_discordant)
            if not tbl.empty:
                tbl["regime_type"] = "market_cap"
                tbl["vol_regime"]  = np.nan
                tbl["interaction"] = np.nan
                blocks.append(tbl)
    else:
        logger.warning("regime-mcnemar: market-cap lookup unavailable — "
                       "skipping market-cap regime block")

    # ── Interaction regimes ─────────────────────────────────────
    if has_vol and has_mcap:
        enr = attach_regimes(signals, vol_lookup)
        if "regime" in enr.columns:
            enr = enr.rename(columns={"regime": "vol_regime"})
        enr = attach_market_cap_regimes(enr, mcap_lookup)
        keep = (("vol_regime" in enr.columns) and ("mcap_regime" in enr.columns))
        if keep:
            enr = enr[enr["vol_regime"].notna() & enr["mcap_regime"].notna()]
        else:
            enr = enr.iloc[0:0]
        if not enr.empty:
            tbl = mcnemar_by_group(enr, ["mcap_regime", "vol_regime"], benchmarks,
                                   min_n_matched, min_discordant)
            if not tbl.empty:
                tbl["regime_type"] = "interaction"
                tbl["interaction"] = (tbl["mcap_regime"].astype(str) + "_"
                                      + tbl["vol_regime"].astype(str))
                blocks.append(tbl)

    if not blocks:
        return pd.DataFrame(columns=_REGIME_FRONT_COLS)

    out = pd.concat(blocks, ignore_index=True)
    # Ensure all regime columns exist before reordering.
    for col in ("vol_regime", "mcap_regime", "interaction"):
        if col not in out.columns:
            out[col] = np.nan
    out = adjust_pvalues_bh(out, "mcnemar_pval")
    ordered = [c for c in _REGIME_FRONT_COLS if c in out.columns]
    rest = [c for c in out.columns if c not in ordered]
    return out[ordered + rest].reset_index(drop=True)
