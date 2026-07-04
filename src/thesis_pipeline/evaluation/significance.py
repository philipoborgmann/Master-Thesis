"""Continuity-corrected McNemar test against the v4 ECON benchmark.

For every non-benchmark signal in categories ``{sentiment, combined}`` we
match observations against the benchmark by ``(timestamp, ticker)`` and
compute:

* ``b`` = benchmark correct, model wrong
* ``c`` = model correct, benchmark wrong
* statistic = (|b − c| − 1)² / (b + c)
* p-value   = χ² survival function with 1 df
* ``b + c == 0`` → ``p_value = 1.0``

v4 (Aufgabe 6.1) uses a **single matched benchmark — ``ECON``** — for the
primary H1 test ("does the sentiment block add information beyond the
shared ECON core?"). The legacy ``B1 / B2`` comparison was removed:
``B1`` is now the NAIVE evaluation reference, ``B2`` is folded into the
v4 ECON information set, and neither is a v4 feature set. The
``DEFAULT_BENCHMARKS`` tuple is therefore a single-element ``("ECON",)``.
Callers that build synthetic benchmark frames (unit tests) can still pass
any ``set_id`` they like via the ``benchmarks=`` kwarg.
"""

from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from scipy.stats import chi2

from ..logging_utils import get_logger
from .metrics import GROUP_KEYS, FRONT_META, _front_order, ensure_group_columns

DEFAULT_BENCHMARKS: tuple[str, ...] = ("ECON",)
# Categories eligible for benchmark comparisons (McNemar, threshold-lift,
# economic-lift). After the family refactor + FinBERT removal:
#
# * ``sentiment_cryptobert`` (S*)        — pure CryptoBERT-only sentiment
# * ``sentiment_vader``      (SV*)       — pure VADER-only sentiment
# * ``combined_cryptobert``  (C*)        — Benchmark + CryptoBERT sentiment
# * ``combined_vader``       (CV*)       — Benchmark + VADER sentiment
# * ``multi``                (M*)        — Benchmark + CryptoBERT + VADER
#
# The legacy ``"sentiment"`` / ``"combined"`` literals are kept so any
# in-flight signal frames produced before the refactor still pass through.
ELIGIBLE_CATEGORIES = (
    "sentiment", "sentiment_cryptobert", "sentiment_vader",
    "combined", "combined_cryptobert", "combined_vader",
    "multi",
)
SIGNIFICANCE_ALPHA = 0.05

# Single matched benchmark for the v4 primary H1 test.
BENCHMARK_SET_ID = "ECON"

# Columns identifying a model family. Benchmark matching is confined to a
# single family so a panel-logit (or HPO) model is never silently compared
# against a per-asset / fixed-C benchmark. ``hpo_variant`` keeps fixed-C and
# each tuned variant in separate families.
FAMILY_COLS = ["horizon", "model_type", "panel_mode", "hpo_variant"]


def _family_benchmark_frames(fam_grp: pd.DataFrame,
                             signals: pd.DataFrame,
                             fam_keys,
                             benchmarks: Sequence[str],
                             allow_cross_model_benchmark: bool,
                             *,
                             context: str = "mcnemar") -> dict[str, pd.DataFrame]:
    """Resolve the benchmark frames available *within one model family*.

    ``fam_keys`` is the ``(horizon, model_type, panel_mode, hpo_variant)``
    tuple of the family. By default a benchmark must live in the same family.
    With ``allow_cross_model_benchmark=True`` a missing same-family benchmark
    falls back to the per-asset benchmark of the same horizon **and the same
    ``hpo_variant``** — fixed-C and HPO are never mixed, even under the opt-in.
    A benchmark that is unavailable under either rule yields an empty frame
    (the caller skips that comparison) and a single WARN is logged — never a
    silent cross-family mix.
    """
    if not isinstance(fam_keys, tuple):
        fam_keys = (fam_keys,)
    horizon = fam_keys[0]
    model_type = fam_keys[1] if len(fam_keys) > 1 else "per_asset"
    panel_mode = fam_keys[2] if len(fam_keys) > 2 else "-"
    hpo_variant = fam_keys[3] if len(fam_keys) > 3 else "fixed"
    frames: dict[str, pd.DataFrame] = {}
    for bid in benchmarks:
        bench = fam_grp[fam_grp["set_id"] == bid]
        if bench.empty and allow_cross_model_benchmark:
            # Relax model_type / panel_mode only — never the HPO variant.
            bench = signals[(signals["horizon"] == horizon)
                            & (signals["set_id"] == bid)
                            & (signals["model_type"] == "per_asset")
                            & (signals.get("hpo_variant", "fixed") == hpo_variant)]
        if bench.empty:
            get_logger().warning(
                "%s: no %s benchmark within family "
                "(horizon=%s, model_type=%s, panel_mode=%s, hpo_variant=%s) — "
                "skipping that comparison (allow_cross_model_benchmark=%s)",
                context, bid, horizon, model_type, panel_mode, hpo_variant,
                allow_cross_model_benchmark,
            )
        frames[bid] = bench
    return frames


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
    """Inner-join model vs benchmark on the strict ``(timestamp, ticker)``
    intersection and return matched correctness.

    Both sides are deduplicated to one row per ``(timestamp, ticker)``
    before the join (commit 12 Task E) so the McNemar ``n_matched``
    equals the candidate's true matched coverage rather than fanning out
    when the ECON benchmark carries coin-timestamps the candidate lacks
    (the 1d/1h coverage-asymmetry case). When both sides already have
    one clean row per key (the 6h case) the dedup is a no-op and the
    statistic is unchanged.
    """
    m = model_df[["timestamp", "ticker", "target", "prediction"]].copy()
    b = benchmark_df[["timestamp", "ticker", "target", "prediction"]].copy()
    m["correct_model"]     = _correctness(m)
    b["correct_benchmark"] = _correctness(b)
    m = m.drop_duplicates(subset=["timestamp", "ticker"], keep="first")
    b = b.drop_duplicates(subset=["timestamp", "ticker"], keep="first")
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
                  benchmarks: Sequence[str] = DEFAULT_BENCHMARKS,
                  allow_cross_model_benchmark: bool = False) -> pd.DataFrame:
    """Long-form McNemar table: one row per
    (horizon × model_type × panel_mode × set × benchmark).

    The benchmark itself is excluded from the rows it benchmarks (you don't
    test ECON against ECON). Rows whose ``category`` is not in
    :data:`ELIGIBLE_CATEGORIES` are skipped — i.e. ECON itself never appears
    on the model side because the thesis H1 question is whether sentiment
    adds value on top of ECON, not whether ECON beats itself.

    Benchmark matching is **confined to the same model family**
    (horizon + model_type + panel_mode). A panel-logit model is therefore
    never compared against a per-asset benchmark unless
    ``allow_cross_model_benchmark=True`` is passed explicitly (default
    ``False`` — no implicit cross-family fallback).
    """
    if signals.empty or "category" not in signals.columns:
        return pd.DataFrame()
    signals = ensure_group_columns(signals)
    rows: list[dict] = []
    for fam_keys, fam_grp in signals.groupby(FAMILY_COLS, dropna=False):
        bench_frames = _family_benchmark_frames(
            fam_grp, signals, fam_keys, benchmarks, allow_cross_model_benchmark,
        )
        for keys, grp in fam_grp.groupby(list(GROUP_KEYS), dropna=False):
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
    front = FRONT_META + ["benchmark", "mcnemar_stat", "mcnemar_pval",
                          "significant_vs_benchmark", "b", "c", "n_matched"]
    return _front_order(out, front)


def mcnemar_wide(mcnemar_long: pd.DataFrame,
                 benchmarks: Sequence[str] = DEFAULT_BENCHMARKS) -> pd.DataFrame:
    """Pivot the long-form McNemar table into one row per (horizon × set).

    Each benchmark contributes its own ``_vs_<bid>`` columns so the result can
    be merged onto the pooled metrics table without colliding.
    """
    if mcnemar_long is None or mcnemar_long.empty:
        return pd.DataFrame()
    mcnemar_long = ensure_group_columns(mcnemar_long)
    id_cols = ["horizon", "set_id", "sentiment_model", "model_type",
               "panel_mode", "hpo_variant"]
    pieces = []
    for bid in benchmarks:
        sub = mcnemar_long[mcnemar_long["benchmark"] == bid].copy()
        if sub.empty:
            continue
        suffix = f"_vs_{bid.lower()}"
        keep = id_cols + ["mcnemar_stat", "mcnemar_pval",
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
        out = out.merge(nxt, on=id_cols, how="outer")
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
    # Identity
    "regime_type", "horizon", "model_type", "panel_mode", "hpo_variant",
    "set_id", "category", "sentiment_model",
    "benchmark", "vol_regime", "mcap_regime", "interaction",
    # Direction + effect size up front so the table reads at a glance.
    "direction", "net_improvement", "abs_net_improvement",
    "improvement_rate", "discordant_advantage",
    # Raw counts
    "b", "c", "n_matched", "discordant_n",
    # Significance (p / q never hidden)
    "mcnemar_stat", "mcnemar_pval", "q_value_bh",
    "significant_5pct", "significant_10pct",
    "significant_bh_5pct",
    # Direction-aware significance
    "model_better_raw_5pct", "benchmark_better_raw_5pct",
    "model_better_bh_5pct", "benchmark_better_bh_5pct",
    "practical_effect_flag", "test_valid",
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
    # Strict intersection: dedupe each side on the observation key so the
    # regime-stratified McNemar matches on the candidate's true coverage
    # (commit 12 Task E). No-op when both sides are already clean.
    m = m.drop_duplicates(subset=["timestamp", "ticker"], keep="first")
    b = b.drop_duplicates(subset=["timestamp", "ticker"], keep="first")
    return m.merge(
        b[["timestamp", "ticker", "correct_benchmark"]],
        on=["timestamp", "ticker"], how="inner",
    )


def mcnemar_by_group(signals: pd.DataFrame,
                     group_cols: list[str],
                     benchmarks: Sequence[str] = DEFAULT_BENCHMARKS,
                     min_n_matched: int = MIN_N_MATCHED_DEFAULT,
                     min_discordant: int = MIN_DISCORDANT_DEFAULT,
                     min_effect_rate: float = 0.01,
                     allow_cross_model_benchmark: bool = False) -> pd.DataFrame:
    """Continuity-corrected McNemar per
    (horizon × set_id × sentiment_model × benchmark × group_cols).

    Only sentiment / combined sets are tested; benchmark sets are excluded
    from the model side. A cell is ``test_valid`` only when
    ``n_matched >= min_n_matched`` **and** ``discordant_n >= min_discordant``;
    invalid cells carry NaN statistics and ``False`` significance flags.

    Effect-size / direction columns (McNemar significance is *symmetric*, so
    direction must be read off b vs c):

    * ``net_improvement = c - b`` — the central effect: > 0 means the model
      corrects more benchmark errors than vice versa; < 0 the opposite.
    * ``abs_net_improvement = |c - b|``
    * ``improvement_rate = (c - b) / n_matched`` — net effect per matched obs.
    * ``discordant_advantage = c / (b + c)`` — share of *disagreement* cases
      the model wins; > 0.5 ⇒ model is right more often when they disagree.
      NaN when ``b + c == 0``.
    * ``direction`` ∈ {model_better, benchmark_better, tie}.
    * ``practical_effect_flag`` — valid AND |improvement_rate| ≥
      ``min_effect_rate`` AND ``discordant_n ≥ min_discordant``.
    * ``model_better_raw_5pct`` / ``benchmark_better_raw_5pct`` —
      raw-significant AND the corresponding direction.
    """
    if signals.empty or "category" not in signals.columns:
        return pd.DataFrame()
    missing = [c for c in group_cols if c not in signals.columns]
    if missing:
        get_logger().warning("regime-mcnemar: missing group columns %s — skipping",
                             missing)
        return pd.DataFrame()
    signals = ensure_group_columns(signals)

    rows: list[dict] = []
    for fam_keys, fam_grp in signals.groupby(FAMILY_COLS, dropna=False):
        bench_frames = _family_benchmark_frames(
            fam_grp, signals, fam_keys, benchmarks, allow_cross_model_benchmark,
            context="regime-mcnemar",
        )
        for keys, grp in fam_grp.groupby(list(GROUP_KEYS), dropna=False):
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

                    # ── Effect size + direction ──────────────────
                    net_improvement = c - b
                    abs_net = abs(net_improvement)
                    improvement_rate = (net_improvement / n_matched) if n_matched else np.nan
                    discordant_advantage = (c / discordant) if discordant > 0 else np.nan
                    if c > b:
                        direction = "model_better"
                    elif b > c:
                        direction = "benchmark_better"
                    else:
                        direction = "tie"
                    practical_effect_flag = bool(
                        test_valid
                        and (improvement_rate == improvement_rate)  # not NaN
                        and abs(improvement_rate) >= min_effect_rate
                        and discordant >= min_discordant
                    )
                    model_better_raw_5pct     = bool(sig5 and direction == "model_better")
                    benchmark_better_raw_5pct = bool(sig5 and direction == "benchmark_better")

                    row = {
                        "horizon":          keys[0],
                        "set_id":           keys[1],
                        "sentiment_model":  keys[2],
                        "model_type":       keys[3],
                        "panel_mode":       keys[4],
                        "hpo_variant":      keys[5],
                        "category":         category,
                        "benchmark":        bid,
                        "direction":        direction,
                        "net_improvement":  net_improvement,
                        "abs_net_improvement": abs_net,
                        "improvement_rate": improvement_rate,
                        "discordant_advantage": discordant_advantage,
                        "b":                b,
                        "c":                c,
                        "n_matched":        n_matched,
                        "discordant_n":     discordant,
                        "mcnemar_stat":     stat,
                        "mcnemar_pval":     pval,
                        "significant_5pct":  sig5,
                        "significant_10pct": sig10,
                        "model_better_raw_5pct":     model_better_raw_5pct,
                        "benchmark_better_raw_5pct": benchmark_better_raw_5pct,
                        "practical_effect_flag":     practical_effect_flag,
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
    """Append BH/FDR columns: ``q_value_bh``, ``significant_bh_5pct``.

    Only valid tests (non-NaN p-value) enter the FDR pool. Uses
    ``statsmodels.stats.multitest.multipletests`` when available, otherwise
    a pure-numpy BH fallback.

    NOTE (commit 15): the regime-McNemar surface is DESCRIPTIVE ONLY —
    it is superseded by the cluster-robust difference-in-improvement
    (confirmatory Families C/D) and is never listed as a confirmatory
    family. The 10% flag was removed globally.
    """
    out = df.copy()
    out["q_value_bh"] = np.nan
    out["significant_bh_5pct"] = False
    if out.empty or p_col not in out.columns:
        return out

    mask = out[p_col].notna()
    pvals = out.loc[mask, p_col].to_numpy(dtype=float)
    if len(pvals) == 0:
        return out

    q5 = rej5 = None
    try:
        from statsmodels.stats.multitest import multipletests
        rej5,  q5,  _, _ = multipletests(pvals, alpha=0.05, method="fdr_bh")
    except Exception:  # noqa: BLE001 — fall back to the numpy BH
        q5 = _bh_qvalues(pvals)
        rej5  = q5 < 0.05

    out.loc[mask, "q_value_bh"]          = q5
    out.loc[mask, "significant_bh_5pct"] = rej5
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
        # ``attach_regimes`` already emits ``vol_regime`` (and a legacy
        # ``regime`` alias). The rename used to handle older lookups
        # that only carried ``regime``; do it only when the alias is
        # the sole regime column to avoid creating duplicate names.
        if "vol_regime" not in enr.columns and "regime" in enr.columns:
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
        # ``attach_regimes`` already emits ``vol_regime`` (and a legacy
        # ``regime`` alias). The rename used to handle older lookups
        # that only carried ``regime``; do it only when the alias is
        # the sole regime column to avoid creating duplicate names.
        if "vol_regime" not in enr.columns and "regime" in enr.columns:
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

    # Direction-aware BH flags (McNemar is symmetric → split by direction).
    direction = out.get("direction", pd.Series(["tie"] * len(out)))
    bh5 = out.get("significant_bh_5pct", pd.Series([False] * len(out))).fillna(False)
    out["model_better_bh_5pct"]     = bh5 & (direction == "model_better")
    out["benchmark_better_bh_5pct"] = bh5 & (direction == "benchmark_better")

    ordered = [c for c in _REGIME_FRONT_COLS if c in out.columns]
    rest = [c for c in out.columns if c not in ordered]
    return out[ordered + rest].reset_index(drop=True)


def build_regime_mcnemar_summary(regime_mcnemar_df: pd.DataFrame) -> pd.DataFrame:
    """Distil the regime McNemar table into thesis-ready highlight rows.

    Sections produced (each contributes 0-1 rows, sorted as described):

    * ``top_model_better_raw`` / ``top_model_better_bh`` — strongest cells
      where the *model* beats the benchmark (raw / BH-significant). Sorted
      BH-significant first, then ``q_value_bh`` ascending, then
      ``net_improvement`` descending.
    * ``top_benchmark_better_raw`` / ``top_benchmark_better_bh`` — strongest
      cells where the *benchmark* wins. Sorted BH-significant first, then
      ``q_value_bh`` ascending, then ``net_improvement`` ascending (most
      negative first).
    * ``best_small_high_model`` / ``best_small_low_model`` — best model cell
      in the small-cap × high-vol / small-cap × low-vol interaction regime.
    * ``best_cryptobert_model_better`` — best model cell for the cryptobert
      scorer.
    * ``strongest_benchmark_win`` — most negative ``net_improvement`` overall.

    Always returns a DataFrame (possibly empty) — never raises on empty
    input. Regime McNemar stays exploratory; interpret with the
    multiple-testing caveat in mind.
    """
    cols = ["section", "horizon", "model_type", "panel_mode", "hpo_variant",
            "set_id", "sentiment_model", "benchmark",
            "regime_type", "mcap_regime", "vol_regime", "interaction",
            "direction", "net_improvement", "improvement_rate",
            "discordant_advantage", "n_matched", "discordant_n",
            "mcnemar_pval", "q_value_bh"]
    if regime_mcnemar_df is None or regime_mcnemar_df.empty:
        return pd.DataFrame(columns=cols)

    df = regime_mcnemar_df[regime_mcnemar_df.get("test_valid", False) == True].copy()  # noqa: E712
    if df.empty:
        return pd.DataFrame(columns=cols)

    # Stable helper to pull the columns we want out of one row.
    def _row(section: str, r: pd.Series) -> dict:
        out = {"section": section}
        for c in cols[1:]:
            out[c] = r.get(c, np.nan)
        return out

    rows: list[dict] = []

    def _sorted_model(frame: pd.DataFrame) -> pd.DataFrame:
        # BH-significant first, then q ascending, then net_improvement desc.
        f = frame.copy()
        f["_bh"] = f.get("significant_bh_5pct", False).fillna(False).astype(int)
        f["_q"] = f.get("q_value_bh", np.nan)
        return f.sort_values(["_bh", "_q", "net_improvement"],
                             ascending=[False, True, False])

    def _sorted_bench(frame: pd.DataFrame) -> pd.DataFrame:
        f = frame.copy()
        f["_bh"] = f.get("significant_bh_5pct", False).fillna(False).astype(int)
        f["_q"] = f.get("q_value_bh", np.nan)
        return f.sort_values(["_bh", "_q", "net_improvement"],
                             ascending=[False, True, True])

    model = df[df["direction"] == "model_better"]
    bench = df[df["direction"] == "benchmark_better"]

    if not model.empty:
        rows.append(_row("top_model_better_raw",
                         _sorted_model(model[model.get("significant_5pct", False) == True]  # noqa: E712
                                       if (model.get("significant_5pct", False) == True).any()  # noqa: E712
                                       else model).iloc[0]))
        bh_model = model[model.get("significant_bh_5pct", False) == True]  # noqa: E712
        if not bh_model.empty:
            rows.append(_row("top_model_better_bh", _sorted_model(bh_model).iloc[0]))

    if not bench.empty:
        rows.append(_row("top_benchmark_better_raw",
                         _sorted_bench(bench[bench.get("significant_5pct", False) == True]  # noqa: E712
                                       if (bench.get("significant_5pct", False) == True).any()  # noqa: E712
                                       else bench).iloc[0]))
        bh_bench = bench[bench.get("significant_bh_5pct", False) == True]  # noqa: E712
        if not bh_bench.empty:
            rows.append(_row("top_benchmark_better_bh", _sorted_bench(bh_bench).iloc[0]))

    # Interaction-specific highlights (model side).
    inter = model[model["regime_type"] == "interaction"]
    for label, mcap_r, vol_r in (("best_small_high_model", "small", "high"),
                                 ("best_small_low_model",  "small", "low")):
        sub = inter[(inter["mcap_regime"] == mcap_r) & (inter["vol_regime"] == vol_r)]
        if not sub.empty:
            rows.append(_row(label, _sorted_model(sub).iloc[0]))

    # Best cryptobert model cell.
    cb = model[model["sentiment_model"].astype(str).str.lower() == "cryptobert"]
    if not cb.empty:
        rows.append(_row("best_cryptobert_model_better", _sorted_model(cb).iloc[0]))

    # Strongest benchmark win overall (most negative net_improvement).
    if not bench.empty:
        rows.append(_row("strongest_benchmark_win",
                         bench.sort_values("net_improvement", ascending=True).iloc[0]))

    return pd.DataFrame(rows, columns=cols)
