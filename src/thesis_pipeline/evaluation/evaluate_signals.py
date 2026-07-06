"""Signal evaluation stage — canonical entry point.

This module is the orchestrator. The actual computations live in sibling
modules so that each piece is independently testable:

* :mod:`.loading`        — robust signal parquet loading
* :mod:`.metrics`        — pooled + per-ticker metrics + confusion diagnostics
* :mod:`.thresholds`     — high-conviction threshold analysis + lift vs benchmark
* :mod:`.volatility`     — Garman-Klass + tertile regime stratification
* :mod:`.significance`   — continuity-corrected McNemar vs ECON
* :mod:`.market_cap`     — daily cross-sectional cap tertiles + interaction
* :mod:`.economic`       — turnover/cost-aware backtest with risk metrics
* :mod:`.reporting`      — Excel + CSV + console summary

CLI:

    python -m thesis_pipeline.cli evaluate-signals --horizon 1d --smoke
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

import pandas as pd

from ..config import load_config, resolve_path
from ..logging_utils import get_logger, log_stage_header
from ..modeling.run_models import load_feature_sets
from .loading import (
    attach_feature_set_metadata, discover_signal_files, load_all_signals,
)
from .market_cap import (
    build_market_cap_lookup, build_market_cap_summary,
    market_cap_stratification_table, regime_interaction_table,
)
from .metrics import pooled_metrics_table, per_ticker_metrics_table
from .economic import (
    attach_benchmark_lifts, buy_and_hold_benchmark,
    economic_group_diagnostics,
    load_backtest_config, load_forward_returns,
    summarize_backtest_by_ticker,
    summarize_high_low_backtest,
    summarize_high_low_threshold_backtest,
)
from .incremental import incremental_sentiment_value_table
from .confirmatory import finalize_confirmatory_families
from .naive_comparison import absolute_vs_naive_table
from .reporting import (
    build_leaderboard, build_summary,
    print_console_summary, write_csv_outputs, write_excel_report,
)
from .significance import (
    DEFAULT_BENCHMARKS, build_regime_mcnemar_summary,
    mcnemar_table, mcnemar_wide, regime_mcnemar_table,
)
from .diff_in_improvement import (
    adjust_pvalues_bh_within_family,
    difference_in_improvement_table,
    H_HYPOTHESIS_FAMILIES,
)
from .incremental import MATCHED_ECONOMIC_BENCHMARK
from .thresholds import threshold_analysis_table, threshold_lift_table
from .volatility import build_regime_lookup, volatility_stratification_table


# ---------------------------------------------------------------------------
# CLI / argparse
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Signal evaluation — pooled / per-ticker metrics, McNemar, "
                    "volatility + market-cap regimes, threshold lift, "
                    "and turnover/cost-aware backtests."
    )
    parser.add_argument("--horizon", default=None, choices=["1h", "6h", "1d"],
                        help="Restrict to a single horizon. Default: all available.")
    parser.add_argument("--output-dir", "--output_dir", dest="output_dir",
                        default=None,
                        help="Override Outputs/Evaluation/.")
    parser.add_argument("--feature_config", "--feature-config",
                        dest="feature_config", default=None,
                        help="Path to feature_sets.xlsx; defaults to configs/paths.yaml.")
    parser.add_argument("--backtest-config", "--backtest_config",
                        dest="backtest_config", default=None,
                        help="Override configs/backtest.yaml.")
    parser.add_argument("--transaction-cost-bps", "--transaction_cost_bps",
                        dest="transaction_cost_bps", type=float, nargs="*",
                        default=None,
                        help="Override the cost grid from configs/backtest.yaml "
                             "(one or more values in basis points).")
    parser.add_argument("--smoke", action="store_true",
                        help="Smoke mode: horizon=1d if not given; writes under smoke_root.")
    parser.add_argument("--dry-run", "--dry_run", dest="dry_run",
                        action="store_true",
                        help="Print planned inputs/outputs and exit.")
    parser.add_argument("--force", action="store_true",
                        help="Allow overwriting full production outputs.")
    parser.add_argument("--no-volatility", dest="no_volatility",
                        action="store_true",
                        help="Skip the volatility-stratification step.")
    parser.add_argument("--no-market-cap", "--no_market_cap",
                        dest="no_market_cap", action="store_true",
                        help="Skip the market-cap stratification + interaction step.")
    parser.add_argument("--no-economic", "--no_economic",
                        dest="no_economic", action="store_true",
                        help="Skip the turnover/cost-aware backtest step.")
    parser.add_argument("--no-regime-mcnemar", "--no_regime_mcnemar",
                        dest="no_regime_mcnemar", action="store_true",
                        help="Skip the SUPPLEMENTARY within-regime McNemar tests. "
                             "Does NOT disable the H2/H3 headline "
                             "difference-in-improvement layer — use "
                             "--no-diff-in-improvement for that.")
    parser.add_argument("--no-diff-in-improvement", "--no_diff_in_improvement",
                        dest="no_diff_in_improvement", action="store_true",
                        help="Skip the HEADLINE H2/H3 cluster-robust "
                             "difference-in-improvement layer. Default ON when "
                             "the corresponding regime lookup is available; "
                             "independent of --no-regime-mcnemar.")
    parser.add_argument("--strict-feature-set-ids", "--strict_feature_set_ids",
                        dest="strict_feature_set_ids", action="store_true",
                        help="Only evaluate signal groups whose set_id appears in "
                             "the active feature_sets.xlsx. Default: evaluate every "
                             "loaded signal file (legacy / stale IDs are still "
                             "scored, just flagged in economic_diagnostics.csv).")
    return parser


# ---------------------------------------------------------------------------
# Core stage
# ---------------------------------------------------------------------------

def _output_root(args: argparse.Namespace) -> Path:
    if args.output_dir:
        return Path(args.output_dir)
    if args.smoke and not args.force:
        return resolve_path("smoke_root") / "evaluation"
    return resolve_path("evaluation_root")


def _xlsx_path(out_root: Path) -> Path:
    return out_root / "signal_evaluation.xlsx"


def run(*, horizon: str | None = None,
        output_dir: str | Path | None = None,
        smoke: bool = False, dry_run: bool = False,
        force: bool = False, no_volatility: bool = False,
        no_market_cap: bool = False, no_economic: bool = False,
        no_regime_mcnemar: bool = False,
        no_diff_in_improvement: bool = False,
        strict_feature_set_ids: bool = False,
        feature_config: str | None = None,
        backtest_config: str | None = None,
        transaction_cost_bps: list[float] | None = None) -> int:
    """Programmatic entry point. Mirrors :func:`main` but takes kwargs."""
    argv: list[str] = []
    if horizon:
        argv += ["--horizon", horizon]
    if output_dir:
        argv += ["--output-dir", str(output_dir)]
    if feature_config:
        argv += ["--feature-config", feature_config]
    if backtest_config:
        argv += ["--backtest-config", backtest_config]
    if transaction_cost_bps:
        argv += ["--transaction-cost-bps", *(str(c) for c in transaction_cost_bps)]
    if smoke:
        argv.append("--smoke")
    if dry_run:
        argv.append("--dry-run")
    if force:
        argv.append("--force")
    if no_volatility:
        argv.append("--no-volatility")
    if no_market_cap:
        argv.append("--no-market-cap")
    if no_economic:
        argv.append("--no-economic")
    if no_regime_mcnemar:
        argv.append("--no-regime-mcnemar")
    if no_diff_in_improvement:
        argv.append("--no-diff-in-improvement")
    if strict_feature_set_ids:
        argv.append("--strict-feature-set-ids")
    return main(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    # ── Smoke defaults ──────────────────────────────────────────
    if args.smoke and not args.horizon:
        args.horizon = "1d"

    # ── Path resolution ─────────────────────────────────────────
    out_root = _output_root(args)
    xlsx_path = _xlsx_path(out_root)
    inputs = [resolve_path("signals_root"),
              Path(args.feature_config) if args.feature_config
              else resolve_path("feature_sets_xlsx"),
              resolve_path("raw_price_1d")]
    if not args.no_market_cap:
        try:
            inputs.append(resolve_path("cmc_market_cap"))
        except KeyError:
            pass
    outputs = [xlsx_path,
               out_root / "pooled_metrics.csv",
               out_root / "per_ticker_metrics.csv",
               out_root / "threshold_analysis.csv",
               out_root / "volatility_stratification.csv",
               out_root / "absolute_vs_naive.csv"]
    if not args.no_market_cap:
        outputs += [out_root / "market_cap_stratification.csv",
                    out_root / "regime_interaction.csv"]
    if not args.no_economic:
        outputs += [out_root / "economic_performance.csv",
                    out_root / "economic_performance_by_threshold.csv",
                    out_root / "economic_performance_by_ticker.csv",
                    out_root / "economic_diagnostics.csv"]
    if not args.no_regime_mcnemar:
        outputs.append(out_root / "regime_mcnemar_tests.csv")
        outputs.append(out_root / "regime_mcnemar_summary.csv")

    # Universe-aware diagnostics for the dry-run (commit 3 Section H).
    from .naive_comparison import (
        NAIVE_IDENTITY_COLUMNS as _abs_identity,
        HYPOTHESIS_FAMILY as _abs_family,
        ABSOLUTE_BENCHMARK_ROLE as _abs_role,
    )
    log_stage_header(
        "evaluate_signals",
        mode="dry-run" if args.dry_run else ("smoke" if args.smoke else "full"),
        inputs=inputs,
        outputs=outputs,
        extra={
            "horizon":               args.horizon or "(all)",
            "output_dir":            str(out_root),
            "no_volatility":         args.no_volatility,
            "no_market_cap":         args.no_market_cap,
            "no_economic":           args.no_economic,
            "no_regime_mcnemar":     args.no_regime_mcnemar,
            "no_diff_in_improvement": getattr(args, "no_diff_in_improvement", False),
            "strict_feature_set_ids": args.strict_feature_set_ids,
            "strict_filter_keeps_naive": True,
            # Section E — every regime consumer is on the shared
            # availability-based strict-< as-of join.
            "regime_join_strategy": "availability-based strict backward as-of",
            "regime_match_strategy": "asof_backward_strict",
            # Section H — surface the absolute_vs_naive identity columns
            # AND the legacy-fallback flag in the dry-run so users can see
            # the exact contract upfront.
            "absolute_vs_naive_csv":        str(out_root / "absolute_vs_naive.csv"),
            "absolute_vs_naive_hypothesis_family": _abs_family,
            "absolute_vs_naive_benchmark_role":   _abs_role,
            "absolute_vs_naive_identity_columns": ",".join(_abs_identity),
            "legacy_universe_fallback_enabled":   True,
            "transaction_cost_bps":  args.transaction_cost_bps or "(from config)",
            "backtest_config":       args.backtest_config or "(default)",
            "force":                 args.force,
        },
    )

    if args.dry_run:
        return 0

    logger = get_logger()

    # ── 1. Discover and load signal files ───────────────────────
    files = discover_signal_files(args.horizon)
    if not files:
        logger.warning("evaluate-signals: no signal parquets found under %s — nothing to do",
                       resolve_path("signals_root"))
        return 0
    logger.info("evaluate-signals: discovered %d signal file(s)", len(files))
    signals = load_all_signals(args.horizon, paths=files)
    if signals.empty:
        logger.warning("evaluate-signals: no usable signal rows after loading")
        return 0
    logger.info("evaluate-signals: loaded %d signal rows across %d tickers",
                len(signals), signals["ticker"].nunique())

    # ── 2. Attach category/label metadata from feature_sets.xlsx ─
    fs_path = args.feature_config or str(resolve_path("feature_sets_xlsx"))
    try:
        feature_sets = load_feature_sets(fs_path)
    except Exception as exc:  # noqa: BLE001
        logger.warning("evaluate-signals: cannot load %s (%s) — category/label "
                       "columns will be empty", fs_path, exc)
        feature_sets = pd.DataFrame(columns=["set_id", "category", "label"])
    signals = attach_feature_set_metadata(signals, feature_sets)

    # ── 2b. Optional strict feature-set-id filter ───────────────
    # Without the flag we evaluate every signal file on disk and let
    # economic_diagnostics.csv flag stale/legacy set_ids. With it, only
    # set_ids registered in the active feature_sets.xlsx (PLUS the
    # documented evaluation references — NAIVE) survive. Aufgabe 6
    # follow-up C: NAIVE is intentionally NOT a feature set in
    # SET_ID_PATTERN, but it IS an explicitly allowed evaluation
    # reference and must survive the filter alongside the 17-set grid.
    if args.strict_feature_set_ids:
        if "set_id" not in feature_sets.columns or feature_sets["set_id"].empty:
            logger.warning(
                "evaluate-signals: --strict-feature-set-ids requested but "
                "feature_sets.xlsx has no set_id column / is empty — filter "
                "would drop everything; ignoring the flag."
            )
        else:
            from .incremental import NAIVE_REFERENCE_LABEL
            registered = set(feature_sets["set_id"].dropna().astype(str))
            evaluation_refs = {NAIVE_REFERENCE_LABEL}
            allowed = registered | evaluation_refs
            before_n = len(signals)
            present  = set(signals["set_id"].astype(str).unique())
            stale    = sorted(present - allowed)
            signals  = signals[signals["set_id"].astype(str).isin(allowed)]
            logger.info(
                "evaluate-signals: --strict-feature-set-ids kept %d / %d rows "
                "(%d registered feature-set IDs, %d evaluation references: %s, "
                "%d stale IDs dropped: %s)",
                len(signals), before_n, len(registered),
                len(evaluation_refs), ", ".join(sorted(evaluation_refs)),
                len(stale), ", ".join(stale) if stale else "(none)",
            )
            if signals.empty:
                logger.warning("evaluate-signals: no signal rows survive strict "
                               "feature-set-id filter — nothing to evaluate")
                return 0

    # ── 3. Pooled + per-ticker metrics ──────────────────────────
    pooled     = pooled_metrics_table(signals)
    per_ticker = per_ticker_metrics_table(signals)

    # ── 4. Threshold / conviction analysis ──────────────────────
    threshold_df      = threshold_analysis_table(signals)
    threshold_lift_df = threshold_lift_table(signals, benchmarks=DEFAULT_BENCHMARKS)

    # ── 5. McNemar significance vs the v4 matched benchmark (ECON) ──
    mcnemar_df_long = mcnemar_table(signals, benchmarks=DEFAULT_BENCHMARKS)
    mcnemar_wide_df = mcnemar_wide(mcnemar_df_long, benchmarks=DEFAULT_BENCHMARKS)
    if not pooled.empty and not mcnemar_wide_df.empty:
        pooled = pooled.merge(
            mcnemar_wide_df,
            on=["horizon", "set_id", "sentiment_model", "model_type",
                "panel_mode", "hpo_variant"],
            how="left",
        )
    # Back-compat aliases for downstream consumers that grep on the bare
    # ``mcnemar_*`` columns. v4 uses ECON as the single matched benchmark,
    # but we still mirror the disambiguated columns under the bare names
    # (mirroring whatever benchmark the wide pivot actually populated).
    _bench_alias_source = None
    for _bid in DEFAULT_BENCHMARKS:
        _src = f"mcnemar_pval_vs_{_bid.lower()}"
        if _src in pooled.columns:
            _bench_alias_source = _bid.lower()
            break
    if _bench_alias_source is not None and "mcnemar_pval" not in pooled.columns:
        pooled["mcnemar_stat"]             = pooled[f"mcnemar_stat_vs_{_bench_alias_source}"]
        pooled["mcnemar_pval"]             = pooled[f"mcnemar_pval_vs_{_bench_alias_source}"]
        pooled["significant_vs_benchmark"] = pooled.get(
            f"significant_vs_{_bench_alias_source}", False,
        )

    tickers = sorted(signals["ticker"].dropna().astype(str).str.upper().unique())

    # ── 6. Volatility regime stratification ─────────────────────
    vol_lookup = pd.DataFrame()
    if args.no_volatility:
        volatility_df = pd.DataFrame()
        logger.info("evaluate-signals: --no-volatility set, skipping stratification")
    else:
        vol_lookup = build_regime_lookup(tickers)
        if vol_lookup.empty:
            logger.warning("evaluate-signals: no daily price data found — "
                           "volatility stratification will be empty")
            volatility_df = pd.DataFrame()
        else:
            volatility_df = volatility_stratification_table(signals, vol_lookup)

    # ── 7. Market-cap regime stratification + interaction ───────
    mcap_df         = pd.DataFrame()
    interaction_df  = pd.DataFrame()
    extra_summary: list[dict] = []
    mcap_lookup     = pd.DataFrame()
    if args.no_market_cap:
        logger.info("evaluate-signals: --no-market-cap set, skipping mcap stratification")
    else:
        # Constrain the cross-section to the model's signal universe.
        mcap_lookup = build_market_cap_lookup(tickers=tickers)
        if mcap_lookup.empty:
            logger.warning("evaluate-signals: market-cap lookup is empty — "
                           "mcap_stratification + regime_interaction tables "
                           "will be empty")
        else:
            mcap_df = market_cap_stratification_table(signals, mcap_lookup)
            if not vol_lookup.empty:
                interaction_df = regime_interaction_table(
                    signals, mcap_lookup, vol_lookup,
                )
            else:
                logger.info("evaluate-signals: skipping regime_interaction — "
                            "volatility lookup is empty")
        extra_summary = build_market_cap_summary(mcap_df, interaction_df)

    # ── 7b. Regime-specific McNemar tests (supplementary) ───────
    regime_mcnemar_df = pd.DataFrame()
    regime_mcnemar_summary_df = pd.DataFrame()
    if args.no_regime_mcnemar:
        logger.info("evaluate-signals: --no-regime-mcnemar set, skipping regime McNemar")
    else:
        has_vol  = not vol_lookup.empty
        has_mcap = not mcap_lookup.empty
        if not has_vol and not has_mcap:
            logger.info("evaluate-signals: no regime lookups available — "
                        "skipping regime McNemar")
        else:
            regime_mcnemar_df = regime_mcnemar_table(
                signals,
                vol_lookup=vol_lookup if has_vol else None,
                mcap_lookup=mcap_lookup if has_mcap else None,
                benchmarks=DEFAULT_BENCHMARKS,
            )
            regime_mcnemar_summary_df = build_regime_mcnemar_summary(regime_mcnemar_df)

    # ── 7c. H2 / H3 difference-in-improvement (Aufgabe 6.4) ─────
    # Cluster-robust regression of d_{i,t} = 1(aug correct) - 1(ECON correct)
    # on a regime dummy, clustered by ticker. The HEADLINE H2/H3 test is
    # independent of the SUPPLEMENTARY within-regime McNemar block above:
    # --no-regime-mcnemar suppresses ONLY the supplementary tests, while
    # --no-diff-in-improvement suppresses this headline block. By default
    # H2 runs whenever the volatility lookup is non-empty and H3 runs
    # whenever the market-cap lookup is non-empty.
    diff_in_improvement_df = pd.DataFrame()
    if not getattr(args, "no_diff_in_improvement", False):
        h_blocks: list[pd.DataFrame] = []
        if not vol_lookup.empty:
            vol_lk = vol_lookup.copy()
            if "regime" in vol_lk.columns and "vol_regime" not in vol_lk.columns:
                vol_lk = vol_lk.rename(columns={"regime": "vol_regime"})
            h_blocks.append(difference_in_improvement_table(
                signals=signals,
                matched_benchmark=MATCHED_ECONOMIC_BENCHMARK,
                regime_lookup=vol_lk,
                regime_col="vol_regime",
                treatment_value="high",
                control_value="low",
                hypothesis_family="H2_volatility",
            ))
        if not mcap_lookup.empty:
            mc_lk = mcap_lookup.copy()
            if "regime" in mc_lk.columns and "mcap_regime" not in mc_lk.columns:
                mc_lk = mc_lk.rename(columns={"regime": "mcap_regime"})
            h_blocks.append(difference_in_improvement_table(
                signals=signals,
                matched_benchmark=MATCHED_ECONOMIC_BENCHMARK,
                regime_lookup=mc_lk,
                regime_col="mcap_regime",
                treatment_value="small",
                control_value="large",
                hypothesis_family="H3_market_cap",
            ))
        if h_blocks:
            diff_in_improvement_df = pd.concat(
                [b for b in h_blocks if not b.empty], ignore_index=True,
            )
            if not diff_in_improvement_df.empty:
                diff_in_improvement_df = adjust_pvalues_bh_within_family(
                    diff_in_improvement_df,
                )

    # ── 8. Economic / backtest layer ────────────────────────────
    economic_df      = pd.DataFrame()
    economic_thr_df  = pd.DataFrame()
    economic_tk_df   = pd.DataFrame()
    economic_diag_df = pd.DataFrame()
    if args.no_economic:
        logger.info("evaluate-signals: --no-economic set, skipping backtest")
    else:
        bcfg = load_backtest_config(args.backtest_config)
        if args.transaction_cost_bps:
            bcfg["transaction_cost_bps"] = list(args.transaction_cost_bps)
        # Forward returns are loaded per-horizon to respect bar granularity.
        for hz in sorted(signals["horizon"].dropna().astype(str).unique()):
            hz_signals = signals[signals["horizon"] == hz]
            if hz_signals.empty:
                continue
            hz_tickers = sorted(hz_signals["ticker"].astype(str).str.upper().unique())
            fr = load_forward_returns(hz_tickers, horizon=hz)
            # Always build diagnostics — even when no FR data exists — so the
            # CSV records *every* attempted (horizon × group) with a reason.
            economic_diag_df = pd.concat(
                [economic_diag_df,
                 economic_group_diagnostics(hz_signals, fr, bcfg, horizon=hz)],
                ignore_index=True,
            )
            if fr.empty:
                logger.warning("evaluate-signals: no close-price data for %s — "
                               "backtest will skip this horizon", hz)
                continue
            economic_df     = pd.concat(
                [economic_df,
                 summarize_high_low_backtest(hz_signals, fr, bcfg, horizon=hz)],
                ignore_index=True,
            )
            economic_thr_df = pd.concat(
                [economic_thr_df,
                 summarize_high_low_threshold_backtest(hz_signals, fr, bcfg,
                                                       horizon=hz)],
                ignore_index=True,
            )
            economic_tk_df  = pd.concat(
                [economic_tk_df,
                 summarize_backtest_by_ticker(hz_signals, fr, bcfg, horizon=hz)],
                ignore_index=True,
            )
            # Buy-and-hold reference rows — one per cost level (was: cost=0 only).
            if bcfg.get("include_buy_and_hold"):
                bh = buy_and_hold_benchmark(hz_tickers, fr, bcfg, hz)
                if bh is not None and not bh.empty:
                    economic_df = pd.concat([economic_df, bh], ignore_index=True)

        # Append per-benchmark lift columns (sharpe / cumulative_return).
        if not economic_df.empty:
            economic_df = attach_benchmark_lifts(
                economic_df, bcfg.get("benchmark_ids", ["ECON"]),
            )

    # ── 8b. Incremental sentiment value vs matched economic benchmark ──
    def _warn_missing_bench(horizon, set_id, sm, bench_set_id):
        logger.warning("evaluate-signals: combined %s/%s (%s) has no matched %s "
                       "economic benchmark in the same family — emitting "
                       "missing_benchmark row.", set_id, sm, horizon, bench_set_id)
    incremental_df = incremental_sentiment_value_table(
        signals, warn_missing=_warn_missing_bench,
    )

    # ── 8c. Absolute vs NAIVE comparison (v4 cleanup commit 2 Section F) ──
    # One row per (model run × matched NAIVE) using the COMPLETE NAIVE
    # identity (horizon, model_type, panel_mode, train_window_*,
    # coin_universe_hash). Hypothesis family ``absolute_vs_naive`` —
    # NEVER pooled with H1 / H2 / H3. It is a DIAGNOSTIC FLOOR: it reports
    # only ``p_value_raw`` + ``significant_raw`` (at ALPHA_PRESPECIFIED) and
    # carries NO BH q-value / confirmatory family.
    absolute_vs_naive_df = absolute_vs_naive_table(signals)
    if not absolute_vs_naive_df.empty:
        absolute_vs_naive_df = absolute_vs_naive_df.copy()
        absolute_vs_naive_df["p_value_raw"] = absolute_vs_naive_df["mcnemar_p_value"]
        absolute_vs_naive_df["significant_raw"] = (
            absolute_vs_naive_df["significant_raw_5pct"].astype(bool))
        # A diagnostic floor never carries a corrected q-value / family.
        for _c in ("q_value_bh", "significant_bh"):
            if _c in absolute_vs_naive_df.columns:
                absolute_vs_naive_df = absolute_vs_naive_df.drop(columns=_c)

    # ── 8d. Confirmatory multiplicity (Parts 1 + 2A/2B) ──────────
    # ONE centralized family-aware BH pooled ACROSS horizons within each
    # confirmatory family (A/B/C/D/E1/E2). Emits the enriched incremental
    # (Family A log-loss DM + Family B directional) and diff-in-improvement
    # (Families C/D) surfaces plus the horizon-comparison table (E1/E2),
    # the multiple-testing manifest, the metric-roles table and the
    # class-balance table. This SUPERSEDES the diff table's own BH.
    confirmatory = finalize_confirmatory_families(
        signals=signals,
        incremental_df=incremental_df,
        diff_df=diff_in_improvement_df,
        pooled_df=pooled,
    )
    incremental_df          = confirmatory.incremental
    diff_in_improvement_df  = confirmatory.diff
    horizon_comparison_df   = confirmatory.horizon_comparison
    multiple_testing_manifest_df = confirmatory.manifest
    metric_roles_df         = confirmatory.metric_roles
    class_balance_df        = confirmatory.class_balance
    if not confirmatory.pooled.empty:
        pooled = confirmatory.pooled

    # ── 9. Leaderboard + thesis-style summary ───────────────────
    leaderboard = build_leaderboard(pooled)
    summary     = build_summary(
        pooled, threshold_df, volatility_df,
        threshold_lift=threshold_lift_df,
        market_cap_df=mcap_df,
        interaction_df=interaction_df,
        economic_df=economic_df,
        economic_threshold_df=economic_thr_df,
        regime_mcnemar_df=regime_mcnemar_df,
        extra_rows=extra_summary,
    )

    # ── 10. Sorting (deterministic output) ──────────────────────
    # Sort order keeps the full family (model_type, panel_mode, hpo_variant)
    # adjacent to the horizon so per-asset/panel and fixed/HPO rows never
    # interleave.
    fam = ["model_type", "panel_mode", "hpo_variant"]
    if not pooled.empty:
        pooled = pooled.sort_values(["horizon", *fam, "accuracy"],
                                    ascending=[True, True, True, True, False]
                                    ).reset_index(drop=True)
    if not per_ticker.empty:
        per_ticker = per_ticker.sort_values(
            ["horizon", *fam, "set_id", "sentiment_model", "ticker"]
        ).reset_index(drop=True)
    if not threshold_df.empty:
        threshold_df = threshold_df.sort_values(
            ["horizon", *fam, "set_id", "sentiment_model", "threshold"]
        ).reset_index(drop=True)
    if not threshold_lift_df.empty:
        threshold_lift_df = threshold_lift_df.sort_values(
            ["horizon", *fam, "set_id", "sentiment_model", "benchmark", "threshold"]
        ).reset_index(drop=True)
    if not volatility_df.empty:
        volatility_df = volatility_df.sort_values(
            ["horizon", *fam, "set_id", "sentiment_model", "vol_regime"]
        ).reset_index(drop=True)
    if not mcap_df.empty:
        mcap_df = mcap_df.sort_values(
            ["horizon", *fam, "set_id", "sentiment_model", "mcap_regime"]
        ).reset_index(drop=True)
    if not interaction_df.empty:
        interaction_df = interaction_df.sort_values(
            ["horizon", *fam, "set_id", "sentiment_model", "mcap_regime", "vol_regime"]
        ).reset_index(drop=True)
    if not mcnemar_df_long.empty:
        mcnemar_df_long = mcnemar_df_long.sort_values(
            ["horizon", *fam, "set_id", "sentiment_model", "benchmark"]
        ).reset_index(drop=True)
    if not economic_df.empty:
        economic_df = economic_df.sort_values(
            ["horizon", *fam, "cost_bps", "sharpe"],
            ascending=[True, True, True, True, True, False]).reset_index(drop=True)
    if not economic_thr_df.empty:
        economic_thr_df = economic_thr_df.sort_values(
            ["horizon", *fam, "threshold", "cost_bps", "sharpe"],
            ascending=[True, True, True, True, True, True, False]).reset_index(drop=True)
    if not economic_tk_df.empty:
        economic_tk_df = economic_tk_df.sort_values(
            ["horizon", *fam, "set_id", "sentiment_model", "ticker", "cost_bps"]
        ).reset_index(drop=True)
    if not regime_mcnemar_df.empty:
        regime_mcnemar_df = regime_mcnemar_df.sort_values(
            ["regime_type", "horizon", *fam, "set_id", "sentiment_model", "benchmark"]
        ).reset_index(drop=True)
    if not incremental_df.empty:
        incremental_df = incremental_df.sort_values(
            ["horizon", *fam, "set_id", "sentiment_model"],
        ).reset_index(drop=True)

    # ── 11. Write outputs ───────────────────────────────────────
    csv_paths = write_csv_outputs(
        out_root,
        pooled=pooled, per_ticker=per_ticker,
        threshold=threshold_df, volatility=volatility_df,
        threshold_lift=threshold_lift_df,
        mcnemar=mcnemar_df_long,
        market_cap=(mcap_df         if not args.no_market_cap else None),
        regime_interaction=(interaction_df if not args.no_market_cap else None),
        economic=(economic_df       if not args.no_economic   else None),
        economic_threshold=(economic_thr_df if not args.no_economic else None),
        economic_by_ticker=(economic_tk_df  if not args.no_economic else None),
        economic_diagnostics=(economic_diag_df if not args.no_economic else None),
        regime_mcnemar=(regime_mcnemar_df if not args.no_regime_mcnemar else None),
        regime_mcnemar_summary=(regime_mcnemar_summary_df
                                if not args.no_regime_mcnemar else None),
        diff_in_improvement=(diff_in_improvement_df
                              if not getattr(args, "no_diff_in_improvement",
                                             False) else None),
        incremental_sentiment=incremental_df,
        absolute_vs_naive=absolute_vs_naive_df,
        horizon_comparison=horizon_comparison_df,
        multiple_testing_manifest=multiple_testing_manifest_df,
        metric_roles=metric_roles_df,
        class_balance=class_balance_df,
    )
    xlsx = write_excel_report(
        xlsx_path,
        pooled=pooled, per_ticker=per_ticker,
        threshold=threshold_df, volatility=volatility_df,
        leaderboard=leaderboard, summary=summary,
        threshold_lift=threshold_lift_df,
        mcnemar=mcnemar_df_long,
        market_cap=(mcap_df         if not args.no_market_cap else None),
        regime_interaction=(interaction_df if not args.no_market_cap else None),
        economic=(economic_df       if not args.no_economic   else None),
        economic_threshold=(economic_thr_df if not args.no_economic else None),
        economic_by_ticker=(economic_tk_df  if not args.no_economic else None),
        regime_mcnemar=(regime_mcnemar_df if not args.no_regime_mcnemar else None),
        regime_mcnemar_summary=(regime_mcnemar_summary_df
                                if not args.no_regime_mcnemar else None),
        diff_in_improvement=(diff_in_improvement_df
                              if not getattr(args, "no_diff_in_improvement",
                                             False) else None),
        incremental_sentiment=incremental_df,
        absolute_vs_naive=absolute_vs_naive_df,
    )
    logger.info("evaluate-signals: wrote Excel report %s", xlsx)
    for label, path in csv_paths.items():
        logger.info("evaluate-signals: wrote %s → %s", label, path)

    # ── 12. Console summary ─────────────────────────────────────
    print_console_summary(
        pooled=pooled, threshold=threshold_df,
        mcnemar=mcnemar_df_long, threshold_lift=threshold_lift_df,
        market_cap=mcap_df, regime_interaction=interaction_df,
        economic=economic_df, economic_threshold=economic_thr_df,
        economic_diagnostics=(economic_diag_df
                              if not args.no_economic else None),
        regime_mcnemar=regime_mcnemar_df,
        incremental_sentiment=incremental_df,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
