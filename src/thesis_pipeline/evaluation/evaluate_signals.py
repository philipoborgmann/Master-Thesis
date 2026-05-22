"""Signal evaluation stage — canonical entry point.

This module is the orchestrator. The actual computations live in sibling
modules so that each piece is independently testable:

* :mod:`.loading`        — robust signal parquet loading
* :mod:`.metrics`        — pooled + per-ticker metrics + confusion diagnostics
* :mod:`.thresholds`     — high-conviction threshold analysis + lift vs benchmark
* :mod:`.volatility`     — Garman-Klass + tertile regime stratification
* :mod:`.significance`   — continuity-corrected McNemar vs B1 / B2
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
    buy_and_hold_benchmark, load_backtest_config, load_forward_returns,
    summarize_backtest, summarize_backtest_by_ticker,
    summarize_threshold_backtest,
)
from .reporting import (
    build_leaderboard, build_summary,
    print_console_summary, write_csv_outputs, write_excel_report,
)
from .significance import DEFAULT_BENCHMARKS, mcnemar_table, mcnemar_wide
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
               out_root / "volatility_stratification.csv"]
    if not args.no_market_cap:
        outputs += [out_root / "market_cap_stratification.csv",
                    out_root / "regime_interaction.csv"]
    if not args.no_economic:
        outputs += [out_root / "economic_performance.csv",
                    out_root / "economic_performance_by_threshold.csv",
                    out_root / "economic_performance_by_ticker.csv"]

    log_stage_header(
        "evaluate_signals",
        mode="dry-run" if args.dry_run else ("smoke" if args.smoke else "full"),
        inputs=inputs,
        outputs=outputs,
        extra={
            "horizon":              args.horizon or "(all)",
            "output_dir":           str(out_root),
            "no_volatility":        args.no_volatility,
            "no_market_cap":        args.no_market_cap,
            "no_economic":          args.no_economic,
            "transaction_cost_bps": args.transaction_cost_bps or "(from config)",
            "backtest_config":      args.backtest_config or "(default)",
            "force":                args.force,
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

    # ── 3. Pooled + per-ticker metrics ──────────────────────────
    pooled     = pooled_metrics_table(signals)
    per_ticker = per_ticker_metrics_table(signals)

    # ── 4. Threshold / conviction analysis ──────────────────────
    threshold_df      = threshold_analysis_table(signals)
    threshold_lift_df = threshold_lift_table(signals, benchmarks=DEFAULT_BENCHMARKS)

    # ── 5. McNemar significance vs B1 and B2 ────────────────────
    mcnemar_df_long = mcnemar_table(signals, benchmarks=DEFAULT_BENCHMARKS)
    mcnemar_wide_df = mcnemar_wide(mcnemar_df_long, benchmarks=DEFAULT_BENCHMARKS)
    if not pooled.empty and not mcnemar_wide_df.empty:
        pooled = pooled.merge(
            mcnemar_wide_df, on=["horizon", "set_id", "sentiment_model"],
            how="left",
        )
    if "mcnemar_pval_vs_b1" in pooled.columns and "mcnemar_pval" not in pooled.columns:
        pooled["mcnemar_stat"]              = pooled["mcnemar_stat_vs_b1"]
        pooled["mcnemar_pval"]              = pooled["mcnemar_pval_vs_b1"]
        pooled["significant_vs_benchmark"]  = pooled.get("significant_vs_b1", False)

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
    if args.no_market_cap:
        logger.info("evaluate-signals: --no-market-cap set, skipping mcap stratification")
    else:
        mcap_lookup = build_market_cap_lookup()
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

    # ── 8. Economic / backtest layer ────────────────────────────
    economic_df      = pd.DataFrame()
    economic_thr_df  = pd.DataFrame()
    economic_tk_df   = pd.DataFrame()
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
            if fr.empty:
                logger.warning("evaluate-signals: no close-price data for %s — "
                               "backtest will skip this horizon", hz)
                continue
            economic_df     = pd.concat(
                [economic_df, summarize_backtest(hz_signals, fr, bcfg)],
                ignore_index=True,
            )
            economic_thr_df = pd.concat(
                [economic_thr_df, summarize_threshold_backtest(hz_signals, fr, bcfg)],
                ignore_index=True,
            )
            economic_tk_df  = pd.concat(
                [economic_tk_df, summarize_backtest_by_ticker(hz_signals, fr, bcfg)],
                ignore_index=True,
            )
            # Buy-and-hold reference row (optional)
            if bcfg.get("include_buy_and_hold"):
                bh = buy_and_hold_benchmark(hz_tickers, fr, bcfg, hz)
                if bh is not None:
                    economic_df = pd.concat(
                        [economic_df, pd.DataFrame([bh])],
                        ignore_index=True,
                    )

    # ── 9. Leaderboard + thesis-style summary ───────────────────
    leaderboard = build_leaderboard(pooled)
    summary     = build_summary(
        pooled, threshold_df, volatility_df,
        threshold_lift=threshold_lift_df,
        market_cap_df=mcap_df,
        interaction_df=interaction_df,
        economic_df=economic_df,
        economic_threshold_df=economic_thr_df,
        extra_rows=extra_summary,
    )

    # ── 10. Sorting (deterministic output) ──────────────────────
    if not pooled.empty:
        pooled = pooled.sort_values(["horizon", "accuracy"],
                                    ascending=[True, False]).reset_index(drop=True)
    if not per_ticker.empty:
        per_ticker = per_ticker.sort_values(
            ["horizon", "set_id", "sentiment_model", "ticker"]).reset_index(drop=True)
    if not threshold_df.empty:
        threshold_df = threshold_df.sort_values(
            ["horizon", "set_id", "sentiment_model", "threshold"]).reset_index(drop=True)
    if not threshold_lift_df.empty:
        threshold_lift_df = threshold_lift_df.sort_values(
            ["horizon", "set_id", "sentiment_model", "benchmark", "threshold"]
        ).reset_index(drop=True)
    if not volatility_df.empty:
        volatility_df = volatility_df.sort_values(
            ["horizon", "set_id", "sentiment_model", "vol_regime"]).reset_index(drop=True)
    if not mcap_df.empty:
        mcap_df = mcap_df.sort_values(
            ["horizon", "set_id", "sentiment_model", "mcap_regime"]).reset_index(drop=True)
    if not interaction_df.empty:
        interaction_df = interaction_df.sort_values(
            ["horizon", "set_id", "sentiment_model", "mcap_regime", "vol_regime"]
        ).reset_index(drop=True)
    if not mcnemar_df_long.empty:
        mcnemar_df_long = mcnemar_df_long.sort_values(
            ["horizon", "set_id", "sentiment_model", "benchmark"]
        ).reset_index(drop=True)
    if not economic_df.empty:
        economic_df = economic_df.sort_values(
            ["horizon", "cost_bps", "sharpe"],
            ascending=[True, True, False]).reset_index(drop=True)
    if not economic_thr_df.empty:
        economic_thr_df = economic_thr_df.sort_values(
            ["horizon", "threshold", "cost_bps", "sharpe"],
            ascending=[True, True, True, False]).reset_index(drop=True)
    if not economic_tk_df.empty:
        economic_tk_df = economic_tk_df.sort_values(
            ["horizon", "set_id", "sentiment_model", "ticker", "cost_bps"]
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
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
