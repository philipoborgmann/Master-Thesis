"""Signal evaluation stage — canonical entry point.

This module is the orchestrator. The actual computations live in sibling
modules so that each piece is independently testable:

* :mod:`.loading`        — robust signal parquet loading
* :mod:`.metrics`        — pooled + per-ticker metrics + confusion diagnostics
* :mod:`.thresholds`     — high-conviction threshold analysis
* :mod:`.volatility`     — Garman-Klass + tertile regime stratification
* :mod:`.significance`   — continuity-corrected McNemar vs B1
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
from .metrics import pooled_metrics_table, per_ticker_metrics_table
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
        description="Signal evaluation — pooled/per-ticker metrics, McNemar, "
                    "volatility-regime and threshold analysis."
    )
    parser.add_argument("--horizon", default=None, choices=["1h", "6h", "1d"],
                        help="Restrict to a single horizon. Default: all available.")
    parser.add_argument("--output-dir", "--output_dir", dest="output_dir",
                        default=None,
                        help="Override Outputs/Evaluation/.")
    parser.add_argument("--feature_config", "--feature-config",
                        dest="feature_config", default=None,
                        help="Path to feature_sets.xlsx; defaults to configs/paths.yaml.")
    parser.add_argument("--smoke", action="store_true",
                        help="Smoke mode: horizon=1d if not given; writes under smoke_root.")
    parser.add_argument("--dry-run", "--dry_run", dest="dry_run",
                        action="store_true",
                        help="Print planned inputs/outputs and exit.")
    parser.add_argument("--force", action="store_true",
                        help="Allow overwriting full production outputs.")
    parser.add_argument("--no-volatility", dest="no_volatility",
                        action="store_true",
                        help="Skip the volatility-stratification step "
                             "(useful if Data/Raw/Price/1d/ is unavailable).")
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
        feature_config: str | None = None) -> int:
    """Programmatic entry point. Mirrors :func:`main` but takes kwargs."""
    argv: list[str] = []
    if horizon:
        argv += ["--horizon", horizon]
    if output_dir:
        argv += ["--output-dir", str(output_dir)]
    if feature_config:
        argv += ["--feature-config", feature_config]
    if smoke:
        argv.append("--smoke")
    if dry_run:
        argv.append("--dry-run")
    if force:
        argv.append("--force")
    if no_volatility:
        argv.append("--no-volatility")
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
    outputs = [xlsx_path,
               out_root / "pooled_metrics.csv",
               out_root / "per_ticker_metrics.csv",
               out_root / "threshold_analysis.csv",
               out_root / "volatility_stratification.csv"]

    log_stage_header(
        "evaluate_signals",
        mode="dry-run" if args.dry_run else ("smoke" if args.smoke else "full"),
        inputs=inputs,
        outputs=outputs,
        extra={
            "horizon":       args.horizon or "(all)",
            "output_dir":    str(out_root),
            "no_volatility": args.no_volatility,
            "force":         args.force,
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
    threshold_df = threshold_analysis_table(signals)
    threshold_lift_df = threshold_lift_table(signals, benchmarks=DEFAULT_BENCHMARKS)

    # ── 5. McNemar significance vs B1 and B2 ────────────────────
    mcnemar_df_long = mcnemar_table(signals, benchmarks=DEFAULT_BENCHMARKS)
    mcnemar_wide_df = mcnemar_wide(mcnemar_df_long, benchmarks=DEFAULT_BENCHMARKS)
    if not pooled.empty and not mcnemar_wide_df.empty:
        pooled = pooled.merge(
            mcnemar_wide_df, on=["horizon", "set_id", "sentiment_model"],
            how="left",
        )
    # Back-compat aliases so external code that still reads `mcnemar_pval`
    # keeps working. They mirror the vs-B1 columns.
    if "mcnemar_pval_vs_b1" in pooled.columns and "mcnemar_pval" not in pooled.columns:
        pooled["mcnemar_stat"]              = pooled["mcnemar_stat_vs_b1"]
        pooled["mcnemar_pval"]              = pooled["mcnemar_pval_vs_b1"]
        pooled["significant_vs_benchmark"]  = pooled.get("significant_vs_b1", False)

    # ── 6. Volatility-regime stratification ─────────────────────
    if args.no_volatility:
        volatility_df = pd.DataFrame()
        logger.info("evaluate-signals: --no-volatility set, skipping stratification")
    else:
        tickers = sorted(signals["ticker"].dropna().astype(str).str.upper().unique())
        lookup = build_regime_lookup(tickers)
        if lookup.empty:
            logger.warning("evaluate-signals: no daily price data found — "
                           "volatility stratification will be empty")
            volatility_df = pd.DataFrame()
        else:
            volatility_df = volatility_stratification_table(signals, lookup)

    # ── 7. Leaderboard + summary ────────────────────────────────
    leaderboard = build_leaderboard(pooled)
    summary     = build_summary(pooled, threshold_df, volatility_df,
                                threshold_lift=threshold_lift_df)

    # ── 8. Sorting (deterministic output) ───────────────────────
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
    if not mcnemar_df_long.empty:
        mcnemar_df_long = mcnemar_df_long.sort_values(
            ["horizon", "set_id", "sentiment_model", "benchmark"]
        ).reset_index(drop=True)

    # ── 9. Write outputs ────────────────────────────────────────
    csv_paths = write_csv_outputs(
        out_root,
        pooled=pooled, per_ticker=per_ticker,
        threshold=threshold_df, volatility=volatility_df,
        threshold_lift=threshold_lift_df,
        mcnemar=mcnemar_df_long,
    )
    xlsx = write_excel_report(
        xlsx_path,
        pooled=pooled, per_ticker=per_ticker,
        threshold=threshold_df, volatility=volatility_df,
        leaderboard=leaderboard, summary=summary,
        threshold_lift=threshold_lift_df,
        mcnemar=mcnemar_df_long,
    )
    logger.info("evaluate-signals: wrote Excel report %s", xlsx)
    for label, path in csv_paths.items():
        logger.info("evaluate-signals: wrote %s → %s", label, path)

    # ── 10. Console summary ─────────────────────────────────────
    print_console_summary(
        pooled=pooled, threshold=threshold_df,
        mcnemar=mcnemar_df_long, threshold_lift=threshold_lift_df,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
