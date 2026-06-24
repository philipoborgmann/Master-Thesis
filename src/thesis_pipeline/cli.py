"""Command-line interface for the thesis pipeline.

Usage:
    python -m thesis_pipeline.cli --help
    python -m thesis_pipeline.cli <command> [options]

Every heavy stage supports ``--smoke`` (small, safe defaults written to
``Outputs/diagnostics/smoke/``) and ``--dry-run`` (no work, no files,
just prints the plan).
"""

from __future__ import annotations

import argparse
import sys
from typing import Sequence

from .config import all_configs, load_config, resolve_path
from .logging_utils import attach_file_handler, get_logger


# ---------------------------------------------------------------------------
# Command handlers
# ---------------------------------------------------------------------------

_FINBERT_REMOVED_MSG = (
    "FinBERT was removed from this pipeline. It is trained on financial / "
    "analyst language and did not produce meaningful variation on Reddit / "
    "crypto sentiment data in this project. Use --sentiment-model cryptobert "
    "(domain-specific transformer) or --sentiment-model vader (transparent "
    "lexicon baseline) instead."
)


def _sentiment_model_choice(value):
    """argparse ``type`` for the ``--sentiment-model`` flag.

    Yields a clear FinBERT-removed message before argparse's stock ``invalid
    choice`` complaint would fire.
    """
    if value is None:
        return value
    v = str(value).strip().lower()
    if v == "finbert":
        raise argparse.ArgumentTypeError(_FINBERT_REMOVED_MSG)
    return v


def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--smoke", action="store_true",
                        help="Run with small smoke-mode defaults.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print planned actions but do not write files or run heavy work.")
    parser.add_argument("--force", action="store_true",
                        help="Allow overwriting full production outputs in non-smoke runs.")


def _add_run_models_args(parser: argparse.ArgumentParser) -> None:
    """Add the run-models argument surface.

    Single source of truth for the ``run-models`` subcommand so the CLI can
    never drift out of sync with ``modeling.run_models.build_parser``. Every
    flag accepts both the dashed and underscored spelling (matching the
    underlying model parsers), so e.g. ``--sentiment-model`` and
    ``--sentiment_model`` are both recognised and bound to ``sentiment_model``.
    """
    parser.add_argument("--horizon", default=None)
    parser.add_argument("--set-id", "--set_id", dest="set_id", default=None)
    parser.add_argument("--coins", "--coin", "--ticker",
                        dest="coins", nargs="*")
    parser.add_argument("--feature-config", "--feature_config",
                        dest="feature_config", default=None,
                        help="Override path to feature_sets.xlsx (used by "
                             "modeling.run_models.load_feature_sets).")
    parser.add_argument("--sentiment-model", "--sentiment_model",
                        dest="sentiment_model", default=None,
                        type=_sentiment_model_choice,
                        choices=[None, "vader", "cryptobert", "multi"],
                        help="Restrict to one sentiment scorer. FinBERT was "
                             "removed and rejected with an informative error.")
    parser.add_argument("--model-type", "--model_type", dest="model_type",
                        default="per_asset",
                        choices=["per_asset", "panel_logit"],
                        help="per_asset (default) or panel_logit.")
    parser.add_argument("--panel-mode", "--panel_mode", dest="panel_mode",
                        default="pooled",
                        choices=["pooled", "ticker_fixed_effects"],
                        help="Panel variant (only with --model-type panel_logit).")
    parser.add_argument("--restart", action="store_true",
                        help="Ignore cached signal parquets and rerun every set.")
    parser.add_argument("--tune-hyperparams", "--tune_hyperparams",
                        dest="tune_hyperparams", action="store_true",
                        help="Enable conservative, leakage-safe grid-search HPO "
                             "inside each walk-forward training window.")
    parser.add_argument("--hpo-objective", "--hpo_objective",
                        dest="hpo_objective", default=None,
                        choices=["brier_score", "log_loss", "accuracy"],
                        help="HPO selection metric (default: model_specs.yaml).")
    parser.add_argument("--hpo-config", "--hpo_config", dest="hpo_config",
                        default=None,
                        help="Path to a YAML with a hyperparameter_tuning section.")
    parser.add_argument("--hpo-grid-C", "--hpo_grid_C", dest="hpo_grid_C",
                        type=float, nargs="+", default=None,
                        help="Override the C search grid.")
    parser.add_argument("--hpo-class-weight", "--hpo_class_weight",
                        dest="hpo_class_weight", nargs="+", default=None,
                        help="Override the class_weight grid (e.g. none balanced).")
    # ── Checkpointing / resume ──────────────────────────────────
    parser.add_argument("--checkpoint", dest="checkpoint",
                        action=argparse.BooleanOptionalAction, default=True,
                        help="Write per-ticker / per-chunk checkpoints so a "
                             "crashed run can resume (default: on; --no-checkpoint "
                             "restores the old behaviour).")
    parser.add_argument("--resume", dest="resume",
                        action=argparse.BooleanOptionalAction, default=True,
                        help="Reuse existing checkpoints instead of recomputing "
                             "(default: on).")
    parser.add_argument("--checkpoint-dir", "--checkpoint_dir",
                        dest="checkpoint_dir", default="Outputs/Checkpoints/Models",
                        help="Root directory for model checkpoints.")
    parser.add_argument("--checkpoint-chunk-size", "--checkpoint_chunk_size",
                        dest="checkpoint_chunk_size", type=int, default=20,
                        help="Number of panel test timestamps per checkpoint chunk.")
    parser.add_argument("--clear-checkpoints", "--clear_checkpoints",
                        dest="clear_checkpoints", action="store_true",
                        help="Delete this run's checkpoint directory before "
                             "starting (does not happen automatically on --restart).")
    parser.add_argument("--train-window", "--train_window",
                        dest="train_window", default="expanding",
                        choices=["expanding", "rolling_fixed"],
                        help="Panel training window (default: expanding). "
                             "rolling_fixed requires --rolling-window-timestamps "
                             "or --rolling-window-days; structural-break "
                             "diagnostics never set this automatically.")
    parser.add_argument("--rolling-window-timestamps", "--rolling_window_timestamps",
                        dest="rolling_window_timestamps", type=int, default=None,
                        help="Manual number of pre-tau unique timestamps to "
                             "include in the rolling training window.")
    parser.add_argument("--rolling-window-days", "--rolling_window_days",
                        dest="rolling_window_days", type=float, default=None,
                        help="Manual day-distance rolling window (alternative "
                             "to --rolling-window-timestamps).")


# ---------------------------------------------------------------------------
# Helper: log a stage plan and short-circuit on --dry-run.
#
# The migrated modules retain their original argparse parsers which do not
# universally understand ``--dry-run`` / ``--smoke``. To keep the CLI's
# uniform UX (every stage supports ``--dry-run``) we filter these flags here:
# on ``--dry-run`` we emit the stage header and return 0 *before* invoking
# the module. The modules that have native dry-run handling
# (``modeling.run_models``, ``evaluation.evaluate_signals``,
# ``features.merge``) still benefit from this short-circuit since both paths
# agree on a noop return value.
# ---------------------------------------------------------------------------

def _stage_dry_run(stage: str, args: argparse.Namespace) -> int | None:
    """Return 0 (handled) when ``--dry-run`` is set, else None (continue)."""
    if not getattr(args, "dry_run", False):
        return None
    from .logging_utils import log_stage_header
    extras = {
        "horizon": getattr(args, "horizon", None) or "(all)",
        "coins":   list(args.coins) if getattr(args, "coins", None) else "(all)",
    }
    for opt in ("set_id", "sentiment_model", "max_rows", "batch_size",
                "winsor_p", "no_plots", "restart", "force",
                "feature_config", "output_dir", "no_volatility",
                "no_market_cap", "no_economic", "no_regime_mcnemar",
                "backtest_config", "transaction_cost_bps"):
        if hasattr(args, opt) and getattr(args, opt) not in (None, False):
            extras[opt] = getattr(args, opt)
    log_stage_header(stage, mode="dry-run", inputs=[], outputs=[], extra=extras)
    return 0


def cmd_validate_price(args: argparse.Namespace) -> int:
    rc = _stage_dry_run("validate_price", args)
    if rc is not None:
        return rc
    from .price import validate as _m
    argv: list[str] = []
    if args.coins:
        for c in args.coins:
            argv += ["--coin", c]
    if getattr(args, "debug", False):
        argv.append("--debug")
    return _m.main(argv)


def cmd_create_price_features(args: argparse.Namespace) -> int:
    rc = _stage_dry_run("create_price_features", args)
    if rc is not None:
        return rc
    from .price import features as _m
    argv: list[str] = []
    if args.horizon:
        argv += ["--horizons", args.horizon]
    if args.coins:
        for c in args.coins:
            argv += ["--coin", c]
    if getattr(args, "winsor_p", None) is not None:
        argv += ["--winsor_p", str(args.winsor_p)]
    return _m.main(argv)


def cmd_load_sentiment(args: argparse.Namespace) -> int:
    rc = _stage_dry_run("load_sentiment", args)
    if rc is not None:
        return rc
    from .sentiment import load as _m
    return _m.main([])


def cmd_score_sentiment(args: argparse.Namespace) -> int:
    if str(getattr(args, "model", "")).lower() == "finbert":
        # Friendly error for legacy callers that bypass argparse.
        raise SystemExit(_FINBERT_REMOVED_MSG)
    rc = _stage_dry_run(f"score_{args.model}", args)
    if rc is not None:
        return rc
    if args.model == "vader":
        from .sentiment import score_vader as _m
    elif args.model == "cryptobert":
        from .sentiment import score_cryptobert as _m
    else:
        raise SystemExit(f"Unknown sentiment model: {args.model}")
    argv: list[str] = []
    if getattr(args, "batch_size", None) is not None:
        argv += ["--batch_size", str(args.batch_size)]
    if getattr(args, "restart", False):
        argv.append("--restart")
    return _m.main(argv)


def cmd_create_sentiment_features(args: argparse.Namespace) -> int:
    rc = _stage_dry_run("create_sentiment_features", args)
    if rc is not None:
        return rc
    from .sentiment import aggregate as _m
    argv: list[str] = []
    if getattr(args, "no_plots", False):
        argv.append("--no_plots")
    return _m.main(argv)


def cmd_stationarity(args: argparse.Namespace) -> int:
    rc = _stage_dry_run("stationarity", args)
    if rc is not None:
        return rc
    from .sentiment import stationarity as _m
    argv: list[str] = []
    source = getattr(args, "source", None)
    if getattr(args, "final_features", False):
        source = "final"
    if source:
        argv += ["--source", source]
    if args.horizon:
        argv += ["--horizon", args.horizon]
    if args.coins:
        # The module only supports a single --ticker; forward the first.
        argv += ["--ticker", args.coins[0]]
    if getattr(args, "no_plots", False):
        argv.append("--no_plots")
    if getattr(args, "no_panel", False):
        argv.append("--no_panel")
    return _m.main(argv)


def cmd_structural_breaks(args: argparse.Namespace) -> int:
    rc = _stage_dry_run("structural_breaks", args)
    if rc is not None:
        return rc
    from .diagnostics import structural_breaks as _m
    argv: list[str] = []
    if getattr(args, "horizon", None):
        argv += ["--horizon", args.horizon]
    if getattr(args, "features", None):
        argv += ["--features", *list(args.features)]
    if getattr(args, "include_market_cap", False):
        argv.append("--include-market-cap")
    if getattr(args, "max_breaks", None) is not None:
        argv += ["--max-breaks", str(args.max_breaks)]
    if getattr(args, "min_segment_frac", None) is not None:
        argv += ["--min-segment-frac", str(args.min_segment_frac)]
    if getattr(args, "n_bkps", None) is not None:
        argv += ["--n-bkps", str(args.n_bkps)]
    if getattr(args, "output_dir", None):
        argv += ["--output-dir", args.output_dir]
    if getattr(args, "dry_run", False):
        argv.append("--dry-run")
    return _m.main(argv)


def cmd_descriptive_final_features(args: argparse.Namespace) -> int:
    rc = _stage_dry_run("descriptive_final_features", args)
    if rc is not None:
        return rc
    from .diagnostics import descriptive_final_features as _m
    argv: list[str] = []
    if getattr(args, "horizon", None):
        argv += ["--horizon", args.horizon]
    if getattr(args, "output_dir", None):
        argv += ["--output-dir", args.output_dir]
    if getattr(args, "min_pairwise_n", None) is not None:
        argv += ["--min-pairwise-n", str(args.min_pairwise_n)]
    if getattr(args, "smoke", False):
        argv.append("--smoke")
    if getattr(args, "dry_run", False):
        argv.append("--dry-run")
    if getattr(args, "force", False):
        argv.append("--force")
    return _m.main(argv)


def cmd_merge_features(args: argparse.Namespace) -> int:
    from .features import merge as _m
    argv: list[str] = []
    if args.horizon:
        argv += ["--horizon", args.horizon]
    if getattr(args, "no_sentiment_coverage_filter", False):
        argv.append("--no-sentiment-coverage-filter")
    # ``neutral_fill_missing_sentiment`` defaults to True; only forward when
    # the caller explicitly turned it off so the legacy CLI surface stays clean.
    if getattr(args, "neutral_fill_missing_sentiment", True) is False:
        argv.append("--no-neutral-fill-missing-sentiment")
    if args.smoke:
        argv.append("--smoke")
    if args.dry_run:
        argv.append("--dry-run")
    return _m.main(argv)


def cmd_run_models(args: argparse.Namespace) -> int:
    """Call ``thesis_pipeline.modeling.run_models.main`` directly.

    Translates the CLI's ``--coins BTC ETH`` to the modeling module's
    ``--coins BTC ETH`` — never to the buggy ``--coin BTC --coin ETH``
    that previous versions emitted.
    """
    from .modeling import run_models as _rm
    argv: list[str] = []
    if args.horizon:
        argv += ["--horizon", args.horizon]
    if args.set_id:
        argv += ["--set-id", args.set_id]
    if args.coins:
        argv += ["--coins", *list(args.coins)]
    if getattr(args, "feature_config", None):
        argv += ["--feature-config", args.feature_config]
    if args.sentiment_model:
        argv += ["--sentiment-model", args.sentiment_model]
    if getattr(args, "model_type", None):
        argv += ["--model-type", args.model_type]
    if getattr(args, "panel_mode", None):
        argv += ["--panel-mode", args.panel_mode]
    if getattr(args, "tune_hyperparams", False):
        argv.append("--tune-hyperparams")
    if getattr(args, "hpo_objective", None):
        argv += ["--hpo-objective", args.hpo_objective]
    if getattr(args, "hpo_config", None):
        argv += ["--hpo-config", args.hpo_config]
    if getattr(args, "hpo_grid_C", None):
        argv += ["--hpo-grid-C", *(str(c) for c in args.hpo_grid_C)]
    if getattr(args, "hpo_class_weight", None):
        argv += ["--hpo-class-weight", *(str(c) for c in args.hpo_class_weight)]
    # Checkpointing — forward only when diverging from the model defaults
    # (checkpoint=True, resume=True) so run-stage / run-pipeline (which don't
    # define these flags) keep the default behaviour.
    if getattr(args, "checkpoint", True) is False:
        argv.append("--no-checkpoint")
    if getattr(args, "resume", True) is False:
        argv.append("--no-resume")
    if getattr(args, "checkpoint_dir", None):
        argv += ["--checkpoint-dir", args.checkpoint_dir]
    if getattr(args, "checkpoint_chunk_size", None) is not None:
        argv += ["--checkpoint-chunk-size", str(args.checkpoint_chunk_size)]
    if getattr(args, "clear_checkpoints", False):
        argv.append("--clear-checkpoints")
    if getattr(args, "train_window", None):
        argv += ["--train-window", args.train_window]
    if getattr(args, "rolling_window_timestamps", None) is not None:
        argv += ["--rolling-window-timestamps", str(args.rolling_window_timestamps)]
    if getattr(args, "rolling_window_days", None) is not None:
        argv += ["--rolling-window-days", str(args.rolling_window_days)]
    if args.smoke:
        argv.append("--smoke")
    if args.dry_run:
        argv.append("--dry-run")
    if getattr(args, "force", False):
        argv.append("--force")
    if getattr(args, "restart", False):
        argv.append("--restart")
    return _rm.main(argv)


def cmd_diagnostics(args: argparse.Namespace) -> int:
    from .diagnostics.sample_report import run
    return run(horizon=args.horizon, smoke=args.smoke, dry_run=args.dry_run)


def cmd_evaluate_signals(args: argparse.Namespace) -> int:
    """Forward CLI flags directly to the evaluation module's argparse-based main()."""
    from .evaluation import evaluate_signals as _es
    argv: list[str] = []
    horizon = getattr(args, "horizon", None)
    if horizon:
        argv += ["--horizon", horizon]
    output_dir = getattr(args, "output_dir", None)
    if output_dir:
        argv += ["--output-dir", str(output_dir)]
    feature_config = getattr(args, "feature_config", None)
    if feature_config:
        argv += ["--feature-config", feature_config]
    backtest_config = getattr(args, "backtest_config", None)
    if backtest_config:
        argv += ["--backtest-config", str(backtest_config)]
    tcb = getattr(args, "transaction_cost_bps", None)
    if tcb:
        argv += ["--transaction-cost-bps", *(str(c) for c in tcb)]
    if getattr(args, "smoke", False):
        argv.append("--smoke")
    if getattr(args, "dry_run", False):
        argv.append("--dry-run")
    if getattr(args, "force", False):
        argv.append("--force")
    if getattr(args, "no_volatility", False):
        argv.append("--no-volatility")
    if getattr(args, "no_market_cap", False):
        argv.append("--no-market-cap")
    if getattr(args, "no_economic", False):
        argv.append("--no-economic")
    if getattr(args, "no_regime_mcnemar", False):
        argv.append("--no-regime-mcnemar")
    return _es.main(argv)


def cmd_run_stage(args: argparse.Namespace) -> int:
    return _dispatch_stage(args.stage, args)


def cmd_run_pipeline(args: argparse.Namespace) -> int:
    pipeline = load_config("pipeline")
    stages = args.stages or pipeline.get("default_order", [])
    rc = 0
    for stage in stages:
        get_logger().info(">>> running stage: %s", stage)
        rc = _dispatch_stage(stage, args) or rc
        if rc:
            get_logger().error("stage %s failed with rc=%s; stopping", stage, rc)
            return rc
    return rc


def _dispatch_stage(stage: str, args: argparse.Namespace) -> int:
    """Map a stage name to the matching command function."""
    table = {
        "validate_price":            cmd_validate_price,
        "create_price_features":     cmd_create_price_features,
        "load_sentiment":            cmd_load_sentiment,
        "score_vader":               lambda a: cmd_score_sentiment(_with_attr(a, model="vader")),
        "score_cryptobert":          lambda a: cmd_score_sentiment(_with_attr(a, model="cryptobert")),
        "create_sentiment_features": cmd_create_sentiment_features,
        "stationarity":              cmd_stationarity,
        "merge_features":            cmd_merge_features,
        "run_models":                cmd_run_models,
        "evaluate_signals":          cmd_evaluate_signals,
        "diagnostics":               cmd_diagnostics,
        "reports":                   cmd_diagnostics,
        "descriptive_final_features": cmd_descriptive_final_features,
        "structural_breaks":          cmd_structural_breaks,
    }
    if stage not in table:
        raise SystemExit(f"Unknown stage: {stage}")
    return table[stage](args)


def _with_attr(ns: argparse.Namespace, **kwargs) -> argparse.Namespace:
    for k, v in kwargs.items():
        setattr(ns, k, v)
    # Ensure defaults exist for fields the sub-command may read
    for default_key, default_val in (("max_rows", None), ("batch_size", None),
                                     ("restart", False)):
        if not hasattr(ns, default_key):
            setattr(ns, default_key, default_val)
    return ns


# ---------------------------------------------------------------------------
# Parser construction
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m thesis_pipeline.cli",
        description="Staged pipeline for the cryptocurrency-sentiment thesis.",
    )
    sub = p.add_subparsers(dest="command", required=True)

    # validate-price
    sp = sub.add_parser("validate-price", help="Run price-data validation.")
    sp.add_argument("--horizon", default=None)
    sp.add_argument("--coins", nargs="*")
    sp.add_argument("--debug", action="store_true")
    _add_common(sp)
    sp.set_defaults(func=cmd_validate_price)

    # create-price-features
    sp = sub.add_parser("create-price-features", help="Compute price features.")
    sp.add_argument("--horizon", default=None)
    sp.add_argument("--coins", nargs="*")
    sp.add_argument("--winsor-p", dest="winsor_p", type=float, default=None)
    _add_common(sp)
    sp.set_defaults(func=cmd_create_price_features)

    # load-sentiment
    sp = sub.add_parser("load-sentiment", help="Load and clean raw Reddit sentiment data.")
    sp.add_argument("--max-rows", dest="max_rows", type=int, default=None)
    _add_common(sp)
    sp.set_defaults(func=cmd_load_sentiment)

    # score-sentiment
    sp = sub.add_parser("score-sentiment", help="Score sentiment with a chosen model.")
    sp.add_argument("--model", required=True,
                    type=_sentiment_model_choice,
                    choices=["vader", "cryptobert"],
                    help="Sentiment scorer to run. FinBERT was removed; the "
                         "CLI rejects it with an informative error.")
    sp.add_argument("--max-rows", dest="max_rows", type=int, default=None)
    sp.add_argument("--batch-size", dest="batch_size", type=int, default=None)
    sp.add_argument("--restart", action="store_true",
                    help="Ignore existing checkpoints and start over.")
    _add_common(sp)
    sp.set_defaults(func=cmd_score_sentiment)

    # create-sentiment-features
    sp = sub.add_parser("create-sentiment-features",
                        help="Aggregate scored posts into per-horizon sentiment features.")
    sp.add_argument("--horizon", default=None)
    sp.add_argument("--max-rows", dest="max_rows", type=int, default=None)
    sp.add_argument("--no-plots", dest="no_plots", action="store_true")
    _add_common(sp)
    sp.set_defaults(func=cmd_create_sentiment_features)

    # stationarity
    sp = sub.add_parser("stationarity",
                        help="Run stationarity tests on the final modelling "
                             "features (default) or the legacy sentiment-only "
                             "features (--source sentiment).")
    sp.add_argument("--source", choices=["final", "sentiment"], default="final",
                    help="Feature source: 'final' (Data/Final/features_{h}.parquet, "
                         "default) or 'sentiment' (legacy).")
    sp.add_argument("--final-features", "--final_features",
                    dest="final_features", action="store_true",
                    help="Alias for --source final.")
    sp.add_argument("--horizon", default=None)
    sp.add_argument("--coins", nargs="*")
    sp.add_argument("--no-plots", dest="no_plots", action="store_true")
    sp.add_argument("--no-panel", dest="no_panel", action="store_true",
                    help="Skip the CIPS panel tests.")
    _add_common(sp)
    sp.set_defaults(func=cmd_stationarity)

    # structural-breaks (diagnostic only — never feeds modelling)
    sp = sub.add_parser("structural-breaks",
                        help="Bai-Perron-style structural-break diagnostics "
                             "for the final modelling features. Diagnostic "
                             "only — does NOT set rolling-window sizes.")
    sp.add_argument("--horizon", default=None, choices=["1h", "6h", "1d"])
    sp.add_argument("--features", nargs="*", default=None,
                    help="Override the default diagnostic basket.")
    sp.add_argument("--include-market-cap", dest="include_market_cap",
                    action="store_true",
                    help="Include log_market_cap_lag1 in the default basket.")
    sp.add_argument("--max-breaks", dest="max_breaks", type=int, default=None)
    sp.add_argument("--min-segment-frac", dest="min_segment_frac", type=float,
                    default=None)
    sp.add_argument("--n-bkps", dest="n_bkps", type=int, default=None,
                    help="Fix the number of breaks; skips BIC selection.")
    sp.add_argument("--output-dir", dest="output_dir", default=None)
    _add_common(sp)
    sp.set_defaults(func=cmd_structural_breaks)

    # descriptive-final-features
    sp = sub.add_parser("descriptive-final-features",
                        help="Extensive descriptive statistics for the final "
                             "modelling feature sets.")
    sp.add_argument("--horizon", default=None, choices=["1h", "6h", "1d"])
    sp.add_argument("--output-dir", dest="output_dir", default=None,
                    help="Override Outputs/deskriptiv/final_feature_sets/.")
    sp.add_argument("--min-pairwise-n", dest="min_pairwise_n", type=int,
                    default=None,
                    help="Drop correlation pairs below this overlap count.")
    _add_common(sp)
    sp.set_defaults(func=cmd_descriptive_final_features)

    # merge-features
    sp = sub.add_parser("merge-features", help="Merge price + sentiment features.")
    sp.add_argument("--horizon", default=None)
    sp.add_argument("--no-sentiment-coverage-filter",
                    dest="no_sentiment_coverage_filter", action="store_true",
                    help="Keep tickers below the per-horizon sentiment coverage "
                         "threshold (useful for panel models).")
    sp.add_argument("--neutral-fill-missing-sentiment",
                    dest="neutral_fill_missing_sentiment",
                    action=argparse.BooleanOptionalAction, default=True,
                    help="Fill NaN sentiment/attention columns with neutral "
                         "values (default: on, current behaviour).")
    _add_common(sp)
    sp.set_defaults(func=cmd_merge_features)

    # run-models
    sp = sub.add_parser("run-models", help="Run walk-forward models.")
    _add_run_models_args(sp)
    _add_common(sp)
    sp.set_defaults(func=cmd_run_models)

    # diagnostics
    sp = sub.add_parser("diagnostics", help="Write a per-horizon sample report.")
    sp.add_argument("--horizon", default="1d")
    _add_common(sp)
    sp.set_defaults(func=cmd_diagnostics)

    # evaluate-signals
    sp = sub.add_parser("evaluate-signals",
                        help="Evaluate walk-forward signals (metrics, McNemar, "
                             "regimes, market-cap interaction, backtest).")
    sp.add_argument("--horizon", default=None, choices=["1h", "6h", "1d"])
    sp.add_argument("--output-dir", dest="output_dir", default=None)
    sp.add_argument("--feature-config", dest="feature_config", default=None,
                    help="Override path to feature_sets.xlsx.")
    sp.add_argument("--backtest-config", dest="backtest_config", default=None,
                    help="Override path to configs/backtest.yaml.")
    sp.add_argument("--transaction-cost-bps", dest="transaction_cost_bps",
                    type=float, nargs="*", default=None,
                    help="Override the transaction-cost grid (bps).")
    sp.add_argument("--no-volatility", dest="no_volatility", action="store_true",
                    help="Skip the volatility-stratification step.")
    sp.add_argument("--no-market-cap", dest="no_market_cap", action="store_true",
                    help="Skip the market-cap stratification + interaction step.")
    sp.add_argument("--no-economic", dest="no_economic", action="store_true",
                    help="Skip the turnover/cost-aware backtest step.")
    sp.add_argument("--no-regime-mcnemar", dest="no_regime_mcnemar",
                    action="store_true",
                    help="Skip the regime-specific McNemar tests.")
    _add_common(sp)
    sp.set_defaults(func=cmd_evaluate_signals)

    # run-stage
    sp = sub.add_parser("run-stage", help="Run a single named stage.")
    sp.add_argument("--stage", required=True)
    sp.add_argument("--horizon", default=None)
    sp.add_argument("--set-id", dest="set_id", default=None)
    sp.add_argument("--coins", nargs="*")
    sp.add_argument("--max-rows", dest="max_rows", type=int, default=None)
    sp.add_argument("--no-plots", dest="no_plots", action="store_true")
    sp.add_argument("--debug", action="store_true")
    sp.add_argument("--winsor-p", dest="winsor_p", type=float, default=None)
    sp.add_argument("--batch-size", dest="batch_size", type=int, default=None)
    sp.add_argument("--restart", action="store_true")
    sp.add_argument("--sentiment-model", dest="sentiment_model", default=None)
    sp.add_argument("--output-dir", dest="output_dir", default=None)
    sp.add_argument("--feature-config", dest="feature_config", default=None)
    sp.add_argument("--no-volatility",  dest="no_volatility",  action="store_true")
    sp.add_argument("--no-market-cap",  dest="no_market_cap",  action="store_true")
    sp.add_argument("--no-economic",    dest="no_economic",    action="store_true")
    sp.add_argument("--no-regime-mcnemar", dest="no_regime_mcnemar", action="store_true")
    sp.add_argument("--backtest-config", dest="backtest_config", default=None)
    sp.add_argument("--transaction-cost-bps", dest="transaction_cost_bps",
                    type=float, nargs="*", default=None)
    _add_common(sp)
    sp.set_defaults(func=cmd_run_stage)

    # run-pipeline
    sp = sub.add_parser("run-pipeline", help="Run multiple stages in order.")
    sp.add_argument("--stages", nargs="*", default=None,
                    help="Subset of stages to run, in the given order. "
                         "Defaults to configs/pipeline.yaml default_order.")
    sp.add_argument("--horizon", default=None)
    sp.add_argument("--set-id", dest="set_id", default=None)
    sp.add_argument("--coins", nargs="*")
    sp.add_argument("--max-rows", dest="max_rows", type=int, default=None)
    sp.add_argument("--no-plots", dest="no_plots", action="store_true")
    sp.add_argument("--debug", action="store_true")
    sp.add_argument("--winsor-p", dest="winsor_p", type=float, default=None)
    sp.add_argument("--batch-size", dest="batch_size", type=int, default=None)
    sp.add_argument("--restart", action="store_true")
    sp.add_argument("--sentiment-model", dest="sentiment_model", default=None)
    sp.add_argument("--output-dir", dest="output_dir", default=None)
    sp.add_argument("--feature-config", dest="feature_config", default=None)
    sp.add_argument("--no-volatility",  dest="no_volatility",  action="store_true")
    sp.add_argument("--no-market-cap",  dest="no_market_cap",  action="store_true")
    sp.add_argument("--no-economic",    dest="no_economic",    action="store_true")
    sp.add_argument("--no-regime-mcnemar", dest="no_regime_mcnemar", action="store_true")
    sp.add_argument("--backtest-config", dest="backtest_config", default=None)
    sp.add_argument("--transaction-cost-bps", dest="transaction_cost_bps",
                    type=float, nargs="*", default=None)
    _add_common(sp)
    sp.set_defaults(func=cmd_run_pipeline)

    return p


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    attach_file_handler(getattr(args, "command", "cli"))
    # Sanity-load configs so missing/invalid YAML is reported up-front.
    all_configs()
    return int(args.func(args) or 0)


if __name__ == "__main__":
    sys.exit(main())
