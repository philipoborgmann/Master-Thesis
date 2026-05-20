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

def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--smoke", action="store_true",
                        help="Run with small smoke-mode defaults.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print planned actions but do not write files or run heavy work.")
    parser.add_argument("--force", action="store_true",
                        help="Allow overwriting full production outputs in non-smoke runs.")


def cmd_validate_price(args: argparse.Namespace) -> int:
    from .price.validate import run
    return run(horizon=args.horizon, coins=args.coins,
               smoke=args.smoke, dry_run=args.dry_run, debug=args.debug)


def cmd_create_price_features(args: argparse.Namespace) -> int:
    from .price.features import run
    return run(horizon=args.horizon, coins=args.coins,
               winsor_p=args.winsor_p,
               smoke=args.smoke, dry_run=args.dry_run)


def cmd_load_sentiment(args: argparse.Namespace) -> int:
    from .sentiment.load import run
    return run(smoke=args.smoke, max_rows=args.max_rows, dry_run=args.dry_run)


def cmd_score_sentiment(args: argparse.Namespace) -> int:
    if args.model == "vader":
        from .sentiment.score_vader import run
        return run(smoke=args.smoke, max_rows=args.max_rows,
                   dry_run=args.dry_run, restart=args.restart)
    if args.model == "finbert":
        from .sentiment.score_finbert import run
        return run(smoke=args.smoke, max_rows=args.max_rows,
                   batch_size=args.batch_size,
                   dry_run=args.dry_run, restart=args.restart)
    if args.model == "cryptobert":
        from .sentiment.score_cryptobert import run
        return run(smoke=args.smoke, max_rows=args.max_rows,
                   batch_size=args.batch_size,
                   dry_run=args.dry_run, restart=args.restart)
    raise SystemExit(f"Unknown sentiment model: {args.model}")


def cmd_create_sentiment_features(args: argparse.Namespace) -> int:
    from .sentiment.aggregate import run
    return run(horizon=args.horizon, smoke=args.smoke,
               max_rows=args.max_rows, no_plots=args.no_plots,
               dry_run=args.dry_run)


def cmd_stationarity(args: argparse.Namespace) -> int:
    from .sentiment.stationarity import run
    return run(horizon=args.horizon, coins=args.coins,
               smoke=args.smoke, no_plots=args.no_plots, dry_run=args.dry_run)


def cmd_merge_features(args: argparse.Namespace) -> int:
    from .features.merge import run
    return run(horizon=args.horizon, smoke=args.smoke, dry_run=args.dry_run)


def cmd_run_models(args: argparse.Namespace) -> int:
    from .modeling.run_models import run
    return run(horizon=args.horizon, set_id=args.set_id,
               coins=args.coins, sentiment_model=args.sentiment_model,
               smoke=args.smoke, dry_run=args.dry_run)


def cmd_diagnostics(args: argparse.Namespace) -> int:
    from .diagnostics.sample_report import run
    return run(horizon=args.horizon, smoke=args.smoke, dry_run=args.dry_run)


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
        "score_finbert":             lambda a: cmd_score_sentiment(_with_attr(a, model="finbert")),
        "score_cryptobert":          lambda a: cmd_score_sentiment(_with_attr(a, model="cryptobert")),
        "create_sentiment_features": cmd_create_sentiment_features,
        "stationarity":              cmd_stationarity,
        "merge_features":            cmd_merge_features,
        "run_models":                cmd_run_models,
        "diagnostics":               cmd_diagnostics,
        "reports":                   cmd_diagnostics,
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
    sp.add_argument("--model", required=True, choices=["vader", "finbert", "cryptobert"])
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
    sp = sub.add_parser("stationarity", help="Run stationarity tests on sentiment features.")
    sp.add_argument("--horizon", default=None)
    sp.add_argument("--coins", nargs="*")
    sp.add_argument("--no-plots", dest="no_plots", action="store_true")
    _add_common(sp)
    sp.set_defaults(func=cmd_stationarity)

    # merge-features
    sp = sub.add_parser("merge-features", help="Merge price + sentiment features.")
    sp.add_argument("--horizon", default=None)
    _add_common(sp)
    sp.set_defaults(func=cmd_merge_features)

    # run-models
    sp = sub.add_parser("run-models", help="Run walk-forward models.")
    sp.add_argument("--horizon", default=None)
    sp.add_argument("--set-id", dest="set_id", default=None)
    sp.add_argument("--coins", nargs="*")
    sp.add_argument("--sentiment-model", dest="sentiment_model", default=None,
                    choices=[None, "vader", "finbert", "cryptobert"])
    _add_common(sp)
    sp.set_defaults(func=cmd_run_models)

    # diagnostics
    sp = sub.add_parser("diagnostics", help="Write a per-horizon sample report.")
    sp.add_argument("--horizon", default="1d")
    _add_common(sp)
    sp.set_defaults(func=cmd_diagnostics)

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
