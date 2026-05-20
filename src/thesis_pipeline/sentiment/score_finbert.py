"""FinBERT sentiment scoring — wraps Sentiment_score_finbert.py."""

from __future__ import annotations

from ..config import resolve_path
from ..logging_utils import get_logger, log_stage_header
from ..utils import run_script_main


def run(*, smoke: bool = False, max_rows: int | None = None,
        batch_size: int | None = None, dry_run: bool = False,
        restart: bool = False) -> int:
    inputs  = [resolve_path("sentiment_combined")]
    outputs = [resolve_path("sentiment_scored_finbert"), resolve_path("finbert_checkpoints")]

    if smoke and max_rows is None:
        max_rows = 200

    log_stage_header(
        "score_finbert",
        mode="dry-run" if dry_run else ("smoke" if smoke else "full"),
        inputs=inputs,
        outputs=outputs,
        extra={
            "max_rows": max_rows if max_rows is not None else "(all)",
            "batch_size": batch_size or "(script default 64)",
            "restart": restart,
        },
    )

    if dry_run:
        get_logger().info("dry-run: not invoking Sentiment_score_finbert.py")
        return 0

    argv: list[str] = [
        "--input",  str(resolve_path("sentiment_combined")),
        "--output", str(resolve_path("sentiment_scored_finbert")),
    ]
    if batch_size is not None:
        argv += ["--batch_size", str(batch_size)]
    if max_rows is not None:
        argv += ["--validate", "--validate_n", str(max_rows)]
    if restart:
        argv += ["--restart"]
    return run_script_main("score_finbert", argv)
