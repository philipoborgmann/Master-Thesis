"""Sentiment loading stage — wraps Sentiment_Data_Load.py.

Reads per-subreddit Reddit dumps under ``Data/Raw/Sentiment/<subreddit>/submission.csv``
plus the global ``st-data-full.{parquet,xlsx}`` and produces
``Data/Processed/Sentiment/sentiment_combined.csv`` together with descriptive
statistics under ``Outputs/deskriptiv/``.
"""

from __future__ import annotations

from ..config import resolve_path
from ..logging_utils import get_logger, log_stage_header
from ..utils import run_script_main


def run(*, smoke: bool = False, max_rows: int | None = None,
        dry_run: bool = False) -> int:
    inputs = [
        resolve_path("raw_sentiment_root"),
        resolve_path("subreddit_ticker_mapping"),
    ]
    outputs = [
        resolve_path("sentiment_combined"),
        resolve_path("descriptive_root"),
    ]
    if smoke and max_rows is None:
        max_rows = 10_000

    log_stage_header(
        "load_sentiment",
        mode="dry-run" if dry_run else ("smoke" if smoke else "full"),
        inputs=inputs,
        outputs=outputs,
        extra={"max_rows": max_rows if max_rows is not None else "(all)"},
    )

    if dry_run:
        get_logger().info("dry-run: not invoking Sentiment_Data_Load.py")
        return 0

    argv: list[str] = [
        "--raw_dir",     str(resolve_path("raw_sentiment_root")),
        "--mapping",     str(resolve_path("subreddit_ticker_mapping")),
        "--output_csv",  str(resolve_path("sentiment_combined")),
        "--output_stats", str(resolve_path("descriptive_root") / "descriptive_statistics.xlsx"),
    ]
    # Sentiment_Data_Load.py does not currently take --max_rows; the smoke
    # cap is enforced downstream in scoring/aggregation. TODO(refactor_log):
    # add native --max_rows support if needed.
    if max_rows is not None:
        get_logger().warning(
            "Sentiment_Data_Load.py does not accept --max_rows directly; "
            "smoke cap will be re-applied at the scoring/aggregation stage."
        )

    return run_script_main("load_sentiment", argv)
