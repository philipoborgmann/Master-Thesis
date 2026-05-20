"""Per-horizon sample reports.

Emits a small markdown summary under
``Outputs/diagnostics/sample_report_<horizon>.md`` describing row counts,
date ranges, and feature columns of the final merged feature parquet.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from ..config import resolve_path
from ..io import ensure_dir
from ..logging_utils import get_logger, log_stage_header


def run(*, horizon: str = "1d", smoke: bool = False, dry_run: bool = False) -> int:
    final_path = resolve_path("final_features_pattern", horizon=horizon)
    out_root = resolve_path("smoke_root") if smoke else resolve_path("diagnostics_root")
    out_path = ensure_dir(out_root) / f"sample_report_{horizon}.md"

    log_stage_header(
        "diagnostics",
        mode="dry-run" if dry_run else ("smoke" if smoke else "full"),
        inputs=[final_path],
        outputs=[out_path],
        extra={"horizon": horizon},
    )

    if dry_run:
        return 0
    if not Path(final_path).exists():
        get_logger().warning("Final features parquet not found: %s — skipping", final_path)
        return 0

    df = pd.read_parquet(final_path)
    n_rows = len(df)
    tickers = sorted(df["ticker"].dropna().unique()) if "ticker" in df.columns else []
    ts = pd.to_datetime(df["timestamp"]) if "timestamp" in df.columns else None

    lines = [
        f"# Sample report — horizon {horizon}",
        "",
        f"- file: `{final_path}`",
        f"- rows: {n_rows}",
        f"- tickers ({len(tickers)}): {', '.join(tickers)}",
    ]
    if ts is not None and len(ts) > 0:
        lines += [
            f"- date range: {ts.min()} → {ts.max()}",
        ]
    lines += ["", "## Columns", ""]
    for c in df.columns:
        lines.append(f"- `{c}` — dtype {df[c].dtype}, nulls {df[c].isna().sum()}")
    out_path.write_text("\n".join(lines), encoding="utf-8")
    get_logger().info("wrote sample report: %s", out_path)
    return 0
