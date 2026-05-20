"""Price validation stage.

Delegates to ``Price_Data_Validation.py`` for the real work. Behavior,
formulas, and outputs are unchanged.
"""

from __future__ import annotations

from typing import Sequence

from ..config import resolve_path, smoke_coins
from ..logging_utils import get_logger, log_stage_header
from ..utils import run_script_main


def run(*, horizon: str | None = None,
        coins: Sequence[str] | None = None,
        smoke: bool = False,
        dry_run: bool = False,
        debug: bool = False) -> int:
    """Run price validation.

    Builds an argv for ``Price_Data_Validation.py`` based on configs, then
    invokes its ``main()``.
    """
    if smoke and not coins:
        coins = smoke_coins()

    inputs  = [resolve_path("raw_price_root"), resolve_path("cmc_root")]
    outputs = [resolve_path("price_validation_root")]

    log_stage_header(
        "validate_price",
        mode="dry-run" if dry_run else ("smoke" if smoke else "full"),
        inputs=inputs,
        outputs=outputs,
        extra={"horizon": horizon or "(all)", "coins": list(coins) if coins else "(all)"},
    )

    if dry_run:
        get_logger().info("dry-run: not invoking Price_Data_Validation.py")
        return 0

    argv: list[str] = [
        "--price_dir",  str(resolve_path("raw_price_root")),
        "--cmc_dir",    str(resolve_path("cmc_root")),
        "--output_dir", str(resolve_path("price_validation_root")),
    ]
    if coins:
        for c in coins:
            argv += ["--coin", c]
    if debug:
        argv += ["--debug"]

    return run_script_main("validate_price", argv)
