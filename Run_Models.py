#!/usr/bin/env python3
"""Legacy redirect for ``python Run_Models.py``.

All modeling logic now lives in :mod:`thesis_pipeline.modeling.run_models`.
This file exists only so historical invocations such as

    python Run_Models.py --horizon 1d --set_id B1 --ticker BTC

continue to work. It contains no logic of its own.

Preferred usage:
    python -m thesis_pipeline.cli run-models
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from thesis_pipeline.modeling.run_models import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
