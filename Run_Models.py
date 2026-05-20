#!/usr/bin/env python3
# This wrapper is kept for backward compatibility.
# Preferred usage: python -m thesis_pipeline.cli run-models
"""Legacy redirect.

All modeling logic now lives in ``thesis_pipeline.modeling.run_models``.
This file exists only so that existing invocations like

    python Run_Models.py --horizon 1d --set_id B1 --ticker BTC

continue to work. It contains no logic of its own.
"""

import sys
from pathlib import Path

# Make the in-repo ``src/`` layout importable when this file is executed
# directly from the repository root.
_SRC = Path(__file__).resolve().parent / "src"
if _SRC.exists() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from thesis_pipeline.modeling.run_models import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
