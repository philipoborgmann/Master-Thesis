#!/usr/bin/env python3
"""Legacy redirect for ``python Merge_Features.py``.

All merge logic now lives in :mod:`thesis_pipeline.features.merge`.

Preferred usage:
    python -m thesis_pipeline.cli merge-features
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from thesis_pipeline.features.merge import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
