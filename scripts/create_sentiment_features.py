#!/usr/bin/env python3
"""Thin entry point for the `create-sentiment-features` stage.

Preferred usage:
    python -m thesis_pipeline.cli create-sentiment-features
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from thesis_pipeline.sentiment.aggregate import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
