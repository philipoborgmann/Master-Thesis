#!/usr/bin/env python3
"""Legacy redirect for `python Sentiment_Data_Load.py`.

All logic now lives in :mod:`thesis_pipeline.sentiment.load`.

Preferred usage:
    python -m thesis_pipeline.cli load-sentiment
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from thesis_pipeline.sentiment.load import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
