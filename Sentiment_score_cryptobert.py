#!/usr/bin/env python3
"""Legacy redirect for `python Sentiment_score_cryptobert.py`.

All logic now lives in :mod:`thesis_pipeline.sentiment.score_cryptobert`.

Preferred usage:
    python -m thesis_pipeline.cli score-sentiment --model cryptobert
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from thesis_pipeline.sentiment.score_cryptobert import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
