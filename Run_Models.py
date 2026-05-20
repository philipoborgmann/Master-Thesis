#!/usr/bin/env python3
"""Legacy redirect for backward compatibility."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from thesis_pipeline.modeling.run_models import main

if __name__ == "__main__":
    raise SystemExit(main())