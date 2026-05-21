#!/usr/bin/env python
"""Thin entry point — delegates to ``thesis_pipeline.evaluation.evaluate_signals.main``."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from thesis_pipeline.evaluation.evaluate_signals import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
