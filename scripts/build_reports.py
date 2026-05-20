#!/usr/bin/env python
"""Thin wrapper for `python -m thesis_pipeline.cli diagnostics`."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from thesis_pipeline.cli import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main(["diagnostics", *sys.argv[1:]]))
