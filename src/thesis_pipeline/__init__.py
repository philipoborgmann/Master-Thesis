"""thesis_pipeline — staged, reproducible pipeline for the crypto sentiment thesis.

Public entry point: ``python -m thesis_pipeline.cli --help``.

This package is an *organizational* refactor of the original root-level
scripts (Create_Price_Features.py, Run_Models.py, …). The original scripts
remain functional as backward-compatible wrappers; the new modules expose
the same behavior behind a cleaner package layout.
"""

__version__ = "0.1.0"
