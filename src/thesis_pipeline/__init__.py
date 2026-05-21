"""thesis_pipeline — staged, reproducible pipeline for the crypto sentiment thesis.

Public entry point: ``python -m thesis_pipeline.cli --help``.

The full pipeline logic lives in this package; ``scripts/`` holds thin
entry points that re-export the matching module's ``main``. The historical
root-level scripts (Create_Price_Features.py, Run_Models.py, …) have been
retired — archived copies live outside the repository.
"""

__version__ = "0.1.0"
