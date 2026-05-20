#!/usr/bin/env python
"""Thin entry point — delegates to ``thesis_pipeline.modeling.run_models.main``.

Accepts the same arguments as the canonical module (``--horizon``, ``--set-id``
/ ``--set_id``, ``--coins`` / ``--ticker``, ``--sentiment-model``, ``--smoke``,
``--dry-run``, ``--force``, ``--restart``, ``--C``).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from thesis_pipeline.modeling.run_models import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
