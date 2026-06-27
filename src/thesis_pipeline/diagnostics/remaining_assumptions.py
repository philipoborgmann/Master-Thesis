"""Machine-readable remaining-assumptions report (Aufgabe 8 Section E.4).

Generates a JSON document describing the methodological assumptions
that survive into the production run. Each entry records:

* ``rule`` — the substantive rule;
* ``status`` — one of ``verified_by_assertion``, ``verified_by_test``,
  ``configuration_assumption`` or ``remaining_manual_review``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import pandas as pd


VERIFIED_ASSERTION = "verified_by_assertion"
VERIFIED_TEST = "verified_by_test"
CONFIG_ASSUMPTION = "configuration_assumption"
REMAINING_MANUAL = "remaining_manual_review"


def build_remaining_assumptions(*,
                                observation_cutoff: Optional[pd.Timestamp] = None,
                                ) -> dict:
    """Return the full remaining-assumptions payload.

    ``observation_cutoff`` is a config value that the production run
    passes through; we only record it here.
    """
    from .leakage_checks import (
        assert_posts_respect_cutoff,
        assert_no_target_interval_posts_in_predictors,
    )
    from .timing_audit import (
        assert_only_completed_slots, CompletedSlotAssumption,
    )

    entries = [
        {
            "rule": "completed_slot",
            "description": "A sentiment slot may be included only when "
                            "its end timestamp <= observation_cutoff.",
            "status": (VERIFIED_ASSERTION if observation_cutoff is not None
                       else REMAINING_MANUAL),
            "assertion": "thesis_pipeline.diagnostics.timing_audit."
                          "assert_only_completed_slots",
            "observation_cutoff_supplied": observation_cutoff is not None,
            "unresolved_warning_text": CompletedSlotAssumption.UNRESOLVED_WARNING,
        },
        {
            "rule": "post_cutoff",
            "description": "Every Reddit post used as a feature source "
                            "must satisfy created <= cutoff. Equality is "
                            "permitted; created > cutoff fails.",
            "status": VERIFIED_ASSERTION,
            "assertion": "thesis_pipeline.diagnostics.leakage_checks."
                          "assert_posts_respect_cutoff",
        },
        {
            "rule": "predictor_target_separation",
            "description": "No post created strictly after the prediction "
                            "timestamp may enter the predictor stack.",
            "status": VERIFIED_ASSERTION,
            "assertion": "thesis_pipeline.diagnostics.leakage_checks."
                          "assert_no_target_interval_posts_in_predictors",
        },
        {
            "rule": "market_cap_availability",
            "description": "market_cap_available_at < prediction_timestamp "
                            "for every matched row; the as-of merge uses "
                            "allow_exact_matches=False.",
            "status": VERIFIED_ASSERTION,
            "assertion": "thesis_pipeline.diagnostics.leakage_checks."
                          "assert_market_cap_asof_correct",
        },
        {
            "rule": "volatility_regime_availability",
            "description": "vol_regime_available_at < prediction_timestamp "
                            "via the shared attach_regime_asof helper.",
            "status": VERIFIED_ASSERTION,
            "assertion": "thesis_pipeline.evaluation.regime_join."
                          "attach_regime_asof",
        },
        {
            "rule": "universe_definitions",
            "description": "requested = --coins / full feature-frame "
                            "universe; available = subset present in the "
                            "feature frame; realized = tickers that produced "
                            "predictions. Matching keys off requested.",
            "status": VERIFIED_ASSERTION,
            "assertion": "thesis_pipeline.modeling.naive_reference."
                          "resolve_universes",
        },
        {
            "rule": "legacy_output_policy",
            "description": "Model cache without requested-universe metadata "
                            "is refused. New runs always produce a "
                            "universe-hashed filename.",
            "status": VERIFIED_TEST,
            "test":   "tests/test_model_cache_universe_identity.py",
        },
        {
            "rule": "h1_bh_family_scope",
            "description": "H1_BH_SCOPE='all_primary_h1_tests' — every "
                            "primary-nested ECON_* vs ECON pair enters "
                            "one BH pool. H1/H2/H3 are never pooled.",
            "status": CONFIG_ASSUMPTION,
            "configuration": "thesis_pipeline.evaluation.incremental."
                              "H1_BH_SCOPE",
        },
        {
            "rule": "small_cluster_warning",
            "description": "Cluster-robust inference flags clusters < 10 "
                            "as small_cluster_warning=True without "
                            "rejecting the test.",
            "status": CONFIG_ASSUMPTION,
            "configuration": "thesis_pipeline.evaluation.diff_in_improvement."
                              "SMALL_CLUSTER_THRESHOLD",
        },
        {
            "rule": "zero_se_tolerance",
            "description": "ZERO_SE_TOLERANCE=1e-12 separates exact-null "
                            "from degenerate cluster-robust SE.",
            "status": CONFIG_ASSUMPTION,
            "configuration": "thesis_pipeline.evaluation.diff_in_improvement."
                              "ZERO_SE_TOLERANCE",
        },
    ]
    return {
        "v4_remaining_assumptions": entries,
        "observation_cutoff": (
            observation_cutoff.isoformat() if isinstance(
                observation_cutoff, pd.Timestamp) else observation_cutoff
        ),
    }


def write_remaining_assumptions_report(path: Path,
                                         *,
                                         observation_cutoff:
                                             Optional[pd.Timestamp] = None,
                                         ) -> Path:
    """Serialise the report to a JSON file under ``path``. Uses an
    atomic write so a killed process never leaves a half-written file."""
    payload = build_remaining_assumptions(observation_cutoff=observation_cutoff)
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, default=str))
    import os
    os.replace(tmp, p)
    return p
