"""Pre-specified config for the evaluation-layer multiplicity plan.

This is the SINGLE source of truth for the multiple-testing family
structure, metric roles and the Diebold-Mariano-type inference settings.
Nothing in this block may be hard-coded elsewhere — the writers import
these constants so a reviewer can audit the entire multiplicity plan in
one place. (No external pre-registration exists; the user-facing wording
is "pre-specified".)

Family roles: A–D are CONFIRMATORY; E1, E2 (pairwise horizon contrasts) and
EXPL_sentiment_only_directional are EXPLORATORY. Every family — confirmatory
or exploratory — is corrected by the SAME central family-aware BH pass,
WITHIN the family only.

Run mode (guardrail): evaluation is a SINGLE all-horizons
``evaluate-signals`` pass. Benjamini-Hochberg pools p-values across
horizons WITHIN each family (24 tests = 3 horizons × 2 models × 4
blocks). BH is NEVER applied per-horizon.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Confirmatory block mode
# ---------------------------------------------------------------------------

#: STANDARD. "all_blocks_pooled" | "single_primary".
#:
#: * ``all_blocks_pooled`` (default): all 4 feature blocks per model are
#:   co-equal confirmatory members; each confirmatory family = 3 horizons
#:   × 2 models × 4 blocks = 24 tests, ONE BH pass. No exploratory block
#:   families exist in this mode.
#: * ``single_primary`` (appendix/robustness): confirmatory families use
#:   ONE block per model (:data:`PRIMARY_SENTIMENT_BLOCK`) → 6 tests;
#:   non-primary blocks go to pooled exploratory families.
#:
#: FORBIDDEN in both modes: a separate BH family per block (that would
#: inflate FDR across the block dimension).
CONFIRMATORY_BLOCK_MODE = "all_blocks_pooled"

#: INACTIVE under the default all_blocks_pooled mode; used only if
#: CONFIRMATORY_BLOCK_MODE is switched to "single_primary".
PRIMARY_SENTIMENT_BLOCK = "L"

#: Per-family significance level. One knob per family; all default 0.05.
ALPHA_PRESPECIFIED = 0.05

#: Coin-level output is DESCRIPTIVE only — no q-values, no per-coin
#: significance claims. Point estimates only.
COIN_LEVEL_MODE = "descriptive"

#: Family E splits into two sub-families, one per metric.
FAMILY_E_METRICS = ["log_loss", "accuracy"]

#: Flag balanced_accuracy as primary-robustness when |base_rate-0.5| > delta.
CLASS_IMBALANCE_DELTA = 0.05

#: Log-loss clipping epsilon — MUST match the value used by the pooled
#: metrics (``run_models.compute_metrics``), ``incremental`` and
#: ``naive_comparison`` so the DM log-loss reproduces the pooled figure
#: on the identical matched sample.
LOGLOSS_CLIP_EPS = 1e-15


# ---------------------------------------------------------------------------
# Diebold-Mariano-type inference (moving-block bootstrap primary)
# ---------------------------------------------------------------------------

DM_INFERENCE = {
    "primary": "moving_block_bootstrap",   # nested models → bootstrap primary
    "n_boot": 10000,
    "block_length": "auto",                # auto = round(T**(1/3)); else int
    "hac_lag": "auto",                     # auto = floor(4*(T/100)**(2/9))
    "clip_eps": LOGLOSS_CLIP_EPS,
    "two_sided": True,
    "timestamp_weight": "equal",           # equal-weight per timestamp (primary)
    "small_sample_correction": False,      # HLN t_{T-1} on HAC stat ONLY, labeled
    #: Below this variance the loss differential is treated as degenerate
    #: (nested-model H0 collapse) and the test is marked invalid instead
    #: of dividing by ~0.
    "degenerate_var_tol": 1e-18,
}


# ---------------------------------------------------------------------------
# Pre-specified confirmatory + exploratory families
# ---------------------------------------------------------------------------

#: Family identifiers. Log-loss (A, E1) and directional (B, E2) p-values
#: NEVER share a family; economic/backtest p-values never share a family
#: with forecast-quality tests.
#:
#: Roles (thesis design):
#:   * CONFIRMATORY : A_H1_logloss, B_H1_directional, C_H2_volatility,
#:                    D_H3_marketcap.
#:   * EXPLORATORY  : E1_horizon_logloss, E2_horizon_accuracy (pairwise
#:                    horizon contrasts) and EXPL_sentiment_only_directional.
#: The horizon-comparison families E1/E2 are EXPLORATORY — they are pairwise
#: horizon contrasts derived from the confirmatory effects, not part of the
#: pre-specified confirmatory hypothesis set.
FAMILY_A_H1_LOGLOSS      = "A_H1_logloss"
FAMILY_B_H1_DIRECTIONAL  = "B_H1_directional"
FAMILY_C_H2_VOLATILITY   = "C_H2_volatility"
FAMILY_D_H3_MARKETCAP    = "D_H3_marketcap"
FAMILY_E1_HORIZON_LOGLOSS  = "E1_horizon_logloss"
FAMILY_E2_HORIZON_ACCURACY = "E2_horizon_accuracy"
FAMILY_EXPL_SENTIMENT_DIRECTIONAL = "EXPL_sentiment_only_directional"

#: The four pre-specified CONFIRMATORY families (A–D). E1/E2 are NOT here.
CONFIRMATORY_FAMILIES = (
    FAMILY_A_H1_LOGLOSS,
    FAMILY_B_H1_DIRECTIONAL,
    FAMILY_C_H2_VOLATILITY,
    FAMILY_D_H3_MARKETCAP,
)

#: EXPLORATORY families. Each is BH-corrected WITHIN the family (exactly like
#: the confirmatory ones — same central family-aware correction), but reported
#: as exploratory: they are NOT part of the pre-specified confirmatory set and
#: must never alter the A–D counts. Listed in the manifest with
#: ``family_role="exploratory"``.
#:
#: * E1_horizon_logloss  — pairwise horizon contrasts of the log-loss effect
#:   (its OWN BH pass; pooled over 3 horizon pairs × 2 sentiment methods × 4
#:   feature blocks; never mixed with E2 or with A–D).
#: * E2_horizon_accuracy — pairwise horizon contrasts of the accuracy effect
#:   (its OWN BH pass; same pooling; never mixed with E1 or with A–D).
#: * EXPL_sentiment_only_directional — sentiment-only SENT_* vs ECON McNemar.
EXPLORATORY_FAMILIES = (
    FAMILY_E1_HORIZON_LOGLOSS,
    FAMILY_E2_HORIZON_ACCURACY,
    FAMILY_EXPL_SENTIMENT_DIRECTIONAL,
)

#: Human-readable descriptions for the manifest.
FAMILY_DESCRIPTIONS = {
    FAMILY_A_H1_LOGLOSS:
        "H1 probabilistic: sentiment has lower OOS log loss. "
        "Test = log-loss timestamp Diebold-Mariano-type (moving-block "
        "bootstrap p-value).",
    FAMILY_B_H1_DIRECTIONAL:
        "H1 directional: sentiment has higher OOS accuracy. "
        "Test = pooled ECON-vs-augmented McNemar.",
    FAMILY_C_H2_VOLATILITY:
        "H2 volatility regime: high-vs-low volatility cluster-robust "
        "difference-in-improvement (accuracy-based).",
    FAMILY_D_H3_MARKETCAP:
        "H3 market-cap regime: small-vs-large cap cluster-robust "
        "difference-in-improvement (accuracy-based).",
    FAMILY_E1_HORIZON_LOGLOSS:
        "EXPLORATORY: pairwise horizon contrasts of the log-loss sentiment "
        "effect (independent-sample z on bootstrap SEs; own BH pass).",
    FAMILY_E2_HORIZON_ACCURACY:
        "EXPLORATORY: pairwise horizon contrasts of the accuracy sentiment "
        "effect (independent-sample z on bootstrap SEs; own BH pass).",
    FAMILY_EXPL_SENTIMENT_DIRECTIONAL:
        "EXPLORATORY (not confirmatory): sentiment-only SENT_* vs ECON "
        "directional McNemar, 3 horizons × 8 SENT sets, BH within family. "
        "These sets are not nested H1 members (Family B covers only the "
        "combined ECON_* sets).",
}

#: Descriptive / separate surfaces that must NEVER be assigned a
#: confirmatory family or a q-value.
DESCRIPTIVE_SURFACES = (
    "absolute_vs_naive",        # diagnostic floor (model vs naive)
    "regime_mcnemar_tests",     # superseded per-regime McNemar (old H2/H3)
    "regime_mcnemar_summary",
    "coin_level",               # per-ticker point estimates
)

#: Economic/backtest p-values, if any, form their own separate family —
#: never mixed with A–D (confirmatory) or the exploratory families.
FAMILY_F_ECONOMIC = "F_economic"


# ---------------------------------------------------------------------------
# Metric roles
# ---------------------------------------------------------------------------

#: (role, used_for_hpo, notes) per metric. Emitted verbatim to
#: metric_roles.csv.
METRIC_ROLES = {
    "log_loss": {
        "role": "primary_probabilistic",
        "used_for_hpo": True,
        "notes": "HPO objective; Family A confirmatory test.",
    },
    "accuracy": {
        "role": "primary_directional",
        "used_for_hpo": False,
        "notes": "Family B confirmatory test (McNemar).",
    },
    "balanced_accuracy": {
        "role": "primary_robustness",
        "used_for_hpo": False,
        "notes": "Elevated to primary-robustness only when a horizon's "
                 "class imbalance is flagged (|base_rate-0.5| > "
                 f"{CLASS_IMBALANCE_DELTA}).",
    },
    "brier_score": {
        "role": "secondary", "used_for_hpo": False,
        "notes": "Proper scoring rule, reported for completeness.",
    },
    "precision": {"role": "secondary", "used_for_hpo": False, "notes": ""},
    "recall":    {"role": "secondary", "used_for_hpo": False, "notes": ""},
    "f1":        {"role": "secondary", "used_for_hpo": False, "notes": ""},
}
