"""Feature-set registry — v4 (Variante A, 17 sets).

Reads ``feature_sets.xlsx`` (the source of truth) and falls back to
``configs/feature_sets.yaml``. The behaviour contract is:

* exactly 17 ``set_id`` values, all listed in :data:`SET_ID_PATTERN`;
* every set has a non-empty feature list (no empty-feature exception);
* ``has_posts`` and ``{model}_directional_post_count`` are **diagnostic-only**
  columns; they MUST NOT appear in any of the 17 set definitions.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

from ..config import load_config, resolve_path


# The canonical 17 feature sets (v4 — Variante A).
#
#   * ECON                 — economic-only benchmark (no sentiment).
#   * SENT_{model}_{block} — sentiment-only, one block per (vader, cryptobert).
#   * ECON_{model}_{block} — combined = ECON core + sentiment block.
#
# Blocks: L = score mean only; LD = + std; DA = bullishness + log1p_post_count;
# F = all four (mean, std, bullishness, log1p_post_count). See feature_sets.yaml
# for the canonical feature lists.
SET_ID_PATTERN = (
    "ECON",
    "SENT_VAD_L", "SENT_VAD_LD", "SENT_VAD_DA", "SENT_VAD_F",
    "SENT_CBT_L", "SENT_CBT_LD", "SENT_CBT_DA", "SENT_CBT_F",
    "ECON_VAD_L", "ECON_VAD_LD", "ECON_VAD_DA", "ECON_VAD_F",
    "ECON_CBT_L", "ECON_CBT_LD", "ECON_CBT_DA", "ECON_CBT_F",
)
assert len(SET_ID_PATTERN) == 17, "v4 registry must contain exactly 17 sets"

# Diagnostic-only columns. Exact-name list (kept for symmetry and for
# clarity in error messages); the canonical primary-grid validation uses
# the pattern matcher :func:`is_diagnostic_only_feature` below so the rule
# extends automatically to all `_mean / _median / _std / _max / _min`
# variants and to both VADER and CryptoBERT.
DIAGNOSTIC_ONLY_COLUMNS = (
    "has_posts",
    "vader_directional_post_count",
    "cryptobert_directional_post_count",
    "vader_combined_score_mean",
    "vader_combined_score_median",
    "vader_combined_score_std",
    "vader_selftext_score_mean",
    "vader_selftext_score_median",
    "vader_selftext_score_std",
    "cryptobert_combined_score_mean",
    "cryptobert_combined_score_median",
    "cryptobert_combined_score_std",
    "cryptobert_selftext_score_mean",
    "cryptobert_selftext_score_median",
    "cryptobert_selftext_score_std",
)


def is_diagnostic_only_feature(feature: str) -> bool:
    """Return True if ``feature`` is a robustness/diagnostic-only column.

    These columns may live in the merged feature frame for supplementary
    analyses, but they must never appear in any of the 17 primary-grid
    sets. Pattern-based so a new ``_max`` / ``_kurtosis`` / etc.
    aggregation cannot silently slip into the grid:

      * ``has_posts``                              — derived indicator.
      * ``*_directional_post_count``               — post-class count.
      * ``*_combined_score_*``                     — title+selftext blend
        (Variante A keeps it for robustness, not in the grid).
      * ``*_selftext_score_*``                     — selftext-only score.
      * ``*_weighted_mean``                        — engagement-weighted
        column (Variante A removed engagement weighting entirely; any
        residual ``*_weighted_mean`` is therefore forbidden).
    """
    if not isinstance(feature, str):
        return False
    if feature == "has_posts":
        return True
    if feature.endswith("_directional_post_count"):
        return True
    if "_combined_score_" in feature:
        return True
    if "_selftext_score_" in feature:
        return True
    if feature.endswith("_weighted_mean"):
        return True
    return False


# Set-IDs removed during the v4 17-set registry refactor + FinBERT removal.
# Each entry is a HUMAN-READABLE migration sentence (not a bare replacement
# ID). The CLI guard concatenates this string into its SystemExit so the
# user sees exactly what to use — and, crucially, when there is no exact
# feature-equivalent v4 replacement, the message says so rather than
# silently mapping to a similarly-named v4 ID with a different information
# set. The previous "S2 → SENT_CBT_LD" shorthand implied a one-to-one swap
# even when the underlying score variant changed (combined vs title-only),
# which would mislead anyone doing a re-run.

_FINBERT_REMOVED = (
    "removed in v4 — FinBERT was dropped from the pipeline "
    "(trained on financial / analyst language, no meaningful variation "
    "on Reddit/crypto data). No v4 replacement exists; rerun with one "
    "of the VADER (SENT_VAD_*) or CryptoBERT (SENT_CBT_*) sets instead."
)

_NO_EXACT_EQUIVALENT_CBT = (
    "No exact v4 equivalent exists — the v3 set used a *combined* "
    "title+selftext score; v4 sentiment sets are *title-only*. Choose "
    "one of SENT_CBT_L, SENT_CBT_LD, SENT_CBT_DA, SENT_CBT_F based on "
    "the intended information set."
)
_NO_EXACT_EQUIVALENT_VAD = (
    "No exact v4 equivalent exists — the v3 set used a *combined* "
    "title+selftext score; v4 sentiment sets are *title-only*. Choose "
    "one of SENT_VAD_L, SENT_VAD_LD, SENT_VAD_DA, SENT_VAD_F based on "
    "the intended information set."
)
_NO_EXACT_EQUIVALENT_ECON_CBT = (
    "No exact v4 equivalent exists — the v3 combined set used a "
    "*combined* title+selftext sentiment score; v4 combined sets are "
    "*title-only*. Choose one of ECON_CBT_L, ECON_CBT_LD, ECON_CBT_DA, "
    "ECON_CBT_F based on the intended information set."
)
_NO_EXACT_EQUIVALENT_ECON_VAD = (
    "No exact v4 equivalent exists — the v3 combined set used a "
    "*combined* title+selftext sentiment score; v4 combined sets are "
    "*title-only*. Choose one of ECON_VAD_L, ECON_VAD_LD, ECON_VAD_DA, "
    "ECON_VAD_F based on the intended information set."
)
_M_REMOVED = (
    "removed in v4 — Variante A no longer runs mixed-scorer (VADER + "
    "CryptoBERT) sets. Pick the relevant single-scorer family instead: "
    "ECON_VAD_F or ECON_CBT_F for the full combined block, SENT_VAD_F "
    "or SENT_CBT_F for the sentiment-only block."
)

# Bare-benchmark / economic-only IDs collapse cleanly into ECON because
# both v3 B1/B2/B3/B4/B5/B6/E1/E2/E3/E4 and v4 ECON used title-free
# economic features. ECON adds cum_log_return_21d and switches market_cap
# to log_market_cap_lag1, so the information set is *broader* — still
# named explicitly here so the user knows what changed.
_ECON_EXPANDED = (
    "Use ECON. The v4 ECON set is a superset of the old v3 economic "
    "benchmark: it adds cum_log_return_21d and replaces market_cap_t "
    "with log_market_cap_lag1 (strict as-of merge)."
)
_NAIVE_NOT_A_SET = (
    "The historical-majority rolling-probability benchmark is no longer "
    "a feature set in v4 — it is a separate evaluation reference (NAIVE), "
    "exposed via thesis_pipeline.modeling.benchmarks.run_rolling_probability "
    "and evaluated outside the 17-set grid."
)

REMOVED_SET_IDS: dict[str, str] = {
    # Old benchmark family.
    # B1 was the rolling-probability benchmark — that is NOT a feature
    # set in v4; it lives in modeling.benchmarks.run_rolling_probability
    # and is evaluated as the NAIVE reference.
    "B1": _NAIVE_NOT_A_SET,
    # B2..B6 / E1..E4 were variants of the economic-only baseline.
    "B2": _ECON_EXPANDED, "B3": _ECON_EXPANDED, "B4": _ECON_EXPANDED,
    "B5": _ECON_EXPANDED, "B6": _ECON_EXPANDED,
    "E1": _ECON_EXPANDED, "E2": _ECON_EXPANDED, "E3": _ECON_EXPANDED,
    "E4": _ECON_EXPANDED,
    # CryptoBERT sentiment-only (v3 S*).
    # S1 was the bare combined-score mean — no exact title-only twin.
    "S1": _NO_EXACT_EQUIVALENT_CBT,
    "S2": _NO_EXACT_EQUIVALENT_CBT,
    "S3": _NO_EXACT_EQUIVALENT_CBT,
    "S4": _NO_EXACT_EQUIVALENT_CBT,
    "S5": _NO_EXACT_EQUIVALENT_CBT,
    "S6": _NO_EXACT_EQUIVALENT_CBT,
    "S7": _NO_EXACT_EQUIVALENT_CBT,
    # VADER sentiment-only.
    "SV1": _NO_EXACT_EQUIVALENT_VAD,
    "SV2": _NO_EXACT_EQUIVALENT_VAD,
    "SV3": _NO_EXACT_EQUIVALENT_VAD,
    # CryptoBERT combined (v3 C*).
    "C1": _NO_EXACT_EQUIVALENT_ECON_CBT,
    "C2": _NO_EXACT_EQUIVALENT_ECON_CBT,
    "C3": _NO_EXACT_EQUIVALENT_ECON_CBT,
    "C4": _NO_EXACT_EQUIVALENT_ECON_CBT,
    "C5": _NO_EXACT_EQUIVALENT_ECON_CBT,
    "C6": _NO_EXACT_EQUIVALENT_ECON_CBT,
    # VADER combined (v3 CV*).
    "CV1": _NO_EXACT_EQUIVALENT_ECON_VAD,
    "CV2": _NO_EXACT_EQUIVALENT_ECON_VAD,
    "CV3": _NO_EXACT_EQUIVALENT_ECON_VAD,
    "CV4": _NO_EXACT_EQUIVALENT_ECON_VAD,
    "CV5": _NO_EXACT_EQUIVALENT_ECON_VAD,
    "CV6": _NO_EXACT_EQUIVALENT_ECON_VAD,
    # Old multi-scorer M family.
    "M1": _M_REMOVED, "M2": _M_REMOVED, "M3": _M_REMOVED,
    "M4": _M_REMOVED, "M5": _M_REMOVED, "M6": _M_REMOVED,
    # FinBERT-era families (SC* / SF*).
    "SC1": _FINBERT_REMOVED, "SC2": _FINBERT_REMOVED, "SC3": _FINBERT_REMOVED,
    "SF1": _FINBERT_REMOVED, "SF2": _FINBERT_REMOVED, "SF3": _FINBERT_REMOVED,
}


def load_feature_sets_yaml() -> Dict[str, dict]:
    cfg = load_config("feature_sets")
    return dict(cfg.get("sets", {}))


def load_feature_sets_xlsx() -> Dict[str, dict] | None:
    """Load ``feature_sets.xlsx`` if available. Returns ``None`` otherwise.

    Header handling is intentionally kept in lock-step with
    :func:`thesis_pipeline.modeling.run_models.load_feature_sets` so the
    registry and the modelling loader can never disagree on which column
    holds the comma-separated feature list.
    """
    path = resolve_path("feature_sets_xlsx")
    if not Path(path).exists():
        return None
    try:
        import pandas as pd
    except ImportError:
        return None
    try:
        df = pd.read_excel(path, sheet_name="feature_sets")
    except Exception:  # noqa: BLE001 — be tolerant; the script handles details
        return None

    df.columns = (df.columns.astype(str)
                  .str.strip()
                  .str.lower()
                  .str.replace(r"[^a-z0-9]+", "_", regex=True)
                  .str.strip("_"))

    for preferred in ("feature_columns_comma_separated", "feature_columns"):
        if preferred in df.columns:
            if "features" in df.columns and preferred != "features":
                df = df.drop(columns=["features"])
            df = df.rename(columns={preferred: "features"})
            break

    sets: Dict[str, dict] = {}
    has_sentiment_col = "sentiment_model" in df.columns
    for _, row in df.iterrows():
        set_id = str(row.get("set_id") or "").strip()
        if not set_id:
            continue
        raw_features = row.get("features", "")
        features = [f.strip() for f in str(raw_features).split(",") if f.strip()]
        sets[set_id] = {
            "features": features,
            "category": str(row.get("category", "") or ""),
            "label":    str(row.get("label", "") or ""),
            "sentiment_model": (str(row.get("sentiment_model")) or None)
                if has_sentiment_col else None,
        }
    return sets


def load_feature_sets() -> Dict[str, dict]:
    """Prefer feature_sets.xlsx when present; otherwise fall back to YAML."""
    src = load_config("feature_sets").get("source_of_truth", "feature_sets.xlsx")
    if src == "feature_sets.xlsx":
        sets = load_feature_sets_xlsx()
        if sets:
            return sets
    return load_feature_sets_yaml()


def validate_registry(sets: Dict[str, dict]) -> List[str]:
    """Return a list of validation problems (empty list → registry is OK).

    v4 contract — every category of problem is reported with a distinct
    prefix so callers can grep:

      * ``[shape] total count …``         — the registry has != 17 sets.
      * ``[shape] missing required IDs …`` — at least one of the canonical
        17 v4 IDs is absent.
      * ``[shape] unexpected IDs …``      — IDs outside SET_ID_PATTERN are
        present (covers legacy v3 leftovers and typos alike).
      * ``[content] <set_id>: empty …``    — feature list is empty.
      * ``[content] <set_id>: duplicate …``— feature list contains
        duplicates.
      * ``[content] <set_id>: diagnostic-only feature …`` — feature
        matches :func:`is_diagnostic_only_feature`.
    """
    problems: list[str] = []
    declared = set(SET_ID_PATTERN)
    present  = set(sets.keys())

    # ── Shape: exact 17-set contract ──────────────────────────────
    if len(sets) != 17:
        problems.append(
            f"[shape] total count {len(sets)} != 17 (v4 registry contract)"
        )
    missing = sorted(declared - present)
    if missing:
        problems.append(f"[shape] missing required IDs: {missing}")
    unexpected = sorted(present - declared)
    if unexpected:
        problems.append(
            f"[shape] unexpected IDs (not in SET_ID_PATTERN): {unexpected}"
        )

    # ── Content: per-set checks ───────────────────────────────────
    for set_id, spec in sorted(sets.items()):
        features = list(spec.get("features", []))
        if not features:
            problems.append(
                f"[content] {set_id}: empty feature list (no v4 set may be empty)"
            )
            continue
        if len(features) != len(set(features)):
            problems.append(f"[content] {set_id}: duplicate features {features}")
        bad_diag = [f for f in features if is_diagnostic_only_feature(f)]
        if bad_diag:
            problems.append(
                f"[content] {set_id}: diagnostic-only feature(s) {bad_diag} "
                f"— keep these for robustness analyses, not in the 17-set grid"
            )
    return problems
