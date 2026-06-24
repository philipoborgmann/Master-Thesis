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

# Diagnostic-only columns. Never expose them as features in the primary grid.
DIAGNOSTIC_ONLY_COLUMNS = (
    "has_posts",
    "vader_directional_post_count",
    "cryptobert_directional_post_count",
    "vader_combined_score_mean",      # robustness / supplementary only
    "vader_selftext_score_mean",      # robustness / supplementary only
    "cryptobert_combined_score_mean", # robustness / supplementary only
    "cryptobert_selftext_score_mean", # robustness / supplementary only
)


# Set-IDs removed during the v4 family-structure refactor + FinBERT removal.
# Looking these up raises a clear migration error rather than silently
# returning an empty config (see CLI guard in
# :mod:`thesis_pipeline.modeling.run_models`).
_FINBERT_REMOVED = (
    "removed (FinBERT was dropped; trained on financial / analyst "
    "language, no meaningful variation on Reddit/crypto data)"
)
_V4_REMOVED = (
    "removed in v4 — the 17-set registry replaces the old B/E/S/SV/C/CV/M "
    "families. See SET_ID_PATTERN; e.g. previous B6/E4 → ECON, SV3 → "
    "SENT_VAD_F, CV3 → ECON_VAD_F, S3 → SENT_CBT_F."
)
REMOVED_SET_IDS: dict[str, str] = {
    # Old benchmark family — folded into ECON.
    "B1": _V4_REMOVED, "B2": _V4_REMOVED, "B3": _V4_REMOVED,
    "B4": _V4_REMOVED, "B5": _V4_REMOVED, "B6": _V4_REMOVED,
    # Old economic family — folded into ECON.
    "E1": _V4_REMOVED, "E2": _V4_REMOVED, "E3": _V4_REMOVED, "E4": _V4_REMOVED,
    # Old sentiment-only family (CryptoBERT in S*, VADER in SV*).
    "S1":  "SENT_CBT_L",  "S2":  "SENT_CBT_LD", "S3":  "SENT_CBT_F",
    "S4":  _V4_REMOVED,   "S5":  _V4_REMOVED,   "S6":  _V4_REMOVED, "S7": _V4_REMOVED,
    "SV1": "SENT_VAD_L",  "SV2": "SENT_VAD_LD", "SV3": "SENT_VAD_F",
    # Old combined families (C* = CryptoBERT, CV* = VADER) — superseded by
    # ECON_CBT_* / ECON_VAD_*.
    "C1":  "ECON_CBT_L",  "C2":  "ECON_CBT_LD", "C3":  "ECON_CBT_F",
    "C4":  _V4_REMOVED,   "C5":  _V4_REMOVED,   "C6":  _V4_REMOVED,
    "CV1": "ECON_VAD_L",  "CV2": "ECON_VAD_LD", "CV3": "ECON_VAD_F",
    "CV4": _V4_REMOVED,   "CV5": _V4_REMOVED,   "CV6": _V4_REMOVED,
    # Old multi-source / M family — Variante A does not run dual-scorer
    # combined sets; mixed-source M* is dropped.
    "M1":  _V4_REMOVED, "M2": _V4_REMOVED, "M3": _V4_REMOVED,
    "M4":  _V4_REMOVED, "M5": _V4_REMOVED, "M6": _V4_REMOVED,
    # FinBERT family — dropped entirely.
    "SC1": "SENT_CBT_L", "SC2": "SENT_CBT_LD", "SC3": "SENT_CBT_F",
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

    v4 contract:
      * ID must be in :data:`SET_ID_PATTERN`.
      * feature list must be non-empty.
      * features must be unique.
      * no feature may be one of the diagnostic-only columns.
    """
    problems: list[str] = []
    declared = set(SET_ID_PATTERN)
    diagnostics = set(DIAGNOSTIC_ONLY_COLUMNS)
    for set_id, spec in sets.items():
        features = list(spec.get("features", []))
        if set_id not in declared:
            problems.append(f"{set_id}: not in SET_ID_PATTERN (v4 expects 17 sets)")
        if not features:
            problems.append(f"{set_id}: empty feature list (no v4 set may be empty)")
        if len(features) != len(set(features)):
            problems.append(f"{set_id}: duplicate features {features}")
        bad_diag = [f for f in features if f in diagnostics]
        if bad_diag:
            problems.append(
                f"{set_id}: includes diagnostic-only columns {bad_diag} — "
                f"keep these for robustness analyses, not in the 17-set grid"
            )
    return problems
