"""Feature-set registry.

Reads ``feature_sets.xlsx`` (the current source of truth) and, when
``configs/feature_sets.yaml`` has been verified, optionally the YAML mirror.
The behavior contract is: every ``set_id`` declared in either source maps
to a non-empty list of feature column names (B1, the rolling-probability
benchmark, is allowed to have an empty feature list).
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

from ..config import load_config, resolve_path


SET_ID_PATTERN = (
    "B1", "B2", "B3", "B4", "B5", "B6",
    "SV1", "SV2", "SV3",
    "SF1", "SF2", "SF3",
    "SC1", "SC2", "SC3",
    "C1", "C2", "C3",
    "M1", "M2", "M3",
)

# Set-IDs removed during the 2026 family-structure refactor. Looking these up
# raises a clear migration error rather than silently returning an empty
# config (see CLI guard in :mod:`thesis_pipeline.modeling.run_models`).
REMOVED_SET_IDS: dict[str, str] = {
    "E1": "B3",   # Momentum
    "E2": "B4",   # Momentum + Volatility
    "E3": "B5",   # Momentum + Volume
    "E4": "B6",   # Full Economic
    "S1": "SV1/SF1/SC1",
    "S2": "SV2/SF2/SC2",
    "S3": "SV3/SF3/SC3",
    "S4": "M1",
    "S5": "M1",
    "S6": "M2",
    "S7": "M3",
    "C4": "M1",
    "C5": "M1",
    "C6": "M3",
}


def load_feature_sets_yaml() -> Dict[str, dict]:
    cfg = load_config("feature_sets")
    return dict(cfg.get("sets", {}))


def load_feature_sets_xlsx() -> Dict[str, dict] | None:
    """Load ``feature_sets.xlsx`` if available. Returns ``None`` otherwise.

    Header handling is intentionally kept in lock-step with
    :func:`thesis_pipeline.modeling.run_models.load_feature_sets` so the
    registry and the modelling loader can never disagree on which column holds
    the comma-separated feature list:

      1. Every header is normalised to snake_case by stripping, lower-casing
         and replacing any run of non-alphanumerics with a single ``_``. After
         this pass ``"Set ID"`` → ``"set_id"``,
         ``"Feature Columns (comma-separated)"`` → ``"feature_columns_comma_separated"``
         and ``"# Features"`` → ``"features"``.
      2. The feature-list column is chosen by explicit priority:
         ``feature_columns_comma_separated`` ≻ ``feature_columns`` ≻
         ``features``. When a higher-priority header is present, any
         pre-existing ``features`` column (e.g. the integer ``# Features``
         count that just normalised to that name) is dropped first so the
         rename actually wins.
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

    # Normalise display headers to snake_case (matches run_models.load_feature_sets).
    df.columns = (df.columns.astype(str)
                  .str.strip()
                  .str.lower()
                  .str.replace(r"[^a-z0-9]+", "_", regex=True)
                  .str.strip("_"))

    # Pick the feature-list column by priority and drop the loser ``features``.
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
    """Return a list of validation problems (empty list → registry is OK)."""
    problems: list[str] = []
    seen_per_set: dict[str, set] = {}
    for set_id, spec in sets.items():
        features = list(spec.get("features", []))
        # Allow B1 (rolling-probability benchmark) to have empty features.
        if set_id != "B1" and not features:
            problems.append(f"{set_id}: empty feature list")
        if len(features) != len(set(features)):
            problems.append(f"{set_id}: duplicate features {features}")
        seen_per_set[set_id] = set(features)
    return problems
