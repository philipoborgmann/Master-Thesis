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
    "B1", "B2",
    "E1", "E2", "E3", "E4",
    "S1", "S2", "S3", "S4", "S5", "S6", "S7",
    "C1", "C2", "C3", "C4", "C5",
)


def load_feature_sets_yaml() -> Dict[str, dict]:
    cfg = load_config("feature_sets")
    return dict(cfg.get("sets", {}))


def load_feature_sets_xlsx() -> Dict[str, dict] | None:
    """Load ``feature_sets.xlsx`` if available. Returns None otherwise.

    Mirrors the loader in Run_Models.py (normalises header variants:
    ``feature_columns`` ↔ ``features``).
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
    # Header normalisation matching Run_Models.py
    df.columns = [str(c).strip().lower() for c in df.columns]
    if "features" not in df.columns and "feature_columns" in df.columns:
        df = df.rename(columns={"feature_columns": "features"})
    sets: Dict[str, dict] = {}
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
                if "sentiment_model" in df.columns else None,
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
