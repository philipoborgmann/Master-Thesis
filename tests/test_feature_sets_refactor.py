"""Tests for the 2026 feature-set family refactor.

Verifies the canonical ``feature_sets.xlsx`` shipped in the repo matches the
new structure:

* Benchmark family (B1 Historical Majority, B2 Single Lag Return, B3–B6
  economic).
* Pure sentiment per scorer (SV1–SV3 vader, SF1–SF3 finbert, SC1–SC3
  cryptobert).
* Combined sets (C1 / C2 / C3 with one sentiment scorer + matched benchmark).
* Multi-source sets (M1 / M2 / M3 covering all three scorers + matched
  benchmark).

The old E1–E4 (economic), S4–S7 (multi-source sentiment) and C4–C6 IDs are
gone; requesting one of them from ``run-models`` raises a clear migration
error instead of silently returning an empty config.
"""
from __future__ import annotations

import pytest

from thesis_pipeline.features import feature_registry as fr


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def sets() -> dict:
    """Load the canonical workbook exactly once for every test."""
    out = fr.load_feature_sets()
    assert out, "feature_sets.xlsx returned an empty registry"
    return out


def _load_all_rows() -> "pd.DataFrame":
    """Return every row from feature_sets.xlsx with the **canonical** header
    normalisation (mirrors ``feature_registry.load_feature_sets_xlsx``) so the
    ``# Features`` count column never wins over the comma-separated list.
    """
    import pandas as pd
    from thesis_pipeline.config import resolve_path
    df = pd.read_excel(resolve_path("feature_sets_xlsx"), sheet_name="feature_sets")
    df.columns = (df.columns.astype(str).str.strip().str.lower()
                  .str.replace(r"[^a-z0-9]+", "_", regex=True).str.strip("_"))
    for preferred in ("feature_columns_comma_separated", "feature_columns"):
        if preferred in df.columns:
            if "features" in df.columns and preferred != "features":
                df = df.drop(columns=["features"])
            df = df.rename(columns={preferred: "features"})
            break
    return df


def _by_set(_sets, set_id: str, sentiment_model: str | None = None) -> dict:
    """Return the spec row for ``(set_id, sentiment_model)`` from the XLSX."""
    df = _load_all_rows()
    mask = df["set_id"].astype(str) == set_id
    if sentiment_model is not None:
        mask &= df["sentiment_model"].astype(str) == sentiment_model
    sub = df[mask]
    assert not sub.empty, f"no row for set_id={set_id}, sentiment_model={sentiment_model}"
    row = sub.iloc[0]
    return {
        "category": str(row["category"]),
        "sentiment_model": str(row["sentiment_model"]),
        "features": [f.strip() for f in str(row["features"]).split(",") if f.strip()],
    }


# ---------------------------------------------------------------------------
# Spec items 1, 2 — no E* sets, all B* present
# ---------------------------------------------------------------------------

def test_no_legacy_economic_set_ids_in_xlsx(sets):
    for legacy in ("E1", "E2", "E3", "E4"):
        assert legacy not in sets, f"legacy {legacy} still in feature_sets.xlsx"


def test_no_legacy_multi_or_combined_set_ids(sets):
    for legacy in ("S4", "S5", "S6", "S7", "C4", "C5", "C6"):
        assert legacy not in sets, f"legacy {legacy} still in feature_sets.xlsx"


def test_full_benchmark_family_loads(sets):
    expected = {
        "B1": ["__majority_class__"],
        "B2": ["log_return_t"],
        "B3": ["log_return_t", "cum_log_return_7", "cum_log_return_14"],
        "B4": ["log_return_t", "cum_log_return_7", "cum_log_return_14",
                "realized_vol_14"],
        "B5": ["log_return_t", "cum_log_return_7", "cum_log_return_14",
                "volume_diff"],
        "B6": ["log_return_t", "cum_log_return_7", "cum_log_return_14",
                "realized_vol_14", "volume_diff", "market_cap_t"],
    }
    for set_id, feats in expected.items():
        assert set_id in sets, f"missing benchmark {set_id}"
        loaded = sets[set_id]["features"]
        assert loaded == feats, f"{set_id} features {loaded} != expected {feats}"


# ---------------------------------------------------------------------------
# Spec item 3 — pure sentiment per scorer loads
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("set_id,sentiment_model,expected", [
    ("SV1", "vader", ["vader_title_score_mean"]),
    ("SV2", "vader", ["vader_title_score_mean", "vader_bullishness_ratio"]),
    ("SV3", "vader", ["vader_title_score_mean", "vader_title_score_weighted_mean",
                       "vader_bullishness_ratio", "vader_title_score_std",
                       "post_count"]),
    ("SF1", "finbert", ["finbert_title_score_mean"]),
    ("SF2", "finbert", ["finbert_title_score_mean", "finbert_bullishness_ratio"]),
    ("SF3", "finbert", ["finbert_title_score_mean", "finbert_title_score_weighted_mean",
                         "finbert_bullishness_ratio", "finbert_title_score_std",
                         "post_count"]),
    ("SC1", "cryptobert", ["cryptobert_title_score_mean"]),
    ("SC2", "cryptobert", ["cryptobert_title_score_mean",
                            "cryptobert_bullishness_ratio"]),
    ("SC3", "cryptobert", ["cryptobert_title_score_mean",
                            "cryptobert_title_score_weighted_mean",
                            "cryptobert_bullishness_ratio",
                            "cryptobert_title_score_std", "post_count"]),
])
def test_pure_sentiment_family(set_id, sentiment_model, expected):
    spec = _by_set(None, set_id, sentiment_model)
    assert spec["sentiment_model"] == sentiment_model
    assert spec["category"] == f"sentiment_{sentiment_model}"
    assert spec["features"] == expected


# ---------------------------------------------------------------------------
# Spec item 4 — combined sets are the exact union of benchmark + sentiment
# ---------------------------------------------------------------------------

# Benchmark→sentiment-level pairing (preserves the historical thesis design).
COMBINED_BENCHMARK = {1: "B4", 2: "B6", 3: "B6"}


def _benchmark_features(set_id: str) -> list[str]:
    return _by_set(None, set_id, "-")["features"]


def _sentiment_features(level: int, scorer: str) -> list[str]:
    prefix = {"vader": "SV", "finbert": "SF", "cryptobert": "SC"}[scorer]
    return _by_set(None, f"{prefix}{level}", scorer)["features"]


@pytest.mark.parametrize("level,scorer", [
    (1, "vader"), (1, "finbert"), (1, "cryptobert"),
    (2, "vader"), (2, "finbert"), (2, "cryptobert"),
    (3, "vader"), (3, "finbert"), (3, "cryptobert"),
])
def test_combined_is_union_of_components(level, scorer):
    combined = _by_set(None, f"C{level}", scorer)["features"]
    bench = _benchmark_features(COMBINED_BENCHMARK[level])
    sent  = _sentiment_features(level, scorer)
    expected_union = list(dict.fromkeys([*sent, *bench]))  # de-dup, order-preserving
    assert combined == expected_union


# ---------------------------------------------------------------------------
# Spec item 5 — multi sets contain benchmark + all three sentiment sources
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("level", [1, 2, 3])
def test_multi_includes_all_three_sentiment_sources(level):
    multi = _by_set(None, f"M{level}", "multi")
    assert multi["category"] == "multi"
    feats = multi["features"]
    # Three scorers each contribute their sentiment columns.
    for scorer in ("vader", "finbert", "cryptobert"):
        scorer_feats = _sentiment_features(level, scorer)
        for feat in scorer_feats:
            assert feat in feats, f"M{level} missing {feat}"
    # Plus the matched benchmark backbone.
    for feat in _benchmark_features(COMBINED_BENCHMARK[level]):
        assert feat in feats, f"M{level} missing benchmark column {feat}"


# ---------------------------------------------------------------------------
# Spec item 6 — no duplicate features inside any set
# ---------------------------------------------------------------------------

def test_no_duplicate_features_anywhere(sets):
    for set_id, spec in sets.items():
        feats = spec["features"]
        assert len(feats) == len(set(feats)), (
            f"{set_id} has duplicates: {feats}"
        )


def test_registry_validator_reports_no_problems(sets):
    assert fr.validate_registry(sets) == []


# ---------------------------------------------------------------------------
# Spec item 7 — CLI accepts every new ID
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("set_id", [
    "B1", "B2", "B3", "B4", "B5", "B6",
    "SV1", "SV2", "SV3", "SF1", "SF2", "SF3", "SC1", "SC2", "SC3",
    "C1", "C2", "C3",
    "M1", "M2", "M3",
])
def test_cli_dry_run_accepts_new_set_ids(set_id):
    """``run-models --dry-run`` must reach the planning log for every set_id —
    no module crashes from unknown identifiers.
    """
    from thesis_pipeline import cli
    rc = cli.main(["run-models", "--horizon", "1d", "--set-id", set_id,
                   "--dry-run"])
    assert rc == 0


# ---------------------------------------------------------------------------
# Spec item 8 — old E* IDs fail with a controlled error
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("legacy,replacement_substr", [
    ("E1", "B3"),
    ("E2", "B4"),
    ("E3", "B5"),
    ("E4", "B6"),
    ("S4", "M1"),
    ("S5", "M1"),
    ("S6", "M2"),
    ("S7", "M3"),
    ("C4", "M1"),
    ("C5", "M1"),
    ("C6", "M3"),
])
def test_legacy_set_ids_raise_clear_migration_error(legacy, replacement_substr):
    from thesis_pipeline import cli
    with pytest.raises(SystemExit) as excinfo:
        cli.main(["run-models", "--horizon", "1d", "--set-id", legacy,
                  "--dry-run"])
    msg = str(excinfo.value)
    assert legacy in msg
    assert "removed" in msg
    assert replacement_substr in msg
    assert "refactor_log" in msg

