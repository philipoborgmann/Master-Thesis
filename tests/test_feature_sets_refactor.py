"""Tests for the FinBERT-removal + S*/SV*/C*/CV*/M* family refactor.

Verifies the canonical ``feature_sets.xlsx`` shipped in the repo matches the
final structure:

* Benchmark family `B1`–`B6` (Historical Majority, Single Lag Return,
  Momentum, Momentum + Volatility, Momentum + Volume, Full Economic).
* Pure sentiment families `S1`–`S3` (CryptoBERT) and `SV1`–`SV3` (VADER).
* Combined families `C1`–`C6` (Benchmark + CryptoBERT) and `CV1`–`CV6`
  (Benchmark + VADER) following the hierarchical mapping
  C_k = B_k + S_min(k,3) (k=1,2,3 use sentiment level k; k=4,5,6 use the rich
  level 3 paired with richer benchmarks).
* Multi-source `M1`–`M6` (Benchmark + CryptoBERT + VADER) with the same
  benchmark / sentiment-level pairing.

The old `E*`, `SC*`, `SF*` IDs and FinBERT scoring are gone; requesting any of
them from `run-models` / `score-sentiment` raises a controlled migration
error.
"""
from __future__ import annotations

import pytest

from thesis_pipeline.features import feature_registry as fr


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_all_rows():
    """Return every row from feature_sets.xlsx with the **canonical** header
    normalisation (mirrors ``feature_registry.load_feature_sets_xlsx``)."""
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


def _by_set(set_id: str, sentiment_model: str | None = None) -> dict:
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


@pytest.fixture(scope="module")
def sets() -> dict:
    out = fr.load_feature_sets()
    assert out
    return out


# ---------------------------------------------------------------------------
# Spec items 1, 2, 3 — no FinBERT, no SF*, no SC*, no finbert_* features
# ---------------------------------------------------------------------------

def test_no_finbert_set_ids(sets):
    for legacy in ("SF1", "SF2", "SF3"):
        assert legacy not in sets, f"FinBERT set {legacy!r} still in workbook"


def test_no_legacy_sc_set_ids(sets):
    for legacy in ("SC1", "SC2", "SC3"):
        assert legacy not in sets, (
            f"legacy CryptoBERT set {legacy!r} still in workbook — "
            f"use S* instead"
        )


def test_no_legacy_economic_set_ids(sets):
    for legacy in ("E1", "E2", "E3", "E4"):
        assert legacy not in sets


def test_no_finbert_features_in_any_set(sets):
    for set_id, spec in sets.items():
        bad = [f for f in spec["features"] if "finbert" in f.lower()]
        assert not bad, f"{set_id} still uses FinBERT columns: {bad}"


def test_no_finbert_sentiment_model_value():
    df = _load_all_rows()
    sm = df["sentiment_model"].astype(str).str.lower().unique().tolist()
    assert "finbert" not in sm, f"sentiment_model column still mentions finbert: {sm}"


# ---------------------------------------------------------------------------
# Spec items 4, 5 — S* means CryptoBERT, SV* means VADER
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("set_id,expected_feats", [
    ("S1", ["cryptobert_title_score_mean"]),
    ("S2", ["cryptobert_title_score_mean", "cryptobert_bullishness_ratio"]),
    ("S3", ["cryptobert_title_score_mean", "cryptobert_title_score_weighted_mean",
             "cryptobert_bullishness_ratio", "cryptobert_title_score_std",
             "post_count"]),
])
def test_s_family_is_cryptobert(set_id, expected_feats):
    spec = _by_set(set_id, "cryptobert")
    assert spec["sentiment_model"] == "cryptobert"
    assert spec["category"] == "sentiment_cryptobert"
    assert spec["features"] == expected_feats


@pytest.mark.parametrize("set_id,expected_feats", [
    ("SV1", ["vader_title_score_mean"]),
    ("SV2", ["vader_title_score_mean", "vader_bullishness_ratio"]),
    ("SV3", ["vader_title_score_mean", "vader_title_score_weighted_mean",
              "vader_bullishness_ratio", "vader_title_score_std",
              "post_count"]),
])
def test_sv_family_is_vader(set_id, expected_feats):
    spec = _by_set(set_id, "vader")
    assert spec["sentiment_model"] == "vader"
    assert spec["category"] == "sentiment_vader"
    assert spec["features"] == expected_feats


# ---------------------------------------------------------------------------
# Spec items 6, 7, 8 — C* / CV* / M* family identities
# ---------------------------------------------------------------------------

def _benchmark_real_features(bench_id: str) -> list[str]:
    spec = _by_set(bench_id, "-")
    if spec["features"] == ["__majority_class__"]:
        return []
    return spec["features"]


# Hierarchical pairing per spec: C_k pairs B_k with sentiment level
# min(k, 3) (k=1→1, k=2→2, k≥3→3).
PAIRING = {1: 1, 2: 2, 3: 3, 4: 3, 5: 3, 6: 3}


def _sentiment_features(level: int, scorer: str) -> list[str]:
    prefix = {"vader": "SV", "cryptobert": "S"}[scorer]
    return _by_set(f"{prefix}{level}", scorer)["features"]


@pytest.mark.parametrize("k", [1, 2, 3, 4, 5, 6])
def test_c_family_is_benchmark_plus_cryptobert(k):
    combined = _by_set(f"C{k}", "cryptobert")
    assert combined["sentiment_model"] == "cryptobert"
    assert combined["category"] == "combined_cryptobert"
    bench = _benchmark_real_features(f"B{k}")
    sent  = _sentiment_features(PAIRING[k], "cryptobert")
    expected = list(dict.fromkeys([*bench, *sent]))
    assert combined["features"] == expected


@pytest.mark.parametrize("k", [1, 2, 3, 4, 5, 6])
def test_cv_family_is_benchmark_plus_vader(k):
    combined = _by_set(f"CV{k}", "vader")
    assert combined["sentiment_model"] == "vader"
    assert combined["category"] == "combined_vader"
    bench = _benchmark_real_features(f"B{k}")
    sent  = _sentiment_features(PAIRING[k], "vader")
    expected = list(dict.fromkeys([*bench, *sent]))
    assert combined["features"] == expected


@pytest.mark.parametrize("k", [1, 2, 3, 4, 5, 6])
def test_m_family_is_benchmark_plus_cryptobert_plus_vader(k):
    multi = _by_set(f"M{k}", "multi")
    assert multi["sentiment_model"] == "multi"
    assert multi["category"] == "multi"
    bench = _benchmark_real_features(f"B{k}")
    s    = _sentiment_features(PAIRING[k], "cryptobert")
    sv   = _sentiment_features(PAIRING[k], "vader")
    expected = list(dict.fromkeys([*bench, *s, *sv]))
    assert multi["features"] == expected


# ---------------------------------------------------------------------------
# Spec items 9-11 — the spec-given canonical examples
# ---------------------------------------------------------------------------

def test_c1_equals_b1_plus_s1():
    c1 = _by_set("C1", "cryptobert")["features"]
    s1 = _by_set("S1", "cryptobert")["features"]
    # B1's sentinel is dropped because it is a routing marker, not a feature.
    assert c1 == s1


def test_cv1_equals_b1_plus_sv1():
    cv1 = _by_set("CV1", "vader")["features"]
    sv1 = _by_set("SV1", "vader")["features"]
    assert cv1 == sv1


def test_m1_equals_b1_plus_s1_plus_sv1():
    m1 = _by_set("M1", "multi")["features"]
    s1 = _by_set("S1", "cryptobert")["features"]
    sv1 = _by_set("SV1", "vader")["features"]
    assert m1 == list(dict.fromkeys([*s1, *sv1]))


# ---------------------------------------------------------------------------
# Spec item 12 — no duplicate features inside any set
# ---------------------------------------------------------------------------

def test_no_duplicate_features_anywhere(sets):
    for set_id, spec in sets.items():
        feats = spec["features"]
        assert len(feats) == len(set(feats)), f"{set_id} has duplicates: {feats}"


def test_registry_validator_reports_no_problems(sets):
    assert fr.validate_registry(sets) == []


# ---------------------------------------------------------------------------
# Spec item 13 — CLI accepts every new ID
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("set_id", [
    "B1", "B2", "B3", "B4", "B5", "B6",
    "S1", "S2", "S3", "SV1", "SV2", "SV3",
    "C1", "C2", "C3", "C4", "C5", "C6",
    "CV1", "CV2", "CV3", "CV4", "CV5", "CV6",
    "M1", "M2", "M3", "M4", "M5", "M6",
])
def test_cli_dry_run_accepts_new_set_ids(set_id):
    from thesis_pipeline import cli
    rc = cli.main(["run-models", "--horizon", "1d", "--set-id", set_id,
                   "--dry-run"])
    assert rc == 0


# ---------------------------------------------------------------------------
# Spec item 14 — every removed ID raises a clear migration error
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("legacy,replacement_substr", [
    ("E1", "B3"),
    ("E2", "B4"),
    ("E3", "B5"),
    ("E4", "B6"),
    ("SC1", "S1"),
    ("SC2", "S2"),
    ("SC3", "S3"),
    ("S4", "M1"),
    ("S5", "M1"),
    ("S6", "M2"),
    ("S7", "M3"),
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


@pytest.mark.parametrize("legacy", ["SF1", "SF2", "SF3"])
def test_finbert_set_ids_raise_with_finbert_message(legacy):
    from thesis_pipeline import cli
    with pytest.raises(SystemExit) as excinfo:
        cli.main(["run-models", "--horizon", "1d", "--set-id", legacy,
                  "--dry-run"])
    msg = str(excinfo.value)
    assert legacy in msg
    assert "FinBERT" in msg


# ---------------------------------------------------------------------------
# FinBERT sentiment-model rejection
# ---------------------------------------------------------------------------

def test_sentiment_model_finbert_raises_friendly_error(capsys):
    """``--sentiment-model finbert`` must raise an informative error mentioning
    why FinBERT was removed.
    """
    from thesis_pipeline import cli
    with pytest.raises(SystemExit):
        cli.main(["run-models", "--horizon", "1d", "--set-id", "S1",
                  "--sentiment-model", "finbert", "--dry-run"])
    msg = capsys.readouterr().err
    assert "FinBERT" in msg
    assert "Reddit" in msg


def test_score_sentiment_finbert_raises(capsys):
    from thesis_pipeline import cli
    with pytest.raises(SystemExit):
        cli.main(["score-sentiment", "--model", "finbert"])
    assert "FinBERT" in capsys.readouterr().err
