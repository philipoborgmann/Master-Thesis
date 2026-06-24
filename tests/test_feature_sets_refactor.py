"""Tests for the v4 17-set feature registry (Variante A).

Supersedes the v3 family-layout tests. Verifies:

* the canonical `feature_sets.xlsx` matches the 17-set v4 structure;
* every removed v3 ID (`B*`, `E*`, `S*`, `SV*`, `C*`, `CV*`, `M*`,
  `SC*`, `SF*`) raises a clear migration error mentioning the v4 successor
  (or "no exact equivalent" where the information set genuinely differs);
* every active v4 ID is accepted by `run-models --dry-run`.
"""
from __future__ import annotations

import pytest

from thesis_pipeline.features import feature_registry as fr


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_all_rows():
    """Return every row from feature_sets.xlsx with the canonical header
    normalisation (mirrors :func:`feature_registry.load_feature_sets_xlsx`)."""
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


def _by_set(set_id: str) -> dict:
    df = _load_all_rows()
    sub = df[df["set_id"].astype(str) == set_id]
    assert not sub.empty, f"no row for set_id={set_id} in feature_sets.xlsx"
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
# Shape: exactly 17 v4 sets, no legacy IDs
# ---------------------------------------------------------------------------

def test_registry_contains_exactly_the_v4_seventeen_sets(sets):
    assert set(sets.keys()) == set(fr.SET_ID_PATTERN)
    assert len(sets) == 17


@pytest.mark.parametrize("legacy", [
    "B1", "B2", "B3", "B4", "B5", "B6",
    "E1", "E2", "E3", "E4",
    "S1", "S2", "S3", "SV1", "SV2", "SV3",
    "C1", "C2", "C3", "C4", "C5", "C6",
    "CV1", "CV2", "CV3", "CV4", "CV5", "CV6",
    "M1", "M2", "M3", "M4", "M5", "M6",
    "SC1", "SC2", "SC3", "SF1", "SF2", "SF3",
])
def test_no_legacy_set_ids_in_workbook(sets, legacy):
    assert legacy not in sets


def test_no_finbert_features_in_any_set(sets):
    for set_id, spec in sets.items():
        bad = [f for f in spec["features"] if "finbert" in f.lower()]
        assert not bad, f"{set_id} still uses FinBERT columns: {bad}"


def test_no_finbert_sentiment_model_value():
    df = _load_all_rows()
    sm = df["sentiment_model"].astype(str).str.lower().unique().tolist()
    assert "finbert" not in sm


def test_no_engagement_weighted_features_anywhere(sets):
    """Variante A: no `_weighted_mean` columns may appear in the registry."""
    for set_id, spec in sets.items():
        bad = [f for f in spec["features"] if f.endswith("_weighted_mean")]
        assert not bad, f"{set_id} still uses engagement-weighted column: {bad}"


# ---------------------------------------------------------------------------
# v4 spec — block definitions and ECON-core constancy
# ---------------------------------------------------------------------------

ECON_CORE = [
    "log_return_t",
    "cum_log_return_7d", "cum_log_return_14d", "cum_log_return_21d",
    "realized_vol_14d", "volume_diff", "log_market_cap_lag1",
]


def test_econ_set_is_econ_core():
    assert _by_set("ECON")["features"] == ECON_CORE


@pytest.mark.parametrize("scorer,prefix", [("vader", "SENT_VAD"), ("cryptobert", "SENT_CBT")])
def test_sentiment_only_blocks(scorer, prefix):
    L  = _by_set(f"{prefix}_L")["features"]
    LD = _by_set(f"{prefix}_LD")["features"]
    DA = _by_set(f"{prefix}_DA")["features"]
    F  = _by_set(f"{prefix}_F")["features"]

    assert L  == [f"{scorer}_title_score_mean"]
    assert LD == [f"{scorer}_title_score_mean", f"{scorer}_title_score_std"]
    assert DA == [f"{scorer}_bullishness_ratio", "log1p_post_count"]
    assert F  == [f"{scorer}_title_score_mean", f"{scorer}_title_score_std",
                   f"{scorer}_bullishness_ratio", "log1p_post_count"]


@pytest.mark.parametrize("scorer,prefix", [("vader", "ECON_VAD"), ("cryptobert", "ECON_CBT")])
def test_combined_sets_equal_econ_core_plus_block(scorer, prefix):
    L  = _by_set(f"{prefix}_L")["features"]
    LD = _by_set(f"{prefix}_LD")["features"]
    DA = _by_set(f"{prefix}_DA")["features"]
    F  = _by_set(f"{prefix}_F")["features"]

    sent_prefix = "SENT_VAD" if scorer == "vader" else "SENT_CBT"
    assert L  == ECON_CORE + _by_set(f"{sent_prefix}_L")["features"]
    assert LD == ECON_CORE + _by_set(f"{sent_prefix}_LD")["features"]
    assert DA == ECON_CORE + _by_set(f"{sent_prefix}_DA")["features"]
    assert F  == ECON_CORE + _by_set(f"{sent_prefix}_F")["features"]


# ---------------------------------------------------------------------------
# Validation invariants
# ---------------------------------------------------------------------------

def test_no_duplicate_features_anywhere(sets):
    for set_id, spec in sets.items():
        feats = spec["features"]
        assert len(feats) == len(set(feats)), f"{set_id} has duplicates: {feats}"


def test_registry_validator_reports_no_problems(sets):
    assert fr.validate_registry(sets) == []


# ---------------------------------------------------------------------------
# CLI accepts every v4 ID
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("set_id", list(fr.SET_ID_PATTERN))
def test_cli_dry_run_accepts_each_v4_set_id(set_id):
    from thesis_pipeline import cli
    rc = cli.main(["run-models", "--horizon", "1d", "--set-id", set_id,
                   "--dry-run"])
    assert rc == 0


# ---------------------------------------------------------------------------
# CLI rejects every removed v3 ID — strictly, regardless of any custom xlsx
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("legacy", list(fr.REMOVED_SET_IDS.keys()))
def test_removed_ids_always_raise(legacy):
    from thesis_pipeline import cli
    with pytest.raises(SystemExit) as excinfo:
        cli.main(["run-models", "--horizon", "1d", "--set-id", legacy,
                  "--dry-run"])
    msg = str(excinfo.value)
    assert legacy in msg
    assert "removed" in msg.lower() or "v4" in msg.lower() or "NAIVE" in msg


def test_removed_id_cannot_be_revived_by_custom_xlsx(tmp_path):
    """Even when a user-supplied feature_sets.xlsx defines the legacy ID,
    the v4 guard must still refuse it. Otherwise a stale fixture could
    silently revive a deprecated information set."""
    import pandas as pd
    fs_path = tmp_path / "legacy_xlsx_with_b1.xlsx"
    pd.DataFrame({
        "set_id":   ["B1"],
        "category": ["benchmark"],
        "sentiment_model": ["-"],
        "label":    ["legacy"],
        "feature_columns_comma_separated": ["__majority_class__"],
    }).to_excel(fs_path, sheet_name="feature_sets", index=False)

    from thesis_pipeline import cli
    with pytest.raises(SystemExit) as excinfo:
        cli.main(["run-models", "--horizon", "1d", "--set-id", "B1",
                  "--feature-config", str(fs_path), "--dry-run"])
    assert "B1" in str(excinfo.value)


# ---------------------------------------------------------------------------
# Migration messages — distinguish exact-equivalent from no-exact-equivalent
# ---------------------------------------------------------------------------

def test_b1_migration_message_points_to_naive_not_a_set():
    """B1 was the rolling-probability benchmark. It is NOT a feature set in
    v4; the message must explain that it lives in the modeling.benchmarks
    module as the NAIVE evaluation reference."""
    msg = fr.REMOVED_SET_IDS["B1"]
    assert "NAIVE" in msg
    assert "run_rolling_probability" in msg


@pytest.mark.parametrize("legacy", ["B2", "B3", "B4", "B5", "B6",
                                     "E1", "E2", "E3", "E4"])
def test_economic_only_migration_messages_point_to_econ(legacy):
    """B2..B6 / E1..E4 were variants of the economic-only baseline. v4
    consolidates them into ECON (with documented additions)."""
    msg = fr.REMOVED_SET_IDS[legacy]
    assert "ECON" in msg


@pytest.mark.parametrize("legacy", ["S1", "S2", "S3"])
def test_cbt_sentiment_no_exact_equivalent_messages(legacy):
    msg = fr.REMOVED_SET_IDS[legacy]
    assert "no exact" in msg.lower() or "No exact" in msg
    # Points the user at the four sentiment-only CBT successors.
    for cand in ("SENT_CBT_L", "SENT_CBT_LD", "SENT_CBT_DA", "SENT_CBT_F"):
        assert cand in msg


@pytest.mark.parametrize("legacy", ["SV1", "SV2", "SV3"])
def test_vad_sentiment_no_exact_equivalent_messages(legacy):
    msg = fr.REMOVED_SET_IDS[legacy]
    assert "no exact" in msg.lower() or "No exact" in msg
    for cand in ("SENT_VAD_L", "SENT_VAD_LD", "SENT_VAD_DA", "SENT_VAD_F"):
        assert cand in msg


@pytest.mark.parametrize("legacy", ["C1", "C2", "C3", "C4", "C5", "C6"])
def test_cbt_combined_no_exact_equivalent_messages(legacy):
    msg = fr.REMOVED_SET_IDS[legacy]
    assert "no exact" in msg.lower() or "No exact" in msg
    for cand in ("ECON_CBT_L", "ECON_CBT_LD", "ECON_CBT_DA", "ECON_CBT_F"):
        assert cand in msg


@pytest.mark.parametrize("legacy", ["CV1", "CV2", "CV3", "CV4", "CV5", "CV6"])
def test_vad_combined_no_exact_equivalent_messages(legacy):
    msg = fr.REMOVED_SET_IDS[legacy]
    assert "no exact" in msg.lower() or "No exact" in msg
    for cand in ("ECON_VAD_L", "ECON_VAD_LD", "ECON_VAD_DA", "ECON_VAD_F"):
        assert cand in msg


@pytest.mark.parametrize("legacy", ["M1", "M2", "M3", "M4", "M5", "M6"])
def test_m_family_removed_messages(legacy):
    msg = fr.REMOVED_SET_IDS[legacy]
    assert "removed" in msg.lower()
    # No exact v4 successor — points at both single-scorer families.
    for cand in ("ECON_VAD_F", "ECON_CBT_F"):
        assert cand in msg


@pytest.mark.parametrize("legacy", ["SC1", "SC2", "SC3", "SF1", "SF2", "SF3"])
def test_finbert_era_messages(legacy):
    msg = fr.REMOVED_SET_IDS[legacy]
    assert "FinBERT" in msg


@pytest.mark.parametrize("legacy,expected_substr", [
    # Removed-id error reaches the SystemExit message with the migration text.
    ("B1", "NAIVE"),
    ("E4", "ECON"),
    ("S3", "SENT_CBT"),
    ("SV3", "SENT_VAD"),
    ("C3", "ECON_CBT"),
    ("CV3", "ECON_VAD"),
    ("M3", "ECON_VAD_F"),
    ("SF1", "FinBERT"),
])
def test_removed_id_error_carries_migration_substr(legacy, expected_substr):
    from thesis_pipeline import cli
    with pytest.raises(SystemExit) as excinfo:
        cli.main(["run-models", "--horizon", "1d", "--set-id", legacy,
                  "--dry-run"])
    assert expected_substr in str(excinfo.value)


# ---------------------------------------------------------------------------
# FinBERT sentiment-model rejection at the score-sentiment level still works
# ---------------------------------------------------------------------------

def test_score_sentiment_finbert_raises(capsys):
    from thesis_pipeline import cli
    with pytest.raises(SystemExit):
        cli.main(["score-sentiment", "--model", "finbert"])
    assert "FinBERT" in capsys.readouterr().err
