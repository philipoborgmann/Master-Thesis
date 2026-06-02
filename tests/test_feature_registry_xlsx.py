"""Tests for the feature-set registry XLSX loader.

The registry must stay in lock-step with the modelling loader in
``thesis_pipeline.modeling.run_models.load_feature_sets``: both apply a
snake_case regex normalisation to the headers, and both pick
``feature_columns_comma_separated`` over the integer ``# Features`` count
column when the workbook carries both. These tests pin that contract on the
exact header layout used by the updated ``feature_sets.xlsx`` —

    Set ID | Category | Sentiment Model | Label | # Features
        | Feature Columns (comma-separated) | Description
"""
from __future__ import annotations

import pandas as pd
import pytest

from thesis_pipeline.features import feature_registry as fr


HEADERS = [
    "Set ID",
    "Category",
    "Sentiment Model",
    "Label",
    "# Features",
    "Feature Columns (comma-separated)",
    "Description",
]

ROWS = [
    ["B1", "benchmark", "-",       "rolling probability",       0,
     "__majority_class__",                           "no features"],
    ["E4", "economic",  "-",       "log-return + cum returns",  3,
     "log_return_t,cum_log_return_7,cum_log_return_14",
     "all price features"],
    ["S1", "sentiment", "vader",   "vader combined mean",       1,
     "vader_combined_mean",                          "single sentiment"],
    ["C6", "combined",  "cryptobert", "cryptobert combo",       4,
     "log_return_t,realized_vol_14,cryptobert_combined_mean,cryptobert_post_count",
     "full combined set"],
]


@pytest.fixture
def feature_sets_xlsx(tmp_path, monkeypatch):
    """Write the synthetic workbook to ``tmp_path/feature_sets.xlsx`` and
    point ``resolve_path('feature_sets_xlsx')`` at it.
    """
    path = tmp_path / "feature_sets.xlsx"
    pd.DataFrame(ROWS, columns=HEADERS).to_excel(
        path, sheet_name="feature_sets", index=False
    )
    monkeypatch.setattr(fr, "resolve_path", lambda key: str(path))
    return path


# ---------------------------------------------------------------------------
# Header normalisation + feature-list parsing
# ---------------------------------------------------------------------------

def test_load_feature_sets_xlsx_parses_capitalised_headers(feature_sets_xlsx):
    sets = fr.load_feature_sets_xlsx()
    assert sets is not None
    # All four set_ids are recognised — "Set ID" must normalise to "set_id".
    assert set(sets.keys()) == {"B1", "E4", "S1", "C6"}


def test_b1_has_empty_or_sentinel_features(feature_sets_xlsx):
    sets = fr.load_feature_sets_xlsx()
    # ``__majority_class__`` is the rolling-probability sentinel; the registry
    # keeps it verbatim (the modelling loader rewrites it). The point is that
    # the integer "# Features" column (0) MUST NOT win — that would yield
    # ``features == ["0"]``.
    assert sets["B1"]["features"] == ["__majority_class__"]


def test_e4_parses_full_price_feature_list(feature_sets_xlsx):
    sets = fr.load_feature_sets_xlsx()
    assert sets["E4"]["features"] == [
        "log_return_t", "cum_log_return_7", "cum_log_return_14",
    ]
    assert sets["E4"]["category"] == "economic"


def test_s1_carries_sentiment_model(feature_sets_xlsx):
    sets = fr.load_feature_sets_xlsx()
    assert sets["S1"]["features"] == ["vader_combined_mean"]
    assert sets["S1"]["category"] == "sentiment"
    assert sets["S1"]["sentiment_model"] == "vader"


def test_c6_parses_four_combined_features(feature_sets_xlsx):
    sets = fr.load_feature_sets_xlsx()
    assert sets["C6"]["features"] == [
        "log_return_t",
        "realized_vol_14",
        "cryptobert_combined_mean",
        "cryptobert_post_count",
    ]
    assert sets["C6"]["sentiment_model"] == "cryptobert"


# ---------------------------------------------------------------------------
# # Features must NEVER win over the comma-separated feature list
# ---------------------------------------------------------------------------

def test_hash_features_column_does_not_overwrite_feature_list(feature_sets_xlsx):
    """If the ``# Features`` *count* column were used as the feature list,
    every row would resolve to ``[<digit>]`` (e.g. ``['0']`` for B1 or
    ``['3']`` for E4). Verify that does not happen.
    """
    sets = fr.load_feature_sets_xlsx()
    for set_id, spec in sets.items():
        feats = spec["features"]
        assert not (len(feats) == 1 and feats[0].isdigit()), (
            f"{set_id} resolved to a digit-only feature list {feats!r} — "
            f"the loader is reading the '# Features' count column."
        )


# ---------------------------------------------------------------------------
# Consistency with the modelling loader
# ---------------------------------------------------------------------------

def test_registry_matches_modeling_loader_feature_lists(feature_sets_xlsx):
    """The registry must agree with ``run_models.load_feature_sets`` on
    which column holds the feature list — otherwise the model run and the
    descriptive/stationarity stages would test different feature universes.
    """
    try:
        from thesis_pipeline.modeling.run_models import load_feature_sets as load_modeling
    except ModuleNotFoundError as exc:  # pragma: no cover — sandbox without sklearn
        pytest.skip(f"modeling loader not importable: {exc}")
    modeling_df = load_modeling(str(feature_sets_xlsx))
    by_set = dict(zip(modeling_df["set_id"].astype(str), modeling_df["features"].astype(str)))

    registry = fr.load_feature_sets_xlsx()
    for set_id, spec in registry.items():
        registry_csv = ",".join(spec["features"])
        modeling_csv = by_set[set_id]
        # The modelling loader rewrites the B1 sentinel; normalise both sides.
        modeling_csv = modeling_csv.replace(
            "__rolling_probability__", "__majority_class__"
        )
        assert registry_csv == modeling_csv, (
            f"{set_id}: registry features {registry_csv!r} disagree with "
            f"modelling loader {modeling_csv!r}"
        )
