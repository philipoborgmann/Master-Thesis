"""Tests for ``load_feature_sets`` — column-priority and benchmark-sentinel."""
from __future__ import annotations

import pandas as pd
import pytest

from thesis_pipeline.modeling import run_models as rm


@pytest.fixture
def xlsx_factory(tmp_path):
    """Return a function that writes a one-sheet feature_sets XLSX from a dict."""
    def _make(rows: dict, sheet_name: str = "feature_sets"):
        df = pd.DataFrame(rows)
        path = tmp_path / "feature_sets.xlsx"
        with pd.ExcelWriter(path, engine="openpyxl") as w:
            df.to_excel(w, sheet_name=sheet_name, index=False)
        return str(path)
    return _make


def test_explicit_comma_separated_column_wins_over_hash_features(xlsx_factory):
    """If both ``# Features`` and ``feature_columns_comma_separated`` are
    present, the explicit comma-separated column must win — *not* the
    ``# Features`` column that happens to normalise to the same name."""
    path = xlsx_factory({
        "set_id":   ["B1", "E1"],
        "category": ["benchmark", "economic"],
        # `# Features` normalises to `features` after header cleanup; this
        # used to silently override the more specific column.
        "# Features": ["WRONG_FROM_HASH_FEATURES", "WRONG_FROM_HASH_FEATURES"],
        "feature_columns_comma_separated": [
            "__majority_class__",
            "log_return_t,cum_log_return_7",
        ],
        "label": ["b1-label", "e1-label"],
    })

    out = rm.load_feature_sets(path)

    assert "features" in out.columns
    feats = out["features"].tolist()
    # The wrong column must NOT have leaked through.
    assert "WRONG_FROM_HASH_FEATURES" not in feats, (
        f"`# Features` column won over `feature_columns_comma_separated`: {feats}"
    )
    # Benchmark sentinel must be unified — B1 must not be parsed as feature "0".
    assert feats[0] == "__rolling_probability__"
    # E1 must keep the comma-separated list intact.
    assert feats[1] == "log_return_t,cum_log_return_7"


def test_falls_back_to_hash_features_when_no_explicit_column(xlsx_factory):
    """When no explicit feature_columns* header is present, the legacy
    ``# Features`` column (after normalisation) is the source. This was the
    pre-fix behaviour and must remain available as a fallback."""
    path = xlsx_factory({
        "set_id":   ["B1"],
        "category": ["benchmark"],
        "# Features": ["__majority_class__"],
        "label":    ["b1"],
    })

    out = rm.load_feature_sets(path)
    assert "features" in out.columns
    assert out["features"].iloc[0] == "__rolling_probability__"


def test_explicit_feature_columns_used_when_only_that_is_present(xlsx_factory):
    """`feature_columns` (without ``_comma_separated``) is also acceptable."""
    path = xlsx_factory({
        "set_id":   ["E1"],
        "category": ["economic"],
        "feature_columns": ["log_return_t,cum_log_return_7"],
        "label":    ["e1"],
    })
    out = rm.load_feature_sets(path)
    assert out["features"].iloc[0] == "log_return_t,cum_log_return_7"


def test_majority_class_sentinel_normalised(xlsx_factory):
    """``__majority_class__`` must be replaced with ``__rolling_probability__``
    so that the benchmark path is taken — *not* parsed as a missing feature
    column called ``"0"``."""
    path = xlsx_factory({
        "set_id":   ["B1"],
        "category": ["benchmark"],
        "feature_columns_comma_separated": ["__majority_class__"],
        "label":    ["b1"],
    })
    out = rm.load_feature_sets(path)
    assert out["features"].iloc[0] == "__rolling_probability__"
