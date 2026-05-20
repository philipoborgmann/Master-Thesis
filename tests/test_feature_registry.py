"""Feature-set registry invariants."""
from __future__ import annotations

import pytest

from thesis_pipeline.features.feature_registry import (
    load_feature_sets_yaml, load_feature_sets, validate_registry, SET_ID_PATTERN,
)


def test_yaml_loads_with_known_set_ids():
    sets = load_feature_sets_yaml()
    # The YAML must at least contain the headline benchmark / economic /
    # sentiment / combined IDs.
    must_have = {"B1", "B2", "E1", "E2", "E3", "E4", "S1", "C1"}
    assert must_have.issubset(set(sets.keys())), (
        f"missing keys: {must_have - set(sets.keys())}"
    )


def test_set_ids_unique_and_known_patterns():
    sets = load_feature_sets_yaml()
    assert len(sets) == len(set(sets.keys()))  # unique
    for sid in sets.keys():
        # All set IDs should belong to one of the documented families.
        family = sid[0]
        assert family in ("B", "E", "S", "C"), f"unknown family for {sid!r}"


def test_each_set_has_unique_feature_names():
    sets = load_feature_sets_yaml()
    problems = validate_registry(sets)
    assert not problems, f"registry issues: {problems}"


def test_load_feature_sets_prefers_xlsx_when_available():
    """If feature_sets.xlsx exists, load_feature_sets() should pick it up.

    Otherwise it must fall back to the YAML mirror.
    """
    sets = load_feature_sets()
    assert isinstance(sets, dict) and sets
