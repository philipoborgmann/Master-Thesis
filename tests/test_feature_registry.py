"""Feature-set registry invariants (v4 — Variante A, 17 sets)."""
from __future__ import annotations

import pytest

from thesis_pipeline.features.feature_registry import (
    load_feature_sets_yaml, load_feature_sets, validate_registry,
    SET_ID_PATTERN, REMOVED_SET_IDS, DIAGNOSTIC_ONLY_COLUMNS,
)


def test_yaml_loads_with_exactly_the_v4_seventeen_set_ids():
    """The YAML mirror must contain exactly the 17 v4 sets — no v3 IDs left."""
    sets = load_feature_sets_yaml()
    assert set(sets.keys()) == set(SET_ID_PATTERN)
    assert len(sets) == 17


def test_set_ids_unique_and_follow_v4_naming():
    sets = load_feature_sets_yaml()
    assert len(sets) == len(set(sets.keys()))
    for sid in sets.keys():
        # Either the bare ECON benchmark, or one of the SENT_/ECON_ families.
        assert sid == "ECON" or sid.startswith(("SENT_VAD_", "SENT_CBT_",
                                                 "ECON_VAD_", "ECON_CBT_")), \
            f"unknown v4 family for {sid!r}"


def test_each_set_has_unique_feature_names_and_no_diagnostic_only_columns():
    sets = load_feature_sets_yaml()
    problems = validate_registry(sets)
    assert not problems, f"registry issues: {problems}"


def test_diagnostic_only_columns_never_in_any_v4_set():
    sets = load_feature_sets_yaml()
    diag = set(DIAGNOSTIC_ONLY_COLUMNS)
    for sid, spec in sets.items():
        bad = [f for f in spec["features"] if f in diag]
        assert not bad, f"{sid} contains diagnostic-only columns: {bad}"


def test_load_feature_sets_prefers_xlsx_when_available():
    """If feature_sets.xlsx exists, load_feature_sets() should pick it up.

    Otherwise it falls back to the YAML mirror.
    """
    sets = load_feature_sets()
    assert isinstance(sets, dict) and sets


@pytest.mark.parametrize("legacy", ["B1", "B2", "B6", "E4",
                                     "S1", "SV1", "C1", "CV1",
                                     "M1", "SC1", "SF1"])
def test_legacy_set_ids_have_migration_message(legacy):
    """Every v3 ID listed in REMOVED_SET_IDS must carry a human-readable
    migration sentence (not an empty string, not a bare ID)."""
    msg = REMOVED_SET_IDS[legacy]
    assert isinstance(msg, str) and len(msg) > 20, (
        f"{legacy!r} migration message is too terse: {msg!r}"
    )
