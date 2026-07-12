"""Unit tests for the centralized family-aware BH correction (Part 1)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from thesis_pipeline.evaluation import multiple_testing as mt
from thesis_pipeline.evaluation.preregistration import (
    ALPHA_PRESPECIFIED, CONFIRMATORY_FAMILIES,
)


def _records(rows):
    return pd.DataFrame(rows, columns=["test_id", "family", "horizon",
                                        "model", "set_id", "metric",
                                        "p_value_raw"])


# ---------------------------------------------------------------------------
# (iv) BH grouping isolates families
# ---------------------------------------------------------------------------

def test_family_isolation_a_does_not_affect_b():
    """Adding tiny p-values to family A must not change family B's
    q-values."""
    b_rows = [("B|1d|S1", "B_H1_directional", "1d", "vad", "S1", "acc", 0.02),
              ("B|6h|S1", "B_H1_directional", "6h", "vad", "S1", "acc", 0.04),
              ("B|1h|S1", "B_H1_directional", "1h", "vad", "S1", "acc", 0.20)]
    only_b = mt.apply_family_bh(_records(b_rows))
    q_b_alone = only_b.set_index("test_id")["q_value_bh"].to_dict()

    a_rows = [("A|1d|S1", "A_H1_logloss", "1d", "vad", "S1", "ll", 1e-6),
              ("A|6h|S1", "A_H1_logloss", "6h", "vad", "S1", "ll", 1e-6),
              ("A|1h|S1", "A_H1_logloss", "1h", "vad", "S1", "ll", 1e-6)]
    both = mt.apply_family_bh(_records(a_rows + b_rows))
    q_b_with_a = both[both["family"] == "B_H1_directional"] \
        .set_index("test_id")["q_value_bh"].to_dict()

    for tid, q in q_b_alone.items():
        assert q_b_with_a[tid] == pytest.approx(q, abs=1e-12)


def test_bh_matches_legacy_helper_byte_identical():
    """The centralized BH must reproduce the legacy per-family helper
    exactly on the same p-value set (guarantees B/C/D unchanged)."""
    from thesis_pipeline.evaluation.diff_in_improvement import (
        adjust_pvalues_bh_within_family,
    )
    pvals = [0.001, 0.02, 0.03, 0.008, 0.5, 0.9, 0.04, 0.11]
    rows = [(f"C|{i}", "C_H2_volatility", "1d", "m", f"S{i}", "acc", p)
            for i, p in enumerate(pvals)]
    centralized = mt.apply_family_bh(_records(rows))

    legacy_in = pd.DataFrame({
        "hypothesis_family": ["C_H2_volatility"] * len(pvals),
        "p_value": pvals,
    })
    legacy = adjust_pvalues_bh_within_family(legacy_in)
    np.testing.assert_allclose(
        centralized["q_value_bh"].to_numpy(dtype=float),
        legacy["q_value_bh"].to_numpy(dtype=float),
        rtol=0, atol=1e-12,
    )


def test_nan_pvalues_excluded_from_pool():
    rows = [("t1", "A_H1_logloss", "1d", "m", "S1", "ll", 0.01),
            ("t2", "A_H1_logloss", "6h", "m", "S2", "ll", np.nan),
            ("t3", "A_H1_logloss", "1h", "m", "S3", "ll", 0.02)]
    out = mt.apply_family_bh(_records(rows))
    nan_row = out[out["test_id"] == "t2"].iloc[0]
    assert np.isnan(nan_row["q_value_bh"])
    assert not bool(nan_row["significant_bh"])
    # The pool size for the valid rows is 2 (not 3).
    valid = out[out["p_value_raw"].notna()]
    assert len(valid) == 2


def test_single_significant_flag_at_alpha():
    rows = [("t1", "B_H1_directional", "1d", "m", "S1", "acc", 0.0001),
            ("t2", "B_H1_directional", "6h", "m", "S2", "acc", 0.9)]
    out = mt.apply_family_bh(_records(rows))
    assert set(out["alpha_prespecified"]) == {ALPHA_PRESPECIFIED}
    assert "significant_bh_10pct" not in out.columns
    assert bool(out[out["test_id"] == "t1"].iloc[0]["significant_bh"])
    assert not bool(out[out["test_id"] == "t2"].iloc[0]["significant_bh"])


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------

def test_manifest_default_lists_only_confirmatory_A_to_D():
    rows = []
    for fam in ("A_H1_logloss", "B_H1_directional", "C_H2_volatility",
                "D_H3_marketcap", "E1_horizon_logloss", "E2_horizon_accuracy"):
        for i in range(3):
            rows.append((f"{fam}|{i}", fam, "1d", "m", f"S{i}", "x", 0.01 * (i + 1)))
    out = mt.apply_family_bh(_records(rows))
    manifest = mt.build_manifest(out)
    # The default manifest lists ONLY the four confirmatory families (A–D);
    # E1/E2 are exploratory and excluded from the confirmatory set.
    assert list(manifest["family"]) == list(CONFIRMATORY_FAMILIES)
    assert set(manifest["family"]) == {"A_H1_logloss", "B_H1_directional",
                                       "C_H2_volatility", "D_H3_marketcap"}
    assert "E1_horizon_logloss" not in set(manifest["family"])
    assert "E2_horizon_accuracy" not in set(manifest["family"])
    assert (manifest["family_role"] == "confirmatory").all()
    assert (manifest["n_tests"] == 3).all()
    assert (manifest["correction"] == "BH").all()
    assert not manifest["family"].str.contains("absolute_vs_naive").any()
    assert not manifest["family"].str.contains("regime_mcnemar").any()


# ---------------------------------------------------------------------------
# Objective A — E1/E2 (and sentiment-only) are EXPLORATORY, A–D confirmatory
# ---------------------------------------------------------------------------

def test_family_roles_partition_confirmatory_and_exploratory():
    from thesis_pipeline.evaluation.preregistration import (
        CONFIRMATORY_FAMILIES, EXPLORATORY_FAMILIES,
        FAMILY_A_H1_LOGLOSS, FAMILY_B_H1_DIRECTIONAL, FAMILY_C_H2_VOLATILITY,
        FAMILY_D_H3_MARKETCAP, FAMILY_E1_HORIZON_LOGLOSS,
        FAMILY_E2_HORIZON_ACCURACY, FAMILY_EXPL_SENTIMENT_DIRECTIONAL,
    )
    assert set(CONFIRMATORY_FAMILIES) == {
        FAMILY_A_H1_LOGLOSS, FAMILY_B_H1_DIRECTIONAL,
        FAMILY_C_H2_VOLATILITY, FAMILY_D_H3_MARKETCAP}
    # E1, E2 and sentiment-only directional are exploratory, not confirmatory.
    for fam in (FAMILY_E1_HORIZON_LOGLOSS, FAMILY_E2_HORIZON_ACCURACY,
                FAMILY_EXPL_SENTIMENT_DIRECTIONAL):
        assert fam in EXPLORATORY_FAMILIES
        assert fam not in CONFIRMATORY_FAMILIES
    # Roles are disjoint.
    assert not (set(CONFIRMATORY_FAMILIES) & set(EXPLORATORY_FAMILIES))


def test_build_manifest_exploratory_role_for_E1_E2():
    from thesis_pipeline.evaluation.preregistration import (
        FAMILY_E1_HORIZON_LOGLOSS, FAMILY_E2_HORIZON_ACCURACY,
    )
    rows = []
    for fam in ("A_H1_logloss", "E1_horizon_logloss", "E2_horizon_accuracy"):
        for i in range(3):
            rows.append((f"{fam}|{i}", fam, "1d", "m", f"S{i}", "x", 0.01 * (i + 1)))
    out = mt.apply_family_bh(_records(rows))
    e_manifest = mt.build_manifest(
        out, families=(FAMILY_E1_HORIZON_LOGLOSS, FAMILY_E2_HORIZON_ACCURACY),
        family_role="exploratory")
    assert set(e_manifest["family"]) == {"E1_horizon_logloss",
                                         "E2_horizon_accuracy"}
    assert (e_manifest["family_role"] == "exploratory").all()


def test_E1_and_E2_get_separate_bh_passes():
    # Give E1 tiny p-values and E2 large ones; BH within each family must NOT
    # borrow across families. E1's q-values reflect only E1's pool.
    e1 = [(f"E1|{i}", "E1_horizon_logloss", "1d", "m", f"S{i}", "ll", p)
          for i, p in enumerate([0.001, 0.002, 0.003])]
    e2 = [(f"E2|{i}", "E2_horizon_accuracy", "1d", "m", f"S{i}", "acc", p)
          for i, p in enumerate([0.4, 0.5, 0.6])]
    out = mt.apply_family_bh(_records(e1 + e2))
    # E1 alone gives the same q-values as E1 pooled with E2 (family isolation).
    e1_only = mt.apply_family_bh(_records(e1))
    q_joint = out[out["family"] == "E1_horizon_logloss"] \
        .set_index("test_id")["q_value_bh"]
    q_alone = e1_only.set_index("test_id")["q_value_bh"]
    for tid in q_alone.index:
        assert q_joint[tid] == pytest.approx(q_alone[tid], abs=1e-12)
    # E2's large p-values do not become significant via E1's small ones.
    assert not out[out["family"] == "E2_horizon_accuracy"]["significant_bh"].any()


def test_reclassification_does_not_change_raw_pvalues_or_qvalues():
    # Reclassifying E1/E2 as exploratory is a labelling change only: the BH
    # math (apply_family_bh, grouped by the family COLUMN) is untouched.
    rows = [(f"E1|{i}", "E1_horizon_logloss", "1d", "m", f"S{i}", "ll", p)
            for i, p in enumerate([0.01, 0.02, 0.03, 0.04])]
    out = mt.apply_family_bh(_records(rows))
    # raw p-values preserved verbatim
    assert list(out["p_value_raw"]) == [0.01, 0.02, 0.03, 0.04]
    # q-values equal a standalone BH over exactly these four p-values
    import numpy as np
    expected_q = mt._bh_qvalues(np.array([0.01, 0.02, 0.03, 0.04]))
    np.testing.assert_allclose(out["q_value_bh"].to_numpy(), expected_q,
                               rtol=0, atol=1e-12)


def test_manifest_counts_significant():
    rows = [("t1", "A_H1_logloss", "1d", "m", "S1", "ll", 1e-6),
            ("t2", "A_H1_logloss", "6h", "m", "S2", "ll", 1e-6),
            ("t3", "A_H1_logloss", "1h", "m", "S3", "ll", 0.9)]
    out = mt.apply_family_bh(_records(rows))
    manifest = mt.build_manifest(out)
    a = manifest[manifest["family"] == "A_H1_logloss"].iloc[0]
    assert a["n_tests"] == 3
    assert a["n_significant_bh"] == 2


# ---------------------------------------------------------------------------
# (v) H1 support-class taxonomy
# ---------------------------------------------------------------------------

def test_h1_support_class_taxonomy_all_four_cases():
    from thesis_pipeline.evaluation.confirmatory import _h1_support_class
    # (probabilistic H1a, directional H1b) -> support class.
    assert _h1_support_class(True, True) == "strong"
    assert _h1_support_class(True, False) == "probabilistic"
    assert _h1_support_class(False, True) == "directional"
    assert _h1_support_class(False, False) == "none"
