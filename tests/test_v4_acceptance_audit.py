"""Tests for the v4 acceptance audit (Aufgabe 8 Part G).

The audit must FAIL on a tampered fixture and PASS on the current code
base. A criterion is never reported PASS merely because a test
function exists.
"""
from __future__ import annotations

import pandas as pd
import pytest

from thesis_pipeline.diagnostics import remaining_assumptions as ra
from thesis_pipeline.diagnostics import v4_acceptance_audit as au


# ---------------------------------------------------------------------------
# Smoke: run the full audit on the current code base
# ---------------------------------------------------------------------------

def test_run_audit_returns_all_sections():
    out = au.run_v4_acceptance_audit()
    assert set(out) >= {
        "feature_path", "feature_registry", "price_features",
        "market_cap", "modeling_defaults", "evaluation",
        "temporal_assumptions",
    }
    for section in out.values():
        for item in section.values():
            assert item["status"] in (au.PASS, au.FAIL,
                                       au.NOT_RUN, au.MANUAL_REVIEW)


def test_feature_registry_audit_passes_on_current_registry():
    out = au.audit_feature_registry()
    assert out["registry_validates"]["status"] == au.PASS
    assert out["econ_core_identical_across_combined"]["status"] == au.PASS


def test_modeling_defaults_audit_passes_on_current_code():
    out = au.audit_modeling_defaults()
    for key in ("model_type_default", "panel_mode_default",
                 "train_window_default", "rolling_window_days_default",
                 "hpo_default_on", "hpo_objective_default",
                 "checkpoint_default_on", "resume_default_on",
                 "generate_naive_default_on"):
        assert out[key]["status"] == au.PASS, (key, out[key])


def test_evaluation_audit_passes_on_current_code():
    out = au.audit_evaluation()
    for k, v in out.items():
        assert v["status"] == au.PASS, (k, v)


def test_temporal_assumptions_audit_reports_manual_review_without_cutoff():
    out = au.audit_temporal_assumptions()
    assert out["completed_slot_assertion"]["status"] == au.MANUAL_REVIEW
    assert out["post_cutoff_assertion"]["status"] == au.PASS
    assert out["predictor_target_separation_assertion"]["status"] == au.PASS


def test_temporal_assumptions_audit_passes_with_full_cutoff_map():
    out = au.audit_temporal_assumptions(observation_cutoffs={
        "1d": pd.Timestamp("2024-01-01", tz="UTC"),
        "6h": pd.Timestamp("2024-01-01", tz="UTC"),
        "1h": pd.Timestamp("2024-01-01", tz="UTC"),
    })
    for k, v in out.items():
        assert v["status"] == au.PASS, (k, v)


def test_temporal_assumptions_partial_cutoff_is_manual_review():
    out = au.audit_temporal_assumptions(observation_cutoffs={
        "1d": pd.Timestamp("2024-01-01", tz="UTC"),
    })
    assert out["completed_slot_assertion"]["status"] == au.MANUAL_REVIEW
    detail = out["completed_slot_assertion"]["detail"]
    assert "1h" in detail and "6h" in detail


def test_market_cap_audit_passes_on_current_code():
    out = au.audit_market_cap()
    for k, v in out.items():
        assert v["status"] == au.PASS, (k, v)


def test_price_features_audit_passes_on_current_code():
    out = au.audit_price_features()
    for k, v in out.items():
        assert v["status"] == au.PASS, (k, v)


# ---------------------------------------------------------------------------
# Targeted FAIL paths — audit must detect a tampered fixture
# ---------------------------------------------------------------------------

def test_feature_path_audit_fails_on_forbidden_column():
    df = pd.DataFrame({"score": [1.0], "log_return_t": [0.1]})
    out = au.audit_feature_path(df)
    assert out["forbidden_engagement_in_frame"]["status"] == au.FAIL


def test_feature_path_audit_passes_on_clean_frame():
    df = pd.DataFrame({"log_return_t": [0.1], "score_mean": [0.5]})
    out = au.audit_feature_path(df)
    assert out["forbidden_engagement_in_frame"]["status"] == au.PASS


# ---------------------------------------------------------------------------
# Audit summariser
# ---------------------------------------------------------------------------

def test_summarize_audit_zero_failures_zero_pending_is_merge_ready():
    audit = au.run_v4_acceptance_audit(
        feature_frame=pd.DataFrame({"log_return_t": [0.0]}),
        observation_cutoffs={
            "1d": pd.Timestamp("2024-01-01", tz="UTC"),
            "6h": pd.Timestamp("2024-01-01", tz="UTC"),
            "1h": pd.Timestamp("2024-01-01", tz="UTC"),
        },
    )
    summary = au.summarize_audit(audit)
    assert summary["counts"][au.FAIL] == 0
    assert summary["counts"][au.MANUAL_REVIEW] == 0
    assert summary["counts"][au.NOT_RUN] == 0
    assert summary["merge_ready"] is True
    assert summary["passed"] is True   # legacy alias


def test_summarize_audit_blocks_on_manual_review():
    """Audit without cutoffs leaves the completed-slot rule in
    MANUAL_REVIEW — the merge gate MUST stay closed even though there
    are no hard failures."""
    audit = au.run_v4_acceptance_audit(
        feature_frame=pd.DataFrame({"log_return_t": [0.0]}),
    )
    summary = au.summarize_audit(audit)
    assert summary["counts"][au.MANUAL_REVIEW] >= 1
    assert summary["merge_ready"] is False
    assert any("completed_slot" in p for p in summary["pending"])


def test_summarize_audit_records_failures():
    audit = au.run_v4_acceptance_audit(
        feature_frame=pd.DataFrame({"score": [1.0]}),
        observation_cutoffs={
            "1d": pd.Timestamp("2024-01-01", tz="UTC"),
            "6h": pd.Timestamp("2024-01-01", tz="UTC"),
            "1h": pd.Timestamp("2024-01-01", tz="UTC"),
        },
    )
    summary = au.summarize_audit(audit)
    assert summary["merge_ready"] is False
    assert any("forbidden_engagement_in_frame" in f
                for f in summary["failures"])


def test_evaluation_audit_includes_clustering_and_classweight():
    out = au.audit_evaluation()
    for key in ("ticker_clustered_inference",
                 "class_weight_grid_v4",
                 "mcnemar_supplementary_layer",
                 "h2_h3_family_identity_columns"):
        assert key in out
        assert out[key]["status"] == au.PASS, (key, out[key])


# ---------------------------------------------------------------------------
# Remaining-assumptions report
# ---------------------------------------------------------------------------

def test_remaining_assumptions_payload_lists_required_rules():
    payload = ra.build_remaining_assumptions()
    rules = {r["rule"] for r in payload["v4_remaining_assumptions"]}
    for required in (
        "completed_slot", "post_cutoff", "predictor_target_separation",
        "market_cap_availability", "volatility_regime_availability",
        "universe_definitions", "legacy_output_policy",
        "h1_bh_family_scope", "small_cluster_warning", "zero_se_tolerance",
    ):
        assert required in rules


def test_remaining_assumptions_completed_slot_status_depends_on_cutoff():
    no_cutoff = ra.build_remaining_assumptions()
    with_cutoff = ra.build_remaining_assumptions(
        observation_cutoff=pd.Timestamp("2024-01-01", tz="UTC"),
    )
    by_rule_no = {r["rule"]: r for r in no_cutoff["v4_remaining_assumptions"]}
    by_rule_yes = {r["rule"]: r for r in with_cutoff["v4_remaining_assumptions"]}
    assert by_rule_no["completed_slot"]["status"] == ra.REMAINING_MANUAL
    assert by_rule_yes["completed_slot"]["status"] == ra.VERIFIED_ASSERTION


def test_remaining_assumptions_writer_creates_json(tmp_path):
    p = tmp_path / "remaining_assumptions.json"
    out = ra.write_remaining_assumptions_report(
        p, observation_cutoff=pd.Timestamp("2024-01-01", tz="UTC"),
    )
    assert out.exists()
    import json
    payload = json.loads(out.read_text())
    assert "v4_remaining_assumptions" in payload
