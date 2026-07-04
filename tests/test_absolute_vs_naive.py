"""Tests for the absolute_vs_naive evaluation layer (commit 2 Section F).

The table compares every model run against the matched NAIVE reference
on the COMPLETE NAIVE identity (horizon, model_type, panel_mode,
train_window_*, coin_universe_hash). Output is written to
``Outputs/Evaluation/absolute_vs_naive.csv`` and the Excel sheet
``absolute_vs_naive``. The frame carries the hypothesis-family tag
``absolute_vs_naive`` so it is NEVER pooled with H1 / H2 / H3.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from thesis_pipeline.evaluation import naive_comparison as nc
from thesis_pipeline.modeling.naive_reference import coin_universe_hash


# ---------------------------------------------------------------------------
# Synthetic signal builders
# ---------------------------------------------------------------------------

def _build_signals(model_edge: float = 0.62,
                   naive_edge: float = 0.50,
                   n: int = 200,
                   tickers=("BTC", "ETH"),
                   seed: int = 0) -> pd.DataFrame:
    """Return a long-form signal frame with one model + matched NAIVE row
    per (ticker, timestamp). The NAIVE rows carry hpo_variant=naive,
    set_id=NAIVE and the coin_universe_hash stamped by the generator.
    """
    rng = np.random.default_rng(seed)
    ts = pd.date_range("2024-01-01", periods=n, freq="D", tz="UTC")
    hash_value = coin_universe_hash(list(tickers))
    rows = []
    for tk in tickers:
        target = rng.integers(0, 2, n).astype(int)
        # Model
        flip_m = rng.random(n) > model_edge
        pred_m = np.where(flip_m, 1 - target, target).astype(int)
        prob_m = np.where(pred_m == 1, rng.uniform(0.55, 0.85, n),
                          rng.uniform(0.15, 0.45, n)).astype(float)
        rows.append(pd.DataFrame({
            "timestamp":   ts, "ticker": tk,
            "target":      target, "prediction": pred_m, "probability": prob_m,
            "horizon":     "1d",
            "set_id":      "ECON_VAD_F", "sentiment_model": "vader",
            "model_type":  "panel_logit",
            "panel_mode":  "ticker_fixed_effects",
            "hpo_variant": "log_loss",
            "hpo_objective": "log_loss",
            "train_window_mode":       "rolling_fixed",
            "rolling_window_days":     180.0,
            "rolling_window_timestamps": np.nan,
            "coin_universe_hash":      hash_value,
        }))
        # NAIVE
        flip_n = rng.random(n) > naive_edge
        pred_n = np.where(flip_n, 1 - target, target).astype(int)
        prob_n = np.where(pred_n == 1, rng.uniform(0.55, 0.85, n),
                          rng.uniform(0.15, 0.45, n)).astype(float)
        rows.append(pd.DataFrame({
            "timestamp":   ts, "ticker": tk,
            "target":      target, "prediction": pred_n, "probability": prob_n,
            "horizon":     "1d",
            "set_id":      "NAIVE", "sentiment_model": "-",
            "model_type":  "panel_logit",
            "panel_mode":  "ticker_fixed_effects",
            "hpo_variant": "naive",
            "hpo_objective": "-",
            "train_window_mode":       "rolling_fixed",
            "rolling_window_days":     180.0,
            "rolling_window_timestamps": np.nan,
            "coin_universe_hash":      hash_value,
        }))
    return pd.concat(rows, ignore_index=True)


# ---------------------------------------------------------------------------
# Identity contract
# ---------------------------------------------------------------------------

def test_identity_columns_include_universe_hash_and_window():
    assert "coin_universe_hash" in nc.NAIVE_IDENTITY_COLUMNS
    assert "train_window_mode" in nc.NAIVE_IDENTITY_COLUMNS
    assert "rolling_window_days" in nc.NAIVE_IDENTITY_COLUMNS
    # No HPO column — NAIVE is by construction untuned.
    assert "hpo_variant" not in nc.NAIVE_IDENTITY_COLUMNS
    assert "hpo_objective" not in nc.NAIVE_IDENTITY_COLUMNS


def test_family_tag_is_separate_from_h1_h2_h3():
    """Sanity check: a downstream BH-within-family pool MUST NOT mix
    absolute_vs_naive into the H1 / H2 / H3 families."""
    assert nc.HYPOTHESIS_FAMILY == "absolute_vs_naive"
    from thesis_pipeline.evaluation.diff_in_improvement import H_HYPOTHESIS_FAMILIES
    assert nc.HYPOTHESIS_FAMILY not in H_HYPOTHESIS_FAMILIES


# ---------------------------------------------------------------------------
# Empty-input behaviour
# ---------------------------------------------------------------------------

def test_empty_signals_returns_schema_only_frame():
    out = nc.absolute_vs_naive_table(pd.DataFrame())
    assert out.empty
    for col in ("hypothesis_family", "accuracy_lift", "brier_improvement",
                "log_loss_improvement", "coin_universe_hash"):
        assert col in out.columns


def test_no_naive_rows_returns_empty():
    sig = _build_signals()
    sig = sig[sig["set_id"] != "NAIVE"]
    out = nc.absolute_vs_naive_table(sig)
    assert out.empty


# ---------------------------------------------------------------------------
# Core matching + metric semantics
# ---------------------------------------------------------------------------

def test_one_row_per_model_run_with_metric_signs_correct():
    sig = _build_signals(model_edge=0.70, naive_edge=0.50, seed=1)
    out = nc.absolute_vs_naive_table(sig)
    assert len(out) == 1
    r = out.iloc[0]
    assert r["hypothesis_family"] == "absolute_vs_naive"
    assert r["set_id"] == "ECON_VAD_F"
    assert r["naive_set_id"] == "NAIVE"
    assert r["status"] == "ok"
    # Sign conventions
    assert r["model_accuracy"] > r["naive_accuracy"]
    assert r["accuracy_lift"] > 0
    # brier_improvement = NAIVE - model (lower brier is better → positive lift)
    assert r["naive_brier"] >= r["model_brier"] - 1e-9
    assert r["brier_improvement"] >= -1e-9
    # log_loss_improvement = NAIVE - model
    assert r["naive_log_loss"] >= r["model_log_loss"] - 1e-9
    assert r["log_loss_improvement"] >= -1e-9


def test_naive_universe_hash_does_not_gate_the_match(monkeypatch):
    """A NAIVE run on a different/larger universe STILL matches via the
    coverage-intersection inner-join (commit 13). NAIVE is intentionally
    estimated on a broader universe than the coverage-filtered models, so
    the universe hash is recorded for transparency but is NOT a hard
    gate. The model and NAIVE share the (ticker, timestamp) keys here, so
    the comparison runs on that overlap."""
    sig = _build_signals()
    # Relabel the NAIVE rows' universe hash to a different universe — the
    # actual (ticker, timestamp) predictions still overlap the model's.
    other_hash = coin_universe_hash(["SOL", "ADA"])
    sig.loc[sig["set_id"] == "NAIVE", "coin_universe_hash"] = other_hash
    out = nc.absolute_vs_naive_table(sig)
    assert len(out) == 1
    r = out.iloc[0]
    # The comparison runs on the overlap rather than being dropped.
    assert r["status"] == "ok"
    assert int(r["n_matched"]) > 0
    # The two universe hashes are recorded distinctly and honestly.
    assert r["naive_coin_universe_hash"] == other_hash
    assert r["coin_universe_hash"] != other_hash


def test_match_rejects_different_window():
    """A NAIVE row built with a different rolling window MUST NOT match."""
    sig = _build_signals()
    sig.loc[sig["set_id"] == "NAIVE", "rolling_window_days"] = 60.0
    out = nc.absolute_vs_naive_table(sig)
    r = out.iloc[0]
    assert r["status"] == "missing_naive"


def test_target_mismatch_is_surfaced():
    """If the model and NAIVE disagree on the target sequence (bug
    upstream), the row must be flagged target_mismatch — not silently
    produce metrics."""
    sig = _build_signals()
    # Flip one NAIVE target so model and NAIVE disagree on that row.
    naive_idx = sig.index[sig["set_id"] == "NAIVE"][0]
    sig.loc[naive_idx, "target"] = 1 - sig.loc[naive_idx, "target"]
    out = nc.absolute_vs_naive_table(sig)
    r = out.iloc[0]
    assert r["status"] == "target_mismatch"
    assert r["targets_identical"] is False or r["targets_identical"] is np.False_


def test_duplicate_keys_flagged():
    sig = _build_signals()
    # Duplicate one model row.
    model_row = sig[sig["set_id"] != "NAIVE"].iloc[0:1]
    sig = pd.concat([sig, model_row], ignore_index=True)
    out = nc.absolute_vs_naive_table(sig)
    r = out.iloc[0]
    assert r["n_duplicate_model_keys"] >= 1
    assert r["status"] in ("duplicate_keys", "ok")
    if r["status"] == "duplicate_keys":
        assert r["skip_reason"] == "duplicate_keys_within_identity"


def test_naive_never_compared_to_itself():
    """The NAIVE row must not appear on the model side — that would be a
    no-op identity comparison."""
    sig = _build_signals()
    out = nc.absolute_vs_naive_table(sig)
    assert "NAIVE" not in set(out["set_id"].astype(str))


# ---------------------------------------------------------------------------
# Hypothesis-family separation
# ---------------------------------------------------------------------------

def test_hypothesis_family_tag_is_constant_per_row():
    sig = _build_signals()
    out = nc.absolute_vs_naive_table(sig)
    assert (out["hypothesis_family"] == "absolute_vs_naive").all()


# ---------------------------------------------------------------------------
# Cross-check vs incremental table — they answer different questions
# ---------------------------------------------------------------------------

def test_does_not_overlap_with_incremental_sentiment_value():
    """The absolute_vs_naive frame and the incremental_sentiment_value
    frame must answer different questions: absolute vs NAIVE here, vs
    ECON over there. The two outputs share identifying columns but
    distinct ``hypothesis_family`` tags, never an overlap."""
    from thesis_pipeline.evaluation.incremental import (
        incremental_sentiment_value_table,
    )
    sig = _build_signals()
    abs_df = nc.absolute_vs_naive_table(sig)
    inc_df = incremental_sentiment_value_table(sig)
    # No row carries the wrong family tag.
    assert (abs_df["hypothesis_family"] == "absolute_vs_naive").all()
    if not inc_df.empty:
        assert (inc_df["hypothesis_family"] == "H1_incremental").all()
