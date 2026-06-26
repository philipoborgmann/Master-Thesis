"""Tests for the smoke-output validator (Aufgabe 8 Section F.3).

The validator does NOT run a real smoke pipeline (the data is too
large for unit tests). Instead it inspects parquet outputs that the
test fabricates with the v4 canonical metadata layout and verifies the
validator catches mismatches.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from thesis_pipeline.diagnostics.smoke_validation import (
    CANONICAL_SMOKE, validate_smoke_outputs,
)
from thesis_pipeline.modeling.naive_reference import coin_universe_hash


def _econ_row(**overrides):
    base = {
        "set_id":             "ECON",
        "horizon":            "1d",
        "model_type":         "panel_logit",
        "panel_mode":         "ticker_fixed_effects",
        "train_window_mode":  "rolling_fixed",
        "rolling_window_days": 180.0,
        "hpo_enabled":        True,
        "hpo_objective":      "log_loss",
        "requested_tickers":  "BTC|ETH",
        "universe_identity_source": "requested_metadata",
        "requested_coin_universe_hash": coin_universe_hash(("BTC", "ETH")),
    }
    base.update(overrides)
    return base


def _naive_row(**overrides):
    base = {
        "set_id":             "NAIVE",
        "horizon":            "1d",
        "model_type":         "panel_logit",
        "panel_mode":         "ticker_fixed_effects",
        "train_window_mode":  "rolling_fixed",
        "rolling_window_days": 180.0,
        "hpo_enabled":        False,
        "hpo_variant":        "naive",
        "hpo_objective":      "-",
        "requested_tickers":  "BTC|ETH",
        "universe_identity_source": "requested_metadata",
        "requested_coin_universe_hash": coin_universe_hash(("BTC", "ETH")),
    }
    base.update(overrides)
    return base


def _write(signal_dir, set_id, row):
    horizon_dir = signal_dir / "1d"
    horizon_dir.mkdir(parents=True, exist_ok=True)
    h = coin_universe_hash(("BTC", "ETH"))
    path = horizon_dir / f"{set_id}_u_{h}.parquet"
    pd.DataFrame([row] * 3).to_parquet(path, index=False)
    return path


def test_validator_passes_on_canonical_smoke_outputs(tmp_path):
    sig_dir = tmp_path / "Signals"
    _write(sig_dir, "ECON", _econ_row())
    _write(sig_dir, "NAIVE", _naive_row())
    mismatches = validate_smoke_outputs(sig_dir, expected_tickers=("BTC", "ETH"))
    assert mismatches == {}


def test_validator_flags_missing_econ(tmp_path):
    sig_dir = tmp_path / "Signals"
    _write(sig_dir, "NAIVE", _naive_row())
    mismatches = validate_smoke_outputs(sig_dir)
    assert any("ECON" in m for m in mismatches.get("econ", []))


def test_validator_flags_missing_naive(tmp_path):
    sig_dir = tmp_path / "Signals"
    _write(sig_dir, "ECON", _econ_row())
    mismatches = validate_smoke_outputs(sig_dir)
    assert any("NAIVE" in m for m in mismatches.get("naive", []))


@pytest.mark.parametrize("col,bad_value", [
    ("model_type",    "per_asset"),
    ("panel_mode",    "pooled"),
    ("train_window_mode", "expanding"),
    ("rolling_window_days", 90.0),
    ("hpo_enabled",   False),
    ("hpo_objective", "brier_score"),
])
def test_validator_flags_econ_misconfiguration(tmp_path, col, bad_value):
    sig_dir = tmp_path / "Signals"
    _write(sig_dir, "ECON", _econ_row(**{col: bad_value}))
    _write(sig_dir, "NAIVE", _naive_row())
    mismatches = validate_smoke_outputs(sig_dir)
    assert any(col in m for m in mismatches.get("econ", []))


def test_validator_flags_universe_mismatch(tmp_path):
    """ECON written with BTC/ETH; NAIVE written with a different universe
    hash — the cross-check must fail."""
    sig_dir = tmp_path / "Signals"
    _write(sig_dir, "ECON", _econ_row())
    # Tamper the NAIVE universe hash so it diverges from ECON.
    _write(sig_dir, "NAIVE", _naive_row(
        requested_coin_universe_hash=coin_universe_hash(("BTC", "ETH", "SOL")),
    ))
    mismatches = validate_smoke_outputs(sig_dir)
    assert any("requested_coin_universe_hash" in m
                for m in mismatches.get("cross", []))


def test_canonical_smoke_constants_are_pinned():
    assert CANONICAL_SMOKE["horizon"] == "1d"
    assert CANONICAL_SMOKE["model_type"] == "panel_logit"
    assert CANONICAL_SMOKE["panel_mode"] == "ticker_fixed_effects"
    assert CANONICAL_SMOKE["train_window_mode"] == "rolling_fixed"
    assert CANONICAL_SMOKE["rolling_window_days"] == 180.0
    assert CANONICAL_SMOKE["hpo_objective"] == "log_loss"
    assert CANONICAL_SMOKE["set_id_econ"] == "ECON"
    assert CANONICAL_SMOKE["set_id_naive"] == "NAIVE"
