"""Tests for the v4 rolling-window output metadata (commit 10).

The smoke validator demanded a ``rolling_window_days`` column on the
ECON parquet — the existing pipeline emitted
``train_window_timestamps`` only, which is horizon-specific and not a
valid alias on 6h / 1h runs. The fix attaches the rolling-window
configuration at the canonical assembly site
(:func:`thesis_pipeline.modeling.panel_logit._stamp_rolling_window_metadata`)
so freshly computed AND checkpoint-resumed paths both carry the
contract.

The horizon-specific equality ``timestamps == days * BARS_PER_DAY[hz]``
is the one source of truth for the 1d / 6h / 1h conversion.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from thesis_pipeline.diagnostics.smoke_validation import (
    CANONICAL_SMOKE, validate_smoke_outputs,
)
from thesis_pipeline.modeling.naive_reference import coin_universe_hash
from thesis_pipeline.modeling.panel_logit import (
    ROLLING_WINDOW_METADATA_COLUMNS,
    _assert_training_window_contract,
    _stamp_rolling_window_metadata,
)
from thesis_pipeline.price.features import BARS_PER_DAY


def _bare_signals(n=4):
    """A minimal panel_logit assembly-stage frame (post-concat,
    pre-metadata-stamp)."""
    return pd.DataFrame({
        "timestamp":   pd.date_range("2024-01-01", periods=n, freq="D", tz="UTC"),
        "ticker":      "BTC",
        "target":      [0, 1, 0, 1],
        "prediction":  [0, 1, 0, 1],
        "probability": [0.4, 0.6, 0.3, 0.7],
        "train_window_mode":        "rolling_fixed",
        "train_window_timestamps":  10,
        "train_start_timestamp":    pd.Timestamp("2023-07-01", tz="UTC"),
        "train_end_timestamp":      pd.Timestamp("2023-12-31", tz="UTC"),
    })


# ---------------------------------------------------------------------------
# Stamp helper — horizon-specific timestamps from days × BARS_PER_DAY
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("horizon,expected_ts", [
    ("1d",   180),
    ("6h",   720),
    ("1h",  4320),
])
def test_rolling_window_days_180_maps_to_horizon_specific_timestamps(
        horizon, expected_ts):
    out = _stamp_rolling_window_metadata(
        _bare_signals(),
        horizon=horizon,
        train_window_mode="rolling_fixed",
        rolling_window_days=180.0,
        rolling_window_timestamps=None,
    )
    assert (out["rolling_window_days"] == 180.0).all()
    assert (out["rolling_window_timestamps"] == expected_ts).all()
    assert (out["train_window_mode"] == "rolling_fixed").all()


@pytest.mark.parametrize("horizon,expected_ts", [
    ("1d",   120),
    ("6h",   480),
    ("1h",  2880),
])
def test_rolling_window_days_120_proves_no_hardcoded_180(horizon, expected_ts):
    """Section 9 Test D: a non-default window must produce horizon-
    correct timestamps. Hardcoding 180 anywhere would fail this."""
    out = _stamp_rolling_window_metadata(
        _bare_signals(),
        horizon=horizon,
        train_window_mode="rolling_fixed",
        rolling_window_days=120.0,
        rolling_window_timestamps=None,
    )
    assert (out["rolling_window_days"] == 120.0).all()
    assert (out["rolling_window_timestamps"] == expected_ts).all()


def test_stamp_does_not_derive_days_from_timestamps():
    """``rolling_window_days`` MUST come from the run configuration.
    Passing only ``rolling_window_timestamps`` leaves the day column
    explicitly NaN — the helper never back-computes."""
    out = _stamp_rolling_window_metadata(
        _bare_signals(),
        horizon="1d",
        train_window_mode="rolling_fixed",
        rolling_window_days=None,
        rolling_window_timestamps=30,
    )
    assert out["rolling_window_days"].isna().all()
    assert (out["rolling_window_timestamps"] == 30).all()


def test_expanding_mode_leaves_window_columns_nan():
    out = _stamp_rolling_window_metadata(
        _bare_signals(),
        horizon="1d",
        train_window_mode="expanding",
        rolling_window_days=None,
        rolling_window_timestamps=None,
    )
    assert (out["train_window_mode"] == "expanding").all()
    assert out["rolling_window_days"].isna().all()


# ---------------------------------------------------------------------------
# Output-schema contract assertion
# ---------------------------------------------------------------------------

def test_assert_contract_passes_on_canonical_v4_smoke():
    out = _stamp_rolling_window_metadata(
        _bare_signals(), horizon="1d",
        train_window_mode="rolling_fixed",
        rolling_window_days=180.0, rolling_window_timestamps=None,
    )
    _assert_training_window_contract(out, horizon="1d",
                                       train_window_mode="rolling_fixed")


def test_assert_contract_rejects_missing_column():
    out = _stamp_rolling_window_metadata(
        _bare_signals(), horizon="1d",
        train_window_mode="rolling_fixed",
        rolling_window_days=180.0, rolling_window_timestamps=None,
    ).drop(columns=["rolling_window_days"])
    with pytest.raises(AssertionError, match="rolling_window_days"):
        _assert_training_window_contract(out, horizon="1d",
                                           train_window_mode="rolling_fixed")


def test_assert_contract_rejects_non_constant_days():
    out = _stamp_rolling_window_metadata(
        _bare_signals(), horizon="1d",
        train_window_mode="rolling_fixed",
        rolling_window_days=180.0, rolling_window_timestamps=None,
    )
    out.loc[0, "rolling_window_days"] = 60.0  # tamper
    with pytest.raises(AssertionError):
        _assert_training_window_contract(out, horizon="1d",
                                           train_window_mode="rolling_fixed")


def test_assert_contract_rejects_zero_or_negative():
    out = _stamp_rolling_window_metadata(
        _bare_signals(), horizon="1d",
        train_window_mode="rolling_fixed",
        rolling_window_days=180.0, rolling_window_timestamps=None,
    )
    out["rolling_window_days"] = 0.0
    with pytest.raises(AssertionError):
        _assert_training_window_contract(out, horizon="1d",
                                           train_window_mode="rolling_fixed")


def test_assert_contract_rejects_horizon_mismatch():
    """6h with days=180 must produce timestamps=720. A tampered row
    where the two diverge must fail the consistency rule."""
    out = _stamp_rolling_window_metadata(
        _bare_signals(), horizon="6h",
        train_window_mode="rolling_fixed",
        rolling_window_days=180.0, rolling_window_timestamps=None,
    )
    out["rolling_window_timestamps"] = 360  # wrong: should be 720
    with pytest.raises(AssertionError, match="BARS_PER_DAY"):
        _assert_training_window_contract(out, horizon="6h",
                                           train_window_mode="rolling_fixed")


# ---------------------------------------------------------------------------
# Smoke-validator integration
# ---------------------------------------------------------------------------

def _smoke_econ_row(**overrides):
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
        "requested_coin_universe_hash":
            coin_universe_hash(("BTC", "ETH")),
    }
    base.update(overrides)
    return base


def _smoke_naive_row(**overrides):
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
        "requested_coin_universe_hash":
            coin_universe_hash(("BTC", "ETH")),
    }
    base.update(overrides)
    return base


def _layout(tmp_path, econ_row, naive_row):
    sig_dir = tmp_path / "Signals"
    horizon_dir = sig_dir / "1d"
    horizon_dir.mkdir(parents=True)
    h = coin_universe_hash(("BTC", "ETH"))
    pd.DataFrame([econ_row] * 3).to_parquet(
        horizon_dir / f"ECON_panel_ticker_fe_hpo_logloss_rw180d_u_{h}.parquet",
        index=False,
    )
    pd.DataFrame([naive_row] * 3).to_parquet(
        horizon_dir / f"NAIVE_panel_ticker_fe_rw180d_u_{h}.parquet",
        index=False,
    )
    return sig_dir


def test_smoke_validator_passes_with_rolling_window_days(tmp_path):
    sig_dir = _layout(tmp_path, _smoke_econ_row(), _smoke_naive_row())
    assert validate_smoke_outputs(sig_dir, expected_tickers=("BTC", "ETH")) == {}


def test_smoke_validator_reports_missing_rolling_window_days(tmp_path):
    """Reproduces the exact error the user observed locally."""
    econ_row = _smoke_econ_row()
    del econ_row["rolling_window_days"]
    sig_dir = _layout(tmp_path, econ_row, _smoke_naive_row())
    out = validate_smoke_outputs(sig_dir, expected_tickers=("BTC", "ETH"))
    assert any("rolling_window_days" in m and "missing" in m
                for m in out.get("econ", []))


def test_smoke_validator_flags_non_numeric_rolling_window_days(tmp_path):
    sig_dir = _layout(tmp_path,
                        _smoke_econ_row(rolling_window_days="not-a-number"),
                        _smoke_naive_row())
    out = validate_smoke_outputs(sig_dir, expected_tickers=("BTC", "ETH"))
    assert any("rolling_window_days" in m for m in out.get("econ", []))


def test_smoke_validator_flags_non_positive_rolling_window_days(tmp_path):
    sig_dir = _layout(tmp_path,
                        _smoke_econ_row(rolling_window_days=0.0),
                        _smoke_naive_row())
    out = validate_smoke_outputs(sig_dir, expected_tickers=("BTC", "ETH"))
    assert any("rolling_window_days" in m for m in out.get("econ", []))


# ---------------------------------------------------------------------------
# Checkpoint / cache assembly path
# ---------------------------------------------------------------------------

def test_metadata_stamped_on_checkpoint_resumed_assembly():
    """Simulate the assembly site receiving chunks produced WITHOUT the
    rolling-window metadata (legacy / pre-fix checkpoints). After the
    canonical stamp it must satisfy the contract."""
    legacy = _bare_signals()
    # Strip the column that legacy chunks lacked.
    legacy = legacy.drop(columns=["rolling_window_days"], errors="ignore")
    stamped = _stamp_rolling_window_metadata(
        legacy, horizon="1d",
        train_window_mode="rolling_fixed",
        rolling_window_days=180.0, rolling_window_timestamps=None,
    )
    assert (stamped["rolling_window_days"] == 180.0).all()
    _assert_training_window_contract(stamped, horizon="1d",
                                       train_window_mode="rolling_fixed")


def test_canonical_smoke_constants_pin_rolling_window_days_180():
    assert float(CANONICAL_SMOKE["rolling_window_days"]) == 180.0
