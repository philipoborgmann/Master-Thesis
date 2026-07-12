"""Forecast-origin sample: horizon offsets, derivation, boundaries, filtering,
metrics, loader compatibility, checkpoint invalidation (Objective B / Sec 12)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from thesis_pipeline.modeling.forecast_sample import (
    FORECAST_ORIGIN_COLUMN, add_forecast_origin, forecast_sample_signature,
    horizon_offset, load_forecast_sample_config, restrict_to_forecast_sample,
    sample_bounds, validate_forecast_origin,
)

# The canonical production window (independent of the test-session env default).
PROD_CFG = {
    "basis": "forecast_origin",
    "start": "2022-01-01T00:00:00Z",
    "end_exclusive": "2023-01-01T00:00:00Z",
    "write_forecast_origin_column": True,
    "enabled": True,
}


def _rows(horizon, raw_ts):
    return pd.DataFrame({
        "timestamp": [pd.Timestamp(t) for t in raw_ts],
        "ticker": ["ADA"] * len(raw_ts),
        "target": [1] * len(raw_ts),
        "prediction": [1] * len(raw_ts),
        "probability": [0.6] * len(raw_ts),
        "horizon": [horizon] * len(raw_ts),
    })


# --- horizon offset mapping + tz handling -----------------------------------

def test_horizon_offset_mapping():
    assert horizon_offset("1h") == pd.Timedelta(hours=1)
    assert horizon_offset("6h") == pd.Timedelta(hours=6)
    assert horizon_offset("1d") == pd.Timedelta(days=1)
    assert horizon_offset("1D") == pd.Timedelta(days=1)  # case-insensitive


def test_unsupported_horizon_fails_clearly():
    with pytest.raises(ValueError, match="Unsupported horizon"):
        horizon_offset("4h")


def test_naive_timestamps_normalised_to_utc():
    df = pd.DataFrame({"timestamp": [pd.Timestamp("2022-06-01 00:00")],  # naive
                       "horizon": ["1d"]})
    out = add_forecast_origin(df, "1d")
    assert out["timestamp"].dt.tz is not None
    assert out[FORECAST_ORIGIN_COLUMN].dt.tz is not None
    assert out[FORECAST_ORIGIN_COLUMN].iloc[0] == pd.Timestamp("2022-06-02", tz="UTC")


def test_forecast_origin_is_timestamp_plus_h():
    for hz, delta in (("1h", pd.Timedelta(hours=1)),
                      ("6h", pd.Timedelta(hours=6)),
                      ("1d", pd.Timedelta(days=1))):
        df = _rows(hz, ["2022-06-01 00:00"])
        out = add_forecast_origin(df, hz)
        assert out[FORECAST_ORIGIN_COLUMN].iloc[0] == pd.Timestamp("2022-06-01", tz="UTC") + delta


# --- boundaries (start-inclusive, end-exclusive, per-horizon year-end) -------

@pytest.mark.parametrize("hz,raw,retain", [
    # end boundary (end-exclusive)
    ("1h", "2022-12-31 22:00", True),
    ("1h", "2022-12-31 23:00", False),
    ("6h", "2022-12-31 12:00", True),
    ("6h", "2022-12-31 18:00", False),
    ("1d", "2022-12-30 00:00", True),
    ("1d", "2022-12-31 00:00", False),
    # start boundary (start-inclusive)
    ("1d", "2021-12-31 00:00", True),   # fo = 2022-01-01 00:00 -> retain
    ("1d", "2021-12-30 00:00", False),  # fo = 2021-12-31 -> exclude
    ("1h", "2021-12-31 23:00", True),   # fo = 2022-01-01 00:00 -> retain
    ("1h", "2021-12-31 22:00", False),  # fo = 2021-12-31 23:00 -> exclude
])
def test_forecast_origin_boundaries(hz, raw, retain):
    out = restrict_to_forecast_sample(_rows(hz, [raw]), hz, cfg=PROD_CFG)
    assert (len(out) == 1) is retain
    if retain:
        assert FORECAST_ORIGIN_COLUMN in out.columns


def test_filter_never_uses_raw_timestamp():
    # A 1d row at raw 2022-12-31 has forecast_origin 2023-01-01 -> excluded,
    # even though the RAW timestamp is inside 2022. Proves filtering is on the
    # forecast origin, not the raw timestamp.
    out = restrict_to_forecast_sample(_rows("1d", ["2022-12-31 00:00"]), "1d",
                                      cfg=PROD_CFG)
    assert len(out) == 0


def test_disabled_window_stamps_but_keeps_all_rows():
    cfg = dict(PROD_CFG, enabled=False)
    out = restrict_to_forecast_sample(_rows("1d", ["2024-05-01 00:00"]), "1d",
                                      cfg=cfg)
    assert len(out) == 1
    assert FORECAST_ORIGIN_COLUMN in out.columns


# --- validation of a supplied forecast_origin -------------------------------

def test_supplied_forecast_origin_mismatch_detected():
    df = _rows("1d", ["2022-06-01 00:00"])
    df[FORECAST_ORIGIN_COLUMN] = pd.Timestamp("2022-06-05", tz="UTC")  # wrong
    ok = validate_forecast_origin(df, "1d")
    assert not ok.all()
    # correct value validates
    df2 = add_forecast_origin(_rows("1d", ["2022-06-01 00:00"]), "1d")
    assert validate_forecast_origin(df2, "1d").all()


# --- signature invalidation -------------------------------------------------

def test_signature_changes_with_window_and_basis():
    base = forecast_sample_signature(PROD_CFG)
    assert base == forecast_sample_signature(dict(PROD_CFG))
    assert base != forecast_sample_signature(dict(PROD_CFG, start="2021-01-01T00:00:00Z"))
    assert base != forecast_sample_signature(dict(PROD_CFG, end_exclusive="2024-01-01T00:00:00Z"))
    assert base != forecast_sample_signature(dict(PROD_CFG, basis="raw_timestamp"))


def test_config_defaults_match_2022_window():
    cfg = load_forecast_sample_config()
    start, end = sample_bounds(cfg)
    assert start == pd.Timestamp("2022-01-01", tz="UTC")
    assert end == pd.Timestamp("2023-01-01", tz="UTC")
    assert cfg["basis"] == "forecast_origin"
