"""Regression tests for the strict market-cap as-of merge dtype contract
(commit 9).

The production failure was:

    pandas.errors.MergeError: incompatible merge keys [0]
    datetime64[ms, UTC] and datetime64[us, UTC], must be the same type

The fix normalises BOTH join keys to ``datetime64[ns, UTC]`` immediately
before :func:`pd.merge_asof` and asserts the dtypes match. These tests
pin every input flavour that has surfaced in real data: ms, us, ns,
strings, ISO datetimes, NaT, and a round-tripped Parquet fixture.

Strict-< availability semantics (``direction="backward"``,
``allow_exact_matches=False``) MUST still hold after the fix.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from thesis_pipeline.price import features as pf


# ---------------------------------------------------------------------------
# Tiny helpers
# ---------------------------------------------------------------------------

def _ohlcv(timestamps) -> pd.DataFrame:
    n = len(timestamps)
    rng = np.random.default_rng(0)
    close = 100 + np.cumsum(rng.normal(0, 0.5, n))
    return pd.DataFrame({
        "timestamp": pd.Index(timestamps),
        "open":   close + rng.normal(0, 0.05, n),
        "high":   close + np.abs(rng.normal(0, 0.10, n)),
        "low":    close - np.abs(rng.normal(0, 0.10, n)),
        "close":  close,
        "volume": np.abs(rng.normal(1000, 50, n)),
    })


def _mcap(timestamps, *, lag_days: int = 1) -> pd.DataFrame:
    """Synthetic CMC long-form series. ``timestamps`` is treated as the
    source-date column; availability is source + ``lag_days`` (matching
    :data:`pf.MARKET_CAP_AVAILABILITY_LAG` semantics)."""
    source = pd.Index(timestamps)
    available_at = pd.Index(timestamps) + pd.Timedelta(days=lag_days)
    n = len(source)
    return pd.DataFrame({
        "date":                    source,
        "market_cap_source_date":  source,
        "market_cap_available_at": available_at,
        "market_cap":              np.linspace(1e9, 1.2e9, n),
        "log_market_cap_lag1":     np.linspace(20.7, 21.0, n),
    })


# ---------------------------------------------------------------------------
# _normalize_utc_ns — the canonical normaliser
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("unit", ["ms", "us", "ns"])
def test_normalize_utc_ns_accepts_every_utc_resolution(unit):
    s = pd.Series(pd.date_range("2024-01-01", periods=5, freq="D", tz="UTC")
                    .astype(f"datetime64[{unit}, UTC]"))
    out = pf._normalize_utc_ns(s)
    assert str(out.dtype) == "datetime64[ns, UTC]"
    # Values preserved.
    pd.testing.assert_series_equal(
        out, s.astype("datetime64[ns, UTC]"), check_names=False,
    )


def test_normalize_utc_ns_accepts_naive_strings():
    """ISO strings (no tz) are interpreted as UTC per the pipeline's
    documented convention."""
    s = pd.Series(["2024-01-01 00:00:00", "2024-01-02 12:34:56"])
    out = pf._normalize_utc_ns(s)
    assert str(out.dtype) == "datetime64[ns, UTC]"
    assert out.iloc[0] == pd.Timestamp("2024-01-01 00:00:00", tz="UTC")


def test_normalize_utc_ns_preserves_nat():
    s = pd.Series([pd.Timestamp("2024-01-01", tz="UTC"), pd.NaT,
                    pd.Timestamp("2024-01-03", tz="UTC")])
    out = pf._normalize_utc_ns(s)
    assert str(out.dtype) == "datetime64[ns, UTC]"
    assert pd.isna(out.iloc[1])
    assert out.iloc[0] == pd.Timestamp("2024-01-01", tz="UTC")


def test_normalize_utc_ns_does_not_strip_timezone():
    s = pd.Series(pd.date_range("2024-01-01", periods=3, freq="D", tz="UTC"))
    out = pf._normalize_utc_ns(s)
    assert out.dt.tz is not None
    assert str(out.dt.tz) == "UTC"


# ---------------------------------------------------------------------------
# Mixed-resolution merge — the original failure
# ---------------------------------------------------------------------------

def _run_merge(left_unit: str, right_unit: str, n_days: int = 60) -> pd.DataFrame:
    """Run the full ``create_features_for_coin_horizon`` against an
    OHLCV / CMC pair whose join-key resolutions differ.

    ``n_days`` defaults to 60 so the 21-day cum_log_return window
    leaves enough surviving rows for the assertions downstream — a
    realistic dimension since the user reported the bug on the real
    multi-year price archive.
    """
    days = pd.date_range("2024-01-10", periods=n_days, freq="D", tz="UTC")
    ohlcv = _ohlcv(days)
    ohlcv["timestamp"] = ohlcv["timestamp"].astype(f"datetime64[{left_unit}, UTC]")
    mc_days = pd.date_range("2024-01-01", periods=n_days, freq="D", tz="UTC")
    mc = _mcap(mc_days)
    mc["market_cap_available_at"] = mc["market_cap_available_at"].astype(
        f"datetime64[{right_unit}, UTC]"
    )
    mc["market_cap_source_date"] = mc["market_cap_source_date"].astype(
        f"datetime64[{right_unit}, UTC]"
    )
    out, _, _ = pf.create_features_for_coin_horizon(
        price_dir=None,
        ticker="BTC", horizon="1d",
        market_cap_series=mc,
        winsor_p=0.005,
        ohlcv_override=ohlcv,
    )
    return out


def test_case_a_ms_left_us_right_no_merge_error():
    """The exact pair that triggered the user's Windows traceback."""
    out = _run_merge(left_unit="ms", right_unit="us")
    assert "market_cap_available_at" in out.columns
    matched = out["market_cap_available_at"].notna()
    assert matched.any(), "at least one row must match"
    assert str(out["timestamp"].dtype) == "datetime64[ns, UTC]"
    assert str(out["market_cap_available_at"].dtype) == "datetime64[ns, UTC]"


def test_case_b_us_left_ns_right_no_merge_error():
    out = _run_merge(left_unit="us", right_unit="ns")
    matched = out["market_cap_available_at"].notna()
    assert matched.any()


def test_case_c_string_loaded_left_against_parquet_right(tmp_path):
    """Real-data case: the OHLCV timestamp arrives as ISO strings from
    a CSV / JSON dump while the CMC parquet carries a native datetime64
    dtype."""
    days = pd.date_range("2024-01-10", periods=60, freq="D", tz="UTC")
    ohlcv = _ohlcv(days)
    ohlcv["timestamp"] = ohlcv["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S")
    mc = _mcap(pd.date_range("2024-01-01", periods=60, freq="D", tz="UTC"))
    # Round-trip via Parquet so the dtype is whatever pyarrow chooses.
    p = tmp_path / "mc.parquet"
    mc.to_parquet(p, index=False)
    mc = pd.read_parquet(p)

    out, _, _ = pf.create_features_for_coin_horizon(
        price_dir=None,
        ticker="BTC", horizon="1d",
        market_cap_series=mc,
        winsor_p=0.005,
        ohlcv_override=ohlcv,
    )
    matched = out["market_cap_available_at"].notna()
    assert matched.any()
    assert str(out["timestamp"].dtype) == "datetime64[ns, UTC]"
    assert str(out["market_cap_available_at"].dtype) == "datetime64[ns, UTC]"


def test_case_d_nat_values_do_not_corrupt_matches():
    days = pd.date_range("2024-01-10", periods=60, freq="D", tz="UTC")
    ohlcv = _ohlcv(days)
    ohlcv["timestamp"] = ohlcv["timestamp"].astype("datetime64[ms, UTC]")
    mc = _mcap(pd.date_range("2024-01-01", periods=60, freq="D", tz="UTC"))
    mc.loc[3, "market_cap_available_at"] = pd.NaT  # tamper one row
    mc["market_cap_available_at"] = mc["market_cap_available_at"].astype(
        "datetime64[us, UTC]"
    )
    out, _, _ = pf.create_features_for_coin_horizon(
        price_dir=None,
        ticker="BTC", horizon="1d",
        market_cap_series=mc,
        winsor_p=0.005,
        ohlcv_override=ohlcv,
    )
    # Surviving rows obey the strict-< rule.
    matched = out["market_cap_available_at"].notna()
    assert (out.loc[matched, "market_cap_available_at"]
            < out.loc[matched, "timestamp"]).all()


# ---------------------------------------------------------------------------
# Strict-< availability — direction='backward', allow_exact_matches=False
# ---------------------------------------------------------------------------

def test_strict_lt_relationship_after_merge():
    out = _run_merge(left_unit="ms", right_unit="us")
    matched = out["market_cap_available_at"].notna()
    # Production semantics: strictly before.
    assert (out.loc[matched, "market_cap_available_at"]
            < out.loc[matched, "timestamp"]).all()
    # And the leakage assertion accepts the frame.
    from thesis_pipeline.diagnostics.leakage_checks import (
        assert_market_cap_asof_correct,
    )
    out["ticker"] = "BTC"
    assert_market_cap_asof_correct(out)


def test_exact_match_is_rejected_by_strict_backward():
    """A market-cap availability instant exactly equal to a price
    timestamp MUST NOT match — ``allow_exact_matches=False``."""
    # Same daily grid on both sides → every availability_at coincides
    # exactly with a price timestamp on the same index. Use 60 days so
    # the 21-day cum-return window keeps the boundary row alive.
    days = pd.date_range("2024-01-01", periods=60, freq="D", tz="UTC")
    ohlcv = _ohlcv(days)
    ohlcv["timestamp"] = ohlcv["timestamp"].astype("datetime64[ms, UTC]")
    mc = pd.DataFrame({
        "date":                    days,
        "market_cap_source_date":  days,
        "market_cap_available_at": days,        # equal to the price ts
        "market_cap":              np.linspace(1e9, 1.2e9, 60),
        "log_market_cap_lag1":     np.linspace(20.7, 21.0, 60),
    })
    mc["market_cap_available_at"] = mc["market_cap_available_at"].astype(
        "datetime64[us, UTC]"
    )
    out, _, _ = pf.create_features_for_coin_horizon(
        price_dir=None,
        ticker="BTC", horizon="1d",
        market_cap_series=mc,
        winsor_p=0.005,
        ohlcv_override=ohlcv,
    )
    # Every matched row obeys strict <. There must NOT be any row where
    # the timestamp equals its market_cap_available_at.
    matched = out["market_cap_available_at"].notna()
    assert matched.any(), "the 60-day fixture must produce some matches"
    assert (out.loc[matched, "market_cap_available_at"]
            < out.loc[matched, "timestamp"]).all()
    equal_match = (out.loc[matched, "market_cap_available_at"]
                   == out.loc[matched, "timestamp"]).any()
    assert not equal_match, (
        "allow_exact_matches=False was bypassed — a same-instant "
        "availability slipped into a matched row"
    )


# ---------------------------------------------------------------------------
# Output dtype + parquet round-trip
# ---------------------------------------------------------------------------

def test_output_columns_are_canonical_ns_utc(tmp_path):
    """In-memory merge inputs land on ``datetime64[ns, UTC]``; after a
    parquet round-trip both timestamp columns remain semantically UTC
    even if the backend serialises at a different resolution."""
    out = _run_merge(left_unit="ms", right_unit="us")
    out["ticker"] = "BTC"
    p = tmp_path / "price_features.parquet"
    out.to_parquet(p, index=False)
    rt = pd.read_parquet(p)
    # ``timestamp`` and ``market_cap_available_at`` are tz-aware UTC.
    for col in ("timestamp", "market_cap_available_at"):
        s = pd.to_datetime(rt[col], utc=True, errors="coerce")
        assert s.dt.tz is not None
        assert str(s.dt.tz) == "UTC"
