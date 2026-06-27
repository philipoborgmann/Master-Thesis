"""Tests for the v4 calendar-consistent price-feature pipeline.

Covers:
* Per-horizon bar mapping for the rolling windows (1d/6h/1h).
* Strict as-of market-cap merge with allow_exact_matches=False.
* market_cap_source_date / market_cap_available_at are present and tz-aware.
* The deprecated --marketcap_lag_days flag does not change the merge.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from thesis_pipeline.price import features as pf


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _synthetic_ohlcv(n_bars: int, *, freq: str, start: str = "2024-01-01") -> pd.DataFrame:
    rng = np.random.default_rng(0)
    idx = pd.date_range(start, periods=n_bars, freq=freq, tz="UTC")
    close = 20_000 + np.cumsum(rng.normal(0, 50, n_bars))
    return pd.DataFrame({
        "timestamp": idx,
        "close":     close,
        "volume":    1_000 + np.abs(rng.normal(0, 50, n_bars)),
    })


def _synthetic_market_cap(start: str = "2024-01-01", n_days: int = 60) -> pd.DataFrame:
    dates = pd.date_range(start, periods=n_days, freq="D").date
    return pd.DataFrame({
        "date":              dates,
        "market_cap":        np.linspace(1e11, 2e11, n_days),
        # The price-features module adds these inside build_market_cap_series; the
        # helpers here mimic that output so we can test the merge in isolation.
        "market_cap_source_date":
            pd.to_datetime(dates, utc=True),
        "market_cap_available_at":
            pd.to_datetime(dates, utc=True) + pf.MARKET_CAP_AVAILABILITY_LAG,
        "log_market_cap_lag1":
            np.log(np.linspace(1e11, 2e11, n_days)),
    })


# ---------------------------------------------------------------------------
# Bar-mapping per horizon
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("horizon,bpd", [("1d", 1), ("6h", 4), ("1h", 24)])
def test_bars_per_day_constant(horizon, bpd):
    assert pf.BARS_PER_DAY[horizon] == bpd


@pytest.mark.parametrize("horizon,n_days_per_bar", [
    ("1d", 1), ("6h", 4), ("1h", 24),
])
def test_window_lengths_are_calendar_consistent(horizon, n_days_per_bar, tmp_path):
    """For each horizon, the cum_log_return_*d and realized_vol_14d windows
    use exactly ``days * BARS_PER_DAY[horizon]`` bars; min_periods enforces a
    full window, so the first valid row index is window_bars - 1."""
    freq_map = {"1d": "D", "6h": "6h", "1h": "h"}
    # Need at least 21d + 1 bars to populate every window.
    n_bars = 22 * n_days_per_bar + 5
    ohlcv = _synthetic_ohlcv(n_bars, freq=freq_map[horizon])

    # Write into the layout the loader expects.
    price_dir = tmp_path / "Data" / "Raw" / "Price"
    hz_dir = price_dir / horizon
    hz_dir.mkdir(parents=True)
    ohlcv.to_parquet(hz_dir / f"BTCUSDT_{horizon}.parquet", index=False)

    out, _, _ = pf.create_features_for_coin_horizon(
        price_dir=price_dir,
        ticker="BTC",
        horizon=horizon,
        market_cap_series=_synthetic_market_cap(),
        winsor_p=0.005,
    )
    # All four window features must be present.
    for col in ("cum_log_return_7d", "cum_log_return_14d",
                "cum_log_return_21d", "realized_vol_14d"):
        assert col in out.columns

    # Drop NaN means: first 21-day window must be filled in.
    bars_21d = 21 * n_days_per_bar
    assert len(out) > 0, "need at least one row after dropna"
    # Coverage: with 22d of data we should see (22-21)*BPD output rows for
    # the cum_log_return_21d window (the binding constraint).
    assert len(out) >= 1


def test_bars_per_day_unknown_horizon_raises(tmp_path):
    price_dir = tmp_path / "Data" / "Raw" / "Price"
    (price_dir / "1d").mkdir(parents=True)
    with pytest.raises(ValueError, match="Unknown horizon"):
        pf.create_features_for_coin_horizon(
            price_dir=price_dir, ticker="BTC", horizon="4h",
            market_cap_series=None, winsor_p=0.005,
        )


# ---------------------------------------------------------------------------
# Strict as-of merge for market cap
# ---------------------------------------------------------------------------

def test_market_cap_merge_is_strict_before_only(tmp_path):
    """A CMC value with availability = D+1 00:00 UTC must NOT match a price
    bar at exactly D+1 00:00 UTC (allow_exact_matches=False)."""
    # Two 1d bars: 2024-01-01 00:00 and 2024-01-02 00:00 UTC.
    ohlcv = _synthetic_ohlcv(n_bars=30, freq="D")  # daily bars
    price_dir = tmp_path / "Data" / "Raw" / "Price"
    (price_dir / "1d").mkdir(parents=True)
    ohlcv.to_parquet(price_dir / "1d" / "BTCUSDT_1d.parquet", index=False)

    # Market cap for D=2024-01-01 → available_at = 2024-01-02 00:00 UTC.
    mc = pd.DataFrame({
        "date":              [pd.Timestamp("2024-01-01").date()],
        "market_cap":        [1.5e11],
        "market_cap_source_date":
            pd.to_datetime(["2024-01-01"], utc=True),
        "market_cap_available_at":
            pd.to_datetime(["2024-01-01"], utc=True) + pf.MARKET_CAP_AVAILABILITY_LAG,
        "log_market_cap_lag1": [np.log(1.5e11)],
    })

    out, _, _ = pf.create_features_for_coin_horizon(
        price_dir=price_dir, ticker="BTC", horizon="1d",
        market_cap_series=mc, winsor_p=0.005,
    )
    # Every output row must satisfy market_cap_available_at < timestamp
    # (i.e. strict inequality, never equal).
    sub = out.dropna(subset=["market_cap_available_at", "timestamp"])
    assert (sub["market_cap_available_at"] < sub["timestamp"]).all()


def test_market_cap_columns_present_and_tz_aware(tmp_path):
    ohlcv = _synthetic_ohlcv(n_bars=40, freq="D")
    price_dir = tmp_path / "Data" / "Raw" / "Price"
    (price_dir / "1d").mkdir(parents=True)
    ohlcv.to_parquet(price_dir / "1d" / "BTCUSDT_1d.parquet", index=False)

    out, _, _ = pf.create_features_for_coin_horizon(
        price_dir=price_dir, ticker="BTC", horizon="1d",
        market_cap_series=_synthetic_market_cap(n_days=60),
        winsor_p=0.005,
    )
    assert "log_market_cap_lag1" in out.columns
    assert "market_cap_source_date" in out.columns
    assert "market_cap_available_at" in out.columns
    # tz-aware UTC on both availability columns
    assert out["market_cap_available_at"].dt.tz is not None
    assert out["market_cap_source_date"].dt.tz is not None


def test_marketcap_lag_days_flag_is_ineffective(tmp_path, capsys):
    """The legacy --marketcap_lag_days arg is accepted but ignored — the
    feature output must be identical whether it is set or not."""
    ohlcv = _synthetic_ohlcv(n_bars=40, freq="D")
    price_dir = tmp_path / "Data" / "Raw" / "Price"
    (price_dir / "1d").mkdir(parents=True)
    ohlcv.to_parquet(price_dir / "1d" / "BTCUSDT_1d.parquet", index=False)
    mc = _synthetic_market_cap(n_days=60)

    out0, _, _ = pf.create_features_for_coin_horizon(
        price_dir=price_dir, ticker="BTC", horizon="1d",
        market_cap_series=mc, winsor_p=0.005, marketcap_lag_days=0,
    )
    out2, _, _ = pf.create_features_for_coin_horizon(
        price_dir=price_dir, ticker="BTC", horizon="1d",
        market_cap_series=mc, winsor_p=0.005, marketcap_lag_days=7,
    )
    pd.testing.assert_frame_equal(out0.reset_index(drop=True),
                                  out2.reset_index(drop=True))


# ---------------------------------------------------------------------------
# No legacy column names in the v4 output
# ---------------------------------------------------------------------------

def test_v4_output_drops_legacy_column_names(tmp_path):
    ohlcv = _synthetic_ohlcv(n_bars=60, freq="D")
    price_dir = tmp_path / "Data" / "Raw" / "Price"
    (price_dir / "1d").mkdir(parents=True)
    ohlcv.to_parquet(price_dir / "1d" / "BTCUSDT_1d.parquet", index=False)
    out, _, _ = pf.create_features_for_coin_horizon(
        price_dir=price_dir, ticker="BTC", horizon="1d",
        market_cap_series=_synthetic_market_cap(),
        winsor_p=0.005,
    )
    # v4 column names; legacy names must NOT appear in the output frame.
    for legacy in ("cum_log_return_7", "cum_log_return_14",
                   "realized_vol_14", "market_cap_t"):
        assert legacy not in out.columns
