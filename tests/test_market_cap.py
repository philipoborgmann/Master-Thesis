"""Unit tests for the market-cap stratification module.

Synthetic data only — no real CMC parquet required.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from thesis_pipeline.evaluation import market_cap as mc


# ---------------------------------------------------------------------------
# Loading + normalisation
# ---------------------------------------------------------------------------

def test_normalise_ticker_handles_aliases():
    assert mc._normalise_ticker("BTC")  == "BTC"
    assert mc._normalise_ticker("btc")  == "BTC"
    assert mc._normalise_ticker("NANO") == "NANO"
    assert mc._normalise_ticker("XNO")  == "NANO"
    assert mc._normalise_ticker("IOTA") == "IOTA"
    assert mc._normalise_ticker("MIOTA") == "IOTA"


def test_parse_cmc_column_handles_wide_format():
    assert mc._parse_cmc_column("000000000001_BTC")  == ("1", "BTC")
    assert mc._parse_cmc_column("000000001027_ETH")  == ("1027", "ETH")
    assert mc._parse_cmc_column("date") == (None, None)
    assert mc._parse_cmc_column("BTC")  == (None, None)


def test_load_market_cap_missing_file_returns_empty(tmp_path):
    out = mc.load_market_cap(tmp_path / "nonexistent.parquet")
    assert out.empty
    assert set(out.columns) == {"date", "ticker", "market_cap"}


def test_load_market_cap_wide_format_reshapes(tmp_path):
    dates = pd.date_range("2024-01-01", periods=5, tz="UTC")
    wide = pd.DataFrame({
        "date": dates,
        "000000000001_BTC":   [50_000, 51_000, 49_000, 52_000, 53_000],
        "000000001027_ETH":   [ 3_000,  3_100,  2_900,  3_200,  3_300],
        "000000001567_XNO":   [    50,     55,     60,     58,     57],
    })
    path = tmp_path / "market_cap.parquet"
    wide.to_parquet(path, index=False)

    long = mc.load_market_cap(path)
    assert not long.empty
    assert set(long["ticker"]) == {"BTC", "ETH", "NANO"}  # XNO → NANO
    assert pd.api.types.is_datetime64_any_dtype(long["date"])
    # Daily-midnight normalisation
    assert (long["date"].dt.hour == 0).all()


# ---------------------------------------------------------------------------
# Lookahead guard + tertile assignment
# ---------------------------------------------------------------------------

def test_normalize_market_cap_lags_by_one():
    mcap = pd.DataFrame({
        "date":       pd.date_range("2024-01-01", periods=4, tz="UTC"),
        "ticker":     "BTC",
        "market_cap": [100.0, 110.0, 105.0, 115.0],
    })
    out = mc.normalize_market_cap(mcap)
    assert pd.isna(out["market_cap_lag"].iloc[0])
    assert out["market_cap_lag"].iloc[1] == 100.0
    assert out["market_cap_lag"].iloc[2] == 110.0
    assert out["market_cap_lag"].iloc[3] == 105.0


def test_daily_tertiles_balanced():
    s = pd.Series([10, 50, 100, 150, 200, 250], dtype=float)
    reg = mc._daily_tertiles(s)
    counts = reg.value_counts()
    assert counts.to_dict() == {"small": 2, "mid": 2, "large": 2}
    assert reg.iloc[0]  == "small"
    assert reg.iloc[-1] == "large"


def test_build_market_cap_lookup_no_lookahead(tmp_path):
    # On every day, BTC is twice as large as ETH which is twice as large as
    # XRP — the lagged regime on day t should reflect day t-1's relative
    # ordering, not day t's.
    dates = pd.date_range("2024-01-01", periods=4, tz="UTC")
    wide = pd.DataFrame({
        "date": dates,
        "000000000001_BTC": [400, 410, 420, 430],
        "000000001027_ETH": [200, 210, 220, 230],
        "000000000052_XRP": [100, 110, 120, 130],
    })
    path = tmp_path / "market_cap.parquet"
    wide.to_parquet(path, index=False)

    lookup = mc.build_market_cap_lookup(path)
    # Day 0 has no lag → no rows in output.
    day1 = lookup[lookup["date"] == dates[1]].set_index("ticker")
    assert day1.loc["BTC", "mcap_regime"] == "large"
    assert day1.loc["ETH", "mcap_regime"] == "mid"
    assert day1.loc["XRP", "mcap_regime"] == "small"


# ---------------------------------------------------------------------------
# Join + stratification
# ---------------------------------------------------------------------------

def _synthetic_signals():
    dates = pd.date_range("2024-01-01", periods=12, tz="UTC")
    sigs = []
    rng = np.random.default_rng(0)
    for tk in ("BTC", "ETH", "XRP"):
        target = rng.integers(0, 2, size=len(dates))
        sigs.append(pd.DataFrame({
            "timestamp":   dates,
            "ticker":      tk,
            "target":      target,
            "prediction":  target,
            "probability": rng.uniform(0.5, 0.9, size=len(dates)),
            "set_id":      "S1",
            "sentiment_model": "vader",
            "horizon":     "1d",
            "category":    "sentiment",
            "label":       "test",
        }))
    return pd.concat(sigs, ignore_index=True)


def _synthetic_mcap_lookup():
    # Includes a leading prior day so the availability-based as-of join
    # (commit 3 Section E) finds an admissible regime for the first
    # signal — strict-< excludes same-day availability instants.
    dates = pd.date_range("2023-12-31", periods=13, tz="UTC")
    frames = []
    for tk, lag_val, regime in (("BTC", 400.0, "large"),
                                ("ETH", 200.0, "mid"),
                                ("XRP", 100.0, "small")):
        frames.append(pd.DataFrame({
            "date":            dates,
            "ticker":          tk,
            "market_cap_lag":  lag_val,
            "mcap_regime":     regime,
        }))
    return pd.concat(frames, ignore_index=True)


def test_attach_market_cap_regimes_joins_per_date_ticker():
    sig = _synthetic_signals()
    lookup = _synthetic_mcap_lookup()
    merged = mc.attach_market_cap_regimes(sig, lookup)
    assert "mcap_regime" in merged.columns
    assert merged["mcap_regime"].notna().all()
    # BTC rows must all be large
    assert (merged.loc[merged["ticker"] == "BTC", "mcap_regime"] == "large").all()


def test_market_cap_stratification_table_has_all_regimes():
    sig = _synthetic_signals()
    lookup = _synthetic_mcap_lookup()
    out = mc.market_cap_stratification_table(sig, lookup)
    assert not out.empty
    assert set(out["mcap_regime"]) == {"small", "mid", "large"}
    for col in ("accuracy", "n_obs", "n_days", "n_tickers"):
        assert col in out.columns


def test_regime_interaction_table_covers_all_combinations(monkeypatch):
    sig = _synthetic_signals()
    mcap_lookup = _synthetic_mcap_lookup()
    # Build a synthetic vol lookup: BTC high, ETH mid, XRP low — exactly the
    # same shape as the volatility module would return.
    vol = []
    dates = pd.date_range("2024-01-01", periods=12, tz="UTC")
    for tk, regime in (("BTC", "high"), ("ETH", "mid"), ("XRP", "low")):
        vol.append(pd.DataFrame({
            "ticker": tk, "date": dates,
            "gk_var": 0.001, "vol": 0.01, "regime": regime,
        }))
    vol_lookup = pd.concat(vol, ignore_index=True)

    out = mc.regime_interaction_table(sig, mcap_lookup, vol_lookup)
    assert not out.empty
    assert set(out["mcap_regime"]) == {"small", "mid", "large"}
    assert set(out["vol_regime"])  == {"low", "mid", "high"}
    # The small-cap × high-vol cell pertains to XRP (small) + BTC (high);
    # there is no ticker that is BOTH small AND high in this synthetic data
    # so the cell exists but with n_obs = 0.
    cell = out[(out["mcap_regime"] == "small") & (out["vol_regime"] == "high")]
    assert not cell.empty
    assert int(cell["n_obs"].iloc[0]) == 0


def test_build_market_cap_summary_picks_best_small_cap():
    strat = pd.DataFrame({
        "horizon":         ["1d", "1d", "1d"],
        "set_id":          ["S1", "S2", "S3"],
        "sentiment_model": ["vader", "vader", "vader"],
        "mcap_regime":     ["small", "small", "small"],
        "accuracy":        [0.55, 0.60, 0.52],
        "n_obs":           [100, 100, 100],
    })
    out = mc.build_market_cap_summary(strat, pd.DataFrame())
    assert any(r["section"] == "best_small_cap_model" and r["set_id"] == "S2"
               for r in out)
