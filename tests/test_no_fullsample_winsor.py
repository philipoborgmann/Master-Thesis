"""Feature generation no longer performs full-sample winsorisation
(task Section 5, criteria 5-7)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from thesis_pipeline.price import features as pf
from thesis_pipeline.sentiment import aggregate as agg


def _synthetic_ohlcv(n=400, seed=0):
    rng = np.random.default_rng(seed)
    ts = pd.date_range("2022-01-01", periods=n, freq="D", tz="UTC")
    steps = rng.normal(0, 0.01, n)
    # Inject a few extreme returns that full-sample winsorisation WOULD clip.
    steps[100] = 0.9
    steps[200] = -0.8
    close = 100 * np.exp(np.cumsum(steps))
    vol = np.abs(rng.normal(1000, 50, n))
    vol[100] = 1e6  # extreme volume jump
    return pd.DataFrame({"timestamp": ts, "close": close, "volume": vol})


def _mc_series(ohlcv):
    """Minimal always-available market-cap series so rows survive the dropna."""
    dates = pd.date_range(ohlcv["timestamp"].min() - pd.Timedelta(days=5),
                          ohlcv["timestamp"].max() + pd.Timedelta(days=1),
                          freq="D", tz="UTC")
    return pd.DataFrame({
        "market_cap_source_date": dates,
        "market_cap_available_at": dates + pd.Timedelta(days=1),
        "market_cap": 1e9,
        "log_market_cap_lag1": np.log(1e9),
    })


# --- criterion 5: price feature generation no longer clips ------------------

def test_price_log_return_is_raw_not_winsorised():
    ohlcv = _synthetic_ohlcv()
    out, report, thresholds = pf.create_features_for_coin_horizon(
        price_dir=None, ticker="ADA", horizon="1d",
        market_cap_series=_mc_series(ohlcv), winsor_p=0.005, ohlcv_override=ohlcv,
    )
    # No winsorisation thresholds are produced any more.
    assert thresholds == []
    # log_return_t equals the RAW log return exactly (extreme survives).
    raw = np.log(ohlcv["close"] / ohlcv["close"].shift(1))
    merged = out.merge(
        pd.DataFrame({"timestamp": ohlcv["timestamp"], "raw": raw.values}),
        on="timestamp", how="left")
    assert np.allclose(merged["log_return_t"].to_numpy(),
                       merged["raw"].to_numpy(), equal_nan=True)
    # The injected extreme return (~0.9) is present, unclipped.
    assert out["log_return_t"].max() > 0.5
    assert out["log_return_t"].min() < -0.5


def test_price_generator_writes_no_winsorization_thresholds_csv(tmp_path):
    # winsorize_series helper must be gone (misleading full-sample artefact).
    assert not hasattr(pf, "winsorize_series")


# --- criterion 7: rolling features are built from RAW returns ---------------

def test_rolling_features_are_from_raw_returns():
    ohlcv = _synthetic_ohlcv()
    out, _, _ = pf.create_features_for_coin_horizon(
        price_dir=None, ticker="ADA", horizon="1d",
        market_cap_series=_mc_series(ohlcv), ohlcv_override=ohlcv,
    )
    raw = np.log(ohlcv["close"] / ohlcv["close"].shift(1))
    ref = pd.DataFrame({"timestamp": ohlcv["timestamp"], "raw": raw.values})
    ref["cum7_raw"] = ref["raw"].rolling(7, min_periods=7).sum()
    ref["rv14_raw"] = ref["raw"].rolling(14, min_periods=14).std()
    m = out.merge(ref, on="timestamp", how="left")
    assert np.allclose(m["cum_log_return_7d"].to_numpy(),
                       m["cum7_raw"].to_numpy(), equal_nan=True)
    assert np.allclose(m["realized_vol_14d"].to_numpy(),
                       m["rv14_raw"].to_numpy(), equal_nan=True)


# --- criterion 6: sentiment aggregation no longer clips ---------------------

def test_sentiment_module_has_no_fullsample_winsoriser():
    assert not hasattr(agg, "winsorize_features")
    assert not hasattr(agg, "WINSOR_LOWER")
    assert not hasattr(agg, "WINSOR_UPPER")


def test_sentiment_aggregation_preserves_extreme_scores():
    # Two posts in one slot with one extreme score → the slot mean must equal
    # the raw arithmetic mean (no clipping).
    df = pd.DataFrame({
        "date": pd.to_datetime(["2022-01-01 00:10", "2022-01-01 00:20"], utc=True),
        "ticker": ["ADA", "ADA"],
        "vader_title_sentiment": ["positive", "negative"],
        "vader_title_score": [0.9, -5.0],   # -5 is an out-of-range extreme
        "cryptobert_title_sentiment": ["positive", "negative"],
        "cryptobert_title_score": [0.8, -0.8],
    })
    out = agg.aggregate_to_horizon(df, "1h", "1h")
    row = out.iloc[0]
    assert row["vader_title_score_mean"] == (0.9 + -5.0) / 2  # unclipped
