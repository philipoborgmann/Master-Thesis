"""Tests for the v4 registry/market-cap/timing hardening (Sections C-H).

Sections:
* C — exact 17-set registry contract in validate_registry.
* D — pattern-based diagnostic-only feature detection.
* E — market-cap preprocessing (drop ≤0 / NaN / inf before log).
* F — effective market-cap source-lag audit.
* G — bar-grid regularity audit.
* H — completed-slot assumption + defensive warning.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import pytest

from thesis_pipeline.features import feature_registry as fr
from thesis_pipeline.diagnostics import timing_audit as ta
from thesis_pipeline.price import features as pf


# ---------------------------------------------------------------------------
# Section C — validate_registry: exact 17-set contract
# ---------------------------------------------------------------------------

def _full_v4_sets():
    """Return a minimally-valid 17-set dict (features = [set_id] each)."""
    return {sid: {"features": [f"feat_{sid}_a"]} for sid in fr.SET_ID_PATTERN}


def test_validate_registry_accepts_all_17_sets():
    assert fr.validate_registry(_full_v4_sets()) == []


def test_validate_registry_reports_missing_set():
    sets = _full_v4_sets()
    del sets["SENT_VAD_F"]
    problems = fr.validate_registry(sets)
    assert any("[shape] total count" in p for p in problems)
    assert any("missing required IDs" in p and "SENT_VAD_F" in p for p in problems)


def test_validate_registry_reports_extra_set():
    sets = _full_v4_sets()
    sets["EXTRA_BOGUS"] = {"features": ["x"]}
    problems = fr.validate_registry(sets)
    assert any("unexpected IDs" in p and "EXTRA_BOGUS" in p for p in problems)


def test_validate_registry_rejects_when_only_econ_present():
    sets = {"ECON": {"features": ["log_return_t"]}}
    problems = fr.validate_registry(sets)
    assert any("total count 1 != 17" in p for p in problems)
    assert any("missing required IDs" in p for p in problems)


def test_validate_registry_reports_duplicate_features():
    sets = _full_v4_sets()
    sets["ECON"] = {"features": ["a", "a", "b"]}
    problems = fr.validate_registry(sets)
    assert any("[content] ECON: duplicate features" in p for p in problems)


def test_validate_registry_reports_empty_feature_list():
    sets = _full_v4_sets()
    sets["ECON"] = {"features": []}
    problems = fr.validate_registry(sets)
    assert any("[content] ECON: empty feature list" in p for p in problems)


# ---------------------------------------------------------------------------
# Section D — pattern-based diagnostic-only feature detection
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("scorer", ["vader", "cryptobert"])
@pytest.mark.parametrize("stat", ["mean", "median", "std"])
def test_combined_score_variants_are_diagnostic_only(scorer, stat):
    f = f"{scorer}_combined_score_{stat}"
    assert fr.is_diagnostic_only_feature(f), f"{f} must be flagged diagnostic-only"


@pytest.mark.parametrize("scorer", ["vader", "cryptobert"])
@pytest.mark.parametrize("stat", ["mean", "median", "std"])
def test_selftext_score_variants_are_diagnostic_only(scorer, stat):
    f = f"{scorer}_selftext_score_{stat}"
    assert fr.is_diagnostic_only_feature(f)


@pytest.mark.parametrize("scorer", ["vader", "cryptobert"])
def test_directional_post_count_is_diagnostic_only(scorer):
    assert fr.is_diagnostic_only_feature(f"{scorer}_directional_post_count")


def test_has_posts_is_diagnostic_only():
    assert fr.is_diagnostic_only_feature("has_posts")


@pytest.mark.parametrize("scorer,stat", [
    ("vader", "mean"), ("cryptobert", "mean"),
    ("vader", "weighted_mean"), ("cryptobert", "weighted_mean"),
])
def test_weighted_mean_columns_are_diagnostic_only(scorer, stat):
    f = f"{scorer}_combined_score_{stat}"
    if stat == "weighted_mean":
        # Even at a fresh top-level (not combined_score_*), *_weighted_mean
        # is always rejected per Variante A.
        assert fr.is_diagnostic_only_feature(f"{scorer}_title_score_weighted_mean")


@pytest.mark.parametrize("good", [
    "log_return_t", "cum_log_return_7d", "realized_vol_14d",
    "vader_title_score_mean", "cryptobert_title_score_std",
    "vader_bullishness_ratio", "log1p_post_count",
])
def test_legitimate_v4_features_are_not_diagnostic(good):
    assert not fr.is_diagnostic_only_feature(good)


def test_diagnostic_only_columns_constant_consistent_with_helper():
    # Every entry in the exact list must agree with the helper.
    for name in fr.DIAGNOSTIC_ONLY_COLUMNS:
        assert fr.is_diagnostic_only_feature(name), name


def test_validate_registry_rejects_diagnostic_only_feature():
    sets = _full_v4_sets()
    sets["ECON"] = {"features": ["log_return_t", "has_posts"]}
    problems = fr.validate_registry(sets)
    assert any("diagnostic-only feature" in p and "has_posts" in p for p in problems)


# ---------------------------------------------------------------------------
# Section E — market-cap preprocessing (drop ≤0 / NaN / inf before log)
# ---------------------------------------------------------------------------

def _wide_cmc(values):
    """Build a minimal wide-format CMC market_cap frame with one ticker."""
    dates = pd.date_range("2024-01-01", periods=len(values), freq="D").date
    return pd.DataFrame({"date": dates, "0000000001_BTC": list(values)})


def test_build_market_cap_series_drops_nonpositive_and_nonfinite():
    cmc = _wide_cmc([1.0e9, 0.0, -5.0, np.nan, np.inf, -np.inf, 2.0e9])
    series_by_ticker, records = pf.build_market_cap_series(cmc, None, ["BTC"])

    btc = series_by_ticker["BTC"]
    # Only the two strictly positive finite values survive.
    assert len(btc) == 2
    assert (btc["market_cap"] > 0).all()
    assert np.isfinite(btc["log_market_cap_lag1"]).all()
    rec = next(r for r in records if r["ticker"] == "BTC")
    assert rec["n_invalid_market_cap_rows_raw"] == 5
    # No residual NaN propagated through log.
    assert rec["n_invalid_market_cap_after_log"] == 0


def test_log_market_cap_lag1_is_always_finite_after_build():
    cmc = _wide_cmc([1.0e8, 1.0e9, 1.0e10])
    series_by_ticker, _ = pf.build_market_cap_series(cmc, None, ["BTC"])
    btc = series_by_ticker["BTC"]
    assert np.isfinite(btc["log_market_cap_lag1"]).all()


def test_market_cap_invalid_rows_never_reach_features(tmp_path):
    # Build a tiny OHLCV file and a market-cap series containing one
    # zero and one NaN — the create_features_for_coin_horizon output
    # must never carry log_market_cap_lag1=±inf or NaN.
    rng = np.random.default_rng(0)
    n_days = 60
    ohlcv = pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=n_days, freq="D", tz="UTC"),
        "close":     20_000 + np.cumsum(rng.normal(0, 50, n_days)),
        "volume":    1_000 + np.abs(rng.normal(0, 50, n_days)),
    })
    price_dir = tmp_path / "Data" / "Raw" / "Price"
    (price_dir / "1d").mkdir(parents=True)
    ohlcv.to_parquet(price_dir / "1d" / "BTCUSDT_1d.parquet", index=False)

    cmc = _wide_cmc([1.0e9] * 30 + [0.0] + [np.nan] + [1.0e9] * (n_days - 32))
    series, _ = pf.build_market_cap_series(cmc, None, ["BTC"])

    out, _, _ = pf.create_features_for_coin_horizon(
        price_dir=price_dir, ticker="BTC", horizon="1d",
        market_cap_series=series["BTC"], winsor_p=0.005,
    )
    assert np.isfinite(out["log_market_cap_lag1"]).all()
    assert (out["log_market_cap_lag1"] > 0).all()  # log(1e9) > 0


# ---------------------------------------------------------------------------
# Section F — effective market-cap lag audit
# ---------------------------------------------------------------------------

def _make_features_with_market_cap(horizon: str, hours_per_bar: int):
    """Synthetic feature frame carrying market_cap_source_date /
    market_cap_available_at, so the audit can be exercised in isolation."""
    n = 30
    bar_freq = pd.Timedelta(hours=hours_per_bar)
    bar_ts = pd.date_range("2024-01-01", periods=n, freq=bar_freq, tz="UTC")
    # Each bar sees the previous day's market cap (D-1 source, D-1 + 1d
    # available_at = D 00:00 UTC).
    source_date = bar_ts.normalize() - pd.Timedelta(days=1)
    return pd.DataFrame({
        "timestamp":               bar_ts,
        "ticker":                  "BTC",
        "horizon":                 horizon,
        "market_cap_source_date":  source_date,
        "market_cap_available_at": source_date + pf.MARKET_CAP_AVAILABILITY_LAG,
    })


def test_market_cap_lag_audit_reports_strictly_before_for_intraday():
    df = _make_features_with_market_cap("6h", hours_per_bar=6)
    # Strip the rows where timestamp coincides with the
    # market_cap_available_at instant — those are exactly the rows that
    # merge_asof(..., allow_exact_matches=False) would have refused to
    # match in production, so they never make it into the feature frame
    # we audit here.
    strict = df["timestamp"] > df["market_cap_source_date"] + pf.MARKET_CAP_AVAILABILITY_LAG
    audit = ta.market_cap_lag_audit(df[strict])
    assert (audit["effective_source_lag_days"] >= 1.0).all()
    assert audit["available_strictly_before"].all()


def test_market_cap_lag_audit_flags_daily_midnight_two_day_lag():
    """A 1d bar at 00:00 UTC with D+1 00:00 availability gets the value
    from D-1 — i.e. effective lag = 2 source days. The audit must surface
    this."""
    bar_ts = pd.date_range("2024-01-03", periods=5, freq="D", tz="UTC")
    source_date = bar_ts.normalize() - pd.Timedelta(days=2)
    df = pd.DataFrame({
        "timestamp":               bar_ts,
        "ticker":                  "BTC",
        "horizon":                 "1d",
        "market_cap_source_date":  source_date,
        "market_cap_available_at": source_date + pf.MARKET_CAP_AVAILABILITY_LAG,
    })
    audit = ta.market_cap_lag_audit(df)
    # Use np.allclose (the pytest.approx form returns a single bool
    # against the whole Series, not an elementwise check).
    assert np.allclose(audit["effective_source_lag_days"].values, 2.0)
    summary = ta.market_cap_lag_summary(audit)
    assert summary["median_effective_market_cap_lag_days"] == pytest.approx(2.0)
    assert summary["share_effective_lag_gt_1_day"] == pytest.approx(1.0)


def test_market_cap_lag_audit_requires_columns():
    df = pd.DataFrame({"timestamp": [pd.Timestamp("2024-01-01", tz="UTC")],
                       "ticker": ["BTC"], "horizon": ["1d"]})
    with pytest.raises(ValueError, match="missing required columns"):
        ta.market_cap_lag_audit(df)


# ---------------------------------------------------------------------------
# Section G — bar-grid regularity audit
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("horizon,hours", [("1d", 24), ("6h", 6), ("1h", 1)])
def test_bar_grid_audit_regular_grid_reports_zero_missing(horizon, hours):
    ts = pd.date_range("2024-01-01", periods=30, freq=pd.Timedelta(hours=hours), tz="UTC")
    df = pd.DataFrame({"timestamp": ts, "ticker": "BTC", "horizon": horizon})
    out = ta.bar_grid_audit(df)
    assert len(out) == 1
    row = out.iloc[0]
    assert row["n_observed_bars"] == 30
    assert row["n_missing_expected_bars"] == 0
    assert row["median_timestamp_gap_secs"] == pytest.approx(hours * 3600.0)


def test_bar_grid_audit_detects_missing_bar():
    # Drop bar #3 → one missing canonical bar.
    ts_full = pd.date_range("2024-01-01", periods=10, freq="D", tz="UTC")
    ts = ts_full.delete(3)
    df = pd.DataFrame({"timestamp": ts, "ticker": "BTC", "horizon": "1d"})
    out = ta.bar_grid_audit(df).iloc[0]
    assert out["n_missing_expected_bars"] >= 1
    assert out["missing_bar_rate"] > 0


def test_bar_grid_audit_per_ticker_per_horizon():
    ts = pd.date_range("2024-01-01", periods=10, freq="D", tz="UTC")
    df = pd.DataFrame({
        "timestamp": list(ts) + list(ts),
        "ticker":    ["BTC"] * 10 + ["ETH"] * 10,
        "horizon":   ["1d"] * 20,
    })
    out = ta.bar_grid_audit(df)
    assert set(out["ticker"]) == {"BTC", "ETH"}


# ---------------------------------------------------------------------------
# Section H — completed-slot assumption + defensive warning
# ---------------------------------------------------------------------------

def test_assert_only_completed_slots_warns_when_cutoff_unknown(caplog):
    df = pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=3, freq="D", tz="UTC"),
        "ticker": "BTC",
        "post_count": [1, 2, 3],
    })
    with caplog.at_level(logging.WARNING, logger="thesis_pipeline.completed_slot"):
        out = ta.assert_only_completed_slots(df, horizon="1d",
                                             observation_cutoff=None)
    # No rows dropped, warning emitted (the assumption is unresolved).
    pd.testing.assert_frame_equal(out, df)
    assert any("REMAINING ASSUMPTION" in r.getMessage() for r in caplog.records)


def test_assert_only_completed_slots_drops_partial_final_slot():
    """A daily slot starting at 2024-01-05 ends at 2024-01-06 00:00 UTC.
    If the observation cutoff is 2024-01-05 12:00 UTC the slot is partial
    and must be dropped."""
    df = pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=5, freq="D", tz="UTC"),
        "ticker": "BTC",
        "post_count": [1, 2, 3, 4, 5],
    })
    out = ta.assert_only_completed_slots(
        df, horizon="1d",
        observation_cutoff=pd.Timestamp("2024-01-05 12:00:00", tz="UTC"),
    )
    assert pd.Timestamp("2024-01-05", tz="UTC") not in set(out["timestamp"])
    assert len(out) == 4


def test_completed_slot_assumption_documents_unresolved_marker():
    """The CompletedSlotAssumption class carries the migration text that
    surfaces this still-unresolved methodological choice."""
    msg = ta.CompletedSlotAssumption.UNRESOLVED_WARNING
    assert "REMAINING ASSUMPTION" in msg
    assert "observation cutoff" in msg
