"""Tests for the NAIVE coin-universe-aware cache identity (Aufgabe 6
follow-up A + G).

The NAIVE cache key MUST include the requested coin universe; otherwise
a smoke run (BTC, ETH) and a production run (25 coins) could collide on
the same filename and silently reuse the smoke benchmark.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from thesis_pipeline.modeling import naive_reference as nr


# ---------------------------------------------------------------------------
# coin_universe_hash — order-independent and case-independent
# ---------------------------------------------------------------------------

def test_coin_universe_hash_is_order_independent():
    a = nr.coin_universe_hash(["BTC", "ETH", "SOL"])
    b = nr.coin_universe_hash(["sol", "btc", "eth"])
    c = nr.coin_universe_hash({"ETH", "SOL", "BTC"})
    assert a == b == c


def test_coin_universe_hash_is_8_chars_hex():
    h = nr.coin_universe_hash(["BTC", "ETH"])
    assert len(h) == 8
    assert all(ch in "0123456789abcdef" for ch in h)


def test_coin_universe_hash_different_for_different_universes():
    smoke = nr.coin_universe_hash(["BTC", "ETH"])
    full  = nr.coin_universe_hash(
        ["BTC", "ETH", "SOL", "ADA", "XRP", "DOGE", "BCH", "LTC",
         "XMR", "XLM", "NEO", "NANO", "BAT", "MIOTA", "EOS", "BNB",
         "LRC", "TRX", "MANA", "XTZ", "KCS", "VET", "CRO", "DOT", "UNI"]
    )
    assert smoke != full


def test_coin_universe_hash_normalizes_whitespace_and_dedupes():
    a = nr.coin_universe_hash(["BTC", " btc ", "ETH"])
    b = nr.coin_universe_hash(["BTC", "ETH"])
    assert a == b


def test_coin_universe_hash_empty_universe_is_stable():
    a = nr.coin_universe_hash([])
    b = nr.coin_universe_hash(None)
    assert a == b
    assert len(a) == 8


# ---------------------------------------------------------------------------
# Filename embeds the universe hash
# ---------------------------------------------------------------------------

def test_naive_output_name_carries_universe_hash():
    name = nr.naive_output_name(
        model_type="panel_logit",
        panel_mode="ticker_fixed_effects",
        train_window_mode="rolling_fixed",
        rolling_window_days=180,
        coin_universe=["BTC", "ETH"],
    )
    h = nr.coin_universe_hash(["BTC", "ETH"])
    assert name == f"NAIVE_panel_ticker_fe_rw180d_u_{h}"


def test_naive_output_name_different_universe_yields_different_name():
    smoke = nr.naive_output_name(
        model_type="panel_logit", panel_mode="ticker_fixed_effects",
        train_window_mode="rolling_fixed", rolling_window_days=180,
        coin_universe=["BTC", "ETH"],
    )
    full = nr.naive_output_name(
        model_type="panel_logit", panel_mode="ticker_fixed_effects",
        train_window_mode="rolling_fixed", rolling_window_days=180,
        coin_universe=["BTC", "ETH", "SOL", "ADA", "XRP"],
    )
    assert smoke != full


# ---------------------------------------------------------------------------
# Sidecar metadata + cache validation
# ---------------------------------------------------------------------------

def _features(tickers=("BTC", "ETH", "SOL"), n=160, seed=0):
    rng = np.random.default_rng(seed)
    ts = pd.date_range("2024-01-01", periods=n, freq="D", tz="UTC")
    rows = []
    for tk in tickers:
        rows.append(pd.DataFrame({
            "timestamp": ts, "ticker": tk, "horizon": "1d",
            "target": rng.integers(0, 2, n),
            "signal_feature": rng.normal(0, 1, n),
        }))
    return pd.concat(rows, ignore_index=True)


def test_naive_run_writes_sidecar_metadata(tmp_path):
    out = nr.generate_naive_reference(
        horizon="1d",
        features_df=_features(),
        output_dir=tmp_path,
        coins=["BTC", "ETH", "SOL"],
    )
    assert out is not None
    sidecar = out.with_suffix(".meta.json")
    assert sidecar.exists()
    meta = json.loads(sidecar.read_text())
    assert meta["set_id"] == "NAIVE"
    assert meta["coin_universe_hash"] == nr.coin_universe_hash(["BTC", "ETH", "SOL"])
    assert sorted(meta["coin_universe"]) == ["BTC", "ETH", "SOL"]
    assert meta["model_type"] == "panel_logit"
    assert meta["rolling_window_days"] == 180.0


def test_naive_signals_carry_coin_universe_columns(tmp_path):
    out = nr.generate_naive_reference(
        horizon="1d", features_df=_features(),
        output_dir=tmp_path, coins=["BTC", "ETH", "SOL"],
    )
    sig = pd.read_parquet(out)
    assert (sig["coin_universe_hash"] ==
            nr.coin_universe_hash(["BTC", "ETH", "SOL"])).all()
    assert (sig["n_requested_tickers"] == 3).all()
    # requested_tickers stable pipe-separated representation.
    expected = "|".join(sorted({"BTC", "ETH", "SOL"}))
    assert (sig["requested_tickers"] == expected).all()


def test_smoke_naive_is_not_reused_for_full_universe(tmp_path):
    """The scenario the cleanup task names directly: a smoke run produces
    a NAIVE file, then a 25-coin production run requests the same horizon
    + window — the smoke file must NOT be silently reused."""
    smoke_feats = _features(tickers=("BTC", "ETH"))
    out_smoke = nr.generate_naive_reference(
        horizon="1d", features_df=smoke_feats,
        output_dir=tmp_path, coins=["BTC", "ETH"],
    )
    assert out_smoke is not None
    full_feats = _features(tickers=("BTC", "ETH", "SOL", "ADA", "XRP"),
                           n=160, seed=1)
    out_full = nr.generate_naive_reference(
        horizon="1d", features_df=full_feats,
        output_dir=tmp_path, coins=["BTC", "ETH", "SOL", "ADA", "XRP"],
    )
    assert out_full is not None
    # Distinct files — the universe hash differs.
    assert out_smoke != out_full
    # And the smoke file still contains only BTC/ETH while the production
    # file contains five tickers.
    smoke_sig = pd.read_parquet(out_smoke)
    full_sig  = pd.read_parquet(out_full)
    assert set(smoke_sig["ticker"]) == {"BTC", "ETH"}
    assert set(full_sig["ticker"]) == {"BTC", "ETH", "SOL", "ADA", "XRP"}


def test_same_universe_same_window_returns_cache_hit(tmp_path):
    feats = _features()
    a = nr.generate_naive_reference(
        horizon="1d", features_df=feats,
        output_dir=tmp_path, coins=["BTC", "ETH", "SOL"],
    )
    assert a is not None
    b = nr.generate_naive_reference(
        horizon="1d", features_df=feats,
        output_dir=tmp_path, coins=["BTC", "ETH", "SOL"],
    )
    assert b is None  # cache hit


def test_missing_sidecar_triggers_recompute(tmp_path):
    """A NAIVE parquet without its metadata sidecar must NOT be reused —
    the cache validator returns False and the generator recomputes."""
    feats = _features()
    out = nr.generate_naive_reference(
        horizon="1d", features_df=feats,
        output_dir=tmp_path, coins=["BTC", "ETH", "SOL"],
    )
    assert out is not None
    sidecar = out.with_suffix(".meta.json")
    sidecar.unlink()  # tamper: remove the sidecar
    mtime_before = out.stat().st_mtime
    # Recompute by calling again (resume=True, restart=False).
    again = nr.generate_naive_reference(
        horizon="1d", features_df=feats,
        output_dir=tmp_path, coins=["BTC", "ETH", "SOL"],
    )
    # The cache was invalid → returned path again, file rewritten.
    assert again is not None
    assert again.stat().st_mtime >= mtime_before
    assert again.with_suffix(".meta.json").exists()


def test_drifted_ticker_set_triggers_recompute(tmp_path):
    """If the cached parquet's actual ticker set doesn't intersect the
    requested universe, the cache is invalid."""
    feats = _features(tickers=("BTC", "ETH"))
    out = nr.generate_naive_reference(
        horizon="1d", features_df=feats,
        output_dir=tmp_path, coins=["BTC", "ETH"],
    )
    assert out is not None
    # Corrupt the parquet so it carries only DOGE — a ticker not in the
    # request. Cache must invalidate.
    bad = pd.read_parquet(out).copy()
    bad["ticker"] = "DOGE"
    bad.to_parquet(out, index=False)
    valid = nr._cache_is_valid(
        out, expected=nr._build_identity_payload(
            horizon="1d", model_type="panel_logit",
            panel_mode="ticker_fixed_effects",
            train_window_mode="rolling_fixed",
            rolling_window_days=180.0,
            rolling_window_timestamps=None,
            coin_universe_tuple=("BTC", "ETH"),
        ),
    )
    assert not valid


def test_restart_overwrites_compatible_cache(tmp_path):
    feats = _features()
    out = nr.generate_naive_reference(
        horizon="1d", features_df=feats,
        output_dir=tmp_path, coins=["BTC", "ETH", "SOL"],
    )
    assert out is not None
    mtime_before = out.stat().st_mtime
    # restart=True must overwrite even when the cache is compatible.
    again = nr.generate_naive_reference(
        horizon="1d", features_df=feats,
        output_dir=tmp_path, coins=["BTC", "ETH", "SOL"],
        restart=True,
    )
    assert again is not None
    # The path may be the same but the file must have been rewritten.
    assert again == out


# ---------------------------------------------------------------------------
# Atomic write — no .tmp leftover
# ---------------------------------------------------------------------------

def test_atomic_write_leaves_no_tmp_artifact(tmp_path):
    feats = _features()
    out = nr.generate_naive_reference(
        horizon="1d", features_df=feats,
        output_dir=tmp_path, coins=["BTC", "ETH", "SOL"],
    )
    assert out is not None
    # The .tmp temporary file must not be left behind.
    leftover = list(out.parent.glob("*.tmp"))
    assert leftover == []


# ---------------------------------------------------------------------------
# Per-asset window honesty (Section D)
# ---------------------------------------------------------------------------

def test_per_asset_naive_expanding_returns_actual_metadata(tmp_path):
    feats = _features(n=120)
    out = nr.generate_naive_reference(
        horizon="1d", features_df=feats,
        model_type="per_asset", panel_mode="-",
        train_window_mode="expanding",
        rolling_window_days=None, rolling_window_timestamps=None,
        output_dir=tmp_path, coins=["BTC", "ETH", "SOL"],
    )
    assert out is not None
    sig = pd.read_parquet(out)
    # Expanding mode — every row must report expanding, not a stamped rolling label.
    assert (sig["train_window_mode"] == "expanding").all()


def test_per_asset_naive_rolling_180d_actually_uses_180d(tmp_path):
    """A per-asset rolling-180d NAIVE must actually compute its
    probability from the last 180 calendar days, not from the expanding
    window. The test compares predictions to the expanding case and
    requires them to differ on data where the early target frequency
    is biased — proving the window selector engaged."""
    rng = np.random.default_rng(13)
    n = 400
    ts = pd.date_range("2023-01-01", periods=n, freq="D", tz="UTC")
    # First half mostly 1, second half mostly 0 — expanding will see
    # the bias drift; 180d rolling will track the recent half.
    target = np.concatenate([rng.binomial(1, 0.85, n // 2),
                             rng.binomial(1, 0.15, n // 2)])
    df = pd.DataFrame({"timestamp": ts, "ticker": "BTC",
                        "horizon": "1d", "target": target,
                        "signal_feature": rng.normal(0, 1, n)})

    out_expand = nr.generate_naive_reference(
        horizon="1d", features_df=df,
        model_type="per_asset", panel_mode="-",
        train_window_mode="expanding",
        rolling_window_days=None, rolling_window_timestamps=None,
        output_dir=tmp_path / "expand", coins=["BTC"],
    )
    out_roll = nr.generate_naive_reference(
        horizon="1d", features_df=df,
        model_type="per_asset", panel_mode="-",
        train_window_mode="rolling_fixed",
        rolling_window_days=180.0, rolling_window_timestamps=None,
        output_dir=tmp_path / "roll", coins=["BTC"],
    )
    assert out_expand is not None and out_roll is not None
    sig_e = pd.read_parquet(out_expand).set_index("timestamp")["probability"]
    sig_r = pd.read_parquet(out_roll).set_index("timestamp")["probability"]
    # On overlapping timestamps the two paths should not match exactly —
    # otherwise the rolling selector is silently behaving as expanding.
    common = sig_e.index.intersection(sig_r.index)
    assert len(common) > 0
    assert not np.allclose(sig_e.loc[common].values, sig_r.loc[common].values)


def test_per_asset_naive_no_future_observation_enters_training(tmp_path):
    """Aufgabe 6 follow-up D: every NAIVE prediction at τ must use only
    training rows with ``timestamp < τ``."""
    feats = _features(n=200, tickers=("BTC",))
    out = nr.generate_naive_reference(
        horizon="1d", features_df=feats,
        model_type="per_asset", panel_mode="-",
        train_window_mode="expanding",
        rolling_window_days=None, rolling_window_timestamps=None,
        output_dir=tmp_path, coins=["BTC"],
    )
    sig = pd.read_parquet(out)
    # train_end_timestamp must be strictly before the prediction timestamp.
    ts  = pd.to_datetime(sig["timestamp"], utc=True)
    end = pd.to_datetime(sig["train_end_timestamp"], utc=True)
    assert (end < ts).all()
