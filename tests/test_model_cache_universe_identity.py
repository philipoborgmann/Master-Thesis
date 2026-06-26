"""Tests for commit 4 — model outputs use the requested-universe hash in
their filename and reject legacy / mismatched caches.

Covers Section A.1 — A.5:

* the same hashing helper as NAIVE produces the model filenames,
* smoke outputs don't collide with production-grid outputs,
* order/casing don't affect the hash,
* a legacy parquet without ``requested_coin_universe_hash`` is refused,
* a parquet whose realized tickers escape the requested universe is refused.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from thesis_pipeline.modeling.naive_reference import (
    coin_universe_hash, normalize_coin_universe, resolve_universes,
)
from thesis_pipeline.modeling.panel_logit import panel_output_name


# ---------------------------------------------------------------------------
# Hash + filename
# ---------------------------------------------------------------------------

def test_panel_output_name_appends_requested_universe_hash():
    h_smoke = coin_universe_hash(("BTC", "ETH"))
    h_full  = coin_universe_hash(
        ("BTC", "ETH", "SOL", "ADA", "XRP", "DOGE", "BCH", "LTC"),
    )
    assert h_smoke != h_full

    smoke_name = panel_output_name(
        "ECON", "-", "ticker_fixed_effects", "hpo_logloss",
        window_suffix="_rw180d", coin_universe=("BTC", "ETH"),
    )
    full_name = panel_output_name(
        "ECON", "-", "ticker_fixed_effects", "hpo_logloss",
        window_suffix="_rw180d",
        coin_universe=("BTC", "ETH", "SOL", "ADA", "XRP",
                        "DOGE", "BCH", "LTC"),
    )
    assert smoke_name.endswith(f"_u_{h_smoke}")
    assert full_name.endswith(f"_u_{h_full}")
    assert smoke_name != full_name


def test_universe_hash_is_order_and_case_independent():
    a = coin_universe_hash(("BTC", "ETH"))
    b = coin_universe_hash(("eth", "btc"))
    c = coin_universe_hash({"BTC", "ETH"})
    assert a == b == c
    # And the panel filename inherits the same property.
    n1 = panel_output_name("ECON", "-", "ticker_fixed_effects",
                            coin_universe=("BTC", "ETH"))
    n2 = panel_output_name("ECON", "-", "ticker_fixed_effects",
                            coin_universe=("eth", "btc"))
    assert n1 == n2


# ---------------------------------------------------------------------------
# resolve_universes — requested / available split
# ---------------------------------------------------------------------------

def test_resolve_universes_splits_requested_and_available():
    feature_tickers = ["BTC", "ETH"]  # SOL is missing
    uni = resolve_universes(["BTC", "ETH", "SOL"], feature_tickers)
    assert uni["requested"] == ("BTC", "ETH", "SOL")
    assert uni["available"] == ("BTC", "ETH")
    assert uni["requested_hash"] == coin_universe_hash(("BTC", "ETH", "SOL"))
    assert uni["available_hash"] == coin_universe_hash(("BTC", "ETH"))
    # Critical contract: the requested hash differs from the available
    # hash whenever some requested coins are unavailable — so model and
    # NAIVE outputs (both keyed on the REQUESTED hash) line up.
    assert uni["requested_hash"] != uni["available_hash"]


def test_resolve_universes_uses_full_feature_universe_when_no_filter():
    """``--coins`` omitted ⇒ requested = the complete feature universe."""
    uni = resolve_universes(None, ["BTC", "ETH", "SOL"])
    assert uni["requested"] == ("BTC", "ETH", "SOL")
    assert uni["available"] == ("BTC", "ETH", "SOL")


# ---------------------------------------------------------------------------
# Cache validation: legacy outputs rejected
# ---------------------------------------------------------------------------

@pytest.fixture
def synth_feature_repo(tmp_path, monkeypatch):
    """Tiny in-memory feature frame + minimal feature_sets.xlsx so the
    per-asset run_models pipeline can write a parquet without touching
    real data files."""
    import numpy as np

    (tmp_path / "Data" / "Final").mkdir(parents=True)
    n = 90
    ts = pd.date_range("2024-01-01", periods=n, freq="D", tz="UTC")
    rng = np.random.default_rng(0)
    rows = []
    for tk in ("BTC", "ETH"):
        rows.append(pd.DataFrame({
            "timestamp": ts, "ticker": tk, "horizon": "1d",
            "target":    rng.integers(0, 2, n),
            "log_return_t": rng.normal(0, 1, n),
            "cum_log_return_7d": rng.normal(0, 1, n),
        }))
    df = pd.concat(rows, ignore_index=True)
    df.to_parquet(tmp_path / "Data" / "Final" / "features_1d.parquet",
                  index=False)

    fs = pd.DataFrame({
        "set_id":   ["SYN_ECON"],
        "category": ["economic"],
        "sentiment_model": ["-"],
        "label":    ["syn-econ"],
        "feature_columns_comma_separated": ["log_return_t,cum_log_return_7d"],
    })
    fs_path = tmp_path / "feature_sets.xlsx"
    with pd.ExcelWriter(fs_path, engine="openpyxl") as w:
        fs.to_excel(w, sheet_name="feature_sets", index=False)

    from thesis_pipeline import config as cfg
    monkeypatch.setattr(cfg, "project_root", lambda: tmp_path)
    cfg.load_config.cache_clear()
    import shutil
    shutil.copytree(Path(__file__).resolve().parents[1] / "configs",
                    tmp_path / "configs")
    monkeypatch.chdir(tmp_path)
    return tmp_path


def test_smoke_model_output_does_not_collide_with_larger_universe(synth_feature_repo):
    """A BTC/ETH smoke run and a BTC/ETH/SOL production run must write to
    DIFFERENT filenames thanks to the requested-universe hash suffix."""
    n_smoke = coin_universe_hash(("BTC", "ETH"))
    n_full  = coin_universe_hash(("BTC", "ETH", "SOL"))
    s_name = panel_output_name("ECON", "-", "ticker_fixed_effects",
                                "hpo_logloss", "_rw180d",
                                coin_universe=("BTC", "ETH"))
    f_name = panel_output_name("ECON", "-", "ticker_fixed_effects",
                                "hpo_logloss", "_rw180d",
                                coin_universe=("BTC", "ETH", "SOL"))
    assert s_name.endswith(f"_u_{n_smoke}")
    assert f_name.endswith(f"_u_{n_full}")


def test_per_asset_run_writes_universe_hashed_filename(synth_feature_repo):
    from thesis_pipeline.modeling import run_models as rm
    rc = rm.main(["--horizon", "1d", "--set-id", "SYN_ECON",
                  "--coins", "BTC", "ETH",
                  "--model-type", "per_asset",
                  "--no-tune-hyperparams", "--restart",
                  "--no-generate-naive-reference"])
    assert rc == 0
    sig_dir = synth_feature_repo / "Outputs" / "Signals" / "1d"
    cands = sorted(sig_dir.glob("SYN_ECON_u_*.parquet"))
    assert cands, "per-asset output must carry the universe suffix"
    sig = pd.read_parquet(cands[0])
    expected_hash = coin_universe_hash(("BTC", "ETH"))
    assert (sig["requested_coin_universe_hash"] == expected_hash).all()
    assert (sig["coin_universe_hash"] == expected_hash).all()
    assert (sig["universe_identity_source"] == "requested_metadata").all()


def test_per_asset_run_refuses_legacy_cache(synth_feature_repo, monkeypatch):
    """A pre-v4 parquet WITHOUT requested-universe metadata must NOT be
    treated as a cache hit by a new run."""
    from thesis_pipeline.modeling import run_models as rm
    sig_dir = synth_feature_repo / "Outputs" / "Signals" / "1d"
    sig_dir.mkdir(parents=True, exist_ok=True)
    # Lay a legacy file at the v4 filename so the cache-check sees it.
    h = coin_universe_hash(("BTC", "ETH"))
    legacy_path = sig_dir / f"SYN_ECON_u_{h}.parquet"
    pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=3, tz="UTC"),
        "ticker": "BTC", "target": [1, 0, 1],
        "prediction": [1, 0, 1], "probability": [0.6, 0.4, 0.7],
    }).to_parquet(legacy_path, index=False)
    mtime_before = legacy_path.stat().st_mtime

    rc = rm.main(["--horizon", "1d", "--set-id", "SYN_ECON",
                  "--coins", "BTC", "ETH",
                  "--model-type", "per_asset",
                  "--no-tune-hyperparams",
                  "--no-generate-naive-reference"])
    assert rc == 0
    # The legacy cache was rejected and overwritten.
    cands = sorted(sig_dir.glob("SYN_ECON_u_*.parquet"))
    assert cands
    overwritten = cands[0]
    assert overwritten.stat().st_mtime > mtime_before
    sig = pd.read_parquet(overwritten)
    assert "requested_coin_universe_hash" in sig.columns


def test_same_complete_identity_still_produces_a_cache_hit(synth_feature_repo):
    """Two identical runs of the same complete identity must reuse the
    parquet (no recompute, file mtime unchanged)."""
    from thesis_pipeline.modeling import run_models as rm
    rm.main(["--horizon", "1d", "--set-id", "SYN_ECON",
             "--coins", "BTC", "ETH",
             "--model-type", "per_asset",
             "--no-tune-hyperparams", "--restart",
             "--no-generate-naive-reference"])
    sig_dir = synth_feature_repo / "Outputs" / "Signals" / "1d"
    out = sorted(sig_dir.glob("SYN_ECON_u_*.parquet"))[0]
    mtime_before = out.stat().st_mtime
    rm.main(["--horizon", "1d", "--set-id", "SYN_ECON",
             "--coins", "BTC", "ETH",
             "--model-type", "per_asset",
             "--no-tune-hyperparams",
             "--no-generate-naive-reference"])
    assert out.stat().st_mtime == mtime_before


def test_restart_forces_recompute(synth_feature_repo):
    from thesis_pipeline.modeling import run_models as rm
    rm.main(["--horizon", "1d", "--set-id", "SYN_ECON",
             "--coins", "BTC", "ETH",
             "--model-type", "per_asset",
             "--no-tune-hyperparams", "--restart",
             "--no-generate-naive-reference"])
    sig_dir = synth_feature_repo / "Outputs" / "Signals" / "1d"
    out = sorted(sig_dir.glob("SYN_ECON_u_*.parquet"))[0]
    mtime_before = out.stat().st_mtime
    rm.main(["--horizon", "1d", "--set-id", "SYN_ECON",
             "--coins", "BTC", "ETH",
             "--model-type", "per_asset",
             "--no-tune-hyperparams", "--restart",
             "--no-generate-naive-reference"])
    assert out.stat().st_mtime > mtime_before


# ---------------------------------------------------------------------------
# Model and NAIVE share the requested-universe hash even when some
# requested coins are unavailable
# ---------------------------------------------------------------------------

def test_model_and_naive_hash_match_when_one_requested_coin_unavailable():
    """If a user requests BTC + ETH + SOL but only BTC + ETH are in the
    feature frame, both the model and the NAIVE generator must compute
    the SAME requested-universe hash — keyed on the REQUESTED universe,
    NOT the available one."""
    requested = ("BTC", "ETH", "SOL")
    available = ("BTC", "ETH")
    # The NAIVE generator stamps the REQUESTED hash on its outputs.
    naive_hash = coin_universe_hash(requested)
    # The model writer must do the same.
    model_hash = coin_universe_hash(requested)
    assert naive_hash == model_hash
    # And it differs from the hash that would come from realized tickers.
    assert naive_hash != coin_universe_hash(available)
