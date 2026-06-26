"""Commit 4 Section A.4 — checkpoint identity covers the requested universe.

A BTC/ETH smoke checkpoint must NEVER resume a BTC/ETH/SOL production
run. The compatibility guard refuses reuse on any mismatch in:

* requested_coin_universe_hash / requested_tickers
* horizon / set_id / sentiment_model
* model_type / panel_mode / hpo_variant / hpo_objective
* feature_cols
* train_window_mode / rolling_window_*

Tests exercise both the panel-chunk path and the per-asset ticker path.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from thesis_pipeline.modeling import checkpointing as ckpt
from thesis_pipeline.modeling.naive_reference import (
    coin_universe_hash, normalize_coin_universe,
)
from thesis_pipeline.modeling.run_models import _guard_checkpoint_universe


def _base(*, horizon="1d", set_id="ECON", sentiment_model="-",
          model_type="per_asset", panel_mode="-",
          hpo_variant="fixed", hpo_objective="-",
          feature_cols=("log_return_t",),
          requested=("BTC", "ETH")) -> dict:
    return {
        "horizon": horizon, "set_id": set_id,
        "sentiment_model": sentiment_model,
        "model_type": model_type, "panel_mode": panel_mode,
        "hpo_variant": hpo_variant, "hpo_objective": hpo_objective,
        "feature_cols": list(feature_cols),
        "requested_tickers":            list(requested),
        "requested_coin_universe_hash": coin_universe_hash(requested),
        "n_requested_tickers":          len(requested),
    }


def test_guard_clears_when_requested_universe_grows(tmp_path):
    root = tmp_path / "ckpt"
    root.mkdir()
    # Stage 1 — BTC/ETH checkpoint persisted.
    smoke_base = _base(requested=("BTC", "ETH"))
    ckpt.init_manifest(root, base=smoke_base)
    # Lay a fake chunk so we can confirm it is cleared.
    chunk_path = ckpt.chunk_checkpoint_path(root, 0)
    chunk_path.parent.mkdir(parents=True, exist_ok=True)
    chunk_path.write_text("placeholder")
    assert chunk_path.exists()

    # Stage 2 — larger universe arrives. The guard MUST clear.
    full_base = _base(requested=("BTC", "ETH", "SOL"))
    _guard_checkpoint_universe(ckpt, root, full_base)
    assert not chunk_path.exists(), "stale BTC/ETH checkpoint must be cleared"


def test_guard_keeps_checkpoint_for_same_universe(tmp_path):
    root = tmp_path / "ckpt"
    root.mkdir()
    base = _base(requested=("BTC", "ETH"))
    ckpt.init_manifest(root, base=base)
    chunk = ckpt.chunk_checkpoint_path(root, 0)
    chunk.parent.mkdir(parents=True, exist_ok=True)
    chunk.write_text("placeholder")
    _guard_checkpoint_universe(ckpt, root, base)
    assert chunk.exists()


def test_guard_clears_on_horizon_change(tmp_path):
    root = tmp_path / "ckpt"
    root.mkdir()
    ckpt.init_manifest(root, base=_base(horizon="1d"))
    chunk = ckpt.chunk_checkpoint_path(root, 0)
    chunk.parent.mkdir(parents=True, exist_ok=True)
    chunk.write_text("placeholder")
    _guard_checkpoint_universe(ckpt, root, _base(horizon="6h"))
    assert not chunk.exists()


def test_guard_clears_on_feature_set_change(tmp_path):
    root = tmp_path / "ckpt"
    root.mkdir()
    ckpt.init_manifest(root, base=_base(set_id="ECON"))
    chunk = ckpt.chunk_checkpoint_path(root, 0)
    chunk.parent.mkdir(parents=True, exist_ok=True)
    chunk.write_text("placeholder")
    _guard_checkpoint_universe(ckpt, root, _base(set_id="ECON_VAD_F"))
    assert not chunk.exists()


def test_guard_clears_on_panel_mode_change(tmp_path):
    root = tmp_path / "ckpt"
    root.mkdir()
    ckpt.init_manifest(root, base=_base(panel_mode="pooled"))
    chunk = ckpt.chunk_checkpoint_path(root, 0)
    chunk.parent.mkdir(parents=True, exist_ok=True)
    chunk.write_text("placeholder")
    _guard_checkpoint_universe(ckpt, root,
                                 _base(panel_mode="ticker_fixed_effects"))
    assert not chunk.exists()


def test_guard_clears_on_hpo_objective_change(tmp_path):
    root = tmp_path / "ckpt"
    root.mkdir()
    ckpt.init_manifest(root, base=_base(hpo_objective="log_loss"))
    chunk = ckpt.chunk_checkpoint_path(root, 0)
    chunk.parent.mkdir(parents=True, exist_ok=True)
    chunk.write_text("placeholder")
    _guard_checkpoint_universe(ckpt, root,
                                 _base(hpo_objective="brier_score"))
    assert not chunk.exists()


def test_clear_checkpoints_only_affects_one_run(tmp_path):
    root_a = tmp_path / "run_a"
    root_b = tmp_path / "run_b"
    ckpt.init_manifest(root_a, base=_base(set_id="ECON"))
    ckpt.init_manifest(root_b, base=_base(set_id="ECON_VAD_F"))
    # Place chunks under each.
    for r in (root_a, root_b):
        p = ckpt.chunk_checkpoint_path(r, 0)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("placeholder")
    ckpt.clear_run_checkpoints(root_a)
    # Only run_a was touched.
    assert not (root_a / "chunks" / "chunk_0000.parquet").exists()
    assert (root_b / "chunks" / "chunk_0000.parquet").exists()


def test_per_asset_checkpoint_universe_persisted(tmp_path):
    root = tmp_path / "ckpt"
    root.mkdir()
    base = _base(model_type="per_asset",
                 requested=("BTC", "ETH"))
    ckpt.init_manifest(root, base=base)
    mf = ckpt.load_manifest(root)
    assert mf["requested_coin_universe_hash"] == coin_universe_hash(("BTC", "ETH"))
    assert mf["requested_tickers"] == ["BTC", "ETH"]
    assert mf["n_requested_tickers"] == 2
