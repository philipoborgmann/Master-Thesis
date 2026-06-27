"""Tests for the smoke-default resolution order (Aufgabe 6 follow-up B).

A bare ``run-models --smoke`` must resolve to horizon=1d, coins=[BTC, ETH],
set_id=ECON BEFORE any downstream resolution (HPO config, NAIVE
generation, dispatch, dry-run logging). Explicit user values always win.
"""
from __future__ import annotations

import argparse

import pytest

from thesis_pipeline.modeling.run_models import (
    apply_smoke_defaults, build_parser,
)


def _parse(argv):
    return build_parser().parse_args(argv)


# ---------------------------------------------------------------------------
# Helper is pure: returns the resolved namespace
# ---------------------------------------------------------------------------

def test_smoke_resolves_horizon_coins_set_id_to_canonical_defaults():
    ns = _parse(["--smoke"])
    out = apply_smoke_defaults(ns)
    assert out.horizon == "1d"
    assert out.coins == ["BTC", "ETH"]
    assert out.set_id == "ECON"


def test_smoke_horizon_override_wins():
    ns = _parse(["--smoke", "--horizon", "6h"])
    out = apply_smoke_defaults(ns)
    assert out.horizon == "6h"
    # The other smoke defaults still fill in.
    assert out.coins == ["BTC", "ETH"]
    assert out.set_id == "ECON"


def test_smoke_coins_override_wins():
    ns = _parse(["--smoke", "--coins", "BTC", "SOL"])
    out = apply_smoke_defaults(ns)
    assert out.coins == ["BTC", "SOL"]


def test_smoke_set_id_override_wins():
    ns = _parse(["--smoke", "--set-id", "ECON_CBT_L"])
    out = apply_smoke_defaults(ns)
    assert out.set_id == "ECON_CBT_L"


def test_apply_smoke_defaults_is_a_noop_without_smoke_flag():
    """Without --smoke the helper must leave the namespace untouched."""
    ns = _parse([])
    before = (ns.horizon, list(ns.coins) if ns.coins else None, ns.set_id)
    out = apply_smoke_defaults(ns)
    after = (out.horizon, list(out.coins) if out.coins else None, out.set_id)
    assert before == after


# ---------------------------------------------------------------------------
# Smoke must be applied before NAIVE / HPO / dispatch
# ---------------------------------------------------------------------------

def test_smoke_naive_universe_hash_uses_smoke_coins(tmp_path):
    """Aufgabe 6 follow-up B: when a bare --smoke run reaches the NAIVE
    generator, the universe hash must reflect smoke coins, not the full
    available universe. We exercise this by calling the generator with
    the smoke-resolved coins and asserting the hash matches."""
    from thesis_pipeline.modeling import naive_reference as nr
    import numpy as np
    import pandas as pd

    rng = np.random.default_rng(1)
    ts = pd.date_range("2024-01-01", periods=120, freq="D", tz="UTC")
    rows = []
    for tk in ("BTC", "ETH", "SOL", "ADA"):  # FULL universe in features
        rows.append(pd.DataFrame({
            "timestamp": ts, "ticker": tk, "horizon": "1d",
            "target": rng.integers(0, 2, 120),
            "signal_feature": rng.normal(0, 1, 120),
        }))
    feats = pd.concat(rows, ignore_index=True)

    # Mimic what main() does after apply_smoke_defaults: pass the smoke
    # universe (BTC, ETH) into generate_naive_reference.
    out = nr.generate_naive_reference(
        horizon="1d", features_df=feats,
        output_dir=tmp_path, coins=["BTC", "ETH"],
    )
    assert out is not None
    expected_hash = nr.coin_universe_hash(["BTC", "ETH"])
    assert f"_u_{expected_hash}.parquet" in str(out)
    sig = pd.read_parquet(out)
    # Even though features contained SOL and ADA, the NAIVE file only
    # carries the smoke universe.
    assert set(sig["ticker"]) == {"BTC", "ETH"}


def test_smoke_dry_run_resolves_before_logging(monkeypatch):
    """The dry-run path must invoke ``apply_smoke_defaults`` BEFORE any
    downstream logic — verify by inspecting the namespace
    ``log_stage_header`` actually receives. Aufgabe 6 follow-up B
    forbids leaking the production universe into the smoke stage
    header."""
    captured: dict = {}

    def fake_header(stage, *, mode, inputs, outputs, extra):
        captured["extra"] = dict(extra)

    monkeypatch.setattr("thesis_pipeline.logging_utils.log_stage_header",
                        fake_header)
    from thesis_pipeline import cli
    rc = cli.main(["run-models", "--smoke", "--dry-run"])
    assert rc == 0
    extra = captured.get("extra", {})
    assert extra.get("horizon") == "1d"
    assert extra.get("set_id")  == "ECON"
    # The "coins" entry must be the smoke list, not the "(all)" sentinel
    # that would mean the run was about to iterate every ticker.
    coins_field = extra.get("coins")
    assert coins_field != "(all)"
    if isinstance(coins_field, (list, tuple)):
        assert set(coins_field) == {"BTC", "ETH"}
