"""Regression tests for the ``run-models`` CLI argument plumbing.

These tests pin the contract between
``thesis_pipeline.cli`` and ``thesis_pipeline.modeling.run_models.main``:

* ``--coins BTC ETH`` is forwarded as a single ``--coins`` flag followed by
  both tickers — *not* as repeated ``--coin BTC --coin ETH``, which would
  reintroduce the previous bug where the underlying module silently ignored
  every coin after the first.
* The argparse parser of ``run_models`` accepts both the new (``--set-id``,
  ``--coins``) and the legacy (``--set_id``, ``--ticker``, ``--coin``) flag
  names, so existing scripts keep working after the refactor.
"""
from __future__ import annotations

import pytest

from thesis_pipeline import cli
from thesis_pipeline.modeling import run_models as rm


def test_cli_forwards_coins_flag_not_repeated_coin(monkeypatch):
    captured: dict[str, list[str]] = {}

    def fake_main(argv):
        captured["argv"] = list(argv) if argv else []
        return 0

    monkeypatch.setattr(rm, "main", fake_main)

    rc = cli.main([
        "run-models",
        "--horizon", "1d",
        "--set-id", "B1",
        "--coins", "BTC", "ETH",
        "--smoke", "--dry-run",
    ])
    assert rc == 0

    argv = captured["argv"]
    # The plural --coins flag must be passed once with all tickers as values.
    assert "--coins" in argv, f"expected --coins in argv, got {argv}"
    coins_idx = argv.index("--coins")
    assert argv[coins_idx + 1] == "BTC"
    assert argv[coins_idx + 2] == "ETH"
    # The buggy singular form must NOT be emitted any more.
    assert "--coin" not in argv, (
        f"CLI must not emit '--coin' for run-models; got: {argv}"
    )


def test_cli_forwards_restart_and_force(monkeypatch):
    captured: dict[str, list[str]] = {}

    def fake_main(argv):
        captured["argv"] = list(argv) if argv else []
        return 0

    monkeypatch.setattr(rm, "main", fake_main)

    rc = cli.main([
        "run-models", "--horizon", "1d", "--set-id", "B1",
        "--dry-run", "--restart", "--force",
    ])
    assert rc == 0
    assert "--restart" in captured["argv"]
    assert "--force" in captured["argv"]


def test_module_parser_accepts_legacy_arg_names():
    parser = rm.build_parser()
    # Legacy long-form (underscore) names
    ns = parser.parse_args(["--set_id", "B1", "--ticker", "BTC", "--horizon", "1d"])
    assert ns.set_id == "B1"
    assert ns.coins == ["BTC"]
    assert ns.horizon == "1d"
    # The legacy singular alias `--coin` must also still work.
    ns2 = parser.parse_args(["--coin", "BTC"])
    assert ns2.coins == ["BTC"]


def test_module_parser_accepts_new_arg_names():
    parser = rm.build_parser()
    ns = parser.parse_args([
        "--set-id", "B1", "--coins", "BTC", "ETH",
        "--sentiment-model", "vader", "--horizon", "1d",
        "--smoke", "--dry-run", "--restart", "--force",
    ])
    assert ns.set_id == "B1"
    assert ns.coins == ["BTC", "ETH"]
    assert ns.sentiment_model == "vader"
    assert ns.smoke is True
    assert ns.dry_run is True
    assert ns.restart is True
    assert ns.force is True


def test_dry_run_via_module_main_returns_zero():
    """``main(['--smoke', '--dry-run'])`` must not touch the filesystem.
    Set-id ECON is the v4 smoke default; B1 was removed in the 17-set
    registry refactor."""
    rc = rm.main(["--smoke", "--dry-run", "--horizon", "1d",
                  "--set-id", "ECON", "--coins", "BTC"])
    assert rc == 0
