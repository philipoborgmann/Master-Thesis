"""Smoke tests that don't require heavy models or real data.

We verify the CLI starts, parses arguments, and runs --dry-run mode for a few
representative stages. Transformer scoring tests are skipped when torch or
the underlying models are unavailable.
"""
from __future__ import annotations

import importlib

import pytest

from thesis_pipeline import cli


def test_cli_help_runs(capsys):
    parser = cli.build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["--help"])


def test_dry_run_create_price_features():
    rc = cli.main(["create-price-features", "--horizon", "1d",
                   "--coins", "BTC", "ETH", "--smoke", "--dry-run"])
    assert rc == 0


def test_dry_run_merge_features():
    rc = cli.main(["merge-features", "--horizon", "1d", "--smoke", "--dry-run"])
    assert rc == 0


def test_dry_run_run_models_econ():
    """v4 smoke set_id is ECON (was B1 before the 17-set registry refactor)."""
    rc = cli.main(["run-models", "--horizon", "1d", "--set-id", "ECON",
                   "--smoke", "--dry-run"])
    assert rc == 0


def test_dry_run_score_vader():
    rc = cli.main(["score-sentiment", "--model", "vader",
                   "--smoke", "--max-rows", "100", "--dry-run"])
    assert rc == 0


def test_dry_run_score_finbert_is_rejected(capsys):
    """FinBERT was removed; the CLI must reject it with a clear, informative
    error rather than silently accept it as a dry-run.
    """
    with pytest.raises(SystemExit):
        cli.main(["score-sentiment", "--model", "finbert",
                  "--smoke", "--max-rows", "10", "--dry-run"])
    assert "FinBERT" in capsys.readouterr().err


@pytest.mark.skipif(
    importlib.util.find_spec("torch") is None
    or importlib.util.find_spec("transformers") is None,
    reason="torch/transformers not installed",
)
def test_dry_run_score_cryptobert():
    rc = cli.main(["score-sentiment", "--model", "cryptobert",
                   "--smoke", "--max-rows", "10", "--dry-run"])
    assert rc == 0


def test_dry_run_pipeline_default_order():
    """run-pipeline --dry-run must visit every default stage without errors."""
    rc = cli.main(["run-pipeline", "--horizon", "1d", "--smoke", "--dry-run"])
    assert rc == 0
