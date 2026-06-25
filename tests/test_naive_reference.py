"""Tests for the NAIVE reference auto-generation (Aufgabe 6 follow-up A).

NAIVE is the historical-majority (rolling-probability) reference. It is
NOT a feature set in :data:`SET_ID_PATTERN` and never appears in
``feature_sets.xlsx``. The v4 modeling stage generates it automatically
once per (horizon × model_type × panel_mode × training-window
configuration × coin universe), independently of the 17-set feature
grid, even when HPO is enabled.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from thesis_pipeline.evaluation.incremental import (
    NAIVE_REFERENCE_LABEL, is_naive_signal_row,
)
from thesis_pipeline.features.feature_registry import (
    SET_ID_PATTERN, REMOVED_SET_IDS,
)
from thesis_pipeline.modeling import naive_reference as nr


# ---------------------------------------------------------------------------
# Naming
# ---------------------------------------------------------------------------

def test_naive_output_name_per_asset_no_suffix():
    assert nr.naive_output_name(model_type="per_asset") == "NAIVE"


def test_naive_output_name_panel_pooled_expanding():
    assert nr.naive_output_name(
        model_type="panel_logit",
        panel_mode="pooled",
        train_window_mode="expanding",
    ) == "NAIVE_panel_pooled"


def test_naive_output_name_panel_ticker_fe_rolling_180_days():
    assert nr.naive_output_name(
        model_type="panel_logit",
        panel_mode="ticker_fixed_effects",
        train_window_mode="rolling_fixed",
        rolling_window_days=180,
    ) == "NAIVE_panel_ticker_fe_rw180d"


def test_naive_output_name_panel_pooled_rolling_timestamps():
    assert nr.naive_output_name(
        model_type="panel_logit",
        panel_mode="pooled",
        train_window_mode="rolling_fixed",
        rolling_window_timestamps=30,
    ) == "NAIVE_panel_pooled_rw30"


def test_naive_output_name_does_not_carry_hpo_suffix():
    """NAIVE is by definition untuned — the name must not include
    ``hpo_logloss`` / ``hpo_brier`` etc.
    """
    name = nr.naive_output_name(
        model_type="panel_logit",
        panel_mode="ticker_fixed_effects",
        train_window_mode="rolling_fixed",
        rolling_window_days=180,
    )
    assert "hpo" not in name


# ---------------------------------------------------------------------------
# Registry contract
# ---------------------------------------------------------------------------

def test_naive_is_not_in_registry_pattern():
    assert NAIVE_REFERENCE_LABEL not in set(SET_ID_PATTERN)


def test_naive_is_not_in_removed_set_ids():
    """NAIVE is a legitimate reference label, not a deprecated v3 id.
    Adding it to REMOVED_SET_IDS would silently make every NAIVE-tagged
    run fail with a migration error."""
    assert NAIVE_REFERENCE_LABEL not in REMOVED_SET_IDS


# ---------------------------------------------------------------------------
# Synthetic generator end-to-end
# ---------------------------------------------------------------------------

def _synthetic_features(n_per_ticker: int = 200,
                        tickers=("BTC", "ETH", "SOL"),
                        horizon: str = "1d",
                        seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    ts = pd.date_range("2024-01-01", periods=n_per_ticker, freq="D", tz="UTC")
    rows = []
    for tk in tickers:
        rows.append(pd.DataFrame({
            "timestamp": ts,
            "ticker":    tk,
            "horizon":   horizon,
            "target":    rng.integers(0, 2, n_per_ticker),
            "signal_feature": rng.normal(0, 1, n_per_ticker),
        }))
    return pd.concat(rows, ignore_index=True)


def test_generate_naive_reference_writes_canonical_panel_signal(tmp_path):
    df = _synthetic_features()
    out = nr.generate_naive_reference(
        horizon="1d",
        model_type="panel_logit",
        panel_mode="ticker_fixed_effects",
        train_window_mode="rolling_fixed",
        rolling_window_days=180.0,
        features_df=df,
        output_dir=tmp_path,
        resume=True, restart=False,
    )
    assert out is not None and out.exists()
    assert out.name == "NAIVE_panel_ticker_fe_rw180d.parquet"

    sig = pd.read_parquet(out)
    assert not sig.empty
    # Canonical NAIVE metadata.
    assert (sig["set_id"] == "NAIVE").all()
    assert (sig["sentiment_model"] == "-").all()
    assert (sig["model_type"] == "panel_logit").all()
    assert (sig["panel_mode"] == "ticker_fixed_effects").all()
    assert (sig["train_window_mode"] == "rolling_fixed").all()
    assert (sig["rolling_window_days"] == 180.0).all()
    # NAIVE is never tuned.
    assert (sig["hpo_enabled"] == False).all()  # noqa: E712
    assert (sig["hpo_objective"] == "-").all()
    assert (sig["hpo_variant"] == "naive").all()
    # Benchmark-model tag identifies the path.
    assert (sig["benchmark_model"]
            == "ticker_rolling_probability_with_pooled_fallback").all()
    # is_naive_signal_row() must recognise the produced rows.
    assert is_naive_signal_row(sig.iloc[0])


def test_generate_naive_reference_caches_on_resume(tmp_path):
    df = _synthetic_features()
    first = nr.generate_naive_reference(
        horizon="1d", features_df=df, output_dir=tmp_path,
        resume=True, restart=False,
    )
    assert first is not None
    mtime_before = first.stat().st_mtime
    # Second call must be a cache hit (returns None, file unchanged).
    second = nr.generate_naive_reference(
        horizon="1d", features_df=df, output_dir=tmp_path,
        resume=True, restart=False,
    )
    assert second is None
    assert first.stat().st_mtime == mtime_before


def test_generate_naive_reference_restart_overwrites(tmp_path):
    df = _synthetic_features()
    first = nr.generate_naive_reference(
        horizon="1d", features_df=df, output_dir=tmp_path,
        resume=True, restart=False,
    )
    assert first is not None
    # restart=True bypasses the cache; re-writes the file.
    second = nr.generate_naive_reference(
        horizon="1d", features_df=df, output_dir=tmp_path,
        resume=True, restart=True,
    )
    assert second is not None
    assert second == first


def test_generate_naive_reference_window_change_lands_in_new_file(tmp_path):
    """Different rolling_window_days → different filename → no cross-cache."""
    df = _synthetic_features()
    a = nr.generate_naive_reference(
        horizon="1d", features_df=df, output_dir=tmp_path,
        rolling_window_days=180.0,
    )
    b = nr.generate_naive_reference(
        horizon="1d", features_df=df, output_dir=tmp_path,
        rolling_window_days=60.0,
    )
    assert a is not None and b is not None
    assert a.name != b.name
    assert "rw180d" in a.name
    assert "rw60d" in b.name


def test_generate_naive_reference_with_hpo_enabled_still_runs(tmp_path):
    """Even when HPO is on at the modeling layer, NAIVE is generated. The
    helper takes no hpo flag — it is by construction untuned — so this
    test exercises that the produced rows carry hpo_enabled=False
    regardless of what the wider run is doing."""
    df = _synthetic_features()
    out = nr.generate_naive_reference(
        horizon="1d", features_df=df, output_dir=tmp_path,
    )
    sig = pd.read_parquet(out)
    assert (sig["hpo_enabled"] == False).all()  # noqa: E712
    assert (sig["hpo_variant"] == "naive").all()


# ---------------------------------------------------------------------------
# CLI: --generate-naive-reference flag is BooleanOptionalAction
# ---------------------------------------------------------------------------

def test_cli_flag_default_is_true():
    from thesis_pipeline.modeling.run_models import build_parser
    ns = build_parser().parse_args([])
    assert ns.generate_naive_reference is True


def test_cli_no_generate_naive_reference_disables():
    from thesis_pipeline.modeling.run_models import build_parser
    ns = build_parser().parse_args(["--no-generate-naive-reference"])
    assert ns.generate_naive_reference is False


def test_package_cli_default_is_true():
    from thesis_pipeline import cli as pcli
    parser = pcli.build_parser()
    ns = parser.parse_args(["run-models"])
    assert ns.generate_naive_reference is True


def test_package_cli_no_generate_naive_reference_disables():
    from thesis_pipeline import cli as pcli
    parser = pcli.build_parser()
    ns = parser.parse_args(["run-models", "--no-generate-naive-reference"])
    assert ns.generate_naive_reference is False
