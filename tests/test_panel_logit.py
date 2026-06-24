"""Tests for the panel-logit alternative model family.

Synthetic panel data only — runs inside tmp_path via monkeypatch.chdir so
the real Data/ and Outputs/ trees are never touched.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from thesis_pipeline.modeling import panel_logit as pl
from thesis_pipeline.modeling.run_models import main as run_models_main


# ---------------------------------------------------------------------------
# Synthetic panel builder
# ---------------------------------------------------------------------------

def _build_panel(n_timestamps: int = 300,
                 tickers=("BTC", "ETH", "SOL"),
                 seed: int = 7) -> pd.DataFrame:
    """A panel where a shared ``signal_feature`` drives the target.

    target = 1 if (2.0 * signal_feature + 0.3 * noise) > 0 — the same
    relationship for every coin, so a pooled logit can learn it. The
    ``noise_feature`` is independent of the target.
    """
    rng = np.random.default_rng(seed)
    ts = pd.date_range("2024-01-01", periods=n_timestamps, freq="D", tz="UTC")
    rows = []
    for tk in tickers:
        signal = rng.normal(0, 1, n_timestamps)
        noise  = rng.normal(0, 1, n_timestamps)
        eps    = rng.normal(0, 1, n_timestamps)
        target = (2.0 * signal + 0.3 * eps > 0).astype(int)
        rows.append(pd.DataFrame({
            "timestamp":      ts,
            "ticker":         tk,
            "horizon":        "1d",
            "target":         target,
            "signal_feature": signal,
            "noise_feature":  noise,
        }))
    return pd.concat(rows, ignore_index=True)


# ---------------------------------------------------------------------------
# 1 + 2. Panel walk-forward learns signal, not noise
# ---------------------------------------------------------------------------

def test_panel_walk_forward_learns_signal():
    df = _build_panel(n_timestamps=300)
    sig = pl.run_panel_walk_forward(df, ["signal_feature"], panel_mode="pooled",
                                    min_init_timestamps=30)
    assert not sig.empty
    acc = (sig["prediction"] == sig["target"]).mean()
    assert acc > 0.65, f"pooled panel on signal should beat 0.65; got {acc:.3f}"


def test_panel_walk_forward_noise_near_random():
    df = _build_panel(n_timestamps=300)
    sig = pl.run_panel_walk_forward(df, ["noise_feature"], panel_mode="pooled",
                                    min_init_timestamps=30)
    assert not sig.empty
    acc = (sig["prediction"] == sig["target"]).mean()
    assert acc < 0.60, f"pure-noise panel should stay near random; got {acc:.3f}"


# ---------------------------------------------------------------------------
# 3. No leakage: train timestamps strictly before tau
# ---------------------------------------------------------------------------

def test_panel_no_lookahead(monkeypatch):
    df = _build_panel(n_timestamps=120)
    captured = []
    real_build = pl.build_panel_design_matrix

    def spy(train_df, test_df, feature_cols, panel_mode="pooled"):
        tau = test_df["timestamp"].iloc[0]
        # Every training timestamp must be strictly before the test timestamp.
        assert (train_df["timestamp"] < tau).all()
        # The test timestamp must not appear in training.
        assert tau not in set(train_df["timestamp"])
        captured.append(tau)
        return real_build(train_df, test_df, feature_cols, panel_mode)

    monkeypatch.setattr(pl, "build_panel_design_matrix", spy)
    sig = pl.run_panel_walk_forward(df, ["signal_feature"], min_init_timestamps=20)
    assert not sig.empty
    assert len(captured) > 0


# ---------------------------------------------------------------------------
# 4. Output schema
# ---------------------------------------------------------------------------

def test_panel_walk_forward_probability_range():
    df = _build_panel(n_timestamps=150)
    sig = pl.run_panel_walk_forward(df, ["signal_feature"], min_init_timestamps=20)
    assert ((sig["probability"] >= 0) & (sig["probability"] <= 1)).all()
    assert {"timestamp", "ticker", "target", "prediction", "probability"}.issubset(sig.columns)


# ---------------------------------------------------------------------------
# 6. Ticker fixed effects
# ---------------------------------------------------------------------------

def test_ticker_fixed_effects_runs():
    df = _build_panel(n_timestamps=200)
    sig = pl.run_panel_walk_forward(df, ["signal_feature"],
                                    panel_mode="ticker_fixed_effects",
                                    min_init_timestamps=30)
    assert not sig.empty
    acc = (sig["prediction"] == sig["target"]).mean()
    assert acc > 0.60


def test_add_ticker_fixed_effects_aligns_and_handles_unknown():
    train_tk = pd.Series(["BTC", "ETH", "SOL", "BTC", "ETH"])
    test_tk  = pd.Series(["BTC", "DOGE"])  # DOGE unseen in training
    tr_d, te_d = pl.add_ticker_fixed_effects(train_tk, test_tk)
    # Test dummies aligned to training columns.
    assert list(te_d.columns) == list(tr_d.columns)
    # Reference category dropped → one fewer column than unique training tickers.
    assert tr_d.shape[1] == 2  # 3 tickers − 1 reference
    # Unknown ticker DOGE → all-zero dummy row (the reference category).
    assert (te_d.iloc[1] == 0).all()


# ---------------------------------------------------------------------------
# Full-pipeline fixture (pooled + ticker FE + B1 sentinel)
# ---------------------------------------------------------------------------

@pytest.fixture
def panel_repo(tmp_path, monkeypatch):
    (tmp_path / "Data" / "Final").mkdir(parents=True)
    (tmp_path / "Outputs" / "Signals").mkdir(parents=True)

    df = _build_panel(n_timestamps=300, tickers=("BTC", "ETH", "SOL"))
    df.to_parquet(tmp_path / "Data" / "Final" / "features_1d.parquet", index=False)

    # Synthetic fixture for the panel-logit unit tests. Set-ids:
    #   * NAIVE_FIXTURE — unit-test-only sentinel that routes to the
    #     rolling-probability benchmark via the __majority_class__ feature.
    #     NOT in SET_ID_PATTERN / REMOVED_SET_IDS — this name only exists
    #     inside the synthetic xlsx the test writes.
    #   * ECON / ECON_VAD_L — real v4 ids; the test deliberately overrides
    #     their production feature lists with synthetic columns so the
    #     panel-logit code path can be exercised on noise + signal.
    fs = pd.DataFrame({
        "set_id":   ["NAIVE_FIXTURE", "ECON", "ECON_VAD_L"],
        "category": ["benchmark", "benchmark", "combined"],
        # sentiment_model uses "-" (not "none") to match the production
        # filename convention: a non-sentiment set produces
        # `<set_id>_panel_<mode>.parquet`, not `<set_id>_none_panel_*`.
        "sentiment_model": ["-", "-", "vader"],
        "label":    ["unit-test NAIVE rolling-prob sentinel",
                     "synthetic ECON (single signal feature)",
                     "synthetic ECON_VAD_L (signal + noise)"],
        "Feature Columns (comma-separated)": [
            "__majority_class__",
            "signal_feature",
            "signal_feature,noise_feature",
        ],
    })
    with pd.ExcelWriter(tmp_path / "feature_sets.xlsx", engine="openpyxl") as w:
        fs.to_excel(w, sheet_name="feature_sets", index=False)

    monkeypatch.chdir(tmp_path)
    return tmp_path


# ---------------------------------------------------------------------------
# 5. Panel pooled full pipeline
# ---------------------------------------------------------------------------

def test_full_pipeline_panel_pooled(panel_repo):
    rc = run_models_main([
        "--horizon", "1d", "--set-id", "ECON",
        "--model-type", "panel_logit", "--panel-mode", "pooled",
        "--restart",
    ])
    assert rc == 0
    out = panel_repo / "Outputs" / "Signals" / "1d" / "ECON_panel_pooled.parquet"
    assert out.exists()
    sdf = pd.read_parquet(out)
    assert not sdf.empty
    assert (sdf["model_type"] == "panel_logit").all()
    assert (sdf["panel_mode"] == "pooled").all()
    for col in ("timestamp", "ticker", "target", "prediction", "probability",
                "set_id", "sentiment_model", "horizon", "model_type", "panel_mode"):
        assert col in sdf.columns
    acc = (sdf["prediction"] == sdf["target"]).mean()
    assert acc > 0.65


def test_full_pipeline_panel_does_not_touch_per_asset_file(panel_repo):
    # Run per-asset first, then panel — both must coexist.
    run_models_main(["--horizon", "1d", "--set-id", "ECON", "--restart"])
    per_asset = panel_repo / "Outputs" / "Signals" / "1d" / "ECON.parquet"
    assert per_asset.exists()

    run_models_main(["--horizon", "1d", "--set-id", "ECON",
                     "--model-type", "panel_logit", "--panel-mode", "pooled",
                     "--restart"])
    panel = panel_repo / "Outputs" / "Signals" / "1d" / "ECON_panel_pooled.parquet"
    assert panel.exists()
    # The per-asset file is untouched and carries no panel columns.
    pa = pd.read_parquet(per_asset)
    assert "model_type" not in pa.columns


# ---------------------------------------------------------------------------
# 6. Ticker FE full pipeline
# ---------------------------------------------------------------------------

def test_full_pipeline_panel_ticker_fe(panel_repo):
    rc = run_models_main([
        "--horizon", "1d", "--set-id", "ECON_VAD_L",
        "--model-type", "panel_logit", "--panel-mode", "ticker_fixed_effects",
        "--restart",
    ])
    assert rc == 0
    out = panel_repo / "Outputs" / "Signals" / "1d" / "ECON_VAD_L_vader_panel_ticker_fe.parquet"
    assert out.exists()
    sdf = pd.read_parquet(out)
    assert not sdf.empty
    assert (sdf["panel_mode"] == "ticker_fixed_effects").all()


# ---------------------------------------------------------------------------
# 7. B1 is now run as a panel rolling-probability benchmark (no longer skipped)
# ---------------------------------------------------------------------------

def test_panel_b1_runs_as_rolling_probability_benchmark(panel_repo):
    rc = run_models_main([
        "--horizon", "1d", "--set-id", "NAIVE_FIXTURE",
        "--model-type", "panel_logit", "--panel-mode", "pooled",
        "--restart",
    ])
    assert rc == 0
    # The panel rolling-probability benchmark now produces a signal file with
    # the same naming scheme as the logistic models.
    out = panel_repo / "Outputs" / "Signals" / "1d" / "NAIVE_FIXTURE_panel_pooled.parquet"
    assert out.exists()
    sdf = pd.read_parquet(out)
    assert not sdf.empty
    assert (sdf["benchmark_model"]
            == "ticker_rolling_probability_with_pooled_fallback").all()
    assert ((sdf["probability"] >= 0) & (sdf["probability"] <= 1)).all()
    assert (sdf["model_type"] == "panel_logit").all()


# ---------------------------------------------------------------------------
# Metrics summary carries model_type / panel_mode
# ---------------------------------------------------------------------------

def test_metrics_summary_has_panel_columns(panel_repo):
    run_models_main(["--horizon", "1d", "--set-id", "ECON",
                     "--model-type", "panel_logit", "--panel-mode", "pooled",
                     "--restart"])
    metrics = panel_repo / "Outputs" / "Signals" / "metrics_summary.csv"
    assert metrics.exists()
    m = pd.read_csv(metrics)
    assert "model_type" in m.columns and "panel_mode" in m.columns
    panel_rows = m[m["model_type"] == "panel_logit"]
    assert not panel_rows.empty
    assert (panel_rows["panel_mode"] == "pooled").any()


# ---------------------------------------------------------------------------
# CLI dry-run
# ---------------------------------------------------------------------------

def test_cli_dry_run_panel():
    import sys
    src = Path(__file__).resolve().parents[1] / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
    from thesis_pipeline import cli
    rc = cli.main(["run-models", "--horizon", "1d", "--set-id", "ECON",
                   "--model-type", "panel_logit", "--panel-mode", "pooled",
                   "--dry-run"])
    assert rc == 0
