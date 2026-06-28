"""Tests for the additive rolling-window + structural-break + coin-filter work.

Covers the 13 items in the task spec:

1. expanding-window behaviour is unchanged
2. rolling_fixed never includes τ or future rows
3. rolling_fixed uses exactly the last N unique timestamps
4. rolling_fixed raises ValueError when no window size is supplied
5. output names differ between expanding and rolling
6. checkpoint manifests prevent reuse when train_window_mode / rolling size differs
7. B1 benchmark is no longer skipped for panel_logit
8. structural_breaks writes break_records.csv + break_summary.csv
9. structural_breaks output is not read automatically by run-models
10. default sentiment-coverage filtering remains unchanged
11. --no-sentiment-coverage-filter retains low-coverage coins
12. --neutral-fill-missing-sentiment fills only sentiment / attention columns
13. non-sentiment columns are not neutral-filled
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from thesis_pipeline.modeling import (
    checkpointing as ckpt,
    panel_logit as pl,
    windowing as win,
)
from thesis_pipeline.modeling.run_models import main as run_models_main
from thesis_pipeline.features import merge as fm
from thesis_pipeline.diagnostics import structural_breaks as sb


# ---------------------------------------------------------------------------
# Part B — windowing helper (items 1, 2, 3, 4)
# ---------------------------------------------------------------------------

def _panel(n_ts=20, tickers=("BTC", "ETH")):
    ts = pd.date_range("2024-01-01", periods=n_ts, freq="D", tz="UTC")
    rows = []
    for tk in tickers:
        rows.append(pd.DataFrame({"timestamp": ts, "ticker": tk,
                                   "target": [0, 1] * (n_ts // 2),
                                   "signal_feature": np.arange(n_ts, dtype=float)}))
    return pd.concat(rows, ignore_index=True)


def test_expanding_returns_all_pre_tau_rows():
    df = _panel()
    unique_ts = sorted(df["timestamp"].unique())
    tau = unique_ts[10]
    train, test, meta = win.select_panel_train_window(df, tau)
    assert (train["timestamp"] < tau).all()
    assert (test["timestamp"] == tau).all()
    assert meta["train_window_mode"] == "expanding"
    assert meta["train_window_timestamps"] == 10
    assert meta["train_start_timestamp"] == pd.Timestamp(unique_ts[0])
    assert meta["train_end_timestamp"]   == pd.Timestamp(unique_ts[9])


def test_rolling_fixed_never_includes_tau_or_future():
    df = _panel()
    unique_ts = sorted(df["timestamp"].unique())
    tau = unique_ts[12]
    train, _, _ = win.select_panel_train_window(
        df, tau, train_window_mode="rolling_fixed",
        rolling_window_timestamps=5)
    assert (train["timestamp"] < tau).all()


def test_rolling_fixed_uses_exactly_last_n_unique_timestamps():
    df = _panel()
    unique_ts = sorted(df["timestamp"].unique())
    tau = unique_ts[14]
    train, _, meta = win.select_panel_train_window(
        df, tau, train_window_mode="rolling_fixed",
        rolling_window_timestamps=4)
    assert meta["train_window_timestamps"] == 4
    assert sorted(train["timestamp"].unique()) == unique_ts[10:14]


def test_rolling_fixed_requires_explicit_window_size():
    df = _panel()
    unique_ts = sorted(df["timestamp"].unique())
    tau = unique_ts[10]
    with pytest.raises(ValueError, match="rolling_fixed"):
        win.select_panel_train_window(df, tau, train_window_mode="rolling_fixed")


def test_window_suffix_keeps_expanding_filenames_unchanged():
    assert win.window_suffix("expanding", None, None) == ""
    assert win.window_suffix("rolling_fixed", 180, None) == "_rw180"
    assert win.window_suffix("rolling_fixed", None, 30.0) == "_rw30d"


# ---------------------------------------------------------------------------
# Part C — panel_logit integration: output names + checkpoint guard (items 5, 6, 7)
# ---------------------------------------------------------------------------

def _panel_repo(tmp_path):
    """Mirror the existing test_panel_logit fixture without sentiment-model."""
    (tmp_path / "Data" / "Final").mkdir(parents=True)
    (tmp_path / "Outputs" / "Signals").mkdir(parents=True)
    df = _build_synthetic_panel(n_timestamps=200, tickers=("BTC", "ETH", "SOL"))
    df.to_parquet(tmp_path / "Data" / "Final" / "features_1d.parquet", index=False)
    fs = pd.DataFrame({
        "set_id":   ["NAIVE_FIXTURE", "ECON", "ECON_VAD_L"],
        "category": ["benchmark", "economic", "combined"],
        "sentiment_model": ["-", "-", "vader"],
        "label":    ["bench", "econ", "combo"],
        "Feature Columns (comma-separated)": [
            "__majority_class__", "signal_feature", "signal_feature,noise_feature",
        ],
    })
    with pd.ExcelWriter(tmp_path / "feature_sets.xlsx", engine="openpyxl") as w:
        fs.to_excel(w, sheet_name="feature_sets", index=False)
    return tmp_path


def _build_synthetic_panel(n_timestamps=200, tickers=("BTC", "ETH", "SOL"), seed=7):
    rng = np.random.default_rng(seed)
    ts = pd.date_range("2024-01-01", periods=n_timestamps, freq="D", tz="UTC")
    rows = []
    for tk in tickers:
        signal = rng.normal(0, 1, n_timestamps)
        eps    = rng.normal(0, 1, n_timestamps)
        rows.append(pd.DataFrame({
            "timestamp": ts, "ticker": tk, "horizon": "1d",
            "target": (2 * signal + 0.3 * eps > 0).astype(int),
            "signal_feature": signal,
            "noise_feature":  rng.normal(0, 1, n_timestamps),
        }))
    return pd.concat(rows, ignore_index=True)


def test_rolling_window_filename_differs_from_expanding(tmp_path, monkeypatch):
    repo = _panel_repo(tmp_path)
    monkeypatch.chdir(repo)
    # NB: v4 default is --train-window rolling_fixed --rolling-window-days 180,
    # so to compare expanding vs. rolling filenames we must opt back into
    # expanding explicitly. Both runs also opt out of HPO (default-on in v4)
    # to keep this test focused on filename mechanics, not estimator depth.
    expanded = run_models_main(["--horizon", "1d", "--set-id", "ECON",
                                 "--model-type", "panel_logit", "--panel-mode", "pooled",
                                 "--train-window", "expanding",
                                 "--no-tune-hyperparams", "--restart"])
    rolling = run_models_main(["--horizon", "1d", "--set-id", "ECON",
                                "--model-type", "panel_logit", "--panel-mode", "pooled",
                                "--train-window", "rolling_fixed",
                                "--rolling-window-timestamps", "30",
                                "--no-tune-hyperparams", "--restart"])
    assert expanded == 0 and rolling == 0
    sig_dir = repo / "Outputs" / "Signals" / "1d"
    assert sorted(sig_dir.glob("ECON_panel_pooled_u_*.parquet")), \
        "expanding output missing"
    rolling_cands = sorted(sig_dir.glob("ECON_panel_pooled_rw30_u_*.parquet"))
    assert rolling_cands, \
        "rolling-window run must NOT overwrite the expanding filename"
    rolling_path = rolling_cands[0]
    sig = pd.read_parquet(rolling_path)
    assert (sig["train_window_mode"] == "rolling_fixed").all()
    assert (sig["train_window_timestamps"] <= 30).all()


def test_checkpoint_manifest_refuses_reuse_when_window_changes(tmp_path, monkeypatch):
    repo = _panel_repo(tmp_path)
    monkeypatch.chdir(repo)
    # Run 1: expanding window — checkpoints + manifest written. v4 defaults
    # would write to a different out_name (...rw180), so opt into expanding
    # explicitly to test the same-out_name manifest-mismatch path.
    run_models_main(["--horizon", "1d", "--set-id", "ECON",
                     "--model-type", "panel_logit", "--panel-mode", "pooled",
                     "--train-window", "expanding",
                     "--no-tune-hyperparams", "--restart"])
    ckpt_root_d = repo / "Outputs" / "Checkpoints" / "Models" / "1d"
    cands = sorted(ckpt_root_d.glob("ECON_panel_pooled_u_*"))
    root = cands[0] if cands else (ckpt_root_d / "ECON_panel_pooled")
    mf = ckpt.load_manifest(root)
    assert mf["train_window_mode"] == "expanding"
    chunk = ckpt.chunk_checkpoint_path(root, 0)
    assert chunk.exists()

    # Run 2: same out_name, DIFFERENT window mode. The new run must NOT reuse
    # the expanding checkpoints — it gets its own out_name and root, but if
    # we point it at the same root via the same out_name (which we can't here
    # because the rolling run appends _rw…), the guard still has to refuse
    # reuse when invoked with a different mode. Simulate the corruption by
    # writing an incompatible manifest then forcing the expanding run again.
    incompatible = dict(mf)
    incompatible["train_window_mode"] = "rolling_fixed"
    incompatible["rolling_window_timestamps"] = 50
    ckpt.write_manifest(root, incompatible)
    mtime_before = chunk.stat().st_mtime

    run_models_main(["--horizon", "1d", "--set-id", "ECON",
                     "--model-type", "panel_logit", "--panel-mode", "pooled",
                     "--train-window", "expanding",
                     "--no-tune-hyperparams", "--restart"])
    # The chunk was rewritten (not silently reused).
    assert chunk.stat().st_mtime != mtime_before


def test_panel_b1_runs_as_rolling_probability_benchmark(tmp_path, monkeypatch):
    repo = _panel_repo(tmp_path)
    monkeypatch.chdir(repo)
    # Opt back into expanding + no-HPO so the legacy filename
    # `NAIVE_FIXTURE_panel_pooled.parquet` is produced (the v4 default
    # would add `_rw180` and `_hpo_logloss` suffixes).
    rc = run_models_main(["--horizon", "1d", "--set-id", "NAIVE_FIXTURE",
                          "--model-type", "panel_logit", "--panel-mode", "pooled",
                          "--train-window", "expanding",
                          "--no-tune-hyperparams", "--restart"])
    assert rc == 0
    sig_dir = repo / "Outputs" / "Signals" / "1d"
    cands = sorted(sig_dir.glob("NAIVE_FIXTURE_panel_pooled_u_*.parquet"))
    assert cands, "panel rolling-prob NAIVE_FIXTURE must be produced, not silently skipped"
    out = cands[0]
    sig = pd.read_parquet(out)
    assert not sig.empty
    assert (sig["benchmark_model"]
            == "ticker_rolling_probability_with_pooled_fallback").all()
    assert ((sig["probability"] >= 0) & (sig["probability"] <= 1)).all()


# ---------------------------------------------------------------------------
# Part A — structural-breaks diagnostic (items 8, 9)
# ---------------------------------------------------------------------------

@pytest.fixture
def break_repo(tmp_path, monkeypatch):
    (tmp_path / "Data" / "Final").mkdir(parents=True)
    n = 120
    rng = np.random.default_rng(0)
    ts = pd.date_range("2024-01-01", periods=n, freq="D", tz="UTC")
    rows = []
    for tk in ("BTC", "ETH"):
        rows.append(pd.DataFrame({
            "timestamp": ts, "ticker": tk, "horizon": "1d",
            "target": rng.integers(0, 2, n),
            "log_return_t": rng.normal(0, 0.02, n),
            "realized_vol_14d": np.abs(rng.normal(0.02, 0.005, n)),
            "volume_diff": rng.normal(0, 0.5, n),
            "post_count": rng.integers(0, 5, n),
            "vader_title_score_mean": rng.normal(0, 0.1, n),
            "cryptobert_title_score_mean": rng.normal(0, 0.1, n),
            "cryptobert_bullishness_ratio": rng.uniform(0.3, 0.7, n),
        }))
    pd.concat(rows, ignore_index=True).to_parquet(
        tmp_path / "Data" / "Final" / "features_1d.parquet", index=False)
    monkeypatch.chdir(tmp_path)
    from thesis_pipeline import config as cfg
    monkeypatch.setattr(cfg, "project_root", lambda: tmp_path)
    cfg.load_config.cache_clear()
    shutil.copytree(Path(__file__).resolve().parents[1] / "configs",
                    tmp_path / "configs")
    return tmp_path


def test_structural_breaks_writes_diagnostic_csvs(break_repo):
    rc = sb.main(["--horizon", "1d"])
    assert rc == 0
    out_dir = break_repo / "Data" / "Features" / "structural_breaks"
    assert (out_dir / "break_records.csv").exists()
    assert (out_dir / "break_summary.csv").exists()
    assert (out_dir / "regime_length_diagnostics.csv").exists()
    records = pd.read_csv(out_dir / "break_records.csv")
    if not records.empty:
        assert {"horizon", "ticker", "feature", "feature_group", "n_breaks",
                "break_indices", "break_timestamps",
                "segment_lengths_timestamps", "median_segment_length",
                "mean_segment_length"} <= set(records.columns)
    # The descriptive-only file is explicitly tagged so downstream code cannot
    # use it as a configuration source.
    regime = pd.read_csv(out_dir / "regime_length_diagnostics.csv")
    if not regime.empty:
        assert regime["descriptive_only"].all()


def test_structural_breaks_output_not_consumed_by_run_models():
    """run_models / panel_logit must not import structural_breaks or read
    any of its CSVs. This rules out the ``rolling_bp_median`` failure mode
    listed in the spec.
    """
    import importlib
    rm_src = importlib.import_module("thesis_pipeline.modeling.run_models").__file__
    pl_src = importlib.import_module("thesis_pipeline.modeling.panel_logit").__file__
    for path in (rm_src, pl_src):
        text = Path(path).read_text(encoding="utf-8")
        assert "structural_breaks" not in text
        assert "break_records.csv" not in text
        assert "regime_length_diagnostics" not in text
        assert "rolling_bp_median" not in text


# ---------------------------------------------------------------------------
# Part E — sentiment-coverage filter + neutral fill (items 10, 11, 12, 13)
# ---------------------------------------------------------------------------

@pytest.fixture
def merge_repo(tmp_path, monkeypatch):
    feature_dir = tmp_path / "Data" / "Features"
    final_dir   = tmp_path / "Data" / "Final"
    feature_dir.mkdir(parents=True); final_dir.mkdir()
    n = 24
    ts = pd.date_range("2024-01-01", periods=n, freq="D", tz="UTC")
    # PRICE: three tickers (BTC, ETH, SOL).
    price_rows = []
    for tk in ("BTC", "ETH", "SOL"):
        price_rows.append(pd.DataFrame({
            "timestamp": ts, "ticker": tk, "horizon": "1d",
            "target": [0, 1] * (n // 2),
            # v4 canonical momentum + realised-volatility names. The
            # schema validator in :mod:`thesis_pipeline.diagnostics
            # .feature_schema` refuses legacy ``cum_log_return_N`` /
            # ``realized_vol_14`` aliases.
            "log_return_t":         np.linspace(-0.01, 0.01, n),
            "cum_log_return_7d":    np.linspace(-0.02, 0.02, n),
            "cum_log_return_14d":   np.linspace(-0.03, 0.03, n),
            "cum_log_return_21d":   np.linspace(-0.04, 0.04, n),
            "realized_vol_14d":     np.linspace(0.01, 0.02, n),
            "volume_diff":          np.linspace(0.0, 1.0, n),
            "log_market_cap_lag1":  np.linspace(20.0, 21.0, n),
            # v4 final-frame leakage assertion (Aufgabe 7) requires the
            # availability stamp on every merged frame. The fixture
            # mirrors the strict-< rule: market_cap_available_at = D 00:00
            # for a daily bar at D 00:00 means availability is one calendar
            # day in the past.
            "market_cap_available_at": ts - pd.Timedelta(days=1),
        }))
    pd.concat(price_rows, ignore_index=True).to_parquet(
        feature_dir / "price_features_1d.parquet", index=False)
    # SENTIMENT: BTC + ETH have full coverage, SOL has *no* rows.
    sent_rows = []
    for tk in ("BTC", "ETH"):
        sent_rows.append(pd.DataFrame({
            "timestamp": ts, "ticker": tk,
            "vader_title_score_mean": np.linspace(-0.5, 0.5, n),
            "vader_bullishness_ratio": np.linspace(0.4, 0.6, n),
            "vader_title_score_std": np.linspace(0.1, 0.2, n),
            "post_count": np.arange(n, dtype=int),
        }))
    pd.concat(sent_rows, ignore_index=True).to_parquet(
        feature_dir / "sentiment_features_1d.parquet", index=False)
    # COVERAGE: SOL is below the 85% threshold.
    pd.DataFrame({
        "ticker":       ["BTC", "ETH", "SOL"],
        "horizon":      ["1d",  "1d",  "1d"],
        "coverage_pct": [100.0, 100.0, 0.0],
    }).to_csv(feature_dir / "sentiment_coverage.csv", index=False)
    monkeypatch.chdir(tmp_path)
    return tmp_path


def test_merge_default_keeps_low_coverage_ticker(merge_repo):
    """v4 pooled main spec: the coverage filter is OFF by default so
    SOL stays in the merged frame even though its sentiment coverage
    is below threshold. Neutral fill remains on."""
    rc = fm.main(["--horizon", "1d"])
    assert rc == 0
    merged = pd.read_parquet(merge_repo / "Data" / "Final" / "features_1d.parquet")
    assert set(merged["ticker"]) == {"BTC", "ETH", "SOL"}
    report = pd.read_csv(merge_repo / "Data" / "Final" / "merge_report.csv")
    assert bool(report.loc[0, "apply_coverage_filter"]) is False
    assert bool(report.loc[0, "neutral_fill_sentiment"]) is True


def test_explicit_apply_coverage_filter_drops_low_coverage_ticker(merge_repo):
    """Explicit robustness path: --apply-coverage-filter re-enables the
    legacy behaviour and drops SOL."""
    rc = fm.main(["--horizon", "1d", "--apply-coverage-filter"])
    assert rc == 0
    merged = pd.read_parquet(merge_repo / "Data" / "Final" / "features_1d.parquet")
    assert set(merged["ticker"]) == {"BTC", "ETH"}
    report = pd.read_csv(merge_repo / "Data" / "Final" / "merge_report.csv")
    assert bool(report.loc[0, "apply_coverage_filter"]) is True


def test_no_sentiment_coverage_filter_retains_low_coverage_coin(merge_repo):
    """Legacy alias: --no-sentiment-coverage-filter still works and
    matches the new default behaviour."""
    rc = fm.main(["--horizon", "1d", "--no-sentiment-coverage-filter"])
    assert rc == 0
    merged = pd.read_parquet(merge_repo / "Data" / "Final" / "features_1d.parquet")
    assert "SOL" in set(merged["ticker"])
    # Default neutral fill replaces SOL's missing sentiment.
    sol = merged[merged["ticker"] == "SOL"]
    assert sol["vader_title_score_mean"].notna().all()
    assert (sol["vader_title_score_mean"] == 0.0).all()
    assert (sol["vader_bullishness_ratio"] == 0.5).all()
    assert (sol["post_count"] == 0).all()
    # Non-sentiment columns are untouched (no neutral fill leakage).
    assert sol["log_return_t"].notna().all()
    assert (sol["log_return_t"] != 0).any()


def test_no_neutral_fill_leaves_missing_sentiment_nan(merge_repo):
    rc = fm.main(["--horizon", "1d", "--no-sentiment-coverage-filter",
                  "--no-neutral-fill-missing-sentiment"])
    assert rc == 0
    merged = pd.read_parquet(merge_repo / "Data" / "Final" / "features_1d.parquet")
    sol = merged[merged["ticker"] == "SOL"]
    # Sentiment columns stay NaN when the fill flag is explicitly disabled.
    assert sol["vader_title_score_mean"].isna().all()
    assert sol["vader_bullishness_ratio"].isna().all()
    # Non-sentiment columns are not touched in either direction.
    assert sol["log_return_t"].notna().all()
    assert sol["realized_vol_14d"].notna().all()


def test_neutral_fill_sets_sentiment_std_to_zero(merge_repo):
    """*_std sentiment columns must be filled with 0.0 so panel_logit's
    dropna(subset=feature_cols + ['target']) does not silently drop rich
    rows for low-coverage tickers when neutral fill is enabled.
    """
    rc = fm.main(["--horizon", "1d", "--no-sentiment-coverage-filter"])
    assert rc == 0
    merged = pd.read_parquet(merge_repo / "Data" / "Final" / "features_1d.parquet")
    sol = merged[merged["ticker"] == "SOL"]
    # SOL had no sentiment rows — *_std must now be 0.0, not NaN.
    assert sol["vader_title_score_std"].notna().all()
    assert (sol["vader_title_score_std"] == 0.0).all()
    # And the BTC/ETH rows that did have real std values stay unchanged.
    btc = merged[merged["ticker"] == "BTC"]["vader_title_score_std"]
    assert btc.notna().all()
    assert (btc > 0).any()


def test_no_neutral_fill_leaves_sentiment_std_nan(merge_repo):
    rc = fm.main(["--horizon", "1d", "--no-sentiment-coverage-filter",
                  "--no-neutral-fill-missing-sentiment"])
    assert rc == 0
    merged = pd.read_parquet(merge_repo / "Data" / "Final" / "features_1d.parquet")
    sol = merged[merged["ticker"] == "SOL"]
    assert sol["vader_title_score_std"].isna().all()


def test_fill_missing_sentiment_fills_each_column_kind_correctly():
    """Spot-check every neutral-fill branch in one go on a synthetic frame
    so a future edit can't quietly regress one of them.
    """
    # Variante A: *_weighted_mean is no longer a valid sentiment column and
    # must NOT be auto-filled; including one here verifies it stays NaN.
    df = pd.DataFrame({
        "ticker":                       ["BTC", "BTC"],
        "post_count":                   [np.nan, 3],
        "vader_post_count":             [np.nan, 7],
        "vader_directional_post_count": [np.nan, 5],
        "vader_title_score_mean":       [np.nan, 0.42],
        "vader_combined_weighted_mean": [np.nan, 0.10],  # forbidden — stays NaN
        "vader_title_score_median":     [np.nan, 0.05],
        "vader_title_score_std":        [np.nan, 0.20],
        "vader_combined_score_std":     [np.nan, 0.30],
        "cryptobert_bullishness_ratio": [np.nan, 0.60],
        # Non-sentiment columns the function must NEVER touch.
        "log_return_t":                 [np.nan, 0.01],
        "realized_vol_14d":             [np.nan, 0.02],
        "target":                       [np.nan, 1.0],
    })
    sent_cols = [c for c in df.columns
                 if c not in ("ticker", "log_return_t", "realized_vol_14d",
                              "target")]
    out = fm.fill_missing_sentiment(df.copy(), sent_cols)
    # post_count / directional_post_count → 0
    assert (out["post_count"] == [0, 3]).all()
    assert (out["vader_post_count"] == [0, 7]).all()
    assert (out["vader_directional_post_count"] == [0, 5]).all()
    # mean / median → 0.0
    assert out["vader_title_score_mean"].iat[0] == 0.0
    assert out["vader_title_score_median"].iat[0] == 0.0
    # std → 0.0
    assert out["vader_title_score_std"].iat[0] == 0.0
    assert out["vader_combined_score_std"].iat[0] == 0.0
    # bullishness_ratio → 0.5
    assert out["cryptobert_bullishness_ratio"].iat[0] == 0.5
    # Variante A: *_weighted_mean must stay NaN — engagement weighting was removed.
    assert pd.isna(out["vader_combined_weighted_mean"].iat[0])
    # Non-sentiment columns stay NaN.
    assert pd.isna(out["log_return_t"].iat[0])
    assert pd.isna(out["realized_vol_14d"].iat[0])
    assert pd.isna(out["target"].iat[0])
    # And real values are preserved on the non-NaN rows.
    assert out["vader_title_score_std"].iat[1] == 0.20


def test_non_sentiment_std_like_columns_are_not_filled():
    """``realized_vol_14d`` looks std-ish in spirit but is NOT a sentiment
    column, so ``sentiment_neutral_columns`` must keep it out of scope.
    """
    cols = ["realized_vol_14d", "vader_title_score_std", "cryptobert_combined_score_std",
            "log_return_t", "target", "timestamp", "ticker"]
    fill_targets = set(fm.sentiment_neutral_columns(cols))
    assert {"vader_title_score_std", "cryptobert_combined_score_std"} <= fill_targets
    assert "realized_vol_14d" not in fill_targets
    assert fill_targets.isdisjoint({"log_return_t", "target", "timestamp", "ticker"})


def test_neutral_fill_targets_only_sentiment_attention_columns():
    cols = ["vader_title_score_mean", "vader_bullishness_ratio",
            "vader_title_score_std", "post_count", "log_return_t",
            "realized_vol_14d", "market_cap_t", "target", "timestamp", "ticker"]
    fill_targets = set(fm.sentiment_neutral_columns(cols))
    # Sentiment / attention columns are in scope.
    assert {"vader_title_score_mean", "vader_bullishness_ratio",
            "vader_title_score_std", "post_count"} <= fill_targets
    # Non-sentiment columns are NEVER neutral-fill targets.
    assert fill_targets.isdisjoint({"log_return_t", "realized_vol_14d",
                                     "market_cap_t", "target",
                                     "timestamp", "ticker"})
