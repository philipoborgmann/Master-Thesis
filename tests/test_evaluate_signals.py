"""Unit + integration tests for the signal evaluation stage.

Synthetic signals + synthetic OHLCV are written to a temporary directory so
the tests run without any real ``Data/`` or ``Outputs/`` content.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import shutil

from thesis_pipeline import cli
from thesis_pipeline.evaluation import (
    loading, metrics, reporting, significance, thresholds, volatility,
)
from thesis_pipeline.evaluation import evaluate_signals as eval_main


# ---------------------------------------------------------------------------
# Fixtures: build a minimal but realistic environment under tmp_path
# ---------------------------------------------------------------------------

def _synthetic_signals(rng: np.random.Generator, n: int = 120,
                       ticker: str = "BTC", horizon: str = "1d",
                       set_id: str = "B1", sentiment_model: str = "-",
                       start: datetime = datetime(2024, 1, 1, tzinfo=timezone.utc),
                       step: timedelta = timedelta(days=1),
                       accuracy_target: float = 0.55) -> pd.DataFrame:
    target = rng.integers(0, 2, size=n).astype(int)
    # Generate predictions that hit `accuracy_target` on average.
    flip = rng.random(n) > accuracy_target
    pred = np.where(flip, 1 - target, target).astype(int)
    proba = np.where(pred == 1,
                     rng.uniform(0.55, 0.85, size=n),
                     rng.uniform(0.15, 0.45, size=n))
    ts = [start + i * step for i in range(n)]
    return pd.DataFrame({
        "timestamp": ts,
        "ticker":    ticker,
        "target":    target,
        "prediction": pred,
        "probability": proba.astype(float),
        "set_id":    set_id,
        "sentiment_model": sentiment_model,
        "horizon":   horizon,
    })


def _synthetic_ohlcv(rng: np.random.Generator, n_days: int = 180,
                     start: datetime = datetime(2023, 12, 1)) -> pd.DataFrame:
    base = 20_000 + rng.normal(0, 200, n_days).cumsum()
    high = base + np.abs(rng.normal(50, 30, n_days))
    low  = base - np.abs(rng.normal(50, 30, n_days))
    open_ = base + rng.normal(0, 20, n_days)
    close = base + rng.normal(0, 20, n_days)
    ts = [start + timedelta(days=i) for i in range(n_days)]
    return pd.DataFrame({
        "timestamp": ts,
        "open": open_, "high": high, "low": low, "close": close,
        "volume": np.abs(rng.normal(1000, 100, n_days)),
    })


@pytest.fixture
def signals_env(tmp_path, monkeypatch):
    """Set up Outputs/Signals/, feature_sets.xlsx and Data/Raw/Price/1d/ in tmp_path,
    then point ``project_root`` (and therefore ``resolve_path``) at it."""
    # Layout
    signals_root = tmp_path / "Outputs" / "Signals"
    (signals_root / "1d").mkdir(parents=True)
    (signals_root / "1h").mkdir(parents=True)
    raw_1d = tmp_path / "Data" / "Raw" / "Price" / "1d"
    raw_1d.mkdir(parents=True)

    rng = np.random.default_rng(0)

    # v4 synthetic signal files. The v4 nested H1 test is "ECON_VAD_F vs ECON"
    # so the fixture must carry ECON (matched benchmark) and at least one
    # combined / sentiment-only set.
    sigs = {
        ("ECON", "-"):           _synthetic_signals(rng, set_id="ECON",       sentiment_model="-",      accuracy_target=0.55),
        ("SENT_VAD_L", "vader"): _synthetic_signals(rng, set_id="SENT_VAD_L", sentiment_model="vader",  accuracy_target=0.58),
        ("ECON_VAD_F", "vader"): _synthetic_signals(rng, set_id="ECON_VAD_F", sentiment_model="vader",  accuracy_target=0.60),
        # An empty file to ensure it is skipped gracefully
    }
    for (sid, sm), df in sigs.items():
        name = f"{sid}_{sm}.parquet" if sm and sm != "-" else f"{sid}.parquet"
        df.to_parquet(signals_root / "1d" / name, index=False)

    # An intentionally empty parquet (should be skipped by the loader)
    pd.DataFrame(columns=df.columns).to_parquet(signals_root / "1d" / "EMPTY.parquet",
                                                index=False)

    # A different-horizon signal so we can verify horizon filtering
    sig_1h = _synthetic_signals(rng, set_id="ECON", sentiment_model="-",
                                horizon="1h", n=80, step=timedelta(hours=1))
    sig_1h.to_parquet(signals_root / "1h" / "ECON.parquet", index=False)

    # Daily OHLCV for BTC
    ohlcv = _synthetic_ohlcv(rng)
    ohlcv.to_parquet(raw_1d / "BTCUSDT_1d.parquet", index=False)

    # Minimal feature_sets.xlsx (v4 ids).
    feature_sets = pd.DataFrame({
        "set_id":   ["ECON", "SENT_VAD_L", "ECON_VAD_F"],
        "category": ["benchmark", "sentiment_vader", "combined_vader"],
        "sentiment_model": ["-", "vader", "vader"],
        "label":    ["econ", "vad-mean", "econ+vad-full"],
        "feature_columns_comma_separated": [
            "log_return_t,cum_log_return_7d,cum_log_return_14d,cum_log_return_21d,realized_vol_14d,volume_diff,log_market_cap_lag1",
            "vader_title_score_mean",
            "log_return_t,vader_title_score_mean",
        ],
    })
    fs_path = tmp_path / "feature_sets.xlsx"
    with pd.ExcelWriter(fs_path, engine="openpyxl") as w:
        feature_sets.to_excel(w, sheet_name="feature_sets", index=False)

    # Re-root path resolution so resolve_path(...) → tmp_path/...
    from thesis_pipeline import config as cfg
    monkeypatch.setattr(cfg, "project_root", lambda: tmp_path)
    cfg.load_config.cache_clear()
    # Also clear configs dir cache by writing a marker — load_config reads from
    # ``configs/`` under the patched project_root.
    shutil.copytree(
        Path(__file__).resolve().parents[1] / "configs",
        tmp_path / "configs",
        )
    return {
        "root": tmp_path,
        "signals_root": signals_root,
        "raw_1d": raw_1d,
        "feature_sets": fs_path,
    }


# ---------------------------------------------------------------------------
# loading.py
# ---------------------------------------------------------------------------

def test_discover_signal_files_filters_by_horizon(signals_env):
    files_all = loading.discover_signal_files()
    files_1d  = loading.discover_signal_files("1d")
    files_1h  = loading.discover_signal_files("1h")
    # v4 fixture: 3 valid (ECON, SENT_VAD_L_vader, ECON_VAD_F_vader)
    # + 1 EMPTY.parquet on the 1d horizon = 4.
    assert len(files_all) >= 4
    assert len(files_1d) == 4
    assert len(files_1h) == 1
    assert all(p.suffix == ".parquet" for p in files_all)


def test_load_signal_file_skips_empty_and_malformed(signals_env, tmp_path):
    empty_path = signals_env["signals_root"] / "1d" / "EMPTY.parquet"
    assert loading.load_signal_file(empty_path) is None

    # Malformed file — missing 'target' column
    bad = pd.DataFrame({"timestamp": [pd.Timestamp.utcnow()], "ticker": ["BTC"]})
    bad_path = tmp_path / "BAD.parquet"
    bad.to_parquet(bad_path, index=False)
    assert loading.load_signal_file(bad_path) is None


def test_load_all_signals_carries_metadata_from_columns(signals_env):
    df = loading.load_all_signals("1d")
    assert not df.empty
    # set_id/sentiment_model/horizon come from columns, never the filename:
    assert set(df["set_id"].unique()) == {"ECON", "SENT_VAD_L", "ECON_VAD_F"}
    assert "vader" in df["sentiment_model"].unique()
    assert (df["horizon"] == "1d").all()


def test_load_all_signals_timezone_safe(signals_env):
    df = loading.load_all_signals("1d")
    # All timestamps must be UTC-aware.
    assert pd.api.types.is_datetime64_any_dtype(df["timestamp"])
    assert df["timestamp"].dt.tz is not None


# ---------------------------------------------------------------------------
# metrics.py
# ---------------------------------------------------------------------------

def test_pooled_and_per_ticker_metrics(signals_env):
    df = loading.load_all_signals("1d")
    pooled = metrics.pooled_metrics_table(df)
    per_tk = metrics.per_ticker_metrics_table(df)

    # 3 sets × 1 horizon × 1 sentiment_model-per-set (v4 fixture)
    assert len(pooled) == 3
    for col in ("accuracy", "balanced_accuracy", "f1", "precision", "recall",
                "log_loss", "brier_score", "TP", "TN", "FP", "FN",
                "positive_accuracy", "negative_accuracy", "n_obs", "n_tickers"):
        assert col in pooled.columns
    # Per-ticker: one ticker (BTC) per set
    assert (per_tk["ticker"] == "BTC").all()
    assert len(per_tk) == 3


def test_confusion_diagnostics_synthetic():
    df = pd.DataFrame({
        "target":     [1, 1, 0, 0, 1, 0],
        "prediction": [1, 0, 0, 1, 1, 0],
        "probability": [0.9, 0.4, 0.2, 0.6, 0.7, 0.3],
        "timestamp":  pd.date_range("2024-01-01", periods=6, tz="UTC"),
    })
    diag = metrics.confusion_diagnostics(df)
    # TP: pred=1,true=1 → rows 0,4 → 2
    # FP: pred=1,true=0 → row 3 → 1
    # TN: pred=0,true=0 → rows 2,5 → 2
    # FN: pred=0,true=1 → row 1 → 1
    assert diag["TP"] == 2 and diag["FP"] == 1
    assert diag["TN"] == 2 and diag["FN"] == 1
    assert diag["positive_accuracy"] == pytest.approx(2 / 3)
    assert diag["negative_accuracy"] == pytest.approx(2 / 3)


def test_per_ticker_single_class_returns_nan_for_class_metrics():
    """When a ticker has only one class, f1/precision/recall/log_loss → NaN."""
    df = pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=10, tz="UTC"),
        "ticker":    "ETH",
        "target":    [1] * 10,
        "prediction": [1] * 10,
        "probability": [0.7] * 10,
        "set_id": "E1",
        "sentiment_model": "-",
        "horizon": "1d",
    })
    out = metrics.per_ticker_metrics_table(df)
    row = out.iloc[0]
    for col in ("f1", "precision", "recall", "log_loss"):
        assert pd.isna(row[col]), f"{col} should be NaN for single-class slice"


# ---------------------------------------------------------------------------
# thresholds.py
# ---------------------------------------------------------------------------

def test_threshold_at_05_covers_everything():
    df = pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=100, tz="UTC"),
        "target":     np.random.default_rng(1).integers(0, 2, 100),
        "prediction": np.random.default_rng(2).integers(0, 2, 100),
        "probability": np.linspace(0.01, 0.99, 100),
        "set_id": "X", "sentiment_model": "-", "horizon": "1d",
    })
    out = thresholds.threshold_metrics_for_group(df, thresholds=(0.5, 0.6))
    row_05 = out[out["threshold"] == 0.5].iloc[0]
    assert row_05["coverage"] == 1.0
    assert row_05["n_trades"] == 100
    row_06 = out[out["threshold"] == 0.6].iloc[0]
    # At threshold 0.6, only |p - 0.5| > 0.1 counts
    expected_trades = int(((df["probability"] > 0.6) | (df["probability"] < 0.4)).sum())
    assert row_06["n_trades"] == expected_trades
    assert row_06["coverage"] == pytest.approx(expected_trades / 100)


def test_threshold_analysis_integration(signals_env):
    df = loading.load_all_signals("1d")
    out = thresholds.threshold_analysis_table(df)
    assert not out.empty
    assert set(out["threshold"].unique()) == {0.50, 0.55, 0.60, 0.65}
    # Coverage at 0.5 is always 1
    assert (out[out["threshold"] == 0.5]["coverage"] == 1.0).all()
    # Coverage is monotonically non-increasing as threshold increases
    for (hz, sid, sm), grp in out.groupby(["horizon", "set_id", "sentiment_model"]):
        cov = grp.sort_values("threshold")["coverage"].tolist()
        assert all(cov[i] >= cov[i + 1] - 1e-9 for i in range(len(cov) - 1)), \
            f"coverage should be non-increasing for {sid}/{sm}: {cov}"


# ---------------------------------------------------------------------------
# volatility.py
# ---------------------------------------------------------------------------

def test_garman_klass_and_rolling_shift():
    df = pd.DataFrame({
        "date":  pd.date_range("2024-01-01", periods=30).date,
        "open":  np.linspace(100, 110, 30),
        "high":  np.linspace(101, 112, 30),
        "low":   np.linspace(99, 108, 30),
        "close": np.linspace(100.5, 111, 30),
    })
    gk = volatility.garman_klass_variance(df)
    assert len(gk) == 30
    assert (gk >= 0).all()
    rolled = volatility.rolling_volatility(gk, window=5)
    # shift(1) → first valid index has NaN
    assert pd.isna(rolled.iloc[0])
    # last value of rolled should equal rolling-mean of [t-5..t-1]
    expected = gk.iloc[-6:-1].mean()
    assert rolled.iloc[-1] == pytest.approx(expected)


def test_assign_tercile_regimes_balanced():
    vol = pd.Series(np.arange(30, dtype=float))
    reg = volatility.assign_tercile_regimes(vol)
    counts = reg.value_counts()
    # 30 values, q=3 → 10 per bucket
    assert counts.to_dict() == {"low": 10, "mid": 10, "high": 10}
    # Lowest values get "low"
    assert reg.iloc[0] == "low"
    assert reg.iloc[-1] == "high"


def test_volatility_stratification_uses_nano_xno_alias(signals_env, tmp_path):
    # Drop in a NANO signal but only an XNO OHLCV file — the loader must
    # still find it via the alias.
    raw_1d = signals_env["raw_1d"]
    rng = np.random.default_rng(7)
    ohlcv = _synthetic_ohlcv(rng, n_days=90)
    ohlcv.to_parquet(raw_1d / "XNOUSDT_1d.parquet", index=False)
    out = volatility.load_daily_ohlcv("NANO")
    assert out is not None and not out.empty


def test_volatility_stratification_integration(signals_env):
    df = loading.load_all_signals("1d")
    lookup = volatility.build_regime_lookup(["BTC"])
    assert not lookup.empty
    out = volatility.volatility_stratification_table(df, lookup)
    assert not out.empty
    # Every (set, regime) combo present
    assert set(out["vol_regime"]) >= {"low", "mid", "high"}
    for col in ("accuracy", "brier_score", "n_obs", "n_days"):
        assert col in out.columns


# ---------------------------------------------------------------------------
# significance.py
# ---------------------------------------------------------------------------

def test_mcnemar_continuity_corrected_known_values():
    # Classic example: b=8, c=2, |8-2|-1=5, 5^2/10 = 2.5
    stat, pval = significance.mcnemar_continuity_corrected(8, 2)
    assert stat == pytest.approx(2.5)
    assert 0.0 < pval < 1.0


def test_mcnemar_b_plus_c_zero_is_p_one():
    stat, pval = significance.mcnemar_continuity_corrected(0, 0)
    assert stat == 0.0
    assert pval == 1.0


def test_mcnemar_table_excludes_benchmarks_and_non_eligible(signals_env):
    df = loading.load_all_signals("1d")
    fs = pd.read_excel(signals_env["feature_sets"], sheet_name="feature_sets")
    fs.columns = [c.strip().lower().replace(" ", "_") for c in fs.columns]
    df = loading.attach_feature_set_metadata(df, fs)
    out = significance.mcnemar_table(df, benchmarks=("ECON",))
    # The benchmark itself never appears as the *tested* set_id.
    assert "ECON" not in out["set_id"].unique()
    # Only sentiment + combined are eligible → SENT_VAD_L and ECON_VAD_F,
    # each tested against ECON.
    assert set(out["set_id"]) == {"SENT_VAD_L", "ECON_VAD_F"}
    assert set(out["benchmark"]) == {"ECON"}


def test_mcnemar_wide_disambiguates_benchmarks():
    long = pd.DataFrame({
        "horizon": ["1d", "1d"],
        "set_id":  ["S1", "S1"],
        "sentiment_model": ["vader", "vader"],
        "benchmark": ["B1", "B2"],
        "mcnemar_stat": [3.5, 1.2],
        "mcnemar_pval": [0.02, 0.27],
        "significant_vs_benchmark": [True, False],
        "b": [10, 12],
        "c": [20, 14],
        "n_matched": [100, 100],
    })
    wide = significance.mcnemar_wide(long, benchmarks=("B1", "B2"))
    assert len(wide) == 1
    row = wide.iloc[0]
    assert row["mcnemar_pval_vs_b1"] == pytest.approx(0.02)
    assert row["mcnemar_pval_vs_b2"] == pytest.approx(0.27)
    assert bool(row["significant_vs_b1"]) is True
    assert bool(row["significant_vs_b2"]) is False


# ---------------------------------------------------------------------------
# Top-level CLI / integration
# ---------------------------------------------------------------------------

def test_cli_dry_run_does_not_write_files(signals_env, capsys):
    out_dir = signals_env["root"] / "Outputs" / "Evaluation"
    rc = cli.main([
        "evaluate-signals", "--horizon", "1d", "--smoke", "--dry-run",
    ])
    assert rc == 0
    # No outputs were produced
    assert not out_dir.exists() or not any(out_dir.iterdir())


def test_full_evaluation_writes_excel_and_csvs(signals_env):
    out_dir = signals_env["root"] / "Outputs" / "Evaluation"
    rc = eval_main.main([
        "--horizon", "1d", "--force",
        "--output-dir", str(out_dir),
        "--feature-config", str(signals_env["feature_sets"]),
    ])
    assert rc == 0
    assert (out_dir / "signal_evaluation.xlsx").exists()
    for name in ("pooled_metrics.csv", "per_ticker_metrics.csv",
                 "threshold_analysis.csv", "volatility_stratification.csv",
                 "threshold_lift.csv", "mcnemar_tests.csv",
                 "regime_mcnemar_tests.csv", "regime_mcnemar_summary.csv"):
        assert (out_dir / name).exists()
    # The regime McNemar table is DESCRIPTIVE ONLY: raw direction + effect
    # columns, but NO BH q-value / BH significance (superseded by the
    # confirmatory difference-in-improvement, Families C/D).
    regime = pd.read_csv(out_dir / "regime_mcnemar_tests.csv")
    for col in ("regime_type", "direction", "net_improvement",
                "discordant_advantage", "mcnemar_pval", "test_valid"):
        assert col in regime.columns
    for col in ("q_value_bh", "significant_bh_5pct", "model_better_bh_5pct",
                "benchmark_better_bh_5pct"):
        assert col not in regime.columns
    # Pooled CSV carries the raw McNemar-vs-econ reference AND a single
    # BH-corrected directional verdict; the uncorrected booleans are gone.
    pooled = pd.read_csv(out_dir / "pooled_metrics.csv")
    assert set(pooled["set_id"]) == {"ECON", "SENT_VAD_L", "ECON_VAD_F"}
    for col in ("mcnemar_pval_vs_econ", "mcnemar_q_value_bh_vs_econ",
                "mcnemar_significant_bh_vs_econ", "family"):
        assert col in pooled.columns
    # No uncorrected directional significance boolean survives.
    assert "significant_vs_econ" not in pooled.columns
    assert "significant_vs_benchmark" not in pooled.columns
    # ECON's McNemar columns are NaN (you don't test ECON vs ECON);
    # sentiment/combined have a real test.
    bench = pooled[pooled["set_id"] == "ECON"].iloc[0]
    assert pd.isna(bench["mcnemar_pval_vs_econ"])
    sent = pooled[pooled["set_id"] == "SENT_VAD_L"].iloc[0]
    assert pd.notna(sent["mcnemar_pval_vs_econ"])
    # The ECON benchmark row carries no directional verdict at all.
    assert pd.isna(bench["mcnemar_q_value_bh_vs_econ"])
    # Combined (ECON_VAD_F) rows are tagged Family B; sentiment-only (SENT_*)
    # rows are tagged the exploratory family.
    comb = pooled[pooled["set_id"] == "ECON_VAD_F"].iloc[0]
    assert comb["family"] == "B_H1_directional"
    assert sent["family"] == "EXPL_sentiment_only_directional"

    # threshold_lift.csv must be populated for sentiment/combined sets and
    # carry lift_accuracy = accuracy_model − accuracy_benchmark.
    lift = pd.read_csv(out_dir / "threshold_lift.csv")
    assert not lift.empty
    assert set(lift["set_id"]) == {"SENT_VAD_L", "ECON_VAD_F"}
    assert set(lift["threshold"]) == {0.50, 0.55, 0.60, 0.65}
    # Algebraic invariant: lift_accuracy == accuracy_model − accuracy_benchmark.
    valid = lift.dropna(subset=["accuracy_model", "accuracy_benchmark", "lift_accuracy"])
    diff = valid["accuracy_model"] - valid["accuracy_benchmark"]
    assert np.allclose(diff.values, valid["lift_accuracy"].values, atol=1e-6)


def test_confirmatory_outputs_and_sanity(signals_env):
    """The centralized confirmatory layer must write its four new CSVs,
    enrich the incremental surface with Families A/B + the H1 decomposition,
    reconcile the Nt-weighted log-loss effect with ``log_loss_improvement``,
    keep the diagnostic floor family-free, and never emit a 10% flag."""
    out_dir = signals_env["root"] / "Outputs" / "Evaluation_conf"
    rc = eval_main.main([
        "--horizon", "1d", "--force",
        "--output-dir", str(out_dir),
        "--feature-config", str(signals_env["feature_sets"]),
    ])
    assert rc == 0

    # ── The four new confirmatory CSVs exist ──────────────────────
    for name in ("horizon_comparison.csv", "multiple_testing_manifest.csv",
                 "metric_roles.csv", "class_balance.csv",
                 "incremental_sentiment_value.csv", "absolute_vs_naive.csv"):
        assert (out_dir / name).exists(), name

    # ── metric_roles taxonomy ─────────────────────────────────────
    roles = pd.read_csv(out_dir / "metric_roles.csv")
    role_map = dict(zip(roles["metric"], roles["role"]))
    assert role_map["log_loss"] == "primary_probabilistic"
    assert role_map["accuracy"] == "primary_directional"
    assert bool(roles.loc[roles["metric"] == "log_loss", "used_for_hpo"].iloc[0])

    # ── class_balance ─────────────────────────────────────────────
    cb = pd.read_csv(out_dir / "class_balance.csv")
    assert {"horizon", "n", "base_rate_up", "delta_from_0.5",
            "imbalance_flag"}.issubset(cb.columns)

    # ── incremental enrichment: Family A + B + H1 decomposition ────
    inc = pd.read_csv(out_dir / "incremental_sentiment_value.csv")
    for col in ("family", "p_value_raw", "q_value_bh", "significant_bh",
                "ll_effect", "ll_effect_ntweighted", "ll_q_value_bh",
                "ll_significant_bh", "h1a_supported_bh", "h1b_supported_bh",
                "h1_support_class"):
        assert col in inc.columns, col
    # No 10% flag anywhere.
    assert "significant_bh_10pct" not in inc.columns
    # Family B is the directional confirmatory family for every row.
    assert set(inc["family"].dropna()) <= {"B_H1_directional"}
    # h1_support_class only ever takes the four taxonomy values.
    assert set(inc["h1_support_class"].dropna()) <= {
        "strong", "probabilistic", "directional", "none"}

    # ── SANITY: Nt-weighted log-loss effect == log_loss_improvement ──
    ok = inc[inc["status"] == "ok"].copy()
    sane = ok.dropna(subset=["ll_effect_ntweighted", "log_loss_improvement"])
    if not sane.empty:
        assert np.allclose(sane["ll_effect_ntweighted"].to_numpy(float),
                           sane["log_loss_improvement"].to_numpy(float),
                           atol=1e-9)

    # ── absolute_vs_naive is a diagnostic floor: raw only, no family/q ──
    if (out_dir / "absolute_vs_naive.csv").exists():
        avn = pd.read_csv(out_dir / "absolute_vs_naive.csv")
        if not avn.empty:
            assert "p_value_raw" in avn.columns
            assert "significant_raw" in avn.columns
            assert "q_value_bh" not in avn.columns
            assert "significant_bh" not in avn.columns

    # ── manifest: confirmatory A–E + exploratory sentiment-only ───
    man = pd.read_csv(out_dir / "multiple_testing_manifest.csv")
    assert not man["family"].astype(str).str.contains("absolute_vs_naive").any()
    assert not man["family"].astype(str).str.contains("regime_mcnemar").any()
    assert "family_role" in man.columns
    conf = man[man["family_role"] == "confirmatory"]
    assert set(conf["family"]) == {
        "A_H1_logloss", "B_H1_directional", "C_H2_volatility",
        "D_H3_marketcap", "E1_horizon_logloss", "E2_horizon_accuracy"}
    # The exploratory sentiment-only directional family is present and
    # marked exploratory (SENT_VAD_L is in the fixture).
    expl = man[man["family_role"] == "exploratory"]
    assert list(expl["family"]) == ["EXPL_sentiment_only_directional"]

    # ── FIX 2: pooled directional dedup — one BH verdict per (h,set) ──
    pooled = pd.read_csv(out_dir / "pooled_metrics.csv")
    assert "significant_vs_econ" not in pooled.columns
    assert "significant_vs_benchmark" not in pooled.columns
    for col in ("mcnemar_q_value_bh_vs_econ", "mcnemar_significant_bh_vs_econ",
                "family"):
        assert col in pooled.columns
    # No NaN verdict on any row that carries a non-null raw p-value.
    tested = pooled[pooled["mcnemar_pval_vs_econ"].notna()]
    assert tested["mcnemar_q_value_bh_vs_econ"].notna().all()
    assert tested["mcnemar_significant_bh_vs_econ"].notna().all()
    # Combined rows equal Family B; sentiment-only rows equal the EXPL BH.
    b_lookup = ok.set_index(["horizon", "set_id"])[
        ["q_value_bh", "significant_bh"]]
    for _, pr in pooled[pooled["set_id"] == "ECON_VAD_F"].iterrows():
        assert pr["family"] == "B_H1_directional"
        key = (pr["horizon"], pr["set_id"])
        if key in b_lookup.index:
            fam_b = b_lookup.loc[key]
            assert np.isclose(float(pr["mcnemar_q_value_bh_vs_econ"]),
                              float(fam_b["q_value_bh"]), atol=1e-12)
            assert (bool(pr["mcnemar_significant_bh_vs_econ"])
                    == bool(fam_b["significant_bh"]))
    for _, pr in pooled[pooled["set_id"] == "SENT_VAD_L"].iterrows():
        assert pr["family"] == "EXPL_sentiment_only_directional"

    # ── FIX 3: accuracy-effect parity column ──────────────────────
    assert "acc_effect_ntweighted" in inc.columns
    parity = ok.dropna(subset=["acc_effect_ntweighted", "accuracy_lift"])
    if not parity.empty:
        assert np.allclose(parity["acc_effect_ntweighted"].to_numpy(float),
                           parity["accuracy_lift"].to_numpy(float), atol=1e-9)


def test_no_volatility_flag_skips_regime_step(signals_env):
    out_dir = signals_env["root"] / "Outputs" / "Evaluation_NV"
    rc = eval_main.main([
        "--horizon", "1d", "--force",
        "--output-dir", str(out_dir),
        "--feature-config", str(signals_env["feature_sets"]),
        "--no-volatility",
    ])
    assert rc == 0
    vol_path = out_dir / "volatility_stratification.csv"
    # An empty volatility table produces a header-less CSV; both an empty
    # frame and an EmptyDataError are acceptable here.
    try:
        vol_df = pd.read_csv(vol_path)
        assert vol_df.empty
    except pd.errors.EmptyDataError:
        pass


def test_cli_evaluate_signals_help_runs():
    parser = cli.build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["evaluate-signals", "--help"])


def test_threshold_lift_algebra_synthetic():
    """The matched-observation lift must equal acc_model − acc_benchmark."""
    rng = np.random.default_rng(42)
    n = 200
    base_ts = pd.date_range("2024-01-01", periods=n, freq="D", tz="UTC")
    target = rng.integers(0, 2, n)

    # Model: 60% accuracy, probability skewed away from 0.5
    flip_m = rng.random(n) > 0.6
    pred_m = np.where(flip_m, 1 - target, target).astype(int)
    prob_m = np.where(pred_m == 1,
                      rng.uniform(0.55, 0.95, n),
                      rng.uniform(0.05, 0.45, n))

    # Benchmark: always 1, prob 0.5
    pred_b = np.ones(n, dtype=int)
    prob_b = np.full(n, 0.5)

    sig = pd.concat([
        pd.DataFrame({"timestamp": base_ts, "ticker": "BTC",
                       "target": target, "prediction": pred_m,
                       "probability": prob_m, "set_id": "S1",
                       "sentiment_model": "vader", "horizon": "1d",
                       "category": "sentiment"}),
        pd.DataFrame({"timestamp": base_ts, "ticker": "BTC",
                       "target": target, "prediction": pred_b,
                       "probability": prob_b, "set_id": "B1",
                       "sentiment_model": "-", "horizon": "1d",
                       "category": "benchmark"}),
    ], ignore_index=True)

    out = thresholds.threshold_lift_table(sig, benchmarks=("B1",),
                                          thresholds=(0.50, 0.65))
    assert not out.empty
    # All sentiment rows present, both thresholds
    assert set(out["threshold"]) == {0.50, 0.65}
    # Coverage at 0.5 == 1.0 (model trades on every observation)
    cov50 = out[out["threshold"] == 0.50]["coverage_model"].iloc[0]
    assert cov50 == pytest.approx(1.0)
    # n_matched must shrink (or equal) when threshold rises
    n_m50 = out[out["threshold"] == 0.50]["n_matched"].iloc[0]
    n_m65 = out[out["threshold"] == 0.65]["n_matched"].iloc[0]
    assert n_m65 <= n_m50


def test_full_evaluation_writes_economic_diagnostics(signals_env):
    """economic_diagnostics.csv must exist after a full run and contain one row
    per attempted signal group (set_id × sentiment_model × model_type × …)."""
    out_dir = signals_env["root"] / "Outputs" / "Evaluation_diag"
    rc = eval_main.main([
        "--horizon", "1d", "--force",
        "--output-dir", str(out_dir),
        "--feature-config", str(signals_env["feature_sets"]),
    ])
    assert rc == 0
    diag_path = out_dir / "economic_diagnostics.csv"
    assert diag_path.exists(), "economic_diagnostics.csv missing"
    diag = pd.read_csv(diag_path)
    # All four signal sets in the fixture must appear in diagnostics.
    assert set(diag["set_id"]) == {"ECON", "SENT_VAD_L", "ECON_VAD_F"}
    for col in ("horizon", "set_id", "sentiment_model", "model_type",
                "panel_mode", "hpo_variant", "category", "label",
                "n_signal_rows", "n_unique_timestamps", "n_unique_tickers",
                "n_forward_return_rows", "n_joined_rows",
                "n_joined_non_null_forward_returns",
                "n_portfolio_periods", "status", "reason"):
        assert col in diag.columns


def test_full_evaluation_economic_csv_contains_all_groups(signals_env):
    """Regression: economic_performance.csv must contain every signal group, not
    just the last-loaded one. Pre-fix this file contained only one (set,
    sentiment) tuple per horizon."""
    out_dir = signals_env["root"] / "Outputs" / "Evaluation_econ"
    rc = eval_main.main([
        "--horizon", "1d", "--force",
        "--output-dir", str(out_dir),
        "--feature-config", str(signals_env["feature_sets"]),
    ])
    assert rc == 0
    econ = pd.read_csv(out_dir / "economic_performance.csv")
    # Each fixture set is single-ticker → can't form a high-low portfolio →
    # only BUY_HOLD survives in this synthetic env. The diagnostic CSV is
    # the one that pins the bug fix here.
    diag = pd.read_csv(out_dir / "economic_diagnostics.csv")
    # All four model sets attempted, none collapsed.
    assert set(diag["set_id"]) == {"ECON", "SENT_VAD_L", "ECON_VAD_F"}
    # BUY_HOLD still present in the metrics CSV (per the rest of the design).
    assert "BUY_HOLD" in econ["set_id"].astype(str).tolist()


def test_strict_feature_set_ids_drops_stale_signals(signals_env, tmp_path):
    """A stale signal file whose set_id is not registered in feature_sets.xlsx
    must be DROPPED before evaluation when --strict-feature-set-ids is set."""
    # Drop a stale signal file into the fixture
    rng = np.random.default_rng(99)
    stale = _synthetic_signals(rng, set_id="ZZ_STALE", sentiment_model="vader")
    stale.to_parquet(signals_env["signals_root"] / "1d" / "ZZ_STALE_vader.parquet",
                     index=False)

    out_dir = signals_env["root"] / "Outputs" / "Evaluation_strict"
    rc = eval_main.main([
        "--horizon", "1d", "--force",
        "--output-dir", str(out_dir),
        "--feature-config", str(signals_env["feature_sets"]),
        "--strict-feature-set-ids",
    ])
    assert rc == 0
    pooled = pd.read_csv(out_dir / "pooled_metrics.csv")
    assert "ZZ_STALE" not in pooled["set_id"].astype(str).unique()
    # Registered sets still present.
    assert {"ECON", "SENT_VAD_L", "ECON_VAD_F"}.issubset(
        set(pooled["set_id"].astype(str)))


def test_non_strict_feature_set_ids_keeps_stale_signals(signals_env):
    """Default (no flag) keeps stale signals; the diagnostic CSV records them."""
    rng = np.random.default_rng(98)
    stale = _synthetic_signals(rng, set_id="YY_STALE", sentiment_model="vader")
    stale.to_parquet(signals_env["signals_root"] / "1d" / "YY_STALE_vader.parquet",
                     index=False)

    out_dir = signals_env["root"] / "Outputs" / "Evaluation_lax"
    rc = eval_main.main([
        "--horizon", "1d", "--force",
        "--output-dir", str(out_dir),
        "--feature-config", str(signals_env["feature_sets"]),
    ])
    assert rc == 0
    pooled = pd.read_csv(out_dir / "pooled_metrics.csv")
    assert "YY_STALE" in pooled["set_id"].astype(str).unique()
    diag = pd.read_csv(out_dir / "economic_diagnostics.csv")
    assert "YY_STALE" in diag["set_id"].astype(str).unique()


def test_volatility_regime_join_dtype_stable(tmp_path):
    """Regression: signal date + OHLCV date must use the same dtype.

    Both sides go through ``pd.to_datetime(..., utc=True).dt.normalize()`` so
    the merge key is ``datetime64[ns, UTC]`` on both sides.
    """
    # Build a synthetic OHLCV with int-ms timestamps (the trickiest case)
    rng = np.random.default_rng(13)
    n_days = 60
    start = pd.Timestamp("2024-01-01", tz="UTC")
    ts_ms = [(start + pd.Timedelta(days=i)).value // 1_000_000 for i in range(n_days)]
    ohlcv = pd.DataFrame({
        "open_time": ts_ms,
        "Open": 100 + rng.normal(0, 1, n_days),
        "High": 102 + rng.normal(0, 1, n_days),
        "Low":  98  + rng.normal(0, 1, n_days),
        "Close": 100 + rng.normal(0, 1, n_days),
    })
    out = volatility._normalise_ohlcv(ohlcv)
    assert not out.empty
    # date column must be tz-aware datetime64
    assert pd.api.types.is_datetime64_any_dtype(out["date"])

    # Build a fake regime lookup. We add one leading row so the
    # availability-based as-of join (commit 3 Section E) can match the
    # very first signal — strict-< excludes same-instant availability.
    lookup_dates = pd.DatetimeIndex(
        [start - pd.Timedelta(days=1)]
    ).append(pd.DatetimeIndex(out["date"]))
    lookup = pd.DataFrame({
        "ticker": "BTC", "date": lookup_dates,
        "regime": ["low"] + (["low"] * 20) + (["mid"] * 20) + (["high"] * 20),
    })

    # Signals with naive timestamps — attach_regimes must still join.
    sig = pd.DataFrame({
        "timestamp": [start + pd.Timedelta(days=i) for i in range(n_days)],
        "ticker":    "BTC",
        "target":    rng.integers(0, 2, n_days),
        "prediction": rng.integers(0, 2, n_days),
        "probability": rng.random(n_days),
        "set_id": "S1", "sentiment_model": "vader", "horizon": "1d",
    })
    merged = volatility.attach_regimes(sig, lookup)
    # Every observation must have found a regime.
    assert merged["regime"].notna().all()
