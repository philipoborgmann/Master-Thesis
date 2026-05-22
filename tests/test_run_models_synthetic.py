"""Synthetic-data validation for the run-models pipeline.

This module *does not* change any modelling methodology. It exists so that a
fresh checkout of the package can be validated against:

1. A known-strong signal — :func:`run_walk_forward` must learn it
   out-of-sample.
2. The full ``main()`` orchestration — given a synthetic
   ``Data/Final/features_1d.parquet`` and a synthetic ``feature_sets.xlsx``,
   the pipeline writes the expected parquet + metrics_summary.csv.
3. B1 vs SYN1 — the rolling-probability benchmark must lose to a logistic
   regression on the true signal feature by a clear margin.
4. Pure-noise feature — must *not* beat B1.
5. Feature-set parsing regression — the loader must keep using
   ``Feature Columns (comma-separated)`` and not regress to a numeric
   ``# Features`` column.

All tests run inside a ``tmp_path`` via ``monkeypatch.chdir`` so the real
``Data/`` and ``Outputs/`` trees are never touched.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from thesis_pipeline.modeling.run_models import (
    load_feature_sets, main, run_walk_forward,
)

# ---------------------------------------------------------------------------
# Synthetic-data helpers
# ---------------------------------------------------------------------------

def _build_synthetic_features(n_per_ticker: int = 300,
                              tickers: tuple[str, ...] = ("BTC", "ETH"),
                              seed: int = 42) -> pd.DataFrame:
    """Produce a balanced binary-target dataset with one strong feature.

    Construction:

    * ``signal_feature`` ~ N(0, 1)
    * ``noise_feature``  ~ N(0, 1) (independent of target)
    * latent = 2.0 · signal_feature + 0.3 · ε         where ε ~ N(0, 1)
    * target = 1 if latent > 0 else 0

    With those coefficients the signal variance dominates the noise by
    ≈ 1.0 / 0.09 ≈ 11 ×, so a logistic regression on ``signal_feature`` is
    nearly deterministic out-of-sample (accuracy > 0.9). Targets are
    roughly 50 / 50 balanced because ``signal_feature`` is mean-zero
    Gaussian — that keeps the rolling-probability B1 benchmark near 0.5.

    No leakage column: ``target`` is *not* a function of ``noise_feature``
    or any future variable.
    """
    rng = np.random.default_rng(seed)
    frames = []
    for tk in tickers:
        signal = rng.normal(0.0, 1.0, size=n_per_ticker)
        noise  = rng.normal(0.0, 1.0, size=n_per_ticker)
        eps    = rng.normal(0.0, 1.0, size=n_per_ticker)
        latent = 2.0 * signal + 0.3 * eps
        target = (latent > 0).astype(int)
        ts = pd.date_range("2024-01-01", periods=n_per_ticker,
                           freq="D", tz="UTC")
        frames.append(pd.DataFrame({
            "timestamp":      ts,
            "ticker":         tk,
            "horizon":        "1d",
            "target":         target,
            "signal_feature": signal,
            "noise_feature":  noise,
        }))
    return pd.concat(frames, ignore_index=True)


def _write_feature_sets_xlsx(path: Path) -> None:
    """Write a feature_sets.xlsx that mirrors the historical column layout.

    Critically, ``# Features`` is present as an *integer count* column —
    the loader has historically had a bug where this numeric column would
    overwrite the real feature list. The presence of ``Feature Columns
    (comma-separated)`` must take priority.
    """
    fs = pd.DataFrame({
        "set_id":   ["B1", "SYN1", "SYN_NOISE"],
        "category": ["benchmark", "synthetic", "synthetic"],
        "sentiment_model": ["-", "-", "-"],
        "label":    ["rolling-probability benchmark",
                     "logistic on signal_feature",
                     "logistic on noise_feature"],
        "# Features": [0, 1, 1],
        "Feature Columns (comma-separated)": [
            "__majority_class__",
            "signal_feature",
            "noise_feature",
        ],
    })
    with pd.ExcelWriter(path, engine="openpyxl") as w:
        fs.to_excel(w, sheet_name="feature_sets", index=False)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def synth_features():
    """Function fixture — call ``synth_features(...)`` to get a fresh frame."""
    return _build_synthetic_features


@pytest.fixture
def synth_repo(tmp_path, monkeypatch):
    """Materialise a synthetic project root at ``tmp_path``.

    Layout:
        tmp_path/Data/Final/features_1d.parquet
        tmp_path/feature_sets.xlsx
        tmp_path/Outputs/Signals/        (created by run-models on write)

    ``monkeypatch.chdir`` then points the script's hardcoded relative paths
    (``Data/Final``, ``Outputs/Signals``, ``feature_sets.xlsx``) at this
    tree, so the real repository is never touched.
    """
    (tmp_path / "Data" / "Final").mkdir(parents=True)
    (tmp_path / "Outputs" / "Signals").mkdir(parents=True)

    df = _build_synthetic_features(n_per_ticker=300, tickers=("BTC", "ETH"))
    df.to_parquet(tmp_path / "Data" / "Final" / "features_1d.parquet", index=False)
    _write_feature_sets_xlsx(tmp_path / "feature_sets.xlsx")

    monkeypatch.chdir(tmp_path)
    return tmp_path


# ---------------------------------------------------------------------------
# Section B — run_walk_forward unit tests
# ---------------------------------------------------------------------------

def test_run_walk_forward_learns_synthetic_signal(synth_features):
    """A logistic regression on the strong feature must hit > 0.65 OOS."""
    df_all = synth_features(n_per_ticker=400, tickers=("BTC",))
    df_btc = df_all[df_all["ticker"] == "BTC"].sort_values("timestamp")

    signals = run_walk_forward(df_btc, feature_cols=["signal_feature"])

    assert not signals.empty
    required = {"timestamp", "ticker", "target", "prediction", "probability"}
    assert required.issubset(set(signals.columns))
    assert ((signals["probability"] >= 0.0) & (signals["probability"] <= 1.0)).all()

    # Walk-forward must place every test prediction in the second half of
    # the sample (init_train = max(0.5·n, 30) → 200 for n = 400).
    init_train = max(int(len(df_btc) * 0.50), 30)
    earliest_test = pd.to_datetime(df_btc["timestamp"].iloc[init_train - 1],
                                    utc=True)
    sig_min = pd.to_datetime(signals["timestamp"].min(), utc=True)
    assert sig_min > earliest_test

    accuracy = float((signals["prediction"] == signals["target"]).mean())
    assert accuracy > 0.65, (
        f"signal-driven walk-forward must reach acc > 0.65; got {accuracy:.3f}"
    )


def test_run_walk_forward_noise_feature_near_random(synth_features):
    """Pure noise → walk-forward accuracy must NOT be high (no leakage)."""
    df_all = synth_features(n_per_ticker=400, tickers=("BTC",))
    df_btc = df_all[df_all["ticker"] == "BTC"].sort_values("timestamp")

    signals = run_walk_forward(df_btc, feature_cols=["noise_feature"])

    assert not signals.empty
    accuracy = float((signals["prediction"] == signals["target"]).mean())
    # If accuracy exceeded ≈ 0.60 on pure noise, the walk-forward would be
    # silently leaking information from the future.
    assert accuracy < 0.60, (
        f"noise-feature walk-forward acc must stay near 0.5; got {accuracy:.3f}"
    )


def test_run_walk_forward_returns_probabilities_consistent_with_predictions(synth_features):
    """``probability >= 0.5`` ⇔ ``prediction == 1`` (sanity check on the LR head)."""
    df_all = synth_features(n_per_ticker=300, tickers=("BTC",))
    df_btc = df_all[df_all["ticker"] == "BTC"].sort_values("timestamp")

    sig = run_walk_forward(df_btc, feature_cols=["signal_feature"])

    assert ((sig["probability"] >= 0.5) == (sig["prediction"] == 1)).all()


# ---------------------------------------------------------------------------
# Section C — full pipeline integration
# ---------------------------------------------------------------------------

def test_full_pipeline_writes_signal_parquet(synth_repo):
    rc = main([
        "--horizon", "1d",
        "--set-id", "SYN1",
        "--coins", "BTC", "ETH",
        "--restart",
    ])
    assert rc == 0

    out = synth_repo / "Outputs" / "Signals" / "1d" / "SYN1.parquet"
    assert out.exists(), f"expected SYN1 output at {out}"
    df = pd.read_parquet(out)
    assert not df.empty
    assert set(df["ticker"].astype(str).unique()) == {"BTC", "ETH"}
    required = {"timestamp", "ticker", "target", "prediction", "probability"}
    assert required.issubset(set(df.columns))

    accuracy = float((df["prediction"] == df["target"]).mean())
    assert accuracy > 0.65, (
        f"full-pipeline SYN1 accuracy should exceed 0.65; got {accuracy:.3f}"
    )


def test_full_pipeline_writes_metrics_summary(synth_repo):
    rc = main([
        "--horizon", "1d",
        "--set-id", "SYN1",
        "--coins", "BTC", "ETH",
        "--restart",
    ])
    assert rc == 0

    metrics_path = synth_repo / "Outputs" / "Signals" / "metrics_summary.csv"
    assert metrics_path.exists()
    df = pd.read_csv(metrics_path)
    assert "SYN1" in df["set_id"].astype(str).unique()


# ---------------------------------------------------------------------------
# Section D — benchmark comparison
# ---------------------------------------------------------------------------

def test_syn1_outperforms_b1_benchmark(synth_repo):
    main(["--horizon", "1d", "--set-id", "B1",
          "--coins", "BTC", "ETH", "--restart"])
    main(["--horizon", "1d", "--set-id", "SYN1",
          "--coins", "BTC", "ETH", "--restart"])

    b1   = pd.read_parquet(synth_repo / "Outputs" / "Signals" / "1d" / "B1.parquet")
    syn1 = pd.read_parquet(synth_repo / "Outputs" / "Signals" / "1d" / "SYN1.parquet")

    acc_b1   = float((b1["prediction"]   == b1["target"]).mean())
    acc_syn1 = float((syn1["prediction"] == syn1["target"]).mean())

    assert acc_syn1 - acc_b1 >= 0.05, (
        f"SYN1 acc={acc_syn1:.3f} must beat B1 acc={acc_b1:.3f} by ≥ 0.05; "
        "if this regresses, walk-forward or the benchmark logic is broken."
    )


def test_syn_noise_does_not_meaningfully_beat_b1(synth_repo):
    """Pure-noise feature should not beat the rolling-probability benchmark
    by more than a small finite-sample margin."""
    main(["--horizon", "1d", "--set-id", "B1",
          "--coins", "BTC", "ETH", "--restart"])
    main(["--horizon", "1d", "--set-id", "SYN_NOISE",
          "--coins", "BTC", "ETH", "--restart"])

    b1    = pd.read_parquet(synth_repo / "Outputs" / "Signals" / "1d" / "B1.parquet")
    noise = pd.read_parquet(synth_repo / "Outputs" / "Signals" / "1d" / "SYN_NOISE.parquet")
    acc_b1    = float((b1["prediction"]    == b1["target"]).mean())
    acc_noise = float((noise["prediction"] == noise["target"]).mean())

    # Allow at most a 5-percentage-point sampling advantage; well below the
    # SYN1 lift of ≥ 5 pp that we test above.
    assert acc_noise <= acc_b1 + 0.05, (
        f"noise-feature SYN_NOISE acc={acc_noise:.3f} should not exceed "
        f"B1 acc={acc_b1:.3f} + 0.05 — the noise feature carries no signal."
    )


# ---------------------------------------------------------------------------
# Section E — feature-set parsing regression
# ---------------------------------------------------------------------------

def test_feature_sets_loader_prefers_comma_separated_over_hash_features(synth_repo):
    """``# Features`` normalises to ``features`` after the loader cleans
    headers. The loader must still prefer the explicit
    ``feature_columns_comma_separated`` column. If this regresses, the
    pipeline would interpret the integer count ``1`` as a feature name and
    crash with ``KeyError: '1'`` further down."""
    sets = load_feature_sets(str(synth_repo / "feature_sets.xlsx"))
    syn1_features = sets[sets["set_id"] == "SYN1"]["features"].iloc[0]
    assert syn1_features == "signal_feature", (
        f"loader regressed: features=({syn1_features!r}). Expected "
        f"'signal_feature' — the loader must use "
        f"'Feature Columns (comma-separated)', not '# Features'."
    )


def test_feature_sets_loader_normalises_majority_class_sentinel(synth_repo):
    """B1's sentinel must be normalised so the benchmark path triggers."""
    sets = load_feature_sets(str(synth_repo / "feature_sets.xlsx"))
    b1_features = sets[sets["set_id"] == "B1"]["features"].iloc[0]
    # ``__majority_class__`` is rewritten to ``__rolling_probability__`` by
    # the loader so the benchmark code path triggers downstream.
    assert b1_features == "__rolling_probability__"
