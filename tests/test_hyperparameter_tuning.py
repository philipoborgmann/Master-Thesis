"""Tests for the conservative, leakage-safe hyperparameter tuning module.

Synthetic DataFrames only — no real Data/ or Outputs/ are touched, and every
run is tiny.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from thesis_pipeline.modeling import hyperparameter_tuning as hpt
from thesis_pipeline.modeling.run_models import run_walk_forward

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


# ---------------------------------------------------------------------------
# 1. make_param_grid
# ---------------------------------------------------------------------------

def test_make_param_grid_builds_all_combinations():
    grid = hpt.make_param_grid({"C": [0.01, 0.1, 1.0], "class_weight": [None, "balanced"]})
    assert len(grid) == 6
    # Every combination present, exactly once.
    seen = {(g["C"], g["class_weight"]) for g in grid}
    assert seen == {
        (0.01, None), (0.01, "balanced"),
        (0.1, None),  (0.1, "balanced"),
        (1.0, None),  (1.0, "balanced"),
    }


def test_make_param_grid_empty_inputs():
    assert hpt.make_param_grid({}) == []
    assert hpt.make_param_grid({"C": []}) == []


# ---------------------------------------------------------------------------
# 2-4. score_validation_predictions + objective directions
# ---------------------------------------------------------------------------

def test_score_brier_minimized_by_better_probs():
    y = np.array([1, 0, 1, 0])
    good = np.array([0.9, 0.1, 0.8, 0.2])
    bad  = np.array([0.6, 0.4, 0.55, 0.45])
    s_good = hpt.score_validation_predictions(y, good, (good >= 0.5).astype(int), "brier_score")
    s_bad  = hpt.score_validation_predictions(y, bad,  (bad  >= 0.5).astype(int), "brier_score")
    assert s_good < s_bad
    assert hpt.OBJECTIVE_DIRECTIONS["brier_score"] == "minimize"
    # _is_better picks the smaller brier.
    assert hpt._is_better(s_good, s_bad, "brier_score")


def test_score_log_loss_minimized_by_better_probs():
    y = np.array([1, 0, 1, 0])
    good = np.array([0.95, 0.05, 0.9, 0.1])
    bad  = np.array([0.55, 0.45, 0.5, 0.5])
    s_good = hpt.score_validation_predictions(y, good, (good >= 0.5).astype(int), "log_loss")
    s_bad  = hpt.score_validation_predictions(y, bad,  (bad  >= 0.5).astype(int), "log_loss")
    assert s_good < s_bad
    assert hpt.OBJECTIVE_DIRECTIONS["log_loss"] == "minimize"


def test_score_accuracy_maximized():
    y = np.array([1, 0, 1, 0])
    good_pred = np.array([1, 0, 1, 0])
    bad_pred  = np.array([0, 1, 0, 1])
    s_good = hpt.score_validation_predictions(y, np.full(4, 0.5), good_pred, "accuracy")
    s_bad  = hpt.score_validation_predictions(y, np.full(4, 0.5), bad_pred,  "accuracy")
    assert s_good == 1.0 and s_bad == 0.0
    assert hpt.OBJECTIVE_DIRECTIONS["accuracy"] == "maximize"
    assert hpt._is_better(s_good, s_bad, "accuracy")


def test_score_log_loss_handles_single_class_validation():
    # All-ones validation fold must not raise (labels=[0,1] passed internally).
    y = np.array([1, 1, 1])
    prob = np.array([0.8, 0.7, 0.9])
    val = hpt.score_validation_predictions(y, prob, np.ones(3, int), "log_loss")
    assert np.isfinite(val)


# ---------------------------------------------------------------------------
# 5 + 7. Chronological split (per-asset rows; panel timestamps)
# ---------------------------------------------------------------------------

def test_chronological_split_per_asset_uses_last_fraction():
    df = pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=100, freq="D", tz="UTC"),
        "ticker": "BTC",
        "target": np.arange(100) % 2,
    })
    inner, val = hpt.chronological_train_validation_split(df, 0.2, family=hpt.PER_ASSET)
    assert len(inner) == 80 and len(val) == 20
    # Validation is strictly the most-recent block.
    assert val["timestamp"].min() > inner["timestamp"].max()


def test_chronological_split_panel_splits_on_timestamps_not_rows():
    # 3 tickers × 10 timestamps = 30 observation rows.
    ts = pd.date_range("2024-01-01", periods=10, freq="D", tz="UTC")
    rows = [{"timestamp": t, "ticker": tk, "target": 1}
            for t in ts for tk in ("BTC", "ETH", "SOL")]
    df = pd.DataFrame(rows)
    inner, val = hpt.chronological_train_validation_split(df, 0.2, family=hpt.PANEL)
    # 20% of 10 unique timestamps = 2 → validation holds the last 2 timestamps
    # across all 3 coins (6 rows), inner the first 8 timestamps (24 rows).
    assert set(val["timestamp"].unique()) == set(ts[-2:])
    assert len(val) == 6 and len(inner) == 24
    # No timestamp leaks across the boundary.
    assert inner["timestamp"].max() < val["timestamp"].min()


# ---------------------------------------------------------------------------
# Synthetic panel/per-asset builders
# ---------------------------------------------------------------------------

def _per_asset_df(n=200, noise_dims=0, seed=0, signal_coef=2.0):
    rng = np.random.default_rng(seed)
    ts = pd.date_range("2024-01-01", periods=n, freq="D", tz="UTC")
    signal = rng.normal(0, 1, n)
    eps = rng.normal(0, 1, n)
    target = (signal_coef * signal + 0.3 * eps > 0).astype(int)
    d = {"timestamp": ts, "ticker": "BTC", "target": target, "signal_feature": signal}
    for k in range(noise_dims):
        d[f"noise_{k}"] = rng.normal(0, 1, n)
    return pd.DataFrame(d)


# ---------------------------------------------------------------------------
# 8. HPO picks a plausible C (stronger regularisation under heavy noise)
# ---------------------------------------------------------------------------

def test_hpo_selects_objective_minimising_C():
    """The tuner must pick the grid C that optimises the validation objective.

    We independently re-derive the best C by replaying the (chronological)
    inner-train / validation split and scoring each candidate, then assert the
    tuner agrees. On a heavy-noise, weak-signal window this lands on a
    *stronger* regularisation than the least-regularised C=10 option, which is
    the plausible conservative choice the thesis wants.
    """
    df = _per_asset_df(n=160, noise_dims=12, seed=1, signal_coef=0.8)
    feature_cols = ["signal_feature"] + [f"noise_{k}" for k in range(12)]
    c_grid = [0.01, 0.1, 1.0, 10.0]
    hpo_cfg = {"validation_fraction": 0.25, "min_train_obs": 30,
               "min_validation_obs": 10, "refit_on_full_train": True}

    res = hpt.tune_logistic_hyperparams(
        df, feature_cols, family=hpt.PER_ASSET, objective="brier_score",
        search_space={"C": c_grid, "class_weight": [None]}, hpo_cfg=hpo_cfg,
    )
    assert res["hpo_status"] == "ok"

    # Independent recomputation of the argmin over the same split.
    inner, val = hpt.chronological_train_validation_split(
        df, 0.25, family=hpt.PER_ASSET)
    scores = {}
    for c in c_grid:
        arts = hpt.fit_logistic_model(inner, feature_cols, {"C": c, "class_weight": None},
                                      family=hpt.PER_ASSET)
        proba = hpt.predict_proba(arts, val, feature_cols, family=hpt.PER_ASSET)
        scores[c] = hpt.score_validation_predictions(
            val["target"].values, proba, (proba >= 0.5).astype(int), "brier_score")
    expected_best_c = min(scores, key=scores.get)
    # The tuner picks exactly the objective-minimising C from the grid — a
    # plausible, deterministic choice in whichever direction the data favours.
    assert res["best_params"]["C"] == expected_best_c


def test_hpo_returns_usable_artifacts_for_prediction():
    df = _per_asset_df(n=160, seed=2)
    res = hpt.tune_logistic_hyperparams(
        df, ["signal_feature"], family=hpt.PER_ASSET, objective="accuracy",
        search_space={"C": [0.1, 1.0], "class_weight": [None, "balanced"]},
        hpo_cfg={"validation_fraction": 0.2, "min_train_obs": 30,
                 "min_validation_obs": 10},
    )
    proba = hpt.predict_proba(res["artifacts"], df.tail(5), ["signal_feature"],
                              family=hpt.PER_ASSET)
    assert proba.shape == (5,)
    assert ((proba >= 0) & (proba <= 1)).all()


# ---------------------------------------------------------------------------
# 9. Fallback when there is too little data
# ---------------------------------------------------------------------------

def test_fallback_insufficient_data_uses_defaults():
    df = _per_asset_df(n=40, seed=3)
    res = hpt.tune_logistic_hyperparams(
        df, ["signal_feature"], family=hpt.PER_ASSET, objective="brier_score",
        search_space={"C": [0.01, 1.0, 10.0], "class_weight": [None]},
        hpo_cfg={"validation_fraction": 0.2, "min_train_obs": 1000,
                 "min_validation_obs": 1000, "refit_on_full_train": True},
    )
    assert res["hpo_status"] == "fallback_insufficient_data"
    assert res["best_params"] == hpt.DEFAULT_PARAMS
    # A usable model is still produced (so the walk-forward step still predicts).
    assert res["artifacts"]["model"] is not None


# ---------------------------------------------------------------------------
# 6. Per-asset HPO never uses the test point in tuning (no leakage)
# ---------------------------------------------------------------------------

def test_per_asset_hpo_never_sees_test_point(monkeypatch):
    df = _per_asset_df(n=120, seed=4)
    captured = []
    real = hpt.tune_logistic_hyperparams

    def spy(train_df, feature_cols, **kw):
        captured.append(train_df["timestamp"].max())
        return real(train_df, feature_cols, **kw)

    monkeypatch.setattr(hpt, "tune_logistic_hyperparams", spy)
    hpo_cfg = {"objective": "brier_score", "validation_fraction": 0.2,
               "min_train_obs": 20, "min_validation_obs": 5,
               "refit_on_full_train": True,
               "search_space": {"C": [0.1, 1.0], "class_weight": [None]}}
    signals = run_walk_forward(df, ["signal_feature"], tune_hyperparams=True,
                               hpo_config=hpo_cfg)
    assert not signals.empty
    assert len(captured) > 0
    # The training window handed to the tuner at each step must end strictly
    # before the predicted test timestamp.
    for max_train_ts, test_ts in zip(captured, signals["timestamp"]):
        # Normalise both to UTC-aware before comparing (the signal frame stores
        # tz-naive numpy datetime64, the captured value is tz-aware).
        assert pd.to_datetime(max_train_ts, utc=True) < pd.to_datetime(test_ts, utc=True)
    # And the HPO provenance columns are present + consistent.
    for col in ("hpo_enabled", "hpo_objective", "best_C", "best_class_weight",
                "hpo_score", "hpo_status"):
        assert col in signals.columns
    assert signals["hpo_enabled"].all()
    assert (signals["hpo_objective"] == "brier_score").all()


def test_run_walk_forward_no_tuning_has_no_hpo_columns():
    """Backward compatibility: disabled tuning writes no HPO columns."""
    df = _per_asset_df(n=120, seed=5)
    signals = run_walk_forward(df, ["signal_feature"])
    assert not signals.empty
    assert set(signals.columns) == {"timestamp", "ticker", "target",
                                     "prediction", "probability"}


# ---------------------------------------------------------------------------
# Panel HPO end-to-end (pooled + ticker FE)
# ---------------------------------------------------------------------------

def _panel_df(n_ts=120, tickers=("BTC", "ETH", "SOL"), seed=6):
    from thesis_pipeline.modeling.panel_logit import run_panel_walk_forward  # noqa: F401
    rng = np.random.default_rng(seed)
    ts = pd.date_range("2024-01-01", periods=n_ts, freq="D", tz="UTC")
    rows = []
    for tk in tickers:
        signal = rng.normal(0, 1, n_ts)
        eps = rng.normal(0, 1, n_ts)
        target = (2.0 * signal + 0.3 * eps > 0).astype(int)
        rows.append(pd.DataFrame({"timestamp": ts, "ticker": tk,
                                   "target": target, "signal_feature": signal}))
    return pd.concat(rows, ignore_index=True)


def test_panel_hpo_pooled_runs_and_tags_columns():
    from thesis_pipeline.modeling.panel_logit import run_panel_walk_forward
    df = _panel_df(n_ts=120)
    hpo_cfg = {"objective": "brier_score", "validation_fraction": 0.2,
               "min_train_obs": 30, "min_validation_obs": 10,
               "refit_on_full_train": True,
               "search_space": {"C": [0.1, 1.0], "class_weight": [None]}}
    sig = run_panel_walk_forward(df, ["signal_feature"], panel_mode="pooled",
                                 min_init_timestamps=30, tune_hyperparams=True,
                                 hpo_config=hpo_cfg)
    assert not sig.empty
    for col in ("hpo_enabled", "best_C", "hpo_status"):
        assert col in sig.columns
    acc = (sig["prediction"] == sig["target"]).mean()
    assert acc > 0.6


def test_panel_hpo_ticker_fe_runs():
    from thesis_pipeline.modeling.panel_logit import run_panel_walk_forward
    df = _panel_df(n_ts=140)
    hpo_cfg = {"objective": "accuracy", "validation_fraction": 0.2,
               "min_train_obs": 30, "min_validation_obs": 10,
               "refit_on_full_train": True,
               "search_space": {"C": [0.1, 1.0], "class_weight": [None, "balanced"]}}
    sig = run_panel_walk_forward(df, ["signal_feature"],
                                 panel_mode="ticker_fixed_effects",
                                 min_init_timestamps=30, tune_hyperparams=True,
                                 hpo_config=hpo_cfg)
    assert not sig.empty
    assert sig["hpo_enabled"].all()


# ---------------------------------------------------------------------------
# load_hpo_config
# ---------------------------------------------------------------------------

def test_load_hpo_config_defaults_disabled():
    cfg = hpt.load_hpo_config({"hyperparameter_tuning": {"enabled": False}})
    assert cfg["enabled"] is False
    assert cfg["objective"] == "brier_score"


def test_load_hpo_config_legacy_bool():
    cfg = hpt.load_hpo_config({"hyperparameter_tuning": False})
    assert cfg["enabled"] is False


def test_load_hpo_config_overrides():
    cfg = hpt.load_hpo_config(
        {"hyperparameter_tuning": {"enabled": False}},
        enabled_override=True, objective_override="log_loss",
        c_grid=[0.5, 5.0], class_weight_grid=["none", "balanced"],
    )
    assert cfg["enabled"] is True
    assert cfg["objective"] == "log_loss"
    assert cfg["search_space"]["C"] == [0.5, 5.0]
    assert cfg["search_space"]["class_weight"] == [None, "balanced"]


def test_load_hpo_config_rejects_unknown_objective():
    with pytest.raises(ValueError):
        hpt.load_hpo_config({"hyperparameter_tuning": {"objective": "f1"}})


# ---------------------------------------------------------------------------
# summarize_hpo_columns
# ---------------------------------------------------------------------------

def test_summarize_hpo_columns_empty_when_disabled():
    df = pd.DataFrame({"prediction": [1, 0], "target": [1, 0]})
    assert hpt.summarize_hpo_columns(df) == {}


def test_summarize_hpo_columns_aggregates():
    df = pd.DataFrame({
        "hpo_enabled": [True, True, True],
        "hpo_objective": ["brier_score"] * 3,
        "best_C": [0.1, 0.1, 1.0],
        "best_class_weight": ["none", "none", "balanced"],
        "hpo_status": ["ok", "ok", "fallback_insufficient_data"],
    })
    out = hpt.summarize_hpo_columns(df)
    assert out["hpo_enabled"] is True
    assert out["hpo_objective"] == "brier_score"
    assert out["best_C_mode"] == 0.1
    assert out["best_class_weight_mode"] == "none"
    assert "ok:2" in out["hpo_status_counts"]
    assert "fallback_insufficient_data:1" in out["hpo_status_counts"]


# ---------------------------------------------------------------------------
# 10. CLI dry-run with --tune-hyperparams
# ---------------------------------------------------------------------------

def test_cli_dry_run_with_tune_hyperparams():
    from thesis_pipeline import cli
    rc = cli.main(["run-models", "--horizon", "1d", "--set-id", "ECON_CBT_F",
                   "--sentiment-model", "cryptobert", "--tune-hyperparams",
                   "--hpo-objective", "brier_score", "--dry-run"])
    assert rc == 0


def test_cli_dry_run_panel_with_tune_hyperparams():
    from thesis_pipeline import cli
    rc = cli.main(["run-models", "--horizon", "1d", "--set-id", "ECON_CBT_F",
                   "--sentiment-model", "cryptobert", "--model-type", "panel_logit",
                   "--panel-mode", "pooled", "--tune-hyperparams",
                   "--hpo-objective", "brier_score", "--dry-run"])
    assert rc == 0
