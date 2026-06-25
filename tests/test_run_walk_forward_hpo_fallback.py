"""Regression test for the per-asset HPO objective fallback.

v3 ``run_walk_forward`` fell back to ``"brier_score"`` when its
``hpo_config`` carried no explicit ``objective`` key. v4 (Aufgabe 5)
flips that fallback to ``"log_loss"`` — the same canonical objective as
``load_hpo_config()`` — so a caller that hands the per-asset HPO path a
bare ``hpo_config`` does NOT silently bypass the v4 default.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from thesis_pipeline.modeling.run_models import run_walk_forward


def _synthetic_ticker(n: int = 80, seed: int = 0) -> pd.DataFrame:
    """One ticker, ``n`` daily rows with a deliberately separable signal so
    the HPO inner-train / validation split has both classes."""
    rng = np.random.default_rng(seed)
    signal = rng.normal(0, 1, n)
    eps    = rng.normal(0, 1, n)
    return pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=n, freq="D", tz="UTC"),
        "ticker":    "BTC",
        "target":    (2.0 * signal + 0.3 * eps > 0).astype(int),
        "signal_feature": signal,
    })


def test_run_walk_forward_hpo_objective_fallback_is_log_loss():
    """When ``tune_hyperparams=True`` and ``hpo_config`` has no
    ``objective`` key, the per-asset HPO path must fall back to
    ``log_loss`` (NOT v3's ``brier_score``). The fallback shows up in
    the HPO metadata columns of every output row."""
    df = _synthetic_ticker(n=80, seed=7)

    # hpo_config without an "objective" key — exactly the case the v4
    # fallback governs. Tighten the search space so the test is cheap.
    hpo_config = {
        "search_space": {"C": [0.1, 1.0], "class_weight": [None]},
        "validation_fraction": 0.2,
        "min_train_obs":       20,
        "min_validation_obs":  10,
        "refit_on_full_train": True,
    }

    sig = run_walk_forward(df, feature_cols=["signal_feature"],
                           tune_hyperparams=True, hpo_config=hpo_config)
    assert not sig.empty

    # HPO metadata must be present on every row and must record the
    # v4 fallback objective (NOT brier_score).
    for col in ("hpo_enabled", "hpo_objective", "best_C", "hpo_status"):
        assert col in sig.columns, f"missing HPO metadata column {col}"
    assert sig["hpo_enabled"].all()
    objectives = set(sig["hpo_objective"].astype(str).unique())
    assert objectives == {"log_loss"}, (
        f"v4 fallback must be log_loss; got {objectives}"
    )
    # Sanity: HPO actually picked something from the supplied grid.
    chosen = set(sig["best_C"].dropna().unique())
    assert chosen.issubset({0.1, 1.0})


def test_run_walk_forward_hpo_explicit_objective_wins_over_fallback():
    """When the caller IS explicit about the objective, the fallback
    must not override it."""
    df = _synthetic_ticker(n=80, seed=11)
    hpo_config = {
        "objective": "brier_score",
        "search_space": {"C": [0.1, 1.0], "class_weight": [None]},
        "validation_fraction": 0.2,
        "min_train_obs":       20,
        "min_validation_obs":  10,
        "refit_on_full_train": True,
    }
    sig = run_walk_forward(df, feature_cols=["signal_feature"],
                           tune_hyperparams=True, hpo_config=hpo_config)
    assert set(sig["hpo_objective"].astype(str).unique()) == {"brier_score"}
