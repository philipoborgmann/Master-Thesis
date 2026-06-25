"""Walk-forward invariants from configs/model_specs.yaml."""
from __future__ import annotations

import numpy as np
import pytest

from thesis_pipeline.config import load_config
from thesis_pipeline.diagnostics.leakage_checks import (
    assert_train_precedes_test, scaler_fit_scope_ok,
)


def test_scheme_is_walk_forward_not_random_split():
    """Walk-forward (either expanding or fixed-rolling) — never a random
    split. v4 default is ``rolling_fixed`` with rolling_window_days=180;
    earlier versions used ``expanding``. Both are walk-forward."""
    cfg = load_config("model_specs")
    assert cfg["walk_forward"]["scheme"] in ("expanding", "rolling_fixed")
    # Random split would imply absence of walk_forward block / shuffle flag.
    assert "shuffle" not in cfg["walk_forward"]


def test_scaler_scope_train_only():
    cfg = load_config("model_specs")
    assert cfg["scaler"]["fit_scope"] == "train_window_only"


def test_random_state_pinned():
    cfg = load_config("model_specs")
    assert cfg["logistic_ridge"]["random_state"] == 42


def test_train_precedes_test_helper():
    # Synthetic expanding window: train = [0..t-1], test = [t]
    n = 50
    for t in range(20, n):
        train_idx = list(range(t))
        test_idx  = [t]
        assert_train_precedes_test(train_idx, test_idx)  # raises on leakage
        assert scaler_fit_scope_ok(train_idx, train_idx)


def test_train_precedes_test_detects_leakage():
    with pytest.raises(AssertionError):
        assert_train_precedes_test([0, 1, 2, 3, 4], [3, 4, 5])
