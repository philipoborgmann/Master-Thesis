"""Walk-forward helpers (description only; the production loop lives in
Run_Models.py — kept identical for behavior preservation).

Scheme: expanding window
  - init_train_obs = max(int(n * INIT_TRAIN_FRAC), MIN_INIT_TRAIN_OBS)
  - for t in [init_train_obs, n):
        train = X[:t], y[:t]
        scaler.fit_transform(train); then scaler.transform(X[t:t+1])
        model.fit(train); predict for row t
  - scaler and model are re-fit at every step; train indices always precede
    test index t → no leakage by construction.

Used by tests/test_no_leakage.py as the reference invariant.
"""

from __future__ import annotations

from ..config import load_config


def get_walk_forward_spec() -> dict:
    return dict(load_config("model_specs")["walk_forward"])


def get_logistic_ridge_spec() -> dict:
    return dict(load_config("model_specs")["logistic_ridge"])
