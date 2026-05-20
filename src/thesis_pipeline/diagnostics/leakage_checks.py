"""Leakage sanity checks (used by tests/test_no_leakage.py)."""

from __future__ import annotations

from typing import Sequence


def assert_train_precedes_test(train_idx: Sequence[int], test_idx: Sequence[int]) -> None:
    if not len(train_idx) or not len(test_idx):
        return
    if max(train_idx) >= min(test_idx):
        raise AssertionError(
            f"Leakage: max(train_idx)={max(train_idx)} >= min(test_idx)={min(test_idx)}"
        )


def scaler_fit_scope_ok(scaler_fit_indices: Sequence[int],
                       train_idx: Sequence[int]) -> bool:
    return set(scaler_fit_indices).issubset(set(train_idx))
