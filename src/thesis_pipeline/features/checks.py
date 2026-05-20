"""Lightweight sanity checks shared across stages."""

from __future__ import annotations

from typing import Iterable, Sequence

import pandas as pd


SUSPICIOUS_LEAKAGE_NAMES = (
    "future_", "next_", "lead_", "target_at_", "_lookahead", "_forward_",
)


def looks_like_future_leakage(column: str) -> bool:
    name = column.lower()
    return any(tok in name for tok in SUSPICIOUS_LEAKAGE_NAMES)


def find_leaky_columns(columns: Iterable[str]) -> list[str]:
    return [c for c in columns if looks_like_future_leakage(c) and c != "target"]


def assert_chronological(timestamps: Sequence) -> None:
    s = pd.Series(timestamps)
    if not s.is_monotonic_increasing:
        raise AssertionError("Timestamps are not monotonically increasing")


def common_sample_row_ids(dfs: list[pd.DataFrame], row_id_col: str = "row_id") -> set:
    """Intersection of ``row_id`` values across all given dataframes."""
    if not dfs:
        return set()
    sets = [set(df[row_id_col]) for df in dfs if row_id_col in df.columns]
    if not sets:
        return set()
    out = sets[0]
    for s in sets[1:]:
        out &= s
    return out
