"""Target should be the direction of the *next* return, not the current one."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from thesis_pipeline.features.checks import find_leaky_columns


def test_target_is_future_direction_synthetic():
    """Reference implementation: target_t = 1[log_return_{t+1} >= 0].

    Asserts that this construction is forward-looking — i.e. target_t depends
    only on data strictly after t.
    """
    rng = np.random.default_rng(0)
    n = 200
    log_ret = rng.normal(0, 0.02, n)
    target = (np.roll(log_ret, -1) >= 0).astype(int)
    target[-1] = 0  # last is undefined; mark conservatively
    # contemporaneous correlation between target and current log_ret should
    # not be perfect — that would mean target is contemporaneous.
    corr = np.corrcoef(target[:-1], (log_ret[:-1] >= 0).astype(int))[0, 1]
    assert abs(corr) < 0.5, ("target should not equal direction of current return; "
                             f"corr={corr:.3f}")


def test_find_leaky_columns_flags_suspicious_names():
    cols = ["log_return_t", "future_return", "next_close", "target", "lead_volume", "close"]
    leaky = find_leaky_columns(cols)
    assert "future_return" in leaky
    assert "next_close" in leaky
    assert "lead_volume" in leaky
    assert "target" not in leaky  # target is allowed
    assert "log_return_t" not in leaky


def test_real_price_features_target_alignment():
    """If a real price_features file exists locally, verify target alignment."""
    from thesis_pipeline.config import resolve_path
    p = resolve_path("price_features_pattern", horizon="1d")
    if not p.exists():
        pytest.skip(f"local file missing: {p}")
    df = pd.read_parquet(p).sort_values(["ticker", "timestamp"])
    # For a random ticker, check that target_t aligns with sign of next log_return
    for tkr, g in df.groupby("ticker"):
        g = g.dropna(subset=["log_return_t", "target"]).reset_index(drop=True)
        if len(g) < 30:
            continue
        next_ret = g["log_return_t"].shift(-1)
        expected = (next_ret >= 0).astype("Int64")
        # Compare on rows where both are non-null
        mask = expected.notna() & g["target"].notna()
        match_rate = (g.loc[mask, "target"] == expected[mask]).mean()
        assert match_rate > 0.95, (
            f"ticker={tkr}: target does not match sign of next log_return "
            f"(match={match_rate:.3f})"
        )
        break
