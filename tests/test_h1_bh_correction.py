"""Tests for the primary-H1 BH correction layer in
``thesis_pipeline.evaluation.incremental`` (Aufgabe 6 follow-up B + C).

Every row produced by ``incremental_sentiment_value_table`` is a v4
``ECON_*`` vs ``ECON`` comparison — by construction a primary nested
H1 test. Only valid rows enter the BH pool; missing/invalid rows keep
NaN q-values and ``False`` significance flags.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from thesis_pipeline.evaluation import incremental as inc


def _signal_frame(*, set_id, sentiment_model, category, acc, seed,
                  n=200, ticker="BTC", horizon="1d"):
    rng = np.random.default_rng(seed)
    ts = pd.date_range("2024-01-01", periods=n, freq="D", tz="UTC")
    target = rng.integers(0, 2, n)
    flip = rng.random(n) > acc
    pred = np.where(flip, 1 - target, target).astype(int)
    prob = np.where(pred == 1,
                    rng.uniform(0.55, 0.85, n),
                    rng.uniform(0.15, 0.45, n))
    return pd.DataFrame({
        "timestamp": ts, "ticker": ticker,
        "target": target, "prediction": pred, "probability": prob.astype(float),
        "set_id": set_id, "sentiment_model": sentiment_model,
        "horizon": horizon, "category": category,
        "model_type": "per_asset", "panel_mode": "-", "hpo_variant": "fixed",
        "hpo_enabled": False, "hpo_objective": "-",
        "train_window_mode": "expanding",
        "train_window_timestamps": None, "rolling_window_days": None,
    })


# ---------------------------------------------------------------------------
# A. Column contract
# ---------------------------------------------------------------------------

def test_incremental_table_carries_h1_bh_columns():
    aug  = _signal_frame(set_id="ECON_VAD_F", sentiment_model="vader",
                         category="combined_vader", acc=0.62, seed=1)
    econ = _signal_frame(set_id="ECON",       sentiment_model="-",
                         category="benchmark",      acc=0.50, seed=2)
    econ["target"] = aug["target"].values
    out = inc.incremental_sentiment_value_table(
        pd.concat([aug, econ], ignore_index=True))
    for col in ("test_role", "hypothesis_family",
                "q_value_bh", "significant_raw_5pct",
                "significant_bh_5pct",
                "interpretation_bh"):
        assert col in out.columns
    # The 10% flag was removed globally (single significant_bh at 0.05).
    assert "significant_bh_10pct" not in out.columns


def test_every_emitted_row_is_primary_h1_nested():
    """The table only emits rows whose set_id is in
    MATCHED_ECONOMIC_BENCHMARK — by construction only primary nested H1
    comparisons are produced."""
    aug  = _signal_frame(set_id="ECON_CBT_F", sentiment_model="cryptobert",
                         category="combined_cryptobert", acc=0.60, seed=3)
    econ = _signal_frame(set_id="ECON",       sentiment_model="-",
                         category="benchmark",            acc=0.50, seed=4)
    econ["target"] = aug["target"].values
    out = inc.incremental_sentiment_value_table(
        pd.concat([aug, econ], ignore_index=True))
    assert (out["test_role"] == inc.TEST_ROLE_PRIMARY).all()
    assert (out["hypothesis_family"] == "H1_incremental").all()


def test_h1_bh_scope_constant_documented():
    """Aufgabe 6 follow-up B explicitly requests a named constant for the
    BH-correction scope so the multiplicity plan is not silently chosen."""
    assert inc.H1_BH_SCOPE == "all_primary_h1_tests"


# ---------------------------------------------------------------------------
# B. BH behaviour with several real comparisons
# ---------------------------------------------------------------------------

def _multi_h1_pool() -> pd.DataFrame:
    """One ECON benchmark plus four ECON_*_F augmented variants, all with
    a small lift — enough p-values to make the BH adjustment meaningful."""
    parts = [
        _signal_frame(set_id="ECON", sentiment_model="-",
                      category="benchmark", acc=0.50, seed=0),
    ]
    for sid, sm, cat, seed in [
        ("ECON_VAD_L", "vader",      "combined_vader",      1),
        ("ECON_VAD_F", "vader",      "combined_vader",      2),
        ("ECON_CBT_L", "cryptobert", "combined_cryptobert", 3),
        ("ECON_CBT_F", "cryptobert", "combined_cryptobert", 4),
    ]:
        f = _signal_frame(set_id=sid, sentiment_model=sm,
                          category=cat, acc=0.60, seed=seed)
        f["target"] = parts[0]["target"].values
        parts.append(f)
    return pd.concat(parts, ignore_index=True)


def test_bh_pool_excludes_missing_benchmark_rows():
    """A combined-set row without a matched ECON benchmark must NOT enter
    the BH pool — its q-value stays NaN."""
    aug = _signal_frame(set_id="ECON_VAD_F", sentiment_model="vader",
                        category="combined_vader", acc=0.62, seed=5)
    # No ECON frame supplied.
    out = inc.incremental_sentiment_value_table(aug)
    row = out.iloc[0]
    assert row["status"] == "missing_benchmark"
    assert pd.isna(row["q_value_bh"])
    assert not bool(row["significant_bh_5pct"])


def test_bh_pool_excludes_no_overlap_rows():
    """If the model and benchmark share no overlapping (timestamp, ticker)
    rows, the status is ``no_overlap`` and the BH pool excludes it."""
    aug  = _signal_frame(set_id="ECON_VAD_F", sentiment_model="vader",
                        category="combined_vader", acc=0.6, n=80, seed=6)
    econ = _signal_frame(set_id="ECON",       sentiment_model="-",
                        category="benchmark",      acc=0.5, n=80, seed=7,
                        ticker="ETH")  # different ticker → no overlap
    out = inc.incremental_sentiment_value_table(
        pd.concat([aug, econ], ignore_index=True))
    assert (out["status"] == "no_overlap").any() \
           or (out["status"] == "missing_benchmark").any()
    bad = out[out["status"] != "ok"]
    assert bad["q_value_bh"].isna().all()
    assert not bad["significant_bh_5pct"].any()


def test_q_values_are_non_decreasing_with_p_in_bh_pool():
    """BH q-values are monotone non-decreasing in the p-values within the
    valid pool — a fundamental property of the procedure."""
    df = _multi_h1_pool()
    out = inc.incremental_sentiment_value_table(df)
    ok = out[(out["status"] == "ok")
              & out["mcnemar_p_value"].notna()].copy()
    if len(ok) < 2:
        pytest.skip("not enough valid H1 rows to test monotonicity")
    ok = ok.sort_values("mcnemar_p_value")
    qs = ok["q_value_bh"].astype(float).values
    assert np.all(np.diff(qs) >= -1e-12)  # monotone non-decreasing


def test_interpretation_bh_distinguishes_significant_from_ns():
    df = _multi_h1_pool()
    out = inc.incremental_sentiment_value_table(df)
    ok = out[out["status"] == "ok"]
    flags = set(ok["interpretation_bh"].dropna().astype(str).unique())
    # Every interpretation_bh label must come from the documented set.
    allowed = {"improved_bh_significant", "improved_bh_ns",
               "degraded_bh_significant", "degraded_bh_ns", "no_change"}
    assert flags.issubset(allowed)


def test_significant_raw_5pct_is_consistent_with_mcnemar_p():
    df = _multi_h1_pool()
    out = inc.incremental_sentiment_value_table(df)
    ok = out[out["status"] == "ok"]
    for _, r in ok.iterrows():
        expected = bool(np.isfinite(r["mcnemar_p_value"])
                        and r["mcnemar_p_value"] < 0.05)
        assert bool(r["significant_raw_5pct"]) == expected
