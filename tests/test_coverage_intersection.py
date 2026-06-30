"""Tests for the shared coverage-intersection fix (commit 12).

Three production evaluation writers failed to populate because of a
sample-coverage asymmetry between the ECON benchmark and the
augmented/sentiment sets, plus two key-normalization bugs:

* incremental_sentiment_value.csv — benchmark key mismatch on
  ``sentiment_model`` ("none" vs "-");
* diff_in_improvement.csv — abort-on-duplicate under coverage asymmetry;
* absolute_vs_naive.csv — NAIVE universe hash not resolved/propagated.

These tests pin:
  (shared) the intersection helper restricts both sides to the common
           key, dedupes defensively, and reports honest counts;
  (i)      a benchmark-key match when sentiment_model="none";
  (ii)     an intersection join where the reference has extra rows the
           candidate lacks (the 1d/1h coverage-asymmetry case);
  (iii)    naive identity matching when the universe hash is present.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from thesis_pipeline.evaluation import incremental as inc
from thesis_pipeline.evaluation import naive_comparison as nc
from thesis_pipeline.evaluation.coverage import coverage_intersection
from thesis_pipeline.evaluation.loading import canonical_sentiment_model
from thesis_pipeline.modeling.naive_reference import coin_universe_hash


# ---------------------------------------------------------------------------
# Shared helper
# ---------------------------------------------------------------------------

def _frame(tickers_ts, *, value=1):
    rows = []
    for tk, ts in tickers_ts:
        rows.append({"ticker": tk, "timestamp": pd.Timestamp(ts, tz="UTC"),
                     "horizon": "1d", "val": value})
    return pd.DataFrame(rows)


def test_intersection_restricts_to_common_keys():
    cand = _frame([("BTC", "2024-01-01"), ("BTC", "2024-01-02"),
                   ("ETH", "2024-01-01")])
    ref = _frame([("BTC", "2024-01-01"), ("BTC", "2024-01-02"),
                  ("BTC", "2024-01-03"), ("ETH", "2024-01-01"),
                  ("SOL", "2024-01-01")])
    ci = coverage_intersection(cand, ref)
    # candidate 3, reference 5 → matched 3 (BTC×2 + ETH×1).
    assert ci.n_candidate == 3
    assert ci.n_reference == 5
    assert ci.n_matched == 3
    assert ci.n_unmatched_candidate == 0
    assert ci.n_unmatched_reference == 2
    # Aligned row-for-row.
    assert list(ci.candidate["ticker"]) == list(ci.reference["ticker"])
    assert list(ci.candidate["timestamp"]) == list(ci.reference["timestamp"])


def test_intersection_dedupes_defensively():
    cand = _frame([("BTC", "2024-01-01"), ("BTC", "2024-01-01"),  # dup
                   ("ETH", "2024-01-01")])
    ref = _frame([("BTC", "2024-01-01"), ("ETH", "2024-01-01")])
    ci = coverage_intersection(cand, ref)
    assert ci.n_duplicate_candidate == 1
    assert ci.n_matched == 2          # deduped, no fan-out
    assert len(ci.candidate) == 2


def test_intersection_no_overlap_returns_empty():
    cand = _frame([("BTC", "2024-01-01")])
    ref = _frame([("ETH", "2024-02-01")])
    ci = coverage_intersection(cand, ref)
    assert ci.n_matched == 0
    assert ci.candidate.empty and ci.reference.empty


def test_intersection_equal_coverage_is_noop():
    """The 6h case: equal clean coverage → matched == both sizes, no
    duplicates, no unmatched."""
    keys = [("BTC", f"2024-01-{d:02d}") for d in range(1, 21)]
    cand = _frame(keys)
    ref = _frame(keys)
    ci = coverage_intersection(cand, ref)
    assert ci.n_matched == 20 == ci.n_candidate == ci.n_reference
    assert ci.n_unmatched_candidate == 0 and ci.n_unmatched_reference == 0
    assert ci.n_duplicate_candidate == 0 and ci.n_duplicate_reference == 0


# ---------------------------------------------------------------------------
# Canonical sentiment_model key
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("raw", ["none", "None", "NONE", "nan", "", "-",
                                  "null", "NA", "n/a"])
def test_canonical_sentiment_model_collapses_absent_tokens(raw):
    assert canonical_sentiment_model(raw) == "-"


@pytest.mark.parametrize("raw,expected", [("vader", "vader"),
                                           ("cryptobert", "cryptobert"),
                                           ("  vader  ", "vader")])
def test_canonical_sentiment_model_keeps_real_scorers(raw, expected):
    # Real scorer names are stripped but case-preserved (no grouping-key
    # behaviour change).
    assert canonical_sentiment_model(raw) == expected


# ---------------------------------------------------------------------------
# (i) Benchmark-key match when ECON's sentiment_model = "none"
# ---------------------------------------------------------------------------

def _sig(*, set_id, sentiment_model, n=120, acc=0.55, seed=0,
         category=None):
    rng = np.random.default_rng(seed)
    ts = pd.date_range("2024-01-01", periods=n, freq="D", tz="UTC")
    target = rng.integers(0, 2, n)
    flip = rng.random(n) > acc
    pred = np.where(flip, 1 - target, target).astype(int)
    prob = np.where(pred == 1, rng.uniform(0.55, 0.85, n),
                                rng.uniform(0.15, 0.45, n)).astype(float)
    cat = category or ("benchmark" if set_id == "ECON" else "combined_vader")
    return pd.DataFrame({
        "timestamp": ts, "ticker": "BTC",
        "target": target, "prediction": pred, "probability": prob,
        "set_id": set_id, "sentiment_model": sentiment_model,
        "horizon": "1d", "category": cat,
        "model_type": "panel_logit", "panel_mode": "ticker_fixed_effects",
        "hpo_variant": "fixed", "hpo_enabled": False, "hpo_objective": "-",
        "train_window_mode": "expanding",
        "train_window_timestamps": None, "rolling_window_days": None,
    })


def test_incremental_matches_econ_when_sentiment_model_is_none():
    """ECON written with sentiment_model='none' must still match the
    combined set's benchmark lookup (commit 12 Task C)."""
    combined = _sig(set_id="ECON_VAD_F", sentiment_model="vader",
                    acc=0.70, seed=1)
    econ     = _sig(set_id="ECON", sentiment_model="none", acc=0.55, seed=2)
    econ["target"] = combined["target"].values  # shared target
    out = inc.incremental_sentiment_value_table(
        pd.concat([combined, econ], ignore_index=True))
    row = out[out["set_id"] == "ECON_VAD_F"].iloc[0]
    assert row["status"] == "ok"
    assert int(row["n_matched"]) > 0
    assert str(row["status"]) != "missing_benchmark"


def test_incremental_no_missing_benchmark_for_combined_sets():
    """End-to-end: no combined set should report missing_benchmark when
    ECON is present under the 'none' spelling."""
    frames = [_sig(set_id="ECON", sentiment_model="none", acc=0.55, seed=99)]
    targets = frames[0]["target"].values
    for i, sid in enumerate(["ECON_VAD_F", "ECON_VAD_L", "ECON_CBT_F"]):
        f = _sig(set_id=sid, sentiment_model="vader" if "VAD" in sid else "cryptobert",
                 acc=0.62, seed=i,
                 category="combined_vader" if "VAD" in sid else "combined_cryptobert")
        f["target"] = targets
        frames.append(f)
    out = inc.incremental_sentiment_value_table(pd.concat(frames, ignore_index=True))
    assert not (out["status"] == "missing_benchmark").any()
    assert (out["n_matched"] > 0).all()


# ---------------------------------------------------------------------------
# (ii) Intersection join where reference (ECON) has extra rows
# ---------------------------------------------------------------------------

def test_incremental_uses_intersection_when_econ_has_extra_rows():
    """ECON covers 120 days, the combined set only 80 (no sentiment
    coverage for the tail). The comparison must run on the 80-day
    overlap — not abort, not fan out."""
    combined = _sig(set_id="ECON_VAD_F", sentiment_model="vader",
                    n=80, acc=0.70, seed=5)
    econ     = _sig(set_id="ECON", sentiment_model="none",
                    n=120, acc=0.55, seed=6)
    # Shared target on the overlapping first 80 days.
    econ.loc[:79, "target"] = combined["target"].values
    out = inc.incremental_sentiment_value_table(
        pd.concat([combined, econ], ignore_index=True))
    row = out[out["set_id"] == "ECON_VAD_F"].iloc[0]
    assert row["status"] == "ok"
    assert int(row["n_matched"]) == 80


# ---------------------------------------------------------------------------
# (iii) NAIVE identity matching when the universe hash is present
# ---------------------------------------------------------------------------

def _naive_or_model_row(*, set_id, hpo_variant, n=80, seed=0,
                        coin_hash, tickers=("BTC", "ETH")):
    rng = np.random.default_rng(seed)
    ts = pd.date_range("2024-01-01", periods=n, freq="D", tz="UTC")
    rows = []
    for tk in tickers:
        target = rng.integers(0, 2, n)
        rows.append(pd.DataFrame({
            "timestamp": ts, "ticker": tk,
            "target": target, "prediction": target,
            "probability": rng.uniform(0.4, 0.6, n),
            "horizon": "1d", "set_id": set_id,
            "sentiment_model": "-" if set_id == "NAIVE" else "vader",
            "model_type": "panel_logit", "panel_mode": "ticker_fixed_effects",
            "hpo_variant": hpo_variant,
            "hpo_objective": "-" if hpo_variant == "naive" else "log_loss",
            "train_window_mode": "rolling_fixed",
            "rolling_window_days": 180.0,
            "rolling_window_timestamps": np.nan,
            "coin_universe_hash": coin_hash,
        }))
    return pd.concat(rows, ignore_index=True)


def test_absolute_vs_naive_matches_when_hash_present():
    h = coin_universe_hash(("BTC", "ETH"))
    model = _naive_or_model_row(set_id="ECON_VAD_F", hpo_variant="log_loss",
                                seed=1, coin_hash=h)
    naive = _naive_or_model_row(set_id="NAIVE", hpo_variant="naive",
                                seed=1, coin_hash=h)
    naive["target"] = model["target"].values  # shared target
    out = nc.absolute_vs_naive_table(pd.concat([model, naive], ignore_index=True))
    assert len(out) == 1
    row = out.iloc[0]
    assert row["status"] == "ok"
    assert str(row["status"]) != "missing_naive"
    assert int(row["n_matched"]) > 0
    assert row["naive_coin_universe_hash"] == h


def test_absolute_vs_naive_resolves_naive_hash_from_realized_tickers():
    """When the NAIVE rows lack a stamped coin_universe_hash, the match
    falls back to the realized-ticker hash — which equals the model's
    realized-ticker hash for the same universe."""
    h = coin_universe_hash(("BTC", "ETH"))
    model = _naive_or_model_row(set_id="ECON_VAD_F", hpo_variant="log_loss",
                                seed=2, coin_hash=h)
    naive = _naive_or_model_row(set_id="NAIVE", hpo_variant="naive",
                                seed=2, coin_hash=h)
    naive["target"] = model["target"].values
    # Strip the NAIVE hash column value (legacy NAIVE file).
    naive["coin_universe_hash"] = ""
    # And strip the model hash so BOTH fall back to realized tickers.
    model["coin_universe_hash"] = ""
    out = nc.absolute_vs_naive_table(pd.concat([model, naive], ignore_index=True))
    row = out.iloc[0]
    assert row["status"] == "ok"
    assert row["naive_coin_universe_hash"] == h


def test_absolute_vs_naive_intersection_on_coverage_asymmetry():
    """NAIVE covers more (ticker,timestamp) than the model — the
    comparison runs on the overlap, reporting honest unmatched counts."""
    h = coin_universe_hash(("BTC", "ETH"))
    model = _naive_or_model_row(set_id="ECON_VAD_F", hpo_variant="log_loss",
                                n=60, seed=7, coin_hash=h)
    naive = _naive_or_model_row(set_id="NAIVE", hpo_variant="naive",
                                n=80, seed=7, coin_hash=h)
    # Align targets on the overlapping 60-day prefix per ticker.
    for tk in ("BTC", "ETH"):
        m_mask = model["ticker"] == tk
        n_mask = naive["ticker"] == tk
        naive.loc[n_mask & (naive.groupby("ticker").cumcount() < 60), "target"] = \
            model.loc[m_mask, "target"].values
    out = nc.absolute_vs_naive_table(pd.concat([model, naive], ignore_index=True))
    row = out.iloc[0]
    assert row["status"] == "ok"
    assert int(row["n_matched"]) == 120          # 60 days × 2 tickers
    assert int(row["n_unmatched_naive"]) == 40   # 80-60 days × 2 tickers
