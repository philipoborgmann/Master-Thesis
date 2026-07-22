"""Process-parallel panel-logit checkpoint chunks (n_jobs).

Covers: --n-jobs parsing/validation, sequential==parallel identity, sorted
output, out-of-order completion, cached-chunk reuse, parallel resume of only
missing chunks, corrupt-chunk recompute, main-only manifest writes, worker
exception propagation, HPO parallel==sequential, and the inner-thread cap.

Fixtures are intentionally tiny so the process pool stays fast.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from thesis_pipeline.modeling import panel_logit as pl
from thesis_pipeline.modeling import checkpointing as ckpt
from thesis_pipeline.modeling.run_models import _n_jobs_arg
import argparse

# Row filtering is disabled for the test session (conftest); these synthetic
# frames use 2022 dates anyway, so results are unaffected either way.


def _panel(n_ts=40, tickers=("BTC", "ETH", "ADA"), seed=0):
    rng = np.random.default_rng(seed)
    ts = pd.date_range("2022-01-01", periods=n_ts, freq="D", tz="UTC")
    rows = []
    for tk in tickers:
        beta = rng.normal(0, 1, 3)
        for t in ts:
            x = rng.normal(0, 1, 3)
            y = int(rng.random() < 1.0 / (1.0 + np.exp(-(x @ beta))))
            rows.append({"timestamp": t, "ticker": tk, "target": y,
                         "f1": x[0], "f2": x[1], "f3": x[2]})
    return pd.DataFrame(rows)


FEATS = ["f1", "f2", "f3"]


def _run(df, root, *, n_jobs, chunk_size=7, resume=True, tune=False):
    ctx = {"enabled": True, "resume": resume, "root": Path(root),
           "chunk_size": chunk_size, "n_jobs": n_jobs}
    hpo = ({"objective": "log_loss",
            "search_space": {"C": [0.1, 1.0], "class_weight": [None]},
            "validation_fraction": 0.2, "min_train_obs": 5,
            "min_validation_obs": 3, "refit_on_full_train": True}
           if tune else None)
    return pl.run_panel_walk_forward(
        df, FEATS, panel_mode="ticker_fixed_effects",
        checkpoint_context=ctx, train_window_mode="rolling_fixed",
        rolling_window_days=30, tune_hyperparams=tune, hpo_config=hpo,
        n_jobs=n_jobs)


# --- 1. parsing / validation -------------------------------------------------

def test_n_jobs_arg_parsing_and_validation():
    assert _n_jobs_arg("1") == 1
    assert _n_jobs_arg("4") == 4
    assert _n_jobs_arg("-1") == -1
    for bad in ("0", "-2", "-10"):
        with pytest.raises(argparse.ArgumentTypeError):
            _n_jobs_arg(bad)


def test_resolve_n_jobs():
    assert pl._resolve_n_jobs(1) == 1
    assert pl._resolve_n_jobs(3) == 3
    assert pl._resolve_n_jobs(-1) == (os.cpu_count() or 1)
    for bad in (0, -2, -99):
        with pytest.raises(ValueError):
            pl._resolve_n_jobs(bad)


# --- 2. n_jobs=1 stays on the sequential path (no worker temp file) ---------

def test_n_jobs_1_uses_sequential_path(tmp_path):
    df = _panel()
    out = _run(df, tmp_path, n_jobs=1)
    assert not out.empty
    # The sequential path never dumps the shared worker parquet.
    assert not list(Path(tmp_path).glob("_worker_input_*.parquet"))


# --- 3/4/5. parallel == sequential, sorted, order-independent ----------------

def test_parallel_matches_sequential(tmp_path):
    df = _panel()
    seq = _run(df, tmp_path / "seq", n_jobs=1)
    par = _run(df, tmp_path / "par", n_jobs=4, chunk_size=5)
    pd.testing.assert_frame_equal(seq, par, check_exact=False,
                                  rtol=1e-10, atol=1e-12)
    # deterministic sort regardless of worker completion order
    assert par.equals(par.sort_values(["timestamp", "ticker"])
                      .reset_index(drop=True))
    # temp worker parquet is cleaned up after the parallel run
    assert not list((tmp_path / "par").glob("_worker_input_*.parquet"))


# --- 6. already-cached chunks are skipped ------------------------------------

def test_cached_chunks_are_skipped(tmp_path):
    df = _panel()
    _run(df, tmp_path, n_jobs=2, chunk_size=6)
    chunk_files = sorted((tmp_path / "chunks").glob("chunk_*.parquet"))
    assert chunk_files
    mtimes = {p.name: p.stat().st_mtime_ns for p in chunk_files}
    # Second run with resume must reuse every checkpoint (no rewrite).
    out2 = _run(df, tmp_path, n_jobs=2, chunk_size=6)
    for p in chunk_files:
        assert p.stat().st_mtime_ns == mtimes[p.name], f"{p.name} recomputed"
    assert not out2.empty


# --- 7. parallel resume computes only the missing chunk ----------------------

def test_parallel_resume_only_missing(tmp_path):
    df = _panel()
    full = _run(df, tmp_path, n_jobs=1, chunk_size=6)
    chunk_files = sorted((tmp_path / "chunks").glob("chunk_*.parquet"))
    assert len(chunk_files) >= 2
    # Delete one chunk → only it should be recomputed on parallel resume.
    victim = chunk_files[1]
    victim.unlink()
    survivor_mtimes = {p.name: p.stat().st_mtime_ns
                       for p in chunk_files if p.exists()}
    out = _run(df, tmp_path, n_jobs=3, chunk_size=6)
    assert victim.exists()  # recomputed
    for p in chunk_files:
        if p.name in survivor_mtimes:
            assert p.stat().st_mtime_ns == survivor_mtimes[p.name]
    pd.testing.assert_frame_equal(full, out, check_exact=False,
                                  rtol=1e-10, atol=1e-12)


# --- 8. corrupt chunk is recomputed ------------------------------------------

def test_corrupt_chunk_is_recomputed(tmp_path):
    df = _panel()
    ref = _run(df, tmp_path / "ref", n_jobs=1, chunk_size=6)
    root = tmp_path / "run"
    _run(df, root, n_jobs=1, chunk_size=6)
    chunk_files = sorted((root / "chunks").glob("chunk_*.parquet"))
    # Corrupt the first chunk file with non-parquet bytes.
    chunk_files[0].write_bytes(b"not a parquet file")
    out = _run(df, root, n_jobs=3, chunk_size=6)  # parallel resume
    pd.testing.assert_frame_equal(ref, out, check_exact=False,
                                  rtol=1e-10, atol=1e-12)


# --- 9. workers never write the manifest/checkpoints (main-only) -------------

def test_worker_function_does_not_touch_manifest_or_checkpoints(tmp_path):
    df = _panel()
    inp = tmp_path / "in.parquet"
    df.sort_values(["timestamp", "ticker"]).reset_index(drop=True) \
      .to_parquet(inp, index=False)
    params = pl._PanelParams(
        feature_cols=FEATS, C=1.0, panel_mode="ticker_fixed_effects",
        min_train_obs=20, tune_hyperparams=False, hpo_config={},
        train_window_mode="rolling_fixed", rolling_window_timestamps=None,
        rolling_window_days=30)
    cid, rows = pl._compute_panel_chunk_from_path(
        3, [25, 26, 27], str(inp), os.path.getmtime(inp), params)
    assert cid == 3
    assert isinstance(rows, list)
    # The worker created no manifest and no chunk checkpoints.
    assert not (tmp_path / "manifest.json").exists()
    assert not list(tmp_path.glob("chunk_*.parquet"))
    assert not (tmp_path / "chunks").exists()


# --- 10. worker exception propagates; no false "complete" --------------------

def test_worker_exception_propagates(tmp_path):
    df = _panel()
    root = tmp_path
    ctx = {"enabled": True, "resume": True, "root": Path(root),
           "chunk_size": 6, "n_jobs": 3}
    # A feature column that does not exist makes every worker chunk raise
    # (KeyError inside dropna) — the exception must surface, not be swallowed.
    with pytest.raises(Exception):
        pl.run_panel_walk_forward(
            df, ["does_not_exist"], panel_mode="pooled",
            checkpoint_context=ctx, train_window_mode="rolling_fixed",
            rolling_window_days=30, n_jobs=3)
    # No chunk checkpoints for the failed feature set, and status is not
    # "complete" (the main-process completion write is never reached).
    assert not list((Path(root) / "chunks").glob("chunk_*.parquet"))
    manifest = ckpt.load_manifest(root)
    assert manifest.get("status") != "complete"


# --- 11. HPO parallel == sequential ------------------------------------------

def test_hpo_parallel_matches_sequential(tmp_path):
    df = _panel(n_ts=48, seed=3)
    seq = _run(df, tmp_path / "seq", n_jobs=1, chunk_size=6, tune=True)
    par = _run(df, tmp_path / "par", n_jobs=4, chunk_size=6, tune=True)
    pd.testing.assert_frame_equal(seq, par, check_exact=False,
                                  rtol=1e-10, atol=1e-12)


# --- 12. inner numerical threads constrained to 1 per worker -----------------

def test_parallel_uses_inner_max_num_threads_one(tmp_path, monkeypatch):
    import joblib
    captured = {}

    class _FakeParallel:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        def __call__(self, tasks):
            # ``delayed(fn)(*a, **k)`` yields (fn, args, kwargs) tuples.
            return [fn(*a, **k) for (fn, a, k) in tasks]

    monkeypatch.setattr(joblib, "Parallel", _FakeParallel)
    df = _panel(n_ts=44)  # > MIN_INIT_TIMESTAMPS so parallel actually dispatches
    out = _run(df, tmp_path, n_jobs=4, chunk_size=6)
    # My orchestrator requests both the loky backend and one inner thread.
    assert captured.get("inner_max_num_threads") == 1
    assert captured.get("backend") == "loky"
    assert captured.get("n_jobs") == 4
    assert not out.empty


def test_loky_worker_thread_limit_is_effective():
    # End-to-end: the mechanism the worker uses (threadpool_limits(1)) caps the
    # numerical threadpools to a single thread inside a loky worker, so
    # n_jobs workers cannot oversubscribe the CPU.
    from joblib import Parallel, delayed
    infos = Parallel(n_jobs=2, backend="loky", inner_max_num_threads=1)(
        delayed(_worker_threadpool_limits)() for _ in range(2))
    for info in infos:
        # Empty means no BLAS pool is loaded (no oversubscription risk either);
        # otherwise every pool must be limited to 1 thread.
        assert all(n == 1 for n in info), f"oversubscribed: {info}"


def _worker_threadpool_limits():
    """Top-level (picklable) helper: report the worker's threadpool limits
    under the same one-thread cap the chunk worker applies."""
    from threadpoolctl import threadpool_info, threadpool_limits
    with threadpool_limits(limits=1):
        return [d.get("num_threads") for d in threadpool_info()]
