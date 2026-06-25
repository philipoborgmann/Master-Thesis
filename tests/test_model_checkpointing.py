"""Tests for robust model-run checkpointing (per-asset tickers + panel chunks).

Synthetic data only — every run is tiny and confined to ``tmp_path`` (the
synthetic repo is entered with ``monkeypatch.chdir``), so no real ``Data/`` or
``Outputs/`` is ever touched.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from thesis_pipeline.modeling import checkpointing as ckpt
from thesis_pipeline.modeling import run_models as rm
from thesis_pipeline.modeling import panel_logit as pl


# ---------------------------------------------------------------------------
# Synthetic repo
# ---------------------------------------------------------------------------

def _features(n_per_ticker=120, tickers=("BTC", "ETH", "SOL"), seed=0):
    rng = np.random.default_rng(seed)
    ts = pd.date_range("2024-01-01", periods=n_per_ticker, freq="D", tz="UTC")
    frames = []
    for tk in tickers:
        s = rng.normal(0, 1, n_per_ticker)
        e = rng.normal(0, 1, n_per_ticker)
        frames.append(pd.DataFrame({
            "timestamp": ts, "ticker": tk, "horizon": "1d",
            "target": (2.0 * s + 0.3 * e > 0).astype(int),
            "signal_feature": s, "noise_feature": rng.normal(0, 1, n_per_ticker),
        }))
    return pd.concat(frames, ignore_index=True)


@pytest.fixture
def repo(tmp_path, monkeypatch):
    (tmp_path / "Data" / "Final").mkdir(parents=True)
    (tmp_path / "Outputs" / "Signals").mkdir(parents=True)
    _features().to_parquet(tmp_path / "Data" / "Final" / "features_1d.parquet",
                           index=False)
    # Synthetic fixture (v4):
    #   * NAIVE_FIXTURE — unit-test-only sentinel that triggers the
    #     rolling-probability benchmark via the __majority_class__ feature.
    #     Not in SET_ID_PATTERN; the name lives only inside this xlsx.
    #   * ECON / ECON_CBT_F — real v4 ids; the test overrides their
    #     production feature lists with synthetic columns so the modeling
    #     code path can be exercised on noise + signal.
    fs = pd.DataFrame({
        "set_id":   ["NAIVE_FIXTURE", "ECON", "ECON_CBT_F"],
        "category": ["benchmark", "benchmark", "combined"],
        "sentiment_model": ["-", "-", "cryptobert"],
        "label":    ["unit-test NAIVE rolling-prob sentinel",
                     "synthetic ECON (single signal feature)",
                     "synthetic ECON_CBT_F (signal + noise)"],
        "Feature Columns (comma-separated)": [
            "__majority_class__", "signal_feature", "signal_feature,noise_feature",
        ],
    })
    with pd.ExcelWriter(tmp_path / "feature_sets.xlsx", engine="openpyxl") as w:
        fs.to_excel(w, sheet_name="feature_sets", index=False)
    monkeypatch.chdir(tmp_path)
    return tmp_path


def _ckpt_dir(repo, out_name, horizon="1d"):
    return repo / "Outputs" / "Checkpoints" / "Models" / horizon / out_name


def _final(repo, name, horizon="1d"):
    return repo / "Outputs" / "Signals" / horizon / f"{name}.parquet"


# ---------------------------------------------------------------------------
# Unit tests for the checkpointing module
# ---------------------------------------------------------------------------

def test_checkpoint_paths_separate_by_out_name():
    base = "Outputs/Checkpoints/Models"
    assert ckpt.checkpoint_root(base, "1d", "ECON_CBT_F_cryptobert").as_posix().endswith(
        "Models/1d/ECON_CBT_F_cryptobert")
    assert ckpt.ticker_checkpoint_path(
        ckpt.checkpoint_root(base, "1d", "ECON_CBT_F"), "BTC").name == "BTC.parquet"
    assert ckpt.chunk_checkpoint_path(
        ckpt.checkpoint_root(base, "1d", "ECON_CBT_F"), 7).name == "chunk_0007.parquet"


def test_save_atomic_and_load_roundtrip(tmp_path):
    df = pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=3, tz="UTC"),
        "ticker": "BTC", "target": [1, 0, 1], "prediction": [1, 0, 1],
        "probability": [0.6, 0.4, 0.7],
    })
    p = tmp_path / "x" / "BTC.parquet"
    ckpt.save_checkpoint_atomic(df, p)
    assert p.exists()
    # No leftover .tmp file.
    assert not (tmp_path / "x" / "BTC.parquet.tmp").exists()
    loaded = ckpt.load_checkpoint(p)
    assert loaded is not None and len(loaded) == 3


def test_validate_schema_rejects_missing_columns():
    assert ckpt.validate_checkpoint_schema(
        pd.DataFrame(columns=list(ckpt.CORE_COLUMNS)))
    assert not ckpt.validate_checkpoint_schema(pd.DataFrame({"ticker": ["BTC"]}))
    assert not ckpt.validate_checkpoint_schema(None)


def test_load_corrupt_checkpoint_returns_none(tmp_path):
    p = tmp_path / "bad.parquet"
    p.write_bytes(b"not a parquet file")
    assert ckpt.load_checkpoint(p) is None


def test_chunk_indices_partitions_without_reordering():
    assert ckpt.chunk_indices(range(60, 70), 4) == [[60, 61, 62, 63],
                                                    [64, 65, 66, 67], [68, 69]]


# ---------------------------------------------------------------------------
# 1. Per-asset writes one checkpoint per ticker
# ---------------------------------------------------------------------------

def test_per_asset_writes_ticker_checkpoints(repo):
    rm.main(["--horizon", "1d", "--set-id", "ECON", "--restart", "--model-type", "per_asset", "--no-tune-hyperparams"])
    tdir = _ckpt_dir(repo, "ECON") / "tickers"
    assert {p.stem for p in tdir.glob("*.parquet")} == {"BTC", "ETH", "SOL"}
    # Manifest written and marked complete.
    mf = ckpt.load_manifest(_ckpt_dir(repo, "ECON"))
    assert mf["status"] == "complete"
    assert mf["model_type"] == "per_asset"
    assert set(mf["completed_tickers"]) == {"BTC", "ETH", "SOL"}


# ---------------------------------------------------------------------------
# 2. Per-asset resume skips existing ticker checkpoints
# ---------------------------------------------------------------------------

def test_per_asset_resume_skips_existing(repo):
    rm.main(["--horizon", "1d", "--set-id", "ECON", "--restart", "--model-type", "per_asset", "--no-tune-hyperparams"])
    cp = _ckpt_dir(repo, "ECON") / "tickers" / "BTC.parquet"
    mtime_before = os.path.getmtime(cp)
    # --restart ignores the final file but resume (default) reuses checkpoints.
    rm.main(["--horizon", "1d", "--set-id", "ECON", "--restart", "--model-type", "per_asset", "--no-tune-hyperparams"])
    assert os.path.getmtime(cp) == mtime_before, "resumed ticker must not be rewritten"


# ---------------------------------------------------------------------------
# 3. Per-asset finalises from checkpoints when the final file is missing
# ---------------------------------------------------------------------------

def test_per_asset_finalizes_from_checkpoints_when_final_missing(repo):
    rm.main(["--horizon", "1d", "--set-id", "ECON", "--restart", "--model-type", "per_asset", "--no-tune-hyperparams"])
    final = _final(repo, "ECON")
    original = pd.read_parquet(final)
    cp = _ckpt_dir(repo, "ECON") / "tickers" / "BTC.parquet"
    mtime_before = os.path.getmtime(cp)
    final.unlink()  # delete only the final signal file

    # Re-run without --restart: final missing → rebuild from checkpoints.
    rm.main(["--horizon", "1d", "--set-id", "ECON", "--model-type", "per_asset", "--no-tune-hyperparams"])
    assert final.exists()
    rebuilt = pd.read_parquet(final)
    assert len(rebuilt) == len(original)
    # Checkpoints were reused, not recomputed.
    assert os.path.getmtime(cp) == mtime_before


# ---------------------------------------------------------------------------
# 4. --restart does NOT delete checkpoints automatically
# ---------------------------------------------------------------------------

def test_restart_does_not_clear_checkpoints(repo):
    rm.main(["--horizon", "1d", "--set-id", "ECON", "--restart", "--model-type", "per_asset", "--no-tune-hyperparams"])
    cp = _ckpt_dir(repo, "ECON") / "tickers" / "BTC.parquet"
    assert cp.exists()
    rm.main(["--horizon", "1d", "--set-id", "ECON", "--restart", "--model-type", "per_asset", "--no-tune-hyperparams"])
    assert cp.exists(), "--restart alone must keep checkpoints"


# ---------------------------------------------------------------------------
# 5. --clear-checkpoints removes only this run's directory
# ---------------------------------------------------------------------------

def test_clear_checkpoints_only_affects_this_run(repo):
    rm.main(["--horizon", "1d", "--set-id", "ECON", "--restart", "--model-type", "per_asset", "--no-tune-hyperparams"])
    rm.main(["--horizon", "1d", "--set-id", "ECON_CBT_F", "--sentiment-model",
             "cryptobert", "--restart", "--model-type", "per_asset", "--no-tune-hyperparams"])
    econ_cp = _ckpt_dir(repo, "ECON") / "tickers" / "BTC.parquet"
    ecbt_cp = _ckpt_dir(repo, "ECON_CBT_F_cryptobert") / "tickers" / "BTC.parquet"
    assert econ_cp.exists() and ecbt_cp.exists()
    econ_mtime = os.path.getmtime(econ_cp)

    # Clear only the C3 run.
    rm.main(["--horizon", "1d", "--set-id", "ECON_CBT_F", "--sentiment-model",
             "cryptobert", "--restart", "--clear-checkpoints", "--model-type", "per_asset", "--no-tune-hyperparams"])
    assert econ_cp.exists()
    assert os.path.getmtime(econ_cp) == econ_mtime, "other run's checkpoints untouched"
    assert ecbt_cp.exists()  # cleared then recomputed


# ---------------------------------------------------------------------------
# 6. Panel writes chunk checkpoints
# ---------------------------------------------------------------------------

def test_panel_writes_chunk_checkpoints(repo):
    rm.main(["--horizon", "1d", "--set-id", "ECON_CBT_F", "--sentiment-model", "cryptobert",
             "--model-type", "panel_logit", "--panel-mode", "pooled",
             "--checkpoint-chunk-size", "10", "--restart", "--train-window", "expanding", "--no-tune-hyperparams"])
    cdir = _ckpt_dir(repo, "ECON_CBT_F_cryptobert_panel_pooled") / "chunks"
    chunks = sorted(cdir.glob("chunk_*.parquet"))
    assert len(chunks) >= 1
    assert chunks[0].name == "chunk_0000.parquet"
    mf = ckpt.load_manifest(_ckpt_dir(repo, "ECON_CBT_F_cryptobert_panel_pooled"))
    assert mf["status"] == "complete"
    assert mf["panel_mode"] == "pooled"


# ---------------------------------------------------------------------------
# 7. Panel resume skips existing chunks
# ---------------------------------------------------------------------------

def test_panel_resume_skips_existing_chunks(repo):
    args = ["--horizon", "1d", "--set-id", "ECON_CBT_F", "--sentiment-model", "cryptobert",
            "--model-type", "panel_logit", "--panel-mode", "pooled",
            "--checkpoint-chunk-size", "10", "--restart", "--train-window", "expanding", "--no-tune-hyperparams"]
    rm.main(args)
    cp = _ckpt_dir(repo, "ECON_CBT_F_cryptobert_panel_pooled") / "chunks" / "chunk_0000.parquet"
    mtime_before = os.path.getmtime(cp)
    rm.main(args)
    assert os.path.getmtime(cp) == mtime_before, "resumed chunk must not be rewritten"


# ---------------------------------------------------------------------------
# 8. Chunking does not change predictions
# ---------------------------------------------------------------------------

def test_panel_chunking_does_not_change_predictions(tmp_path):
    df = _features(n_per_ticker=120, seed=3)
    plain = pl.run_panel_walk_forward(df, ["signal_feature"], panel_mode="pooled",
                                      min_init_timestamps=30)
    ctx = {"enabled": True, "resume": True,
           "root": tmp_path / "ck", "chunk_size": 7}
    chunked = pl.run_panel_walk_forward(df, ["signal_feature"], panel_mode="pooled",
                                        min_init_timestamps=30,
                                        checkpoint_context=ctx)
    a = plain.sort_values(["timestamp", "ticker"]).reset_index(drop=True)
    b = chunked.sort_values(["timestamp", "ticker"]).reset_index(drop=True)
    assert len(a) == len(b)
    assert (a["prediction"].values == b["prediction"].values).all()
    assert np.allclose(a["probability"].values, b["probability"].values)


# ---------------------------------------------------------------------------
# 9. Corrupt checkpoint is ignored and recomputed
# ---------------------------------------------------------------------------

def test_corrupt_ticker_checkpoint_is_recomputed(repo):
    rm.main(["--horizon", "1d", "--set-id", "ECON", "--restart", "--model-type", "per_asset", "--no-tune-hyperparams"])
    cp = _ckpt_dir(repo, "ECON") / "tickers" / "BTC.parquet"
    cp.write_bytes(b"garbage not parquet")  # corrupt it
    # Re-run with --restart (final ignored) → corrupt BTC recomputed, others reused.
    rm.main(["--horizon", "1d", "--set-id", "ECON", "--restart", "--model-type", "per_asset", "--no-tune-hyperparams"])
    rebuilt = ckpt.load_checkpoint(cp)
    assert rebuilt is not None and not rebuilt.empty
    final = pd.read_parquet(_final(repo, "ECON"))
    assert "BTC" in set(final["ticker"])


# ---------------------------------------------------------------------------
# 10 + 11. HPO uses its own checkpoint path; fixed and HPO never mix
# ---------------------------------------------------------------------------

def test_hpo_checkpoint_path_isolated_from_fixed(repo):
    # Both calls are per-asset so the checkpoint directory names match
    # the legacy expectation (no `_panel_*_rw180d` suffix). The first
    # establishes the FIXED dir; the second the HPO dir.
    rm.main(["--horizon", "1d", "--set-id", "ECON_CBT_F", "--sentiment-model",
             "cryptobert", "--restart",
             "--model-type", "per_asset", "--no-tune-hyperparams"])
    rm.main(["--horizon", "1d", "--set-id", "ECON_CBT_F", "--sentiment-model",
             "cryptobert", "--restart",
             "--model-type", "per_asset",
             "--tune-hyperparams", "--hpo-objective", "brier_score"])
    fixed_dir = _ckpt_dir(repo, "ECON_CBT_F_cryptobert")
    hpo_dir = _ckpt_dir(repo, "ECON_CBT_F_cryptobert_hpo_brier")
    assert fixed_dir.exists() and hpo_dir.exists()
    assert fixed_dir != hpo_dir
    hpo_mf = ckpt.load_manifest(hpo_dir)
    assert hpo_mf["hpo_variant"] == "hpo_brier"
    assert ckpt.load_manifest(fixed_dir)["hpo_variant"] == "fixed"


# ---------------------------------------------------------------------------
# 12. Panel pooled and ticker-FE checkpoints do not mix
# ---------------------------------------------------------------------------

def test_panel_modes_have_separate_checkpoints(repo):
    common = ["--horizon", "1d", "--set-id", "ECON_CBT_F", "--sentiment-model", "cryptobert",
              "--model-type", "panel_logit", "--checkpoint-chunk-size", "10",
              "--restart", "--train-window", "expanding", "--no-tune-hyperparams"]
    rm.main(common + ["--panel-mode", "pooled"])
    rm.main(common + ["--panel-mode", "ticker_fixed_effects"])
    assert _ckpt_dir(repo, "ECON_CBT_F_cryptobert_panel_pooled").exists()
    assert _ckpt_dir(repo, "ECON_CBT_F_cryptobert_panel_ticker_fe").exists()


# ---------------------------------------------------------------------------
# 13. --no-checkpoint writes no checkpoint directory
# ---------------------------------------------------------------------------

def test_no_checkpoint_writes_no_checkpoints(repo):
    rm.main(["--horizon", "1d", "--set-id", "ECON", "--restart", "--no-checkpoint", "--model-type", "per_asset", "--no-tune-hyperparams"])
    assert not _ckpt_dir(repo, "ECON").exists()
    # Final signal file is still produced.
    assert _final(repo, "ECON").exists()


# ---------------------------------------------------------------------------
# Dry-run shows the checkpoint configuration (rc == 0, parses flags)
# ---------------------------------------------------------------------------

def test_dry_run_accepts_checkpoint_flags():
    import sys
    src = Path(__file__).resolve().parents[1] / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
    from thesis_pipeline import cli
    assert cli.main(["run-models", "--horizon", "1d", "--set-id", "ECON_CBT_F",
                     "--sentiment-model", "cryptobert", "--checkpoint",
                     "--checkpoint-chunk-size", "10", "--dry-run"]) == 0
    assert cli.main(["run-models", "--horizon", "1d", "--set-id", "ECON_CBT_F",
                     "--no-checkpoint", "--dry-run"]) == 0


# ---------------------------------------------------------------------------
# Programmatic run(...) wrappers accept checkpoint args
# ---------------------------------------------------------------------------

def test_run_wrappers_accept_checkpoint_args():
    assert rm.run(horizon="1d", set_id="ECON_CBT_F", sentiment_model="cryptobert",
                  checkpoint=False, resume=False, checkpoint_chunk_size=5,
                  dry_run=True) == 0
    assert pl.run(horizon="1d", set_id="ECON_CBT_F", sentiment_model="cryptobert",
                  panel_mode="pooled", checkpoint=True, checkpoint_chunk_size=8,
                  clear_checkpoints=True, dry_run=True) == 0
