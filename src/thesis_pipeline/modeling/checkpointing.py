"""Robust, optional checkpointing for model runs.

Long model runs — especially with ``--tune-hyperparams`` — can take a long
time and previously had no intermediate persistence: a crash after several
coins (per-asset) or many panel timestamps lost everything computed so far.

This module adds two granularities of *intermediate* checkpointing, written
under ``Outputs/Checkpoints/Models/{horizon}/{out_name}/``:

* **per-asset** — one parquet per ticker (``tickers/{ticker}.parquet``),
* **panel-logit** — one parquet per timestamp *chunk*
  (``chunks/chunk_0000.parquet`` …).

``out_name`` is the same variant-specific name used for the final signal file
(e.g. ``C3_cryptobert_hpo_brier`` or ``C3_cryptobert_panel_pooled_hpo_brier``),
so fixed / HPO / per-asset / panel-pooled / panel-ticker-FE runs never share a
checkpoint directory.

Design notes
------------
* **A chunk is a storage partition, not a training partition.** The walk-forward
  contract is untouched: every test timestamp ``τ`` still trains on *all* rows
  with ``timestamp < τ``. Chunking only decides how many ``τ`` are persisted
  together — there is no leakage.
* **Atomic writes.** Every checkpoint (and the manifest) is written to a
  sibling ``*.tmp`` file and then ``os.replace``-d into place, so a crash mid
  write never leaves a half-written checkpoint.
* **Corruption-tolerant.** An unreadable / schema-invalid checkpoint is logged
  and treated as missing (recomputed), never crashes the run.
"""

from __future__ import annotations

import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import pandas as pd

from ..logging_utils import get_logger

# Minimum columns every checkpoint parquet must carry to be considered valid.
CORE_COLUMNS = ("timestamp", "ticker", "target", "prediction", "probability")

DEFAULT_CHECKPOINT_DIR = "Outputs/Checkpoints/Models"
DEFAULT_CHUNK_SIZE = 20


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

def checkpoint_root(checkpoint_dir: str | Path, horizon: str, out_name: str) -> Path:
    """``{checkpoint_dir}/{horizon}/{out_name}`` — the per-run checkpoint root."""
    return Path(checkpoint_dir) / str(horizon) / str(out_name)


def ticker_checkpoint_path(root: str | Path, ticker: str) -> Path:
    return Path(root) / "tickers" / f"{ticker}.parquet"


def chunk_checkpoint_path(root: str | Path, chunk_id: int) -> Path:
    return Path(root) / "chunks" / f"chunk_{int(chunk_id):04d}.parquet"


def manifest_path(root: str | Path) -> Path:
    return Path(root) / "manifest.json"


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------

def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def code_version() -> str | None:
    """Best-effort code version (``THESIS_CODE_VERSION`` env var, else ``None``).

    Intentionally cheap and side-effect free — no subprocess — so it is safe to
    call inside tests and tight loops.
    """
    val = os.environ.get("THESIS_CODE_VERSION")
    return val.strip() if val else None


def load_manifest(root: str | Path) -> dict:
    """Return the manifest dict, or ``{}`` when missing/unreadable."""
    p = manifest_path(root)
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001 — corrupt manifest is non-fatal
        get_logger().warning("checkpoint: cannot read manifest %s (%s)", p, exc)
        return {}


def write_manifest(root: str | Path, manifest: dict) -> Path:
    """Atomically write the manifest JSON, stamping ``updated_at``."""
    p = manifest_path(root)
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(manifest)
    payload["updated_at"] = _utcnow()
    tmp = p.with_name(p.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    os.replace(tmp, p)
    return p


def init_manifest(root: str | Path, *, base: dict) -> dict:
    """Create or refresh the manifest for a run, preserving ``created_at``.

    ``base`` carries the run identity (horizon, set_id, sentiment_model,
    model_type, panel_mode, hpo_variant, hpo_objective, feature_cols) plus any
    partition-tracking keys already known.
    """
    existing = load_manifest(root)
    now = _utcnow()
    manifest = {
        "horizon":         base.get("horizon"),
        "set_id":          base.get("set_id"),
        "sentiment_model": base.get("sentiment_model"),
        "model_type":      base.get("model_type"),
        "panel_mode":      base.get("panel_mode"),
        "hpo_variant":     base.get("hpo_variant"),
        "hpo_objective":   base.get("hpo_objective"),
        "feature_cols":    list(base.get("feature_cols") or []),
        "created_at":      existing.get("created_at", now),
        "updated_at":      now,
        "status":          "running",
        "code_version":    base.get("code_version", code_version()),
    }
    for key in ("completed_tickers", "total_tickers",
                "completed_chunks", "total_chunks"):
        if key in base:
            manifest[key] = base[key]
        elif key in existing:
            manifest[key] = existing[key]
    # Pass through any caller-supplied identity keys that aren't already in
    # the canonical schema (e.g. ``train_window_mode``, ``rolling_window_*``,
    # ``benchmark``). Without this, downstream incompatibility guards that
    # diff caller intent against the persisted manifest would always fire on
    # the second run because the new keys would silently disappear.
    for k, v in base.items():
        if k not in manifest:
            manifest[k] = v
    write_manifest(root, manifest)
    return manifest


# ---------------------------------------------------------------------------
# Schema validation + atomic IO
# ---------------------------------------------------------------------------

def validate_checkpoint_schema(df: pd.DataFrame | None) -> bool:
    """True when ``df`` carries the core signal columns (metadata optional)."""
    if df is None:
        return False
    return all(c in df.columns for c in CORE_COLUMNS)


def save_checkpoint_atomic(df: pd.DataFrame, path: str | Path) -> Path:
    """Write ``df`` to ``path`` atomically (``*.tmp`` then ``os.replace``)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    df.to_parquet(tmp, index=False, engine="pyarrow")
    os.replace(tmp, path)
    return path


def load_checkpoint(path: str | Path) -> pd.DataFrame | None:
    """Load one checkpoint parquet, or ``None`` if missing / corrupt / invalid.

    Never raises: a corrupt or schema-incomplete checkpoint is logged and
    treated as absent so the caller recomputes it.
    """
    path = Path(path)
    if not path.exists():
        return None
    try:
        df = pd.read_parquet(path)
    except Exception as exc:  # noqa: BLE001 — corrupt parquet → recompute
        get_logger().warning(
            "checkpoint: %s is unreadable (%s) — recomputing", path, exc)
        return None
    if not validate_checkpoint_schema(df):
        get_logger().warning(
            "checkpoint: %s is missing required columns %s — recomputing",
            path, [c for c in CORE_COLUMNS if c not in df.columns])
        return None
    return df


# ---------------------------------------------------------------------------
# Listing + loading completed partitions
# ---------------------------------------------------------------------------

def list_completed_tickers(root: str | Path) -> list[str]:
    d = Path(root) / "tickers"
    if not d.exists():
        return []
    return sorted(p.stem for p in d.glob("*.parquet") if not p.name.endswith(".tmp"))


def list_completed_chunks(root: str | Path) -> list[int]:
    d = Path(root) / "chunks"
    if not d.exists():
        return []
    ids: list[int] = []
    for p in d.glob("chunk_*.parquet"):
        if p.name.endswith(".tmp"):
            continue
        try:
            ids.append(int(p.stem.split("_", 1)[1]))
        except (IndexError, ValueError):
            continue
    return sorted(ids)


def load_checkpoints(root: str | Path, kind: str = "tickers") -> list[pd.DataFrame]:
    """Load every valid checkpoint frame for a run, in deterministic order."""
    frames: list[pd.DataFrame] = []
    if kind == "tickers":
        for tk in list_completed_tickers(root):
            df = load_checkpoint(ticker_checkpoint_path(root, tk))
            if df is not None and not df.empty:
                frames.append(df)
    elif kind == "chunks":
        for cid in list_completed_chunks(root):
            df = load_checkpoint(chunk_checkpoint_path(root, cid))
            if df is not None and not df.empty:
                frames.append(df)
    else:
        raise ValueError(f"Unknown checkpoint kind {kind!r}")
    return frames


def finalize_from_checkpoints(root: str | Path, kind: str = "tickers") -> pd.DataFrame:
    """Concatenate all valid checkpoints into one signal frame.

    Chunk frames are re-sorted by ``(timestamp, ticker)`` so the reconstructed
    panel signal matches a non-checkpointed run regardless of chunk order.
    """
    frames = load_checkpoints(root, kind)
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    if kind == "chunks" and {"timestamp", "ticker"}.issubset(out.columns):
        out = out.sort_values(["timestamp", "ticker"]).reset_index(drop=True)
    return out


def clear_run_checkpoints(root: str | Path) -> None:
    """Delete the checkpoint directory for exactly this run (if present)."""
    root = Path(root)
    if root.exists():
        shutil.rmtree(root)
        get_logger().info("checkpoint: cleared %s", root)


# ---------------------------------------------------------------------------
# Chunk helper
# ---------------------------------------------------------------------------

def chunk_indices(test_indices: Iterable[int], chunk_size: int) -> list[list[int]]:
    """Partition test-timestamp indices into consecutive chunks of ``chunk_size``.

    This only groups *storage*; the indices and their order are preserved, so
    the per-``τ`` walk-forward computation is identical to a non-chunked run.
    """
    idx = list(test_indices)
    size = max(int(chunk_size), 1)
    return [idx[i:i + size] for i in range(0, len(idx), size)]
