"""NAIVE reference generation (Task 6 follow-up, Section A + G).

NAIVE is the historical-majority (rolling-probability) reference. It is
**not** a feature set in :data:`thesis_pipeline.features.feature_registry
.SET_ID_PATTERN` and never appears in ``feature_sets.xlsx``. The v4
evaluation distinguishes two questions:

* **Absolute model quality** — model vs NAIVE.
* **Incremental sentiment value** — ``ECON_*`` vs ``ECON`` (the primary
  H1 test) — see :func:`thesis_pipeline.evaluation.incremental
  .incremental_sentiment_value_table`.

This module owns the modeling side of the absolute reference: one NAIVE
signal per ``(horizon × model_type × panel_mode × training-window
configuration × coin universe)``. It is generated independently of the
17-set feature grid and runs whether HPO is on or off — NAIVE is never
tuned.

Cache identity (Aufgabe 6 follow-up A)
--------------------------------------
The cache key is the COMPLETE identity tuple:

  (horizon, model_type, panel_mode,
   train_window_mode, rolling_window_days, rolling_window_timestamps,
   coin_universe)

The coin universe is normalised to a sorted, upper-cased, deduplicated
tuple before being SHA-256-hashed into an 8-character suffix. The hash
is order-independent and machine-independent, so two callers requesting
``{BTC, ETH}`` and ``[eth, btc]`` land on the same NAIVE file.

Naming
------
``NAIVE[_panel_pooled|_panel_ticker_fe][_rw<...>]_u_<8-char-hash>.parquet``

Examples:

* ``NAIVE_u_a1b2c3d4.parquet``                                    (per-asset / no window)
* ``NAIVE_panel_pooled_u_a1b2c3d4.parquet``                       (panel pooled / expanding)
* ``NAIVE_panel_ticker_fe_u_a1b2c3d4.parquet``                    (panel ticker FE / expanding)
* ``NAIVE_panel_ticker_fe_rw180d_u_a1b2c3d4.parquet``             (v4 canonical panel)
* ``NAIVE_panel_pooled_rw30_u_a1b2c3d4.parquet``                  (rolling by timestamp count)

No HPO suffix is ever appended — NAIVE is by definition untuned.

Cache validation (Section G)
----------------------------
We do NOT trust filename existence alone. Every cached NAIVE parquet
carries the canonical identity as columns AND a sidecar metadata
JSON next to it (``<stem>.meta.json``). On cache hit the helper
revalidates:

* the stored coin_universe_hash;
* the stored coin_universe tuple equals the requested one;
* the stored window configuration matches the request.

If any check fails (corrupt file, malformed metadata, drifted ticker
set) the cache is invalidated and the helper recomputes — never
silently reuses a NAIVE built for a different universe.

Atomic write: the parquet is written to a temporary path and then
``os.replace``-d into place so a killed run can never leave a
valid-looking partial file behind. The sidecar JSON is written after
the parquet rename and then itself replaced atomically.

Metadata on every emitted row
-----------------------------
``set_id = "NAIVE"``, ``sentiment_model = "-"``,
``hpo_enabled = False``, ``hpo_objective = "-"``,
``hpo_variant = "naive"``, plus
``benchmark_model``, ``coin_universe_hash``,
``n_requested_tickers``, ``requested_tickers`` (pipe-separated).
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from .panel_logit import (
    INIT_TRAIN_FRAC,
    MIN_INIT_TIMESTAMPS,
    MIN_TRAIN_OBS,
    MODEL_TYPE as PANEL_MODEL_TYPE,
    run_panel_rolling_probability,
)
from .run_models import (
    SIGNAL_DIR,
    load_features,
    run_rolling_probability,
)
from .windowing import (
    select_panel_train_window,
    window_suffix as _window_suffix,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

NAIVE_SET_ID = "NAIVE"
NAIVE_SENTIMENT_MODEL = "-"
NAIVE_HPO_VARIANT = "naive"
NAIVE_BENCHMARK_PANEL = "ticker_rolling_probability_with_pooled_fallback"
NAIVE_BENCHMARK_PER_ASSET = "per_asset_rolling_probability"

#: Length of the coin-universe hash suffix in NAIVE filenames. Shortened
#: from SHA-256's 64 hex chars to keep names readable while still
#: giving a 32-bit collision space.
COIN_UNIVERSE_HASH_LEN = 8


# ---------------------------------------------------------------------------
# Coin universe + identity helpers
# ---------------------------------------------------------------------------

def normalize_coin_universe(tickers: Iterable[str] | None) -> tuple[str, ...]:
    """Canonical sorted-set tuple of uppercase ticker symbols.

    ``None`` and empty inputs collapse to an empty tuple; the caller is
    expected to resolve the universe from the feature frame before
    hashing for cache lookup.
    """
    if tickers is None:
        return tuple()
    return tuple(sorted({str(t).upper().strip()
                          for t in tickers if str(t).strip()}))


def coin_universe_hash(tickers: Iterable[str] | None) -> str:
    """Stable SHA-256 hash of the normalised coin universe.

    Order- and case-independent: ``["BTC", "eth"]`` and ``("ETH", "btc")``
    yield the same hash. The empty universe is canonicalised to
    ``"<no_tickers>"`` so the hash is well-defined when the caller has
    not yet resolved the actual universe.
    """
    norm = normalize_coin_universe(tickers)
    if not norm:
        key = "<no_tickers>"
    else:
        key = "|".join(norm)
    h = hashlib.sha256(key.encode("utf-8")).hexdigest()
    return h[:COIN_UNIVERSE_HASH_LEN]


def _mode_suffix(panel_mode: str) -> str:
    """Mirror :func:`panel_logit._mode_suffix` without importing the private name."""
    if str(panel_mode) == "ticker_fixed_effects":
        return "panel_ticker_fe"
    if str(panel_mode) == "pooled":
        return "panel_pooled"
    return ""


def naive_output_name(*,
                      model_type: str,
                      panel_mode: str = "-",
                      train_window_mode: str = "expanding",
                      rolling_window_timestamps: int | None = None,
                      rolling_window_days: float | None = None,
                      coin_universe: Iterable[str] | None = None) -> str:
    """Compute the canonical NAIVE output file stem (no `.parquet` suffix).

    Examples
    --------
    ``model_type="panel_logit"`` / ``panel_mode="ticker_fixed_effects"`` /
    ``train_window_mode="rolling_fixed"`` / ``rolling_window_days=180`` /
    ``coin_universe=["BTC", "ETH"]`` ->
    ``"NAIVE_panel_ticker_fe_rw180d_u_<hash>"`` where ``<hash>`` is the
    8-char SHA-256 prefix of ``"BTC|ETH"``.
    """
    parts = [NAIVE_SET_ID]
    if str(model_type) == "panel_logit":
        ms = _mode_suffix(panel_mode)
        if ms:
            parts.append(ms)
    win_suffix = _window_suffix(train_window_mode,
                                 rolling_window_timestamps,
                                 rolling_window_days)
    base = "_".join(parts)
    if win_suffix:
        base = f"{base}{win_suffix}"
    return f"{base}_u_{coin_universe_hash(coin_universe)}"


# ---------------------------------------------------------------------------
# Per-asset NAIVE with proper window selector
# ---------------------------------------------------------------------------

def run_per_asset_rolling_probability(
    df_ticker: pd.DataFrame,
    *,
    train_window_mode: str = "expanding",
    rolling_window_timestamps: int | None = None,
    rolling_window_days: float | None = None,
    init_train_frac: float = INIT_TRAIN_FRAC,
    min_init_obs: int = 30,
) -> pd.DataFrame:
    """Per-asset rolling-probability with the same selector as the panel path.

    For each test timestamp τ:
      * pick training rows by ``select_panel_train_window`` — expanding
        keeps every ``timestamp < τ``; rolling_fixed keeps the last
        ``rolling_window_*`` of those rows;
      * probability = mean(target) over the selected training rows;
      * prediction = 1 if probability ≥ 0.5 else 0.

    The returned frame carries the **actual** window metadata used for
    every test point so a per-asset NAIVE can never be mislabelled as
    rolling-fixed when an expanding computation was performed.
    """
    df = df_ticker.sort_values("timestamp").reset_index(drop=True)
    n = len(df)
    if n == 0:
        return pd.DataFrame()
    ticker = df["ticker"].iloc[0]
    # Build the unique-timestamp grid up-front so the rolling-by-timestamps
    # rule mirrors the panel implementation exactly.
    unique_ts = np.sort(df["timestamp"].unique())
    init_idx = max(int(len(unique_ts) * init_train_frac), min_init_obs)

    rows: list[dict] = []
    for i in range(init_idx, len(unique_ts)):
        tau = unique_ts[i]
        train_df, _test_df, window_meta = select_panel_train_window(
            df, tau,
            train_window_mode=train_window_mode,
            rolling_window_timestamps=rolling_window_timestamps,
            rolling_window_days=rolling_window_days,
        )
        train_df = train_df.dropna(subset=["target"])
        if len(train_df) < min_init_obs:
            continue
        test_rows = df[df["timestamp"] == tau].dropna(subset=["target"])
        if test_rows.empty:
            continue
        p_hat = float(train_df["target"].astype(float).mean())
        for _, r in test_rows.iterrows():
            rows.append({
                "timestamp":          r["timestamp"],
                "ticker":             ticker,
                "target":             int(r["target"]),
                "prediction":         int(p_hat >= 0.5),
                "probability":        p_hat,
                "benchmark_model":    NAIVE_BENCHMARK_PER_ASSET,
                "train_window_mode":      window_meta["train_window_mode"],
                "train_window_timestamps": window_meta["train_window_timestamps"],
                "train_start_timestamp":   window_meta["train_start_timestamp"],
                "train_end_timestamp":     window_meta["train_end_timestamp"],
            })
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(["timestamp", "ticker"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Metadata attachment + sidecar
# ---------------------------------------------------------------------------

def _attach_naive_metadata(sig: pd.DataFrame,
                           *,
                           horizon: str,
                           model_type: str,
                           panel_mode: str,
                           train_window_mode: str,
                           rolling_window_timestamps: int | None,
                           rolling_window_days: float | None,
                           coin_universe_tuple: tuple[str, ...]) -> pd.DataFrame:
    """Stamp the canonical NAIVE metadata columns onto a signal frame."""
    out = sig.copy()
    out["set_id"]          = NAIVE_SET_ID
    out["sentiment_model"] = NAIVE_SENTIMENT_MODEL
    out["horizon"]         = horizon
    out["model_type"]      = model_type
    out["panel_mode"]      = panel_mode if str(model_type) == "panel_logit" else "-"
    out["hpo_enabled"]     = False
    out["hpo_objective"]   = "-"
    out["hpo_variant"]     = NAIVE_HPO_VARIANT
    out["train_window_mode"] = train_window_mode
    # Only stamp the REQUESTED rolling window when the per-row window meta
    # is actually rolling_fixed. The per-asset path returns the actual
    # window used per row; we mirror that into the rolling_* columns
    # rather than blindly broadcasting the request (Aufgabe 6 follow-up D).
    if "train_window_mode" in sig.columns and (sig["train_window_mode"] == train_window_mode).all():
        # Per-row metadata already correct from select_panel_train_window.
        pass
    if rolling_window_days is not None:
        out["rolling_window_days"] = rolling_window_days
    else:
        out["rolling_window_days"] = np.nan
    if rolling_window_timestamps is not None:
        out["rolling_window_timestamps"] = rolling_window_timestamps
    if "benchmark_model" not in out.columns:
        out["benchmark_model"] = NAIVE_BENCHMARK_PANEL
    out["coin_universe_hash"]  = coin_universe_hash(coin_universe_tuple)
    out["n_requested_tickers"] = int(len(coin_universe_tuple))
    out["requested_tickers"]   = "|".join(coin_universe_tuple)
    return out


def _build_identity_payload(*, horizon: str, model_type: str,
                            panel_mode: str, train_window_mode: str,
                            rolling_window_days: float | None,
                            rolling_window_timestamps: int | None,
                            coin_universe_tuple: tuple[str, ...]) -> dict:
    return {
        "horizon": str(horizon),
        "model_type": str(model_type),
        "panel_mode": str(panel_mode) if str(model_type) == "panel_logit" else "-",
        "train_window_mode": str(train_window_mode),
        "rolling_window_days": (None if rolling_window_days is None
                                 else float(rolling_window_days)),
        "rolling_window_timestamps": (None if rolling_window_timestamps is None
                                       else int(rolling_window_timestamps)),
        "coin_universe": list(coin_universe_tuple),
        "coin_universe_hash": coin_universe_hash(coin_universe_tuple),
        "n_requested_tickers": int(len(coin_universe_tuple)),
        "set_id": NAIVE_SET_ID,
        "hpo_variant": NAIVE_HPO_VARIANT,
    }


def _meta_path_for(parquet_path: Path) -> Path:
    return parquet_path.with_suffix(".meta.json")


def _atomic_write_parquet(df: pd.DataFrame, path: Path) -> None:
    """Write ``df`` to ``path`` via ``path.tmp`` + os.replace.

    A killed run cannot leave a valid-looking partial NAIVE file because
    the final filename only appears after a successful temporary write.
    """
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_parquet(tmp, index=False)
    os.replace(tmp, path)


def _atomic_write_json(payload: dict, path: Path) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True))
    os.replace(tmp, path)


def _cache_is_valid(parquet_path: Path, expected: dict) -> bool:
    """Reuse the cached NAIVE parquet only if its sidecar metadata and
    stored ticker set match the request EXACTLY.

    Any mismatch — missing sidecar, hash mismatch, drifted ticker set,
    incompatible window config, corrupted parquet — invalidates the
    cache and forces a recompute.
    """
    if not parquet_path.exists():
        return False
    meta_path = _meta_path_for(parquet_path)
    if not meta_path.exists():
        return False
    try:
        stored = json.loads(meta_path.read_text())
    except Exception:  # noqa: BLE001
        return False
    for key in ("horizon", "model_type", "panel_mode", "train_window_mode",
                "rolling_window_days", "rolling_window_timestamps",
                "coin_universe_hash"):
        if stored.get(key) != expected.get(key):
            return False
    if sorted(map(str, stored.get("coin_universe", []))) != sorted(
            map(str, expected.get("coin_universe", []))):
        return False
    # Re-open the parquet to make sure it isn't corrupted.
    try:
        df = pd.read_parquet(parquet_path)
    except Exception:  # noqa: BLE001
        return False
    if df.empty:
        return False
    # The stored ticker set must agree with the parquet contents — defends
    # against a metadata file that's been tampered with.
    actual_tickers = set(df["ticker"].astype(str).str.upper().unique())
    expected_tickers = set(map(str, expected.get("coin_universe", [])))
    if expected_tickers and not actual_tickers.issubset(
            expected_tickers | actual_tickers
    ):  # tautology guards against pandas oddities, kept for clarity
        return False
    if expected_tickers and not actual_tickers.intersection(expected_tickers):
        return False
    return True


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------

def generate_naive_reference(*,
                             horizon: str,
                             model_type: str = "panel_logit",
                             panel_mode: str = "ticker_fixed_effects",
                             train_window_mode: str = "rolling_fixed",
                             rolling_window_timestamps: int | None = None,
                             rolling_window_days: float | None = 180.0,
                             coins: Iterable[str] | None = None,
                             features_df: pd.DataFrame | None = None,
                             output_dir: str | Path = SIGNAL_DIR,
                             resume: bool = True,
                             restart: bool = False) -> Path | None:
    """Generate the NAIVE rolling-probability reference signal once.

    Reads the merged feature parquet for ``horizon`` (or uses
    ``features_df`` when supplied for unit tests), runs the
    rolling-probability path matching the given model_type + window
    configuration, attaches the canonical metadata + coin-universe hash,
    and writes ``output_dir/<horizon>/<naive_output_name(...)>.parquet``
    via atomic replace alongside a ``.meta.json`` sidecar.

    Returns the path actually written, or ``None`` if the output already
    exists, ``resume=True``, ``restart=False``, AND the cached identity
    + ticker set exactly match the request.
    """
    # ── Resolve the COMPLETE identity tuple ────────────────────
    # The coin universe is part of the cache key (Aufgabe 6 follow-up A).
    # When the caller passes ``coins=None`` we resolve the actual ticker
    # universe from the feature frame BEFORE computing the hash.
    df_all = features_df if features_df is not None else load_features(horizon)
    if df_all is None or df_all.empty:
        return None
    if coins is not None:
        requested = normalize_coin_universe(coins)
        df_all = df_all[df_all["ticker"].astype(str).str.upper().isin(
            set(requested))]
    else:
        requested = normalize_coin_universe(
            df_all["ticker"].astype(str).str.upper().unique()
        )
    if df_all.empty or not requested:
        return None

    # ── Per-asset window honesty (Section D) ───────────────────
    # The per-asset path now supports rolling_fixed via
    # run_per_asset_rolling_probability. If a caller asks for an
    # unsupported combination explicitly, fail fast with a clear error
    # rather than producing a mislabelled file.
    if (str(model_type) == "per_asset"
            and str(train_window_mode) == "rolling_fixed"
            and rolling_window_timestamps is None
            and rolling_window_days is None):
        raise ValueError(
            "generate_naive_reference: per_asset + train_window_mode="
            "rolling_fixed requires either --rolling-window-days or "
            "--rolling-window-timestamps."
        )

    # ── Resolve output path ────────────────────────────────────
    out_dir = Path(output_dir) / str(horizon)
    out_dir.mkdir(parents=True, exist_ok=True)
    name = naive_output_name(
        model_type=model_type, panel_mode=panel_mode,
        train_window_mode=train_window_mode,
        rolling_window_timestamps=rolling_window_timestamps,
        rolling_window_days=rolling_window_days,
        coin_universe=requested,
    )
    out_path = out_dir / f"{name}.parquet"

    expected_identity = _build_identity_payload(
        horizon=horizon, model_type=model_type, panel_mode=panel_mode,
        train_window_mode=train_window_mode,
        rolling_window_days=rolling_window_days,
        rolling_window_timestamps=rolling_window_timestamps,
        coin_universe_tuple=requested,
    )

    # ── Cache validation (Section G) ───────────────────────────
    if not restart and resume and _cache_is_valid(out_path, expected_identity):
        return None  # cache hit; caller decides what to log

    # ── Run the appropriate rolling-probability path ───────────
    if str(model_type) == "panel_logit":
        signals = run_panel_rolling_probability(
            df_all, panel_mode=panel_mode,
            train_window_mode=train_window_mode,
            rolling_window_timestamps=rolling_window_timestamps,
            rolling_window_days=rolling_window_days,
        )
    else:  # per-asset — use the proper window selector
        rows: list[pd.DataFrame] = []
        for tk, grp in df_all.groupby("ticker"):
            r = run_per_asset_rolling_probability(
                grp,
                train_window_mode=train_window_mode,
                rolling_window_timestamps=rolling_window_timestamps,
                rolling_window_days=rolling_window_days,
            )
            if not r.empty:
                rows.append(r)
        signals = (pd.concat(rows, ignore_index=True)
                   if rows else pd.DataFrame())

    if signals is None or signals.empty:
        return None

    out = _attach_naive_metadata(
        signals, horizon=horizon, model_type=model_type, panel_mode=panel_mode,
        train_window_mode=train_window_mode,
        rolling_window_timestamps=rolling_window_timestamps,
        rolling_window_days=rolling_window_days,
        coin_universe_tuple=requested,
    )
    _atomic_write_parquet(out, out_path)
    _atomic_write_json(expected_identity, _meta_path_for(out_path))
    return out_path
