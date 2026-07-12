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

#: Cache-format version embedded in every sidecar JSON. Bump this when
#: the sidecar schema changes so older caches are invalidated
#: automatically on the next run. Schema v2 (this version) introduced
#: the requested-vs-realized ticker split.
CACHE_SCHEMA_VERSION = 2

#: Label written onto every NAIVE signal row identifying the source of
#: its universe identity. Production NAIVE signals always carry
#: ``"requested_metadata"`` — the universe is fixed by the run config,
#: not inferred from the rows that happened to materialise. The
#: ``"legacy_realized_tickers_fallback"`` value is reserved for
#: backwards-compatibility paths in the evaluation layer.
UNIVERSE_IDENTITY_SOURCE_REQUESTED = "requested_metadata"
UNIVERSE_IDENTITY_SOURCE_LEGACY = "legacy_realized_tickers_fallback"


# ---------------------------------------------------------------------------
# Coin universe + identity helpers
# ---------------------------------------------------------------------------

def stamp_universe_metadata(df: pd.DataFrame,
                            *,
                            requested_universe: Iterable[str] | None,
                            available_universe: Iterable[str] | None = None,
                            source: str = UNIVERSE_IDENTITY_SOURCE_REQUESTED,
                            ) -> pd.DataFrame:
    """Stamp the requested/available/realized universe identity onto every row.

    Used by both the NAIVE generator and the production model writers
    so the same hashing helper (and the same column layout) feeds the
    matching identity downstream in
    :mod:`thesis_pipeline.evaluation.naive_comparison`.

    Parameters
    ----------
    df
        Long-form signal frame. Must carry ``ticker``.
    requested_universe
        The universe resolved from ``--coins`` (or the feature-frame
        ticker set when no coin filter was given). ``None`` is allowed
        only for legacy-fallback callers.
    available_universe
        Subset of requested actually present in the feature frame for
        the current horizon. Defaults to ``requested_universe`` when
        omitted.
    source
        Provenance label written into ``universe_identity_source``.
        Production v4 outputs always use ``"requested_metadata"``; the
        ``"legacy_realized_tickers_fallback"`` value is reserved for
        evaluation-time inference on historical files.
    """
    if df is None or df.empty:
        return df
    out = df.copy()
    requested_tuple = (
        normalize_coin_universe(requested_universe)
        if requested_universe is not None else tuple()
    )
    available_tuple = (
        normalize_coin_universe(available_universe)
        if available_universe is not None else requested_tuple
    )
    realized_tuple = normalize_coin_universe(
        out["ticker"].astype(str).unique() if "ticker" in out.columns else []
    )
    req_hash = coin_universe_hash(requested_tuple) if requested_tuple else ""
    avl_hash = coin_universe_hash(available_tuple) if available_tuple else ""
    rea_hash = coin_universe_hash(realized_tuple) if realized_tuple else ""
    # ``coin_universe_hash`` is the public alias for the REQUESTED hash —
    # downstream matching (absolute_vs_naive, checkpoints) keys off it.
    out["coin_universe_hash"]            = req_hash or rea_hash
    out["requested_coin_universe_hash"]  = req_hash
    out["n_requested_tickers"]           = int(len(requested_tuple))
    out["requested_tickers"]             = "|".join(requested_tuple)
    out["available_coin_universe_hash"]  = avl_hash
    out["n_available_tickers"]           = int(len(available_tuple))
    out["available_tickers"]             = "|".join(available_tuple)
    out["realized_coin_universe_hash"]   = rea_hash
    out["n_realized_tickers"]            = int(len(realized_tuple))
    out["realized_tickers"]              = "|".join(realized_tuple)
    out["universe_identity_source"]      = source
    return out


def resolve_universes(args_coins: Iterable[str] | None,
                      df_all_tickers: Iterable[str]) -> dict:
    """Resolve the three v4 universe tuples for one (horizon, args).

    Returns a dict with ``requested``, ``available`` and per-side hashes.
    Both are normalized via :func:`normalize_coin_universe` so the
    hashes coincide with NAIVE's.
    """
    feature_tickers = normalize_coin_universe(df_all_tickers)
    if args_coins:
        requested = normalize_coin_universe(args_coins)
    else:
        requested = feature_tickers
    available = tuple(t for t in requested if t in set(feature_tickers))
    return {
        "requested":      requested,
        "available":      available,
        "requested_hash": coin_universe_hash(requested) if requested else "",
        "available_hash": coin_universe_hash(available) if available else "",
    }


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
    """Stamp the canonical NAIVE metadata columns onto a signal frame.

    Each row carries BOTH the requested-universe identity (immutable —
    derived from ``--coins`` or the feature-frame universe) and the
    realized-universe identity (the tickers actually present in the
    output). The ``coin_universe_hash`` column aliases the requested
    hash so existing consumers keep matching by request identity.
    """
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
    if rolling_window_days is not None:
        out["rolling_window_days"] = rolling_window_days
    else:
        out["rolling_window_days"] = np.nan
    if rolling_window_timestamps is not None:
        out["rolling_window_timestamps"] = rolling_window_timestamps
    if "benchmark_model" not in out.columns:
        out["benchmark_model"] = NAIVE_BENCHMARK_PANEL
    # Requested-universe identity (immutable).
    requested_hash = coin_universe_hash(coin_universe_tuple)
    out["coin_universe_hash"]            = requested_hash
    out["requested_coin_universe_hash"]  = requested_hash
    out["n_requested_tickers"]           = int(len(coin_universe_tuple))
    out["requested_tickers"]             = "|".join(coin_universe_tuple)
    # Realized-universe identity (derived from the produced rows).
    realized = normalize_coin_universe(
        out["ticker"].astype(str).unique() if "ticker" in out.columns else []
    )
    realized_hash = coin_universe_hash(realized) if realized else ""
    out["realized_coin_universe_hash"] = realized_hash
    out["n_realized_tickers"]          = int(len(realized))
    out["realized_tickers"]            = "|".join(realized)
    out["universe_identity_source"]    = UNIVERSE_IDENTITY_SOURCE_REQUESTED
    return out


def _forecast_sample_signature() -> str:
    """Signature of the active forecast-origin sample contract (lazy import to
    avoid any import cycle)."""
    from .forecast_sample import forecast_sample_signature
    return forecast_sample_signature()


def _build_identity_payload(*, horizon: str, model_type: str,
                            panel_mode: str, train_window_mode: str,
                            rolling_window_days: float | None,
                            rolling_window_timestamps: int | None,
                            coin_universe_tuple: tuple[str, ...],
                            realized_universe_tuple: tuple[str, ...] | None = None,
                            ) -> dict:
    """Schema-v2 sidecar payload (Section A + G).

    Distinguishes the REQUESTED universe (immutable, derived from
    ``--coins`` or the feature-frame ticker set) from the REALIZED
    universe (tickers that actually produced predictions — may be a
    proper subset of requested when a coin lacked training data).

    ``coin_universe_hash`` is retained as an alias for
    ``requested_coin_universe_hash`` so older readers keep working.
    """
    requested = tuple(coin_universe_tuple)
    realized = tuple(realized_universe_tuple) if realized_universe_tuple is not None else ()
    return {
        "cache_schema_version":     CACHE_SCHEMA_VERSION,
        "horizon":                  str(horizon),
        "model_type":               str(model_type),
        "panel_mode":               (str(panel_mode)
                                     if str(model_type) == "panel_logit" else "-"),
        "train_window_mode":        str(train_window_mode),
        "rolling_window_days":      (None if rolling_window_days is None
                                     else float(rolling_window_days)),
        "rolling_window_timestamps": (None if rolling_window_timestamps is None
                                       else int(rolling_window_timestamps)),
        # Requested universe — the cache key proper.
        "requested_tickers":            list(requested),
        "requested_coin_universe_hash": coin_universe_hash(requested),
        "n_requested_tickers":          int(len(requested)),
        # Forecast-origin sample contract (Objective B): a NAIVE file written
        # under a different sample window is a cache MISS and is recomputed.
        "forecast_sample_signature":    _forecast_sample_signature(),
        # Realized universe — recorded for validation against the parquet.
        "realized_tickers":            list(realized),
        "realized_coin_universe_hash": coin_universe_hash(realized) if realized else "",
        "n_realized_tickers":          int(len(realized)),
        # Legacy aliases (alias the REQUESTED side).
        "coin_universe":      list(requested),
        "coin_universe_hash": coin_universe_hash(requested),
        # Constants.
        "set_id":      NAIVE_SET_ID,
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
    """Strict schema-v2 NAIVE cache validation (Section A + G).

    A cached NAIVE parquet is reusable only when EVERY check below
    passes:

    1. parquet file exists and is readable + non-empty;
    2. sidecar metadata exists and parses as JSON;
    3. ``cache_schema_version`` matches :data:`CACHE_SCHEMA_VERSION`;
    4. all identity fields (horizon, model_type, panel_mode, training
       window configuration) match the request;
    5. the stored REQUESTED ticker set equals the expected requested
       set, and the stored requested hash matches the expected hash;
    6. the parquet's actual ticker set equals the stored REALIZED set
       AND is a subset of the expected requested set.

    Importantly, the cache is NOT considered valid merely because the
    stored and requested ticker sets overlap (the previous tautological
    rule). A realized subset of the requested universe is allowed —
    some tickers may legitimately produce no signals — but the stored
    requested set must equal the expected requested set exactly.
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
    if int(stored.get("cache_schema_version", 0)) != CACHE_SCHEMA_VERSION:
        return False
    for key in ("horizon", "model_type", "panel_mode", "train_window_mode",
                "rolling_window_days", "rolling_window_timestamps",
                "forecast_sample_signature"):
        if stored.get(key) != expected.get(key):
            return False
    # ── Requested universe must match exactly ──────────────────
    stored_requested = set(map(str, stored.get("requested_tickers", []) or []))
    expected_requested = set(map(str, expected.get("requested_tickers", []) or []))
    if not expected_requested:
        # Defensive: the helper should never produce an empty requested
        # universe in practice — fail safe to recompute.
        return False
    if stored_requested != expected_requested:
        return False
    if stored.get("requested_coin_universe_hash") != expected.get(
            "requested_coin_universe_hash"):
        return False
    # ── Re-open the parquet to detect corruption ───────────────
    try:
        df = pd.read_parquet(parquet_path)
    except Exception:  # noqa: BLE001
        return False
    if df.empty or "ticker" not in df.columns:
        return False
    actual_tickers = set(df["ticker"].astype(str).str.upper().unique())
    if not actual_tickers:
        return False
    # ── Realized universe MUST be a subset of requested ────────
    if not actual_tickers.issubset(expected_requested):
        return False
    # ── Sidecar's realized set must agree with the parquet ─────
    stored_realized = set(map(str, stored.get("realized_tickers", []) or []))
    if stored_realized != actual_tickers:
        return False
    return True


def _purge_stale_tmp_files(parquet_path: Path) -> None:
    """Remove any leftover ``*.tmp`` files next to a cache target.

    Atomic-replace failures from earlier runs (or interrupted writes)
    can leave ``<stem>.parquet.tmp`` / ``<stem>.meta.json.tmp`` behind;
    they never block a fresh write but are nice to scrub before
    rewriting.
    """
    for tmp in (
        parquet_path.with_suffix(parquet_path.suffix + ".tmp"),
        _meta_path_for(parquet_path).with_suffix(".json.tmp"),
    ):
        try:
            tmp.unlink(missing_ok=True)
        except Exception:  # noqa: BLE001
            pass


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

    # ── Cache validation (Section A + G) ───────────────────────
    # The expected_identity payload used for validation only carries
    # the requested universe; we rebuild it with the realized universe
    # once the run produces signals so the sidecar records both sides.
    expected_for_validation = _build_identity_payload(
        horizon=horizon, model_type=model_type, panel_mode=panel_mode,
        train_window_mode=train_window_mode,
        rolling_window_days=rolling_window_days,
        rolling_window_timestamps=rolling_window_timestamps,
        coin_universe_tuple=requested,
    )

    if not restart and resume and _cache_is_valid(out_path, expected_for_validation):
        return None  # cache hit; caller decides what to log

    # Stale partial-write residue (if any) is scrubbed before the
    # recompute. Atomic replace is still the primary guarantee.
    _purge_stale_tmp_files(out_path)

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
    realized = normalize_coin_universe(
        out["ticker"].astype(str).unique() if "ticker" in out.columns else []
    )
    # Final sidecar carries BOTH the requested universe (cache key) and
    # the realized universe (validated against the parquet on hit).
    sidecar = _build_identity_payload(
        horizon=horizon, model_type=model_type, panel_mode=panel_mode,
        train_window_mode=train_window_mode,
        rolling_window_days=rolling_window_days,
        rolling_window_timestamps=rolling_window_timestamps,
        coin_universe_tuple=requested,
        realized_universe_tuple=realized,
    )
    # Canonical output contract (Objective B): stamp forecast_origin and keep
    # only rows whose forecast origin is in the configured 2022 sample window.
    # NAIVE follows the SAME rule as every model specification.
    from .forecast_sample import restrict_to_forecast_sample
    out = restrict_to_forecast_sample(out, horizon)
    _atomic_write_parquet(out, out_path)
    _atomic_write_json(sidecar, _meta_path_for(out_path))
    return out_path
