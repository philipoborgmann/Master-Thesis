"""Central forecast-origin sample utilities (single source of truth wiring).

Timestamp convention
--------------------
The stored signal ``timestamp`` is the **interval-START** label. For a row
labelled ``t`` at horizon ``h``:

* the completed information interval is ``[t, t + h)``;
* the **forecast origin** is ``t + h`` (where the model stands to forecast);
* the model forecasts the NEXT interval ``[t + h, t + 2h)``.

Inclusion in the canonical production signal sample is decided on the
**forecast origin**, NOT the raw stored timestamp. The correct final raw
timestamp therefore differs by horizon. All of the sample-window constants and
the horizon→offset mapping live here (and in ``model_specs.yaml``) so no module
hard-codes 2022 date literals or re-derives the offset from a filename.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping

import pandas as pd

from ..config import load_config

# ---------------------------------------------------------------------------
# Central horizon -> offset mapping (the ONLY one in the codebase)
# ---------------------------------------------------------------------------

HORIZON_OFFSET: dict[str, pd.Timedelta] = {
    "1h": pd.Timedelta(hours=1),
    "6h": pd.Timedelta(hours=6),
    "1d": pd.Timedelta(days=1),
}

FORECAST_ORIGIN_COLUMN = "forecast_origin"


def horizon_offset(horizon: Any) -> pd.Timedelta:
    """Return the bar width for ``horizon`` (``1h``/``6h``/``1d``).

    Fails clearly on an unsupported label. The horizon must be supplied
    explicitly (from signal metadata or the model-run horizon) — never inferred
    from a filename.
    """
    key = str(horizon).strip().lower()
    if key not in HORIZON_OFFSET:
        raise ValueError(
            f"Unsupported horizon {horizon!r}; expected one of "
            f"{sorted(HORIZON_OFFSET)}."
        )
    return HORIZON_OFFSET[key]


def _ensure_utc(values: Any) -> pd.Series:
    """Coerce to timezone-aware UTC. Naive timestamps are interpreted as UTC
    (the pipeline's documented convention) — done explicitly, never silently
    shifted."""
    s = pd.Series(values) if not isinstance(values, pd.Series) else values
    if isinstance(s.dtype, pd.DatetimeTZDtype):
        return s.dt.tz_convert("UTC")
    if pd.api.types.is_datetime64_any_dtype(s):
        return s.dt.tz_localize("UTC")
    return pd.to_datetime(s, utc=True)


# ---------------------------------------------------------------------------
# forecast_origin derivation
# ---------------------------------------------------------------------------

def add_forecast_origin(df: pd.DataFrame,
                        horizon: Any,
                        *,
                        time_col: str = "timestamp",
                        origin_col: str = FORECAST_ORIGIN_COLUMN,
                        overwrite: bool = False) -> pd.DataFrame:
    """Return ``df`` with a tz-aware ``forecast_origin = timestamp + h`` column.

    The raw ``timestamp`` column is preserved (normalised to UTC). When
    ``origin_col`` already exists and ``overwrite`` is False it is left as-is
    (but still normalised to UTC).
    """
    out = df.copy()
    out[time_col] = _ensure_utc(out[time_col])
    if origin_col in out.columns and not overwrite:
        out[origin_col] = _ensure_utc(out[origin_col])
    else:
        out[origin_col] = out[time_col] + horizon_offset(horizon)
    return out


def validate_forecast_origin(df: pd.DataFrame,
                             horizon: Any,
                             *,
                             time_col: str = "timestamp",
                             origin_col: str = FORECAST_ORIGIN_COLUMN
                             ) -> pd.Series:
    """Boolean Series: True where ``forecast_origin == timestamp + h``.

    Only defined for rows that carry ``origin_col``; a missing column returns an
    all-True Series (the caller should derive it first).
    """
    if origin_col not in df.columns:
        return pd.Series(True, index=df.index)
    expected = _ensure_utc(df[time_col]) + horizon_offset(horizon)
    actual = _ensure_utc(df[origin_col])
    return actual.reset_index(drop=True) == expected.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

_DEFAULT_FORECAST_SAMPLE = {
    "basis": "forecast_origin",
    "start": "2022-01-01T00:00:00Z",
    "end_exclusive": "2023-01-01T00:00:00Z",
    "retain_raw_timestamp": True,
    "write_forecast_origin_column": True,
    # ``enabled`` gates ROW FILTERING only — forecast_origin is always stamped.
    # Defaults ON (canonical production). Env override lets the test suite run
    # its out-of-window synthetic fixtures without being emptied; the real
    # boundary behaviour is covered by dedicated tests that pass an explicit
    # config.
    "enabled": True,
}

#: Environment overrides (used mainly by tests). ``THESIS_FORECAST_SAMPLE_
#: ENABLED`` ∈ {0,1}; ``_START`` / ``_END`` accept ISO-8601 UTC instants.
_ENV_ENABLED = "THESIS_FORECAST_SAMPLE_ENABLED"
_ENV_START = "THESIS_FORECAST_SAMPLE_START"
_ENV_END = "THESIS_FORECAST_SAMPLE_END"


def load_forecast_sample_config(model_specs: Mapping[str, Any] | None = None
                                ) -> dict[str, Any]:
    """Resolve the ``forecast_sample`` block from ``model_specs.yaml``.

    Returns a fully-defaulted dict — the SINGLE source of truth for the
    canonical production sample window. Environment variables (see
    :data:`_ENV_ENABLED` / start / end) override the resolved values so a test
    harness can widen or disable the window without editing config files.
    """
    import os
    if model_specs is None:
        try:
            model_specs = load_config("model_specs")
        except FileNotFoundError:
            model_specs = {}
    raw = dict((model_specs or {}).get("forecast_sample") or {})
    cfg = dict(_DEFAULT_FORECAST_SAMPLE)
    cfg.update({k: raw[k] for k in _DEFAULT_FORECAST_SAMPLE if k in raw})
    if _ENV_ENABLED in os.environ:
        cfg["enabled"] = os.environ[_ENV_ENABLED].strip() not in ("0", "false", "False", "")
    if os.environ.get(_ENV_START):
        cfg["start"] = os.environ[_ENV_START]
    if os.environ.get(_ENV_END):
        cfg["end_exclusive"] = os.environ[_ENV_END]
    return cfg


def sample_bounds(cfg: Mapping[str, Any] | None = None
                  ) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Return ``(start, end_exclusive)`` as tz-aware UTC Timestamps."""
    cfg = cfg or load_forecast_sample_config()

    def _ts(v: Any) -> pd.Timestamp:
        t = pd.Timestamp(v)
        return t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")

    return _ts(cfg["start"]), _ts(cfg["end_exclusive"])


# ---------------------------------------------------------------------------
# Filtering — canonical entry point used by every signal writer
# ---------------------------------------------------------------------------

def restrict_to_forecast_sample(df: pd.DataFrame,
                                horizon: Any,
                                *,
                                cfg: Mapping[str, Any] | None = None,
                                time_col: str = "timestamp",
                                origin_col: str = FORECAST_ORIGIN_COLUMN
                                ) -> pd.DataFrame:
    """Stamp ``forecast_origin`` and keep only rows whose forecast origin lies
    in ``[start, end_exclusive)``.

    This is the canonical output-contract filter. It NEVER filters on the raw
    ``timestamp`` and NEVER touches predictions/probabilities of retained
    rows — it only drops out-of-period rows and adds the ``forecast_origin``
    column (dropped again only if ``write_forecast_origin_column`` is False).
    """
    cfg = cfg or load_forecast_sample_config()
    if df is None or len(df) == 0:
        return df if df is not None else pd.DataFrame()
    out = add_forecast_origin(df, horizon, time_col=time_col, origin_col=origin_col)
    # ``enabled`` gates row filtering only — forecast_origin is always stamped.
    if cfg.get("enabled", True):
        start, end = sample_bounds(cfg)
        mask = (out[origin_col] >= start) & (out[origin_col] < end)
        out = out[mask].reset_index(drop=True)
    if not cfg.get("write_forecast_origin_column", True):
        out = out.drop(columns=[origin_col])
    return out


# ---------------------------------------------------------------------------
# Cache-invalidation signature
# ---------------------------------------------------------------------------

def forecast_sample_signature(cfg: Mapping[str, Any] | None = None) -> str:
    """Stable short hash of the forecast-sample contract (basis + window +
    horizon-offset logic). Folded into the model-run preprocessing/output
    signature so a signal file / checkpoint written under a different sample
    contract is detected and rejected."""
    cfg = cfg or load_forecast_sample_config()
    payload = {
        "basis": str(cfg.get("basis")),
        "start": str(cfg.get("start")),
        "end_exclusive": str(cfg.get("end_exclusive")),
        "write_forecast_origin_column": bool(
            cfg.get("write_forecast_origin_column", True)),
        # horizon-offset logic is part of the contract — a change here changes
        # which raw timestamps map into the window.
        "horizon_offsets": {k: str(v) for k, v in sorted(HORIZON_OFFSET.items())},
        "schema": "v5_forecast_origin_sample",
    }
    blob = json.dumps(payload, sort_keys=True)
    return hashlib.sha1(blob.encode("utf-8")).hexdigest()[:16]
