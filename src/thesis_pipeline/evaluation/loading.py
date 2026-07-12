"""Robust signal discovery and parquet loading.

The PRD requires that metadata is always read from parquet columns — never
inferred from filenames. Files that cannot be evaluated are skipped with a
warning; no hard crashes.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd

from ..config import resolve_path
from ..logging_utils import get_logger

REQUIRED_COLUMNS = ("timestamp", "ticker", "target", "prediction", "probability")
META_COLUMNS     = ("set_id", "sentiment_model", "horizon", "model_type",
                    "panel_mode", "hpo_objective", "hpo_variant")

# Per-column defaults applied when a meta column is absent or blank. These are
# read from the parquet columns only — never inferred from the filename. Old
# per-asset signal files (written before the panel-logit / HPO families
# existed) carry none of these; they collapse to the canonical per-asset,
# fixed-C identity here.
META_DEFAULTS = {
    "set_id":          "",
    "sentiment_model": "",
    "horizon":         "",
    "model_type":      "per_asset",
    "panel_mode":      "-",
    "hpo_objective":   "-",
    "hpo_variant":     "fixed",
}

#: Tokens that all denote "no sentiment model" and must canonicalise to
#: the single ``"-"`` sentinel. ECON / economic-only sets are written
#: under the literal ``"none"`` in feature_sets.xlsx (filename
#: ``ECON_none_...``); the incremental joiner looks them up with ``"-"``.
#: Normalising both sides to ``"-"`` is the canonical key fix (commit 12
#: Task C) — preferred over hard-coding a one-sided literal.
_NO_SENTIMENT_TOKENS = ("", "nan", "none", "null", "na", "n/a")


def canonical_sentiment_model(value) -> str:
    """Return the canonical sentiment-model key.

    Every "absent sentiment" spelling (``""``, ``"nan"``, ``"none"``,
    ``"None"``, ``"NULL"``, ``"-"`` …) collapses to ``"-"``; any real
    scorer name (``vader`` / ``cryptobert``) is returned stripped with
    its case preserved (so existing grouping keys are unaffected). This
    makes the benchmark-lookup key identical on the ECON (``none``) and
    combined (``-``) sides.
    """
    s = str(value).strip()
    if s == "-" or s.lower() in _NO_SENTIMENT_TOKENS:
        return "-"
    return s


def discover_signal_files(horizon: str | None = None,
                          signals_root: str | Path | None = None) -> list[Path]:
    """List every signal parquet under ``{signals_root}/{horizon}/``.

    When ``horizon`` is ``None``, all known horizons are searched.
    ``signals_root`` overrides the configured ``paths.yaml`` root for discovery
    only (used by the evaluate-signals ``--signals-root`` flag); it never
    mutates any configuration file.
    """
    root = Path(signals_root) if signals_root else resolve_path("signals_root")
    if not root.exists():
        return []
    if horizon:
        sub = root / horizon
        if not sub.exists():
            return []
        return sorted(sub.glob("*.parquet"))
    out: list[Path] = []
    for sub in sorted(p for p in root.iterdir() if p.is_dir()):
        out.extend(sorted(sub.glob("*.parquet")))
    return out


def _as_bool(value) -> bool:
    """Coerce parquet/CSV truthiness (bool, "True"/"False", 1/0) to a bool."""
    if isinstance(value, str):
        return value.strip().lower() in ("true", "1", "yes", "y")
    return bool(value)


def _normalise_signal_frame(df: pd.DataFrame, *, source: Path) -> pd.DataFrame:
    """Coerce dtypes and ensure required columns exist; tag with source path."""
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"{source.name}: missing columns {missing}")

    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df = df.dropna(subset=["timestamp", "target", "prediction", "probability"])
    df["target"]     = df["target"].astype(int)
    df["prediction"] = df["prediction"].astype(int)
    df["probability"] = df["probability"].astype(float)

    # Fill metadata columns from columns if missing — never from the filename.
    # Defaults are applied only when the column is absent or its value is blank;
    # a present, non-empty column value always wins.
    for col in META_COLUMNS:
        default = META_DEFAULTS.get(col, "")
        if col not in df.columns:
            df[col] = default
        else:
            s = df[col].astype(str).fillna("").replace({"nan": "", "None": ""})
            df[col] = s.replace("", default) if default else s

    # ``hpo_enabled`` is boolean, not a string identity column — handle it
    # separately so old files (no column) default to False without becoming
    # the string "False".
    if "hpo_enabled" not in df.columns:
        df["hpo_enabled"] = False
    else:
        df["hpo_enabled"] = (
            df["hpo_enabled"].map(_as_bool).fillna(False).astype(bool)
        )

    # Forecast-origin compatibility (Objective B / Section 9): accept a
    # ``forecast_origin`` column, derive it for older files that lack one,
    # validate it equals ``timestamp + h`` (reporting mismatches), and enforce
    # the configured sample window.
    df = _apply_forecast_origin(df, source=source)
    return df


def _apply_forecast_origin(df: pd.DataFrame, *, source: Path) -> pd.DataFrame:
    """Accept / derive / validate ``forecast_origin`` and enforce the sample
    period. Backward compatible with files that predate the column.

    * present column  → validate ``== timestamp + h``; inconsistent rows are
      dropped with a clear WARNING (never silently corrected);
    * absent column   → derived from ``timestamp`` + the per-row ``horizon``
      metadata (never the filename) via the central utility;
    * finally, rows whose forecast origin is outside the configured
      ``[start, end_exclusive)`` window are dropped (when filtering is enabled).
    """
    from ..modeling.forecast_sample import (
        FORECAST_ORIGIN_COLUMN, add_forecast_origin, load_forecast_sample_config,
        sample_bounds, validate_forecast_origin,
    )
    logger = get_logger()
    hz_vals = [h for h in df["horizon"].astype(str).unique()
               if h and h.lower() not in ("nan", "none", "")]
    if len(hz_vals) != 1 or str(hz_vals[0]).lower() not in ("1h", "6h", "1d"):
        # Cannot unambiguously determine the horizon → leave the frame as-is
        # (defensive backward compatibility; do not guess the offset).
        if FORECAST_ORIGIN_COLUMN not in df.columns:
            logger.warning(
                "evaluate-signals: %s has no usable 'horizon' metadata — "
                "forecast_origin not derived; period filter skipped.",
                source.name)
        return df
    hz = str(hz_vals[0]).lower()

    if FORECAST_ORIGIN_COLUMN in df.columns:
        ok = validate_forecast_origin(df, hz).to_numpy()
        n_mis = int((~ok).sum())
        if n_mis:
            logger.warning(
                "evaluate-signals: %s has %d/%d rows where forecast_origin != "
                "timestamp + %s — dropping the inconsistent rows (reported, "
                "NOT silently corrected).", source.name, n_mis, len(df), hz)
            df = df[ok].reset_index(drop=True)
    # Normalise dtype (keeps a valid existing column) or derive for old files.
    df = add_forecast_origin(df, hz)

    fs = load_forecast_sample_config()
    if fs.get("enabled", True) and not df.empty:
        start, end = sample_bounds(fs)
        fo = df[FORECAST_ORIGIN_COLUMN]
        in_period = ((fo >= start) & (fo < end)).to_numpy()
        n_out = int((~in_period).sum())
        if n_out:
            logger.warning(
                "evaluate-signals: %s — dropping %d/%d rows with forecast_origin "
                "outside the sample window [%s, %s).",
                source.name, n_out, len(df), start, end)
            df = df[in_period].reset_index(drop=True)
    return df


def load_signal_file(path: Path) -> pd.DataFrame | None:
    """Load a single signal parquet. Returns ``None`` for unusable files."""
    logger = get_logger()
    try:
        df = pd.read_parquet(path)
    except Exception as exc:  # noqa: BLE001
        logger.warning("evaluate-signals: cannot read %s (%s) — skipping", path, exc)
        return None
    if df.empty:
        logger.warning("evaluate-signals: %s is empty — skipping", path)
        return None
    try:
        df = _normalise_signal_frame(df, source=path)
    except ValueError as exc:
        logger.warning("evaluate-signals: %s skipped (%s)", path, exc)
        return None
    if df.empty:
        logger.warning("evaluate-signals: %s has no usable rows after cleaning — skipping",
                       path)
        return None
    df["__source__"] = str(path)
    return df


def load_all_signals(horizon: str | None = None,
                     paths: Iterable[Path] | None = None) -> pd.DataFrame:
    """Discover (or accept) signal files and return one long-form dataframe.

    Each row carries its own ``set_id``, ``sentiment_model`` and ``horizon``
    metadata so downstream code never has to consult the filename.
    """
    files = list(paths) if paths is not None else discover_signal_files(horizon)
    frames = []
    for f in files:
        df = load_signal_file(f)
        if df is None:
            continue
        if horizon and "horizon" in df.columns:
            # Filter rows whose ``horizon`` column does not match the request.
            hz_col = df["horizon"].astype(str)
            df = df[(hz_col == horizon) | (hz_col == "")]
            if df.empty:
                continue
        frames.append(df)
    if not frames:
        return pd.DataFrame(columns=list(REQUIRED_COLUMNS) + list(META_COLUMNS))
    out = pd.concat(frames, ignore_index=True)
    # Canonicalise sentiment_model: every "no-sentiment" spelling
    # (""/"nan"/"none"/"None"/…) → "-" so ECON ("none") and combined
    # ("-") sets share the same benchmark-lookup key (commit 12 Task C).
    out["sentiment_model"] = out["sentiment_model"].map(canonical_sentiment_model)
    # Model-family identity: blank → canonical defaults (defensive — the
    # per-file normaliser already applies these, but a concat of mixed-vintage
    # files is cheap to re-guard).
    out["model_type"] = out["model_type"].replace({"": "per_asset", "nan": "per_asset"})
    out["panel_mode"] = out["panel_mode"].replace({"": "-", "nan": "-"})
    # HPO identity: blank → fixed-C defaults.
    out["hpo_objective"] = out["hpo_objective"].replace({"": "-", "nan": "-"})
    out["hpo_variant"] = out["hpo_variant"].replace({"": "fixed", "nan": "fixed"})
    if "hpo_enabled" not in out.columns:
        out["hpo_enabled"] = False
    else:
        out["hpo_enabled"] = out["hpo_enabled"].map(_as_bool).fillna(False).astype(bool)
    return out


def attach_feature_set_metadata(signals: pd.DataFrame,
                                feature_sets: pd.DataFrame) -> pd.DataFrame:
    """Left-join ``category`` and ``label`` from the feature_sets config."""
    if signals.empty:
        for col in ("category", "label"):
            if col not in signals.columns:
                signals[col] = ""
        return signals
    meta_cols = [c for c in ("set_id", "category", "label") if c in feature_sets.columns]
    if "set_id" not in meta_cols:
        signals = signals.copy()
        signals["category"] = signals.get("category", "")
        signals["label"]    = signals.get("label", "")
        return signals
    meta = feature_sets[meta_cols].drop_duplicates(subset=["set_id"])
    out = signals.merge(meta, on="set_id", how="left")
    for col in ("category", "label"):
        if col in out.columns:
            out[col] = out[col].fillna("").astype(str)
        else:
            out[col] = ""
    return out
