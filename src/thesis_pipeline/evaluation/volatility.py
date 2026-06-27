"""Volatility-regime stratification.

Pipeline:

1. Load daily OHLCV per ticker from ``Data/Raw/Price/1d/<TICKER>USDT_1d.parquet``
   (with the NANO ↔ XNO alias).
2. Compute Garman-Klass daily variance:

       σ²_GK = 0.5 · ln(H/L)²  −  (2 · ln(2) − 1) · ln(C/O)²

3. Smooth with a 14-day rolling mean, then ``.shift(1)`` so no observation
   uses its own day's variance (lookahead guard).
4. Split each ticker's history into terciles → ``low / mid / high`` regimes.
5. Look up the regime for each signal observation; for intraday horizons,
   first aggregate to daily accuracy / brier per (date, ticker), then assign
   the regime at the day level.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Mapping

import numpy as np
import pandas as pd

from ..config import resolve_path
from ..logging_utils import get_logger
from .metrics import GROUP_KEYS, FRONT_META, _front_order, _group_meta, ensure_group_columns

REGIME_LABELS = ("low", "mid", "high")

# Column-name candidates that map onto the canonical OHLC fields. Pandas
# parquet writers do not normalise these so we have to handle the common
# variants explicitly (lowercase, Title, UPPER, plus a few Binance-style
# aliases such as "Open Price").
_OHLCV_KEYS = {
    "open":  ["open",  "Open",  "OPEN",  "open_price",  "Open Price"],
    "high":  ["high",  "High",  "HIGH",  "high_price",  "High Price"],
    "low":   ["low",   "Low",   "LOW",   "low_price",   "Low Price"],
    "close": ["close", "Close", "CLOSE", "close_price", "Close Price"],
}

# Timestamp-column candidates, in priority order. The first column found wins.
_TS_CANDIDATES = (
    "timestamp", "Timestamp", "TIMESTAMP",
    "open_time", "Open time", "Open Time", "OpenTime",
    "close_time", "Close time", "Close Time", "CloseTime",
    "date", "Date", "DATE",
    "datetime", "Datetime", "DATETIME", "DateTime",
    "time", "Time",
)


# ---------------------------------------------------------------------------
# Price loading helpers
# ---------------------------------------------------------------------------

def _ticker_candidates(ticker: str) -> list[str]:
    """Return the file-name candidates for a ticker, honouring NANO ↔ XNO."""
    base = ticker.upper()
    if base == "NANO":
        return ["NANO", "XNO"]
    if base == "XNO":
        return ["XNO", "NANO"]
    return [base]


def _resolve_ohlcv_path(ticker: str) -> Path | None:
    root = resolve_path("raw_price_1d")
    for cand in _ticker_candidates(ticker):
        p = root / f"{cand}USDT_1d.parquet"
        if p.exists():
            return p
    return None


def _coerce_to_utc_datetime(series: pd.Series) -> pd.Series:
    """Best-effort coercion of any timestamp-like series to tz-aware UTC."""
    if pd.api.types.is_integer_dtype(series) or pd.api.types.is_float_dtype(series):
        # Detect epoch unit by magnitude.
        try:
            mx = float(series.dropna().abs().max())
        except (TypeError, ValueError):
            mx = 0.0
        if mx > 1e17:
            unit = "ns"
        elif mx > 1e14:
            unit = "us"
        elif mx > 1e11:
            unit = "ms"
        else:
            unit = "s"
        return pd.to_datetime(series, unit=unit, utc=True, errors="coerce")
    return pd.to_datetime(series, utc=True, errors="coerce")


def _pick_timestamp_column(df: pd.DataFrame) -> pd.Series | None:
    """Return a UTC-aware timestamp series sourced from the first matching column."""
    for candidate in _TS_CANDIDATES:
        if candidate in df.columns:
            return _coerce_to_utc_datetime(df[candidate])
    # Case-insensitive fallback for any column whose lowered name matches.
    lowered = {str(c).lower(): c for c in df.columns}
    for candidate in _TS_CANDIDATES:
        key = candidate.lower()
        if key in lowered:
            return _coerce_to_utc_datetime(df[lowered[key]])
    return None


def _normalise_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """Return a frame with ``date, open, high, low, close`` columns.

    ``date`` is a ``pd.Timestamp`` normalised to UTC midnight so that it has
    a single ``datetime64[ns, UTC]`` dtype and can be merged against signal
    timestamps without dtype mismatches.
    """
    df = df.copy()
    if df.index.name and str(df.index.name).lower() in {
        c.lower() for c in _TS_CANDIDATES
    }:
        df = df.reset_index()

    ts = _pick_timestamp_column(df)
    if ts is None:
        # Last-ditch: use the index, regardless of name.
        try:
            ts = _coerce_to_utc_datetime(pd.Series(df.index))
        except Exception:  # noqa: BLE001
            return pd.DataFrame()

    out = pd.DataFrame({"timestamp": ts.values})
    for name, candidates in _OHLCV_KEYS.items():
        chosen = None
        for c in candidates:
            if c in df.columns:
                chosen = df[c]
                break
        if chosen is None:
            # Case-insensitive secondary pass.
            lowered = {str(c).lower(): c for c in df.columns}
            for c in candidates:
                if c.lower() in lowered:
                    chosen = df[lowered[c.lower()]]
                    break
        if chosen is None:
            return pd.DataFrame()
        out[name] = pd.to_numeric(chosen, errors="coerce").values

    out = out.dropna(subset=["timestamp", "open", "high", "low", "close"])
    if out.empty:
        return out
    # Normalise to UTC midnight so the merge key dtype is stable.
    out["date"] = pd.to_datetime(out["timestamp"], utc=True).dt.normalize()
    out = out.drop_duplicates(subset=["date"]).sort_values("date").reset_index(drop=True)
    return out[["date", "open", "high", "low", "close"]]


def load_daily_ohlcv(ticker: str) -> pd.DataFrame | None:
    """Return cleaned daily OHLCV for a ticker, or ``None`` if unavailable."""
    path = _resolve_ohlcv_path(ticker)
    if path is None:
        return None
    try:
        df = pd.read_parquet(path)
    except Exception as exc:  # noqa: BLE001
        get_logger().warning("evaluate-signals: cannot read %s (%s)", path, exc)
        return None
    out = _normalise_ohlcv(df)
    return out if not out.empty else None


# ---------------------------------------------------------------------------
# Garman-Klass variance + regime assignment
# ---------------------------------------------------------------------------

def garman_klass_variance(df: pd.DataFrame) -> pd.Series:
    """Vectorised σ²_GK series indexed by ``date``."""
    if df.empty:
        return pd.Series(dtype=float)
    o = df["open"].astype(float).values
    h = df["high"].astype(float).values
    l = df["low"].astype(float).values
    c = df["close"].astype(float).values
    with np.errstate(divide="ignore", invalid="ignore"):
        term1 = 0.5 * np.log(h / l) ** 2
        term2 = (2.0 * np.log(2.0) - 1.0) * np.log(c / o) ** 2
        gk = term1 - term2
    gk = np.where(np.isfinite(gk), gk, np.nan)
    return pd.Series(gk, index=df["date"].values, name="gk_var")


def rolling_volatility(gk_var: pd.Series, window: int = 14) -> pd.Series:
    """14-day rolling mean of σ²_GK, shifted by 1 to avoid lookahead."""
    if gk_var.empty:
        return gk_var
    return gk_var.rolling(window=window, min_periods=max(2, window // 2)).mean().shift(1)


def assign_tercile_regimes(vol: pd.Series,
                           labels: Iterable[str] = REGIME_LABELS) -> pd.Series:
    """Split a positive volatility series into terciles (low/mid/high).

    Ties are broken by ``rank(method="first")`` so the result is deterministic
    even if multiple values collide on a tercile cut.
    """
    if vol.empty or vol.notna().sum() == 0:
        return pd.Series(index=vol.index, dtype=object)
    ranked = vol.rank(method="first")
    try:
        regimes = pd.qcut(ranked, q=3, labels=list(labels))
    except ValueError:
        # Not enough distinct values to form three buckets.
        regimes = pd.Series(index=vol.index, dtype=object)
    return regimes.astype(object).where(vol.notna(), other=np.nan)


def build_ticker_regime_lookup(ticker: str) -> pd.DataFrame | None:
    """Return a ``date → regime`` table for a single ticker.

    Returns ``None`` when the OHLCV file is missing or empty.

    Each row also carries availability metadata:

    * ``regime_source_date`` — the underlying source-data date used to
      compute the regime on this row. Because :func:`rolling_volatility`
      applies a ``.shift(1)`` lookahead guard, the regime on lookup
      ``date = D`` is computed from data on day ``D − 1``.
    * ``regime_available_at`` — the UTC instant at which the regime
      becomes available for use by downstream code. Equal to
      ``regime_source_date + 1 day`` (the next 00:00 UTC after the source
      bar closes). With the current shifted lookup that is exactly
      ``date`` itself.
    """
    ohlcv = load_daily_ohlcv(ticker)
    if ohlcv is None or ohlcv.empty:
        return None
    gk = garman_klass_variance(ohlcv)
    vol = rolling_volatility(gk)
    regimes = assign_tercile_regimes(vol)
    date = pd.to_datetime(ohlcv["date"], utc=True).dt.normalize()
    # ``rolling_volatility`` already shifts by 1 so vol[D] uses data up to D-1.
    source_date = date - pd.Timedelta(days=1)
    available_at = source_date + pd.Timedelta(days=1)
    out = pd.DataFrame({
        "date":                 date.values,
        "regime_source_date":   source_date.values,
        "regime_available_at":  available_at.values,
        "gk_var":               gk.values,
        "vol":                  vol.values,
        "regime":               regimes.values,
    })
    out["ticker"] = ticker.upper()
    return out


def build_regime_lookup(tickers: Iterable[str]) -> pd.DataFrame:
    """Concatenated ``(ticker, date) → regime`` lookup for every ticker.

    The output also carries the availability columns
    ``regime_source_date`` and ``regime_available_at`` so callers can
    use ``pd.merge_asof`` for strict-availability joins instead of a
    fixed calendar-day shift.
    """
    frames = []
    for tk in sorted(set(t.upper() for t in tickers)):
        per = build_ticker_regime_lookup(tk)
        if per is not None:
            frames.append(per)
    columns = ["ticker", "date", "regime_source_date", "regime_available_at",
               "gk_var", "vol", "regime"]
    if not frames:
        return pd.DataFrame(columns=columns)
    return pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------------------------
# Joining regimes back onto signals and aggregating
# ---------------------------------------------------------------------------

def _attach_date(signals: pd.DataFrame) -> pd.DataFrame:
    out = signals.copy()
    out["date"] = pd.to_datetime(out["timestamp"], utc=True, errors="coerce").dt.normalize()
    out["ticker"] = out["ticker"].astype(str).str.upper()
    return out


def attach_regimes(signals: pd.DataFrame,
                   regime_lookup: pd.DataFrame) -> pd.DataFrame:
    """Availability-based per-ticker as-of join (commit 3 Section E).

    Delegates to the shared :func:`thesis_pipeline.evaluation
    .regime_join.attach_regime_asof` helper so every production regime
    consumer uses the same strict-< availability semantics. The
    diagnostics columns (source date, availability instant, effective
    lag) are prefixed with ``vol`` so an enriched frame can also carry
    market-cap diagnostics under the ``mcap`` prefix without collision.

    The returned frame keeps the legacy ``regime`` column name — callers
    that historically read ``regime`` keep working — while exposing the
    same data via ``vol_regime`` for the supplementary McNemar path.
    """
    from .regime_join import attach_regime_asof

    if signals.empty:
        return signals
    sig = _attach_date(signals)
    if regime_lookup is None or regime_lookup.empty:
        out = sig.copy()
        out["regime"] = np.nan
        out["vol_regime"] = np.nan
        out["vol_regime_source_date"]  = pd.NaT
        out["vol_regime_available_at"] = pd.NaT
        out["vol_regime_lag_days"]     = np.nan
        return out

    lk = regime_lookup.copy()
    if "regime" in lk.columns and "vol_regime" not in lk.columns:
        lk = lk.rename(columns={"regime": "vol_regime"})
    elif "vol_regime" not in lk.columns:
        raise ValueError("attach_regimes: lookup must carry either "
                         "'regime' or 'vol_regime'")
    out = attach_regime_asof(
        sig, lk, regime_col="vol_regime", column_prefix="vol",
    )
    # Legacy alias — downstream code that reads ``regime`` keeps working.
    out["regime"] = out["vol_regime"]
    matched = int(out["regime"].notna().sum())
    if matched == 0:
        get_logger().warning(
            "evaluate-signals: regime as-of join produced 0 matches "
            "(signals=%d rows, lookup=%d rows). Common causes: "
            "ticker mismatch, dtype mismatch, or every signal "
            "predates the first regime availability instant.",
            len(sig), len(regime_lookup),
        )
    return out


def _daily_aggregate(group: pd.DataFrame) -> pd.DataFrame:
    """Reduce intraday rows to one (date, ticker) row with daily metrics."""
    g = group.copy()
    g["correct"] = (g["prediction"].astype(int) == g["target"].astype(int)).astype(int)
    g["sq_err"]  = (g["probability"].astype(float) - g["target"].astype(float)) ** 2
    daily = g.groupby(["ticker", "date", "regime"], dropna=False).agg(
        daily_accuracy=("correct", "mean"),
        daily_brier=("sq_err",  "mean"),
        n_intraday=("correct", "size"),
    ).reset_index()
    return daily


def volatility_stratification_table(signals: pd.DataFrame,
                                    regime_lookup: pd.DataFrame) -> pd.DataFrame:
    """One row per (horizon × set_id × sentiment_model × regime)."""
    if signals.empty:
        return pd.DataFrame()
    enriched = ensure_group_columns(attach_regimes(signals, regime_lookup))
    if "regime" not in enriched.columns or enriched["regime"].isna().all():
        return pd.DataFrame()

    rows = []
    for keys, grp in enriched.groupby(list(GROUP_KEYS), dropna=False):
        horizon = keys[0]
        ident = dict(zip(GROUP_KEYS, keys))
        meta = _group_meta(grp)
        if str(horizon) == "1d":
            # Observation-level: each signal row already represents one day.
            grp = grp.assign(
                correct=lambda d: (d["prediction"].astype(int) == d["target"].astype(int)).astype(int),
                sq_err=lambda d: (d["probability"].astype(float) - d["target"].astype(float)) ** 2,
            )
            for regime in REGIME_LABELS:
                sub = grp[grp["regime"] == regime]
                rows.append({
                    **ident,
                    "vol_regime": regime,
                    "accuracy":   float(sub["correct"].mean()) if not sub.empty else np.nan,
                    "brier_score": float(sub["sq_err"].mean()) if not sub.empty else np.nan,
                    "n_obs":  int(len(sub)),
                    "n_days": int(sub.groupby(["ticker", "date"]).ngroups) if not sub.empty else 0,
                    **meta,
                })
        else:
            daily = _daily_aggregate(grp)
            for regime in REGIME_LABELS:
                sub = daily[daily["regime"] == regime]
                rows.append({
                    **ident,
                    "vol_regime": regime,
                    "accuracy":   float(sub["daily_accuracy"].mean()) if not sub.empty else np.nan,
                    "brier_score": float(sub["daily_brier"].mean()) if not sub.empty else np.nan,
                    "n_obs":  int(sub["n_intraday"].sum()) if not sub.empty else 0,
                    "n_days": int(len(sub)),
                    **meta,
                })
    out = pd.DataFrame(rows)
    return _front_order(out, FRONT_META + ["vol_regime"])
