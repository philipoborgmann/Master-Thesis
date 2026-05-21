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
from .metrics import GROUP_KEYS, _group_meta

REGIME_LABELS = ("low", "mid", "high")
_OHLCV_KEYS = {"open": ["open", "Open", "OPEN"],
               "high": ["high", "High", "HIGH"],
               "low":  ["low",  "Low",  "LOW"],
               "close":["close","Close","CLOSE"]}


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


def _normalise_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """Return a frame with ``date, open, high, low, close`` columns."""
    if df.index.name in ("timestamp", "date", "Date", "Timestamp"):
        df = df.reset_index()
    cols = {c.lower(): c for c in df.columns}
    out = pd.DataFrame()
    if "timestamp" in cols:
        out["timestamp"] = pd.to_datetime(df[cols["timestamp"]], utc=True, errors="coerce")
    elif "date" in cols:
        out["timestamp"] = pd.to_datetime(df[cols["date"]], utc=True, errors="coerce")
    else:
        # Fall back to the index when no time column is present.
        out["timestamp"] = pd.to_datetime(df.index, utc=True, errors="coerce")
    for name, candidates in _OHLCV_KEYS.items():
        for c in candidates:
            if c in df.columns:
                out[name] = pd.to_numeric(df[c], errors="coerce")
                break
        if name not in out.columns:
            return pd.DataFrame()
    out = out.dropna(subset=["timestamp", "open", "high", "low", "close"])
    out["date"] = out["timestamp"].dt.tz_convert("UTC").dt.date
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
    """
    ohlcv = load_daily_ohlcv(ticker)
    if ohlcv is None or ohlcv.empty:
        return None
    gk = garman_klass_variance(ohlcv)
    vol = rolling_volatility(gk)
    regimes = assign_tercile_regimes(vol)
    out = pd.DataFrame({
        "date":   ohlcv["date"].values,
        "gk_var": gk.values,
        "vol":    vol.values,
        "regime": regimes.values,
    })
    out["ticker"] = ticker.upper()
    return out


def build_regime_lookup(tickers: Iterable[str]) -> pd.DataFrame:
    """Concatenated ``(ticker, date) → regime`` lookup for every ticker."""
    frames = []
    for tk in sorted(set(t.upper() for t in tickers)):
        per = build_ticker_regime_lookup(tk)
        if per is not None:
            frames.append(per)
    if not frames:
        return pd.DataFrame(columns=["ticker", "date", "gk_var", "vol", "regime"])
    return pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------------------------
# Joining regimes back onto signals and aggregating
# ---------------------------------------------------------------------------

def _attach_date(signals: pd.DataFrame) -> pd.DataFrame:
    out = signals.copy()
    out["date"] = pd.to_datetime(out["timestamp"], utc=True, errors="coerce") \
                    .dt.tz_convert("UTC").dt.date
    out["ticker"] = out["ticker"].astype(str).str.upper()
    return out


def attach_regimes(signals: pd.DataFrame,
                   regime_lookup: pd.DataFrame) -> pd.DataFrame:
    """Left-merge ``regime`` onto signals via ``(ticker, date)``."""
    if signals.empty:
        return signals
    if regime_lookup is None or regime_lookup.empty:
        out = _attach_date(signals)
        out["regime"] = np.nan
        return out
    sig = _attach_date(signals)
    return sig.merge(regime_lookup[["ticker", "date", "regime"]],
                     on=["ticker", "date"], how="left")


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
    enriched = attach_regimes(signals, regime_lookup)
    if "regime" not in enriched.columns or enriched["regime"].isna().all():
        return pd.DataFrame()

    rows = []
    for keys, grp in enriched.groupby(list(GROUP_KEYS), dropna=False):
        horizon = keys[0] if isinstance(keys, tuple) else keys
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
                    "horizon": horizon, "set_id": keys[1], "sentiment_model": keys[2],
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
                    "horizon": horizon, "set_id": keys[1], "sentiment_model": keys[2],
                    "vol_regime": regime,
                    "accuracy":   float(sub["daily_accuracy"].mean()) if not sub.empty else np.nan,
                    "brier_score": float(sub["daily_brier"].mean()) if not sub.empty else np.nan,
                    "n_obs":  int(sub["n_intraday"].sum()) if not sub.empty else 0,
                    "n_days": int(len(sub)),
                    **meta,
                })
    out = pd.DataFrame(rows)
    front = ["horizon", "set_id", "category", "sentiment_model", "label", "vol_regime"]
    rest = [c for c in out.columns if c not in front]
    return out[front + rest].reset_index(drop=True)
