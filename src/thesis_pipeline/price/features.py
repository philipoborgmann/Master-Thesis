#!/usr/bin/env python3
"""
Create_Price_Features_v4.py
===========================

Creates ML-ready feature parquet files from OHLCV price data.

Version 4 — calendar-consistent windows and strict as-of market-cap merge,
designed to eliminate identified market-cap-related look-ahead leakage.

For each horizon (1h, 6h, 1d) and each coin, the script computes:
  - target: 0 if log return in t+1 is negative, otherwise 1
  - log_return_t: log return at t
  - cum_log_return_7d  / 14d / 21d: cumulative log return over the past
    7 / 14 / 21 CALENDAR days. The window length in bars is therefore
    ``days * BARS_PER_DAY[horizon]`` and identical wall-clock between
    horizons (1d → 7/14/21 bars; 6h → 28/56/84; 1h → 168/336/504).
  - realized_vol_14d: rolling std of log returns over 14 calendar days.
  - volume_diff: simple first difference in volume at t (horizon-native).
  - log_market_cap_lag1 = log(market_cap_at_t-1).
  - market_cap_source_date: the CMC source date D the value came from.
  - market_cap_available_at: assumed availability of the value
    (D + 1 day, 00:00 UTC). This is a documented conservative DEFAULT
    ASSUMPTION about CMC publication latency, not an empirically
    measured release time.

Market-cap merge is strictly as-of: only values with
``market_cap_available_at < prediction_timestamp`` enter the feature row
(``pd.merge_asof(..., direction='backward', allow_exact_matches=False)``).

Timestamp convention:
  ``timestamp`` is the **interval-start label**. A row labelled ``t`` refers
  to the completed interval ``[t, t+h)``; its market and Reddit information
  become usable at ``t+h``, and the row forecasts the sign of the return of
  the NEXT interval ``[t+h, t+2h)`` (``target``). No timestamp shift is applied
  — see :mod:`thesis_pipeline.diagnostics.timing_invariant`.

Outlier treatment:
  NONE at feature-construction time. Full-sample winsorisation was removed
  because clipping earlier observations at quantiles computed over the whole
  series uses future information (look-ahead leakage). All feature columns are
  therefore RAW point-in-time values:

    - log_return_t        — raw log return
    - volume_diff         — raw first difference of volume
    - cum_log_return_*d   — cumulative sum of RAW log returns
    - realized_vol_14d    — rolling std of RAW log returns

  Outlier control now happens INSIDE the model as leakage-safe,
  training-window winsorisation fitted only on the current training data —
  see :mod:`thesis_pipeline.modeling.preprocessing`.

Expected input structure:
  Data/Raw/Price/1h/{SYMBOL}USDT_1h.parquet
  Data/Raw/Price/6h/{SYMBOL}USDT_6h.parquet
  Data/Raw/Price/1d/{SYMBOL}USDT_1d.parquet
  Data/Raw/Price/CoinMarketCap/market_cap.parquet
  Data/Raw/Price/CoinMarketCap/MetaData.csv       optional but recommended

Output:
  Data/Features/price_features_1h.parquet
  Data/Features/price_features_6h.parquet
  Data/Features/price_features_1d.parquet
  Data/Features/feature_generation_report.csv
  Data/Features/cmc_marketcap_column_matches.csv

Usage:
  python -m thesis_pipeline.price.features
  python -m thesis_pipeline.price.features --coin BTC
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


# =============================================================================
# 1. DEFAULT CONFIGURATION
# =============================================================================

# Repository-root data layout. The price generator was previously
# anchored on ``Path(__file__).resolve().parent`` which put the defaults
# under ``src/thesis_pipeline/price/Data/`` and broke ``python -m
# thesis_pipeline.cli create-price-features``. Use the shared
# :func:`thesis_pipeline.config.resolve_path` helper instead so the
# generator reads from and writes to the canonical repo-root layout on
# every OS / install mode.
from ..config import resolve_path as _resolve_path
DEFAULT_PRICE_DIR = _resolve_path("raw_price_root")
DEFAULT_OUTPUT_DIR = Path(_resolve_path(
    "price_features_pattern", horizon="1d"
)).parent

HORIZONS = ["1h", "6h", "1d"]

# Bar-count per calendar day for each modelling horizon. Window features
# (cum_log_return_*d, realized_vol_14d) use ``days * BARS_PER_DAY[horizon]``
# so the wall-clock window length is identical across horizons.
BARS_PER_DAY: dict[str, int] = {"1d": 1, "6h": 4, "1h": 24}

# Calendar days for each rolling window. 21d is new for v4.
CUM_RETURN_WINDOW_DAYS: tuple[int, ...] = (7, 14, 21)
REALIZED_VOL_WINDOW_DAYS: int = 14

# Assumed availability lag for the daily CMC market-cap snapshot. The
# raw file carries a single calendar date D per row; we model the value
# as becoming usable for a prediction at D + MARKET_CAP_AVAILABILITY_LAG.
# This is a **conservative default assumption** about CMC publication
# latency, not an empirically measured release time.
MARKET_CAP_AVAILABILITY_LAG: pd.Timedelta = pd.Timedelta(days=1)

# File aliases: local exchange files sometimes use IOTA although the thesis
# ticker is MIOTA; Nano can appear as NANO or XNO depending on source/date.
FILE_TICKER_ALIASES: dict[str, list[str]] = {
    "MIOTA": ["MIOTA", "IOTA"],
    "IOTA": ["IOTA", "MIOTA"],
    "NANO": ["NANO", "XNO"],
    "XNO": ["XNO", "NANO"],
}

# Output naming aliases. These keep the thesis ticker naming stable.
OUTPUT_TICKER_ALIASES: dict[str, str] = {
    "IOTA": "MIOTA",
    "XNO": "NANO",
}

# Canonical CMC IDs for the coins in your sample. This prevents false matches
# for duplicated symbols such as UNI, SOL, BTC, NANO/XNO etc.
CANONICAL_CMC_IDS: dict[str, int] = {
    "BTC": 1,
    "LTC": 2,
    "XRP": 52,
    "DOGE": 74,
    "XMR": 328,
    "XLM": 512,
    "ETH": 1027,
    "NEO": 1376,
    "NANO": 1567,
    "XNO": 1567,
    "BAT": 1697,
    "MIOTA": 1720,
    "IOTA": 1720,
    "EOS": 1765,
    "BCH": 1831,
    "BNB": 1839,
    "LRC": 1934,
    "TRX": 1958,
    "MANA": 1966,
    "ADA": 2010,
    "XTZ": 2011,
    "KCS": 2087,
    "VET": 3077,
    "CRO": 3635,
    "SOL": 5426,
    "DOT": 6636,
    "UNI": 7083,
}

# Symbols accepted when matching CMC columns.
CMC_SYMBOL_ALIASES: dict[str, list[str]] = {
    "MIOTA": ["MIOTA", "IOTA"],
    "IOTA": ["IOTA", "MIOTA"],
    "NANO": ["NANO", "XNO"],
    "XNO": ["XNO", "NANO"],
}

# Preferred symbol when multiple CMC columns have the same canonical CMC ID.
# This is necessary because CMC keeps old and new ticker columns after rebrandings.
# The choice below is aligned with the available thesis sample period.
PREFERRED_CMC_SYMBOL_BY_TICKER: dict[str, str] = {
    "MIOTA": "MIOTA",   # 000000001720_MIOTA covers the 2021-12 to 2023-01 sample
    "IOTA": "MIOTA",
    "NANO": "XNO",     # 000000001567_XNO covers the 2022+ sample
    "XNO": "XNO",
}


# =============================================================================
# 2. GENERAL HELPERS
# =============================================================================

def normalise_output_ticker(ticker: str) -> str:
    ticker = str(ticker).upper().strip()
    return OUTPUT_TICKER_ALIASES.get(ticker, ticker)


def ticker_file_candidates(ticker: str) -> list[str]:
    ticker = str(ticker).upper().strip()
    candidates = FILE_TICKER_ALIASES.get(ticker, [ticker])
    out = []
    for c in candidates:
        c = c.upper().strip()
        if c not in out:
            out.append(c)
    return out


def cmc_symbol_candidates(ticker: str) -> list[str]:
    ticker = str(ticker).upper().strip()
    candidates = CMC_SYMBOL_ALIASES.get(ticker, [ticker])
    out = []
    for c in candidates:
        c = c.upper().strip()
        if c not in out:
            out.append(c)
    return out


def parse_cmc_column(col: object) -> tuple[Optional[int], Optional[str]]:
    """
    Parses CMC wide-format columns like '000000007083_UNI'.
    Returns (id_as_int, symbol) if possible.
    """
    s = str(col)
    m = re.match(r"^(\d+)_([^_]+)$", s)
    if not m:
        return None, s.upper() if s and s.lower() != "date" else None
    return int(m.group(1)), m.group(2).upper()


def infer_datetime_from_any(series: pd.Series) -> pd.Series:
    """
    Robust timestamp conversion.
    Handles:
      - unix seconds, milliseconds, microseconds, nanoseconds
      - date strings
      - pandas datetime columns
    Returns UTC-aware pandas timestamps.
    """
    s = series.copy()

    if pd.api.types.is_datetime64_any_dtype(s):
        return pd.to_datetime(s, utc=True, errors="coerce")

    if pd.api.types.is_numeric_dtype(s):
        x = pd.to_numeric(s, errors="coerce")
        med = x.dropna().abs().median()

        # Typical epoch scales:
        # seconds:      ~1e9
        # milliseconds: ~1e12
        # microseconds: ~1e15
        # nanoseconds:  ~1e18
        if pd.isna(med):
            return pd.to_datetime(s, utc=True, errors="coerce")
        if med > 1e17:
            unit = "ns"
        elif med > 1e14:
            unit = "us"
        elif med > 1e11:
            unit = "ms"
        else:
            unit = "s"
        return pd.to_datetime(x, unit=unit, utc=True, errors="coerce")

    return pd.to_datetime(s, utc=True, errors="coerce")


def detect_timestamp_column(df: pd.DataFrame) -> str:
    cols_lower = {str(c).lower(): c for c in df.columns}
    for candidate in ["timestamp", "datetime", "date", "time", "open_time", "close_time"]:
        if candidate in cols_lower:
            return cols_lower[candidate]

    # If the timestamp is in the index, reset it and use it.
    if df.index.name is not None:
        name = str(df.index.name).lower()
        if any(x in name for x in ["time", "date", "timestamp"]):
            df.reset_index(inplace=True)
            return df.columns[0]

    # Fallback: first column.
    return df.columns[0]


# NOTE: full-sample winsorisation was removed from feature construction (it
# used future observations to clip earlier ones). Outlier control now lives in
# the leakage-safe, training-window winsoriser
# (:mod:`thesis_pipeline.modeling.preprocessing`).


# =============================================================================
# 3. PRICE DATA LOADING
# =============================================================================

def find_parquet_file(price_dir: Path, ticker: str, horizon: str) -> Optional[Path]:
    """Finds {SYMBOL}USDT_{horizon}.parquet using ticker aliases."""
    hz_dir = price_dir / horizon
    if not hz_dir.is_dir():
        return None

    for sym in ticker_file_candidates(ticker):
        path = hz_dir / f"{sym}USDT_{horizon}.parquet"
        if path.is_file():
            return path

    return None


def load_ohlcv(price_dir: Path, ticker: str, horizon: str) -> Optional[pd.DataFrame]:
    """Loads and normalises one OHLCV parquet file."""
    path = find_parquet_file(price_dir, ticker, horizon)
    if path is None:
        return None

    df = pd.read_parquet(path)
    df.columns = [str(c).lower().strip() for c in df.columns]

    ts_col = detect_timestamp_column(df)
    df["timestamp"] = infer_datetime_from_any(df[ts_col])

    required = ["close", "volume"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{path} is missing required column(s): {missing}")

    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    df["volume"] = pd.to_numeric(df["volume"], errors="coerce")

    df = (
        df[["timestamp", "close", "volume"]]
        .dropna(subset=["timestamp"])
        .sort_values("timestamp")
        .drop_duplicates(subset=["timestamp"], keep="last")
        .reset_index(drop=True)
    )

    # Keep source path for auditability.
    df.attrs["source_file"] = str(path)
    return df


def discover_tickers(price_dir: Path) -> list[str]:
    """Discovers tickers from the daily directory."""
    daily_dir = price_dir / "1d"
    if not daily_dir.is_dir():
        return []

    tickers = []
    for path in sorted(daily_dir.glob("*USDT_1d.parquet")):
        m = re.match(r"(.+?)USDT_1d\.parquet$", path.name)
        if not m:
            continue
        ticker = normalise_output_ticker(m.group(1).upper())
        if ticker not in tickers:
            tickers.append(ticker)
    return tickers


# =============================================================================
# 4. MARKET CAP LOADING AND MATCHING
# =============================================================================

def find_cmc_file(cmc_dir: Path, names: list[str]) -> Optional[Path]:
    for name in names:
        path = cmc_dir / name
        if path.is_file():
            return path
    return None


def find_market_cap_path(cmc_dir: Path) -> Optional[Path]:
    return find_cmc_file(
        cmc_dir,
        [
            "market_cap.parquet",
            "marketcap.parquet",
            "MarketCap.parquet",
            "market_capitalization.parquet",
        ],
    )


def find_metadata_path(cmc_dir: Path) -> Optional[Path]:
    return find_cmc_file(cmc_dir, ["MetaData.csv", "metadata.csv", "Metadata.csv"])


def load_metadata(cmc_dir: Path) -> Optional[pd.DataFrame]:
    meta_path = find_metadata_path(cmc_dir)
    if meta_path is None:
        return None
    try:
        meta = pd.read_csv(meta_path)
        meta.columns = [str(c).strip() for c in meta.columns]
        return meta
    except Exception as exc:
        print(f"[WARN] Could not read metadata file {meta_path}: {exc}")
        return None


def build_metadata_candidates(meta: Optional[pd.DataFrame], ticker: str) -> set[int]:
    """Returns possible CMC IDs from MetaData.csv for a ticker/symbol."""
    ids: set[int] = set()
    if meta is None or "symbol" not in meta.columns:
        return ids

    id_col = "id" if "id" in meta.columns else ("ID" if "ID" in meta.columns else None)
    if id_col is None:
        return ids

    symbols = set(cmc_symbol_candidates(ticker))
    sub = meta[meta["symbol"].astype(str).str.upper().str.strip().isin(symbols)]
    for x in sub[id_col].dropna():
        try:
            ids.add(int(x))
        except Exception:
            pass
    return ids


def load_market_cap_data(cmc_dir: Path) -> tuple[pd.DataFrame, Optional[pd.DataFrame], Path]:
    """Loads the wide CMC market cap parquet and optional metadata."""
    market_cap_path = find_market_cap_path(cmc_dir)
    if market_cap_path is None:
        raise FileNotFoundError(
            f"Could not find market_cap.parquet in {cmc_dir}. "
            "Expected Data/Raw/Price/CoinMarketCap/market_cap.parquet"
        )

    print(f"[INFO] Loading market cap parquet: {market_cap_path}")
    market_cap = pd.read_parquet(market_cap_path)
    market_cap.columns = [str(c).strip() for c in market_cap.columns]

    if "date" not in [c.lower() for c in market_cap.columns]:
        # Fallback: if date is the first column, rename it.
        first_col = market_cap.columns[0]
        market_cap = market_cap.rename(columns={first_col: "date"})
    else:
        for c in market_cap.columns:
            if c.lower() == "date" and c != "date":
                market_cap = market_cap.rename(columns={c: "date"})
                break

    market_cap["date"] = pd.to_datetime(market_cap["date"], errors="coerce").dt.date
    market_cap = market_cap.dropna(subset=["date"]).drop_duplicates(subset=["date"], keep="last")

    meta = load_metadata(cmc_dir)
    return market_cap, meta, market_cap_path


def choose_market_cap_column(
    market_cap: pd.DataFrame,
    meta: Optional[pd.DataFrame],
    ticker: str,
) -> Optional[str]:
    """
    Chooses the correct wide CMC column for ticker.
    Priority:
      1. canonical CMC ID from CANONICAL_CMC_IDS
      2. metadata IDs for matching symbol aliases
      3. direct symbol/suffix match, highest non-null coverage
    """
    ticker = normalise_output_ticker(ticker)
    symbols = set(cmc_symbol_candidates(ticker))
    canonical_id = CANONICAL_CMC_IDS.get(ticker)
    metadata_ids = build_metadata_candidates(meta, ticker)

    preferred_symbol = PREFERRED_CMC_SYMBOL_BY_TICKER.get(ticker)

    candidates: list[tuple[int, str, int, Optional[str], int, int, int]] = []
    # tuple = (priority, column, cmc_id, symbol, coverage, last_non_null_date_ordinal, preferred_symbol_rank)
    #
    # Important: CMC can contain several columns for the same canonical ID after
    # rebrandings. Nano is the key example: 000000001567_NANO vs 000000001567_XNO.
    # MIOTA/IOTA is the opposite problem for this sample: 000000001720_IOTA starts
    # after the thesis price sample, while 000000001720_MIOTA overlaps. Therefore
    # we use a preferred-symbol rank before recency/coverage for known rebrandings.

    if "date" in market_cap.columns:
        mc_dates = pd.to_datetime(market_cap["date"], errors="coerce").dt.date
    else:
        mc_dates = pd.Series([pd.NaT] * len(market_cap))

    for col in market_cap.columns:
        if col == "date":
            continue
        cmc_id, sym = parse_cmc_column(col)
        s_notna = market_cap[col].notna()
        coverage = int(s_notna.sum())
        if coverage > 0:
            try:
                last_ord = max(d.toordinal() for d in mc_dates[s_notna] if pd.notna(d))
            except ValueError:
                last_ord = -1
        else:
            last_ord = -1

        preferred_rank = 0 if preferred_symbol is not None and sym == preferred_symbol else 1

        if canonical_id is not None and cmc_id == canonical_id:
            # Highest priority. This fixes duplicated symbols such as UNI/SOL and
            # rebrandings such as NANO/XNO and MIOTA/IOTA.
            candidates.append((0, col, cmc_id or -1, sym, coverage, last_ord, preferred_rank))
        elif cmc_id is not None and cmc_id in metadata_ids and sym in symbols:
            candidates.append((1, col, cmc_id, sym, coverage, last_ord, preferred_rank))
        elif sym in symbols:
            candidates.append((2, col, cmc_id or -1, sym, coverage, last_ord, preferred_rank))
        elif str(col).upper() in symbols:
            candidates.append((3, col, cmc_id or -1, sym, coverage, last_ord, preferred_rank))

    if not candidates:
        return None

    # Sort by priority first, then preferred symbol, then by most recent non-null
    # date, then coverage. Preferred symbol fixes MIOTA/IOTA and NANO/XNO in the
    # thesis sample while canonical ID still protects against duplicated symbols.
    candidates = sorted(candidates, key=lambda x: (x[0], x[6], -x[5], -x[4], x[1]))
    return candidates[0][1]


def build_market_cap_series(
    market_cap: pd.DataFrame,
    meta: Optional[pd.DataFrame],
    tickers: list[str],
) -> tuple[dict[str, pd.DataFrame], list[dict]]:
    """Builds one date -> market_cap series per ticker."""
    series_by_ticker: dict[str, pd.DataFrame] = {}
    match_records: list[dict] = []

    for ticker in tickers:
        col = choose_market_cap_column(market_cap, meta, ticker)
        if col is None:
            print(f"[WARN] No CMC market cap column found for {ticker}")
            match_records.append({
                "ticker": ticker,
                "market_cap_column": "",
                "n_non_null_market_cap": 0,
                "status": "not_found",
            })
            continue

        tmp = market_cap[["date", col]].rename(columns={col: "market_cap"}).copy()
        tmp["market_cap"] = pd.to_numeric(tmp["market_cap"], errors="coerce")
        # ── Defensive pre-log filtering ────────────────────────────
        # log() of non-positive / non-finite values yields NaN or -inf
        # and would silently leak into log_market_cap_lag1. Drop ≤0,
        # NaN and ±inf market-caps here so only finite, strictly positive
        # source values survive into the feature pipeline.
        valid_mask  = tmp["market_cap"].notna() & np.isfinite(tmp["market_cap"]) & (tmp["market_cap"] > 0)
        n_invalid_raw = int((~valid_mask).sum())
        tmp = tmp[valid_mask].sort_values("date").reset_index(drop=True)
        # Add documented availability columns:
        #   market_cap_source_date  – the calendar date D the snapshot belongs to.
        #   market_cap_available_at – D + 1 day, 00:00 UTC (conservative default
        #     assumption about CMC publication latency — see module docstring).
        tmp["market_cap_source_date"]  = pd.to_datetime(tmp["date"], utc=True)
        tmp["market_cap_available_at"] = (
            tmp["market_cap_source_date"] + MARKET_CAP_AVAILABILITY_LAG
        )
        # log_market_cap_lag1 — log transform once, here. After the
        # pre-filter above this can only produce finite values; the
        # ``replace`` is a belt-and-braces safety net.
        tmp["log_market_cap_lag1"] = (
            np.log(tmp["market_cap"]).replace([np.inf, -np.inf], np.nan)
        )
        n_invalid_after_log = int(tmp["log_market_cap_lag1"].isna().sum())
        if n_invalid_after_log:
            tmp = tmp.dropna(subset=["log_market_cap_lag1"]).reset_index(drop=True)
        series_by_ticker[ticker] = tmp

        cmc_id, sym = parse_cmc_column(col)
        match_records.append({
            "ticker": ticker,
            "market_cap_column": col,
            "parsed_cmc_id": cmc_id,
            "parsed_symbol": sym,
            "n_non_null_market_cap":           int(tmp["market_cap"].notna().sum()),
            "n_invalid_market_cap_rows_raw":   n_invalid_raw,
            "n_invalid_market_cap_after_log":  n_invalid_after_log,
            "first_market_cap_date": str(tmp["date"].min()) if len(tmp) else "",
            "last_market_cap_date":  str(tmp["date"].max()) if len(tmp) else "",
            "status": "ok",
        })

    return series_by_ticker, match_records


# =============================================================================
# 5. FEATURE ENGINEERING
# =============================================================================

def _normalize_utc_ns(series: pd.Series) -> pd.Series:
    """Coerce a datetime-like Series to ``datetime64[ns, UTC]``.

    Handles every input class that has surfaced in the production
    data flow: ``datetime64[{ms,us,ns}, UTC]``, naive datetimes
    (interpreted as UTC — the pipeline's documented convention),
    ISO date / datetime strings, and ``NaT``.

    The conversion uses :func:`pd.to_datetime(..., utc=True,
    errors='coerce')` and then explicitly downcasts to the
    nanosecond UTC dtype so callers can rely on a single canonical
    dtype across the merge-asof boundary.
    """
    parsed = pd.to_datetime(series, utc=True, errors="coerce")
    # ``.astype("datetime64[ns, UTC]")`` works across the pandas
    # versions the project supports (2.x + 3.x). On older versions
    # ``parsed`` may already be ``[ns, UTC]`` — astype is a no-op
    # in that case.
    return parsed.astype("datetime64[ns, UTC]")


def create_features_for_coin_horizon(
    price_dir: Path,
    ticker: str,
    horizon: str,
    market_cap_series: Optional[pd.DataFrame],
    winsor_p: float = 0.005,  # noqa: ARG001 — DEPRECATED / IGNORED (see below).
    marketcap_lag_days: int = 0,  # noqa: ARG001 — accepted for CLI back-compat; ignored.
    ohlcv_override: Optional[pd.DataFrame] = None,
) -> tuple[pd.DataFrame, dict, list[dict]]:
    """Creates features for one ticker-horizon pair.

    All feature columns are RAW point-in-time values — feature construction no
    longer winsorises (full-sample clipping leaked future data). The
    ``winsor_p`` argument is accepted for call-site back-compat but IGNORED;
    outlier control is handled downstream by the leakage-safe training-window
    winsoriser (:mod:`thesis_pipeline.modeling.preprocessing`).

    Windows are calendar-consistent across horizons: cum_log_return_7d /
    14d / 21d and realized_vol_14d use ``days * BARS_PER_DAY[horizon]``
    bars and ``min_periods = window_bars``.

    Market-cap merge is strictly as-of with documented availability:
    a CMC value with source date D is modelled as available at
    D + MARKET_CAP_AVAILABILITY_LAG (default = D + 1 day, 00:00 UTC) and
    only enters a row whose ``timestamp`` is **strictly** after that
    instant (``allow_exact_matches=False``). The legacy
    ``--marketcap_lag_days`` flag is ignored.

    ``ohlcv_override`` is a unit-test hook: pass a fully-formed OHLCV
    frame and the function skips the parquet read. It must NOT be used
    in production code paths.
    """
    if horizon not in BARS_PER_DAY:
        raise ValueError(
            f"Unknown horizon {horizon!r}; expected one of {list(BARS_PER_DAY)}"
        )

    if ohlcv_override is not None:
        df = ohlcv_override.copy()
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    else:
        df = load_ohlcv(price_dir, ticker, horizon)
    if df is None or df.empty:
        report = {
            "ticker": ticker,
            "horizon": horizon,
            "status": "price_file_missing_or_empty",
            "n_input_rows": 0,
            "n_output_rows": 0,
        }
        return pd.DataFrame(), report, []

    source_file = df.attrs.get("source_file", "")

    df = df.copy()
    df["ticker"] = ticker
    df["horizon"] = horizon
    df["date"] = df["timestamp"].dt.date

    # RAW point-in-time transformations. Full-sample winsorisation was removed
    # (it leaked future observations into earlier rows). Outlier control now
    # happens inside the model via the leakage-safe training-window winsoriser.
    df["log_return_raw"] = np.log(df["close"] / df["close"].shift(1))
    df["log_return_raw"] = df["log_return_raw"].replace([np.inf, -np.inf], np.nan)
    df["volume_diff_raw"] = df["volume"].diff().replace([np.inf, -np.inf], np.nan)

    # ``log_return_t`` / ``volume_diff`` are the RAW values verbatim (no clip).
    df["log_return_t"] = df["log_return_raw"]
    df["volume_diff"] = df["volume_diff_raw"]

    # Calendar-consistent rolling windows, computed from the RAW log returns.
    # ``bars`` is the count of horizon bars that fits the requested calendar
    # days exactly (1d→1, 6h→4, 1h→24 bars per day). ``min_periods = bars`` so
    # we never emit a partial-window value, and the windows are identical
    # wall-clock between horizons.
    bpd = BARS_PER_DAY[horizon]
    for n_days in CUM_RETURN_WINDOW_DAYS:
        bars = n_days * bpd
        df[f"cum_log_return_{n_days}d"] = (
            df["log_return_t"].rolling(window=bars, min_periods=bars).sum()
        )

    # Realized volatility: rolling std of RAW log returns over the
    # 14-calendar-day window. Captures the second moment of returns,
    # complementing the first-moment level features (volatility clustering;
    # Sung et al., 2022; Tang et al., 2024).
    rv_bars = REALIZED_VOL_WINDOW_DAYS * bpd
    df["realized_vol_14d"] = (
        df["log_return_t"].rolling(window=rv_bars, min_periods=rv_bars).std()
    )

    # Target: next-period log return. No winsorization needed for sign.
    next_log_return = df["log_return_raw"].shift(-1)
    df["target"] = np.where(next_log_return < 0, 0, 1).astype("float")
    df.loc[next_log_return.isna(), "target"] = np.nan

    # ── Market-cap merge: strict as-of with documented availability ──
    # No same-day or future market cap may enter a row.
    #
    # Production raw OHLCV parquet files and the CMC dump land on disk
    # with DIFFERENT datetime resolutions (e.g. ``datetime64[ms, UTC]``
    # for OHLCV vs ``datetime64[us, UTC]`` for the reshaped CMC
    # series). ``pd.merge_asof`` refuses to join across mismatched
    # resolutions with ``MergeError: incompatible merge keys [0]
    # datetime64[ms, UTC] and datetime64[us, UTC]``. Normalise BOTH
    # join keys to ``datetime64[ns, UTC]`` immediately before the
    # merge, then assert they actually share a dtype — a defensive
    # guard so a future pandas upgrade cannot regress silently.
    if market_cap_series is not None and not market_cap_series.empty:
        mc = market_cap_series[
            ["market_cap_source_date", "market_cap_available_at",
             "market_cap", "log_market_cap_lag1"]
        ].copy()
        df["timestamp"] = _normalize_utc_ns(df["timestamp"])
        mc["market_cap_available_at"] = _normalize_utc_ns(
            mc["market_cap_available_at"]
        )
        if df["timestamp"].dtype != mc["market_cap_available_at"].dtype:
            raise TypeError(
                "Market-cap merge_asof keys are not the same dtype after "
                f"normalisation: left 'timestamp'={df['timestamp'].dtype} "
                "vs right 'market_cap_available_at'="
                f"{mc['market_cap_available_at'].dtype}"
            )
        # ``pd.merge_asof`` refuses null keys on either side — drop NaT
        # availability rows BEFORE the merge so a tampered CMC dump
        # never crashes the pipeline. Unmatched price rows fall through
        # as NaN, which is the documented behaviour.
        mc = mc[mc["market_cap_available_at"].notna()].copy()
        # Sort AFTER normalisation so the merge sees a canonical, sorted
        # column on both sides.
        df = df.sort_values("timestamp").reset_index(drop=True)
        mc = mc.sort_values("market_cap_available_at").reset_index(drop=True)
        df = pd.merge_asof(
            df,
            mc,
            left_on="timestamp",
            right_on="market_cap_available_at",
            direction="backward",
            allow_exact_matches=False,
        )
    else:
        df = df.sort_values("timestamp").reset_index(drop=True)
        df["market_cap"]              = np.nan
        df["log_market_cap_lag1"]     = np.nan
        df["market_cap_source_date"]  = pd.NaT
        df["market_cap_available_at"] = pd.NaT

    feature_cols = [
        "timestamp",
        "date",
        "ticker",
        "horizon",
        "target",
        "log_return_t",
        "cum_log_return_7d",
        "cum_log_return_14d",
        "cum_log_return_21d",
        "realized_vol_14d",
        "volume_diff",
        "log_market_cap_lag1",
        "market_cap_source_date",
        "market_cap_available_at",
    ]

    before_drop = len(df)
    out = df[feature_cols].copy()
    out = out.dropna(subset=[
        "target",
        "log_return_t",
        "cum_log_return_7d",
        "cum_log_return_14d",
        "cum_log_return_21d",
        "realized_vol_14d",
        "volume_diff",
        "log_market_cap_lag1",
    ]).reset_index(drop=True)
    out["target"] = out["target"].astype("int8")

    report = {
        "ticker": ticker,
        "horizon": horizon,
        "status": "ok" if len(out) > 0 else "empty_after_feature_na_drop",
        "source_file": source_file,
        "n_input_rows": int(before_drop),
        "n_output_rows": int(len(out)),
        "n_dropped_rows": int(before_drop - len(out)),
        "first_timestamp": str(out["timestamp"].min()) if len(out) else "",
        "last_timestamp": str(out["timestamp"].max()) if len(out) else "",
        "share_positive_target": float(out["target"].mean()) if len(out) else np.nan,
        "n_missing_market_cap_before_drop":
            int(df["log_market_cap_lag1"].isna().sum()),
    }

    # Feature construction no longer winsorises (leakage-free raw features).
    # The third tuple element is retained as an empty list for call-site
    # backward compatibility; no winsorization_thresholds.csv is produced.
    thresholds: list[dict] = []

    return out, report, thresholds


# =============================================================================
# 6. MAIN
# =============================================================================

def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Create log-return and volume-difference features from OHLCV parquet data.")
    parser.add_argument("--price_dir", type=str, default=str(DEFAULT_PRICE_DIR), help="Path to Data/Raw/Price")
    parser.add_argument("--output_dir", type=str, default=str(DEFAULT_OUTPUT_DIR), help="Path to Data/Features")
    parser.add_argument("--coin", type=str, default=None, help="Optional: process only one ticker, e.g. BTC")
    parser.add_argument("--horizons", nargs="+", default=HORIZONS, choices=HORIZONS, help="Horizons to process")
    parser.add_argument("--winsor_p", type=float, default=0.005, help=(
        "DEPRECATED / IGNORED. Feature construction no longer winsorises "
        "(full-sample clipping leaked future data). Outlier control is now "
        "leakage-safe training-window winsorisation inside the model."))
    parser.add_argument(
        "--marketcap_lag_days",
        type=int,
        default=0,
        help=(
            "DEPRECATED — accepted for back-compat only and IGNORED. "
            "Market-cap merge is now strictly as-of with a documented "
            "availability lag (MARKET_CAP_AVAILABILITY_LAG, default = "
            "D + 1 day, 00:00 UTC); see module docstring."
        ),
    )
    args = parser.parse_args(argv)
    if args.marketcap_lag_days != 0:
        print(
            "[WARN] --marketcap_lag_days is deprecated and IGNORED. "
            "The merge is now strictly as-of (allow_exact_matches=False) "
            "with a documented availability lag of "
            f"{MARKET_CAP_AVAILABILITY_LAG}."
        )

    price_dir = Path(args.price_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    cmc_dir = price_dir / "CoinMarketCap"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] PRICE_DIR: {price_dir}")
    print(f"[INFO] CMC_DIR:   {cmc_dir}")
    print(f"[INFO] OUTPUT:    {output_dir}")
    print("[INFO] Winsorization: DISABLED at feature time — raw point-in-time "
          "features; leakage-safe training-window winsorisation runs in-model.")

    tickers = [normalise_output_ticker(args.coin)] if args.coin else discover_tickers(price_dir)
    if not tickers:
        print("[ERROR] No tickers found. Check Data/Raw/Price/1d.")
        sys.exit(1)

    print(f"[INFO] Processing {len(tickers)} ticker(s): {', '.join(tickers)}")

    # Load market cap once and build per-ticker series.
    market_cap, meta, _ = load_market_cap_data(cmc_dir)
    market_cap_by_ticker, mc_match_records = build_market_cap_series(market_cap, meta, tickers)

    pd.DataFrame(mc_match_records).to_csv(output_dir / "cmc_marketcap_column_matches.csv", index=False)
    print(f"[INFO] Saved market cap column matches: {output_dir / 'cmc_marketcap_column_matches.csv'}")

    all_reports: list[dict] = []
    all_thresholds: list[dict] = []
    # Audit accumulators (effective market-cap lag + bar-grid regularity).
    lag_audit_rows: list[pd.DataFrame] = []
    bar_audit_rows: list[pd.DataFrame] = []

    for horizon in args.horizons:
        print("\n" + "=" * 70)
        print(f"Creating features for horizon: {horizon}")
        print("=" * 70)

        horizon_parts: list[pd.DataFrame] = []

        for ticker in tickers:
            mc_series = market_cap_by_ticker.get(ticker)
            features, report, thresholds = create_features_for_coin_horizon(
                price_dir=price_dir,
                ticker=ticker,
                horizon=horizon,
                market_cap_series=mc_series,
                winsor_p=args.winsor_p,
                marketcap_lag_days=args.marketcap_lag_days,
            )

            all_reports.append(report)
            all_thresholds.extend(thresholds)

            if not features.empty:
                horizon_parts.append(features)

            print(
                f"  {ticker:6s} rows_in={report.get('n_input_rows', 0):>6} "
                f"rows_out={report.get('n_output_rows', 0):>6} "
                f"status={report.get('status')}"
            )

        if horizon_parts:
            df_horizon = pd.concat(horizon_parts, ignore_index=True)
            df_horizon = df_horizon.sort_values(["timestamp", "ticker"]).reset_index(drop=True)

            # Section F / G — timing audits per horizon (best-effort; the
            # main feature pipeline is never broken by an audit failure).
            try:
                from ..diagnostics.timing_audit import (
                    market_cap_lag_audit, bar_grid_audit,
                )
                if {"market_cap_source_date", "market_cap_available_at"}.issubset(
                        df_horizon.columns):
                    lag_audit_rows.append(market_cap_lag_audit(df_horizon))
                bar_audit_rows.append(bar_grid_audit(df_horizon))
            except Exception as exc:  # noqa: BLE001
                print(f"[WARN] timing audit skipped for {horizon}: {exc}")

            # Canonical v4 filename: ``price_features_{horizon}.parquet``
            # (matches ``configs/paths.yaml :: price_features_pattern``
            # and the downstream merge stage). Legacy installations
            # produced ``features_{horizon}.parquet``; the v4 merge
            # never reads that name.
            output_path = output_dir / f"price_features_{horizon}.parquet"
            # Generator-boundary schema validation. Refuses to write a
            # parquet that lacks any required ECON column.
            from ..diagnostics.feature_schema import validate_price_feature_schema
            validate_price_feature_schema(df_horizon, horizon=horizon,
                                            source=output_path)
            df_horizon.to_parquet(output_path, index=False)
            print(f"\n[INFO] Saved {output_path} ({len(df_horizon):,} rows, {df_horizon.shape[1]} columns)")
        else:
            print(f"[WARN] No features created for horizon {horizon}")

    # Audit files.
    report_df = pd.DataFrame(all_reports)
    # ── Section F: market-cap effective-lag summary appended to the report ─
    if lag_audit_rows:
        from ..diagnostics.timing_audit import market_cap_lag_summary
        lag_full = pd.concat(lag_audit_rows, ignore_index=True)
        per_h = (lag_full.groupby("horizon", as_index=False)
                 .apply(lambda g: pd.Series(market_cap_lag_summary(g)))
                 .reset_index(drop=True))
        per_h.to_csv(output_dir / "market_cap_lag_summary.csv", index=False)
        # Append global summary columns to the per-(ticker,horizon) report.
        global_summary = market_cap_lag_summary(lag_full)
        for k, v in global_summary.items():
            report_df[k] = v
        print(f"[INFO] market_cap_lag_summary: {global_summary}")
    # ── Section G: bar-grid regularity report ─────────────────────────────
    if bar_audit_rows:
        bar_full = pd.concat(bar_audit_rows, ignore_index=True)
        bar_full.to_csv(output_dir / "bar_grid_audit.csv", index=False)
        n_irregular = int((bar_full["n_missing_expected_bars"] > 0).sum())
        if n_irregular:
            print(f"[WARN] bar-grid: {n_irregular} (ticker, horizon) entries "
                  f"have missing bars — see bar_grid_audit.csv.")
    report_path = output_dir / "feature_generation_report.csv"
    report_df.to_csv(report_path, index=False)

    # Full-sample winsorisation was removed — no winsorization_thresholds.csv
    # is written (a leftover file would misleadingly imply the old procedure is
    # still active). Remove a stale copy from a previous run if present.
    stale_thresholds = output_dir / "winsorization_thresholds.csv"
    if stale_thresholds.exists():
        try:
            stale_thresholds.unlink()
            print(f"[INFO] Removed stale {stale_thresholds.name} "
                  "(feature construction no longer winsorises).")
        except OSError as exc:  # noqa: BLE001 — best-effort cleanup
            print(f"[WARN] Could not remove stale {stale_thresholds.name}: {exc}")

    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)
    print(f"[INFO] Feature files saved in: {output_dir}")
    print(f"[INFO] Report saved:          {report_path}")
    print("[INFO] Outlier control: none at feature time — leakage-safe "
          "training-window winsorisation happens inside the model.")


if __name__ == "__main__":
    main()
