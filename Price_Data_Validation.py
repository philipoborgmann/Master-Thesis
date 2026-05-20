#!/usr/bin/env python3
# This wrapper is kept for backward compatibility. Preferred usage: python -m thesis_pipeline.cli validate-price
"""
Price_Data_Validation_v4_cmc_symbol_match_fixed.py
==============================
Research-grade validation for crypto OHLCV price data.

Checks
------
1) Cross-horizon consistency:
   - last 1h close of day t vs. 1d close of day t
   - last 6h close of day t vs. 1d close of day t
   - sum of 1h volume over day t vs. 1d volume of day t
   - sum of 6h volume over day t vs. 1d volume of day t

2) CoinMarketCap benchmark consistency:
   - 1d close vs. CMC close
   - 1d exchange base volume vs. CMC volume (diagnostic only)
   - 1d exchange quote/USD volume vs. CMC volume, using a direct quote-volume
     column when available and otherwise base volume × daily close.

Important CMC fix
-----------------
This version fixes timestamp unit inference and supports wide CMC parquet files with columns such as:
    date, 000000000001_BTC, 000000001027_ETH, ...
It reads the full metadata mapping, matches zero-padded ID + symbol columns,
handles duplicate CMC symbols by preferring canonical IDs before raw data
coverage, handles the Nano/NANO->XNO rename, and treats CMC volume as
USD/quote volume rather than base-asset units.

Expected folder layout
----------------------
Default:
    Data/Raw/Price/1h/{SYMBOL}USDT_1h.parquet
    Data/Raw/Price/6h/{SYMBOL}USDT_6h.parquet
    Data/Raw/Price/1d/{SYMBOL}USDT_1d.parquet
    Data/Raw/Price/CoinMarketCap/price.parquet
    Data/Raw/Price/CoinMarketCap/volume.parquet
    Data/Raw/Price/CoinMarketCap/MetaData.csv

Usage
-----
    python Price_Data_Validation_v4_cmc_symbol_match_fixed.py
    python Price_Data_Validation_v4_cmc_symbol_match_fixed.py --coin BTC
    python Price_Data_Validation_v4_cmc_symbol_match_fixed.py --price_dir "Data/Raw/Price"
    python Price_Data_Validation_v4_cmc_symbol_match_fixed.py --price_dir "Data/Raw/Price" --cmc_dir "Data/Raw/Price/CoinMarketCap"

Outputs
-------
    Data/Raw/Price/validation/cross_horizon_close.csv
    Data/Raw/Price/validation/cross_horizon_volume.csv
    Data/Raw/Price/validation/cmc_benchmark_close.csv
    Data/Raw/Price/validation/cmc_benchmark_volume.csv
    Data/Raw/Price/validation/data_quality.csv
    Data/Raw/Price/validation/validation_summary.csv
    Data/Raw/Price/validation/cmc_column_matches.csv
"""

from __future__ import annotations

import argparse
import glob
import os
import re
import sys
from collections import defaultdict
from typing import Any

import numpy as np
import pandas as pd


# ══════════════════════════════════════════════════════════════════════════════
# 1. CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

PRICE_DIR = os.path.join("Data", "Raw", "Price")
CMC_DIR = os.path.join(PRICE_DIR, "CoinMarketCap")
OUTPUT_DIR = os.path.join(PRICE_DIR, "validation")

HORIZONS = ["1h", "6h", "1d"]
TICKER_ALIASES = {
    "NANO": "XNO",   # exchange/CMC files may use XNO while thesis ticker is NANO
    "MIOTA": "IOTA", # exchange files may use IOTA while thesis ticker is MIOTA
}

# Canonical CoinMarketCap IDs for the main thesis universe.
# Reason: MetaData.csv can contain duplicate symbols, e.g. SOL/UNI. If the
# script simply picks the first symbol match, it may choose an unrelated or
# inactive token with the same ticker. These IDs are used only as a priority;
# the script still falls back to metadata/coverage-based matching.
PREFERRED_CMC_IDS: dict[str, list[str]] = {
    "BTC": ["1"],
    "LTC": ["2"],
    "DOGE": ["74"],
    "XRP": ["52"],
    "XLM": ["512"],
    "XMR": ["328"],
    "ETH": ["1027"],
    "NEO": ["1376"],
    "BAT": ["1697"],
    "IOTA": ["1720"],
    "MIOTA": ["1720"],
    "EOS": ["1765"],
    "BCH": ["1831"],
    "BNB": ["1839"],
    "LRC": ["1934"],
    "TRX": ["1958"],
    "MANA": ["1966"],
    "ADA": ["2010"],
    "XTZ": ["2011"],
    "KCS": ["2087"],
    "VET": ["3077"],
    "CRO": ["3635"],
    "SOL": ["5426"],
    "DOT": ["6636"],
    "UNI": ["7083"],
    "XNO": ["1567"],
    # Important: the real Nano asset was renamed from NANO to XNO.
    # In the wide CMC parquet there may be an old 000000001567_NANO column
    # ending before the thesis sample and a newer 000000001567_XNO column.
    # Therefore NANO is handled through the XNO alias rather than preferring
    # the stale *_NANO column.
    "NANO": ["24211"],
}

# Debug is deliberately off by default to keep full-batch runs readable.
DEBUG = False

# Cache for the CMC wide parquet data; loaded once and reused across tickers.
_CMC_CACHE: dict[str, Any] = {}


# ══════════════════════════════════════════════════════════════════════════════
# 2. SMALL HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _log_debug(msg: str) -> None:
    if DEBUG:
        print(msg)


def _normalise_symbol(x: Any) -> str:
    return str(x).strip().upper()


def _symbol_candidates(ticker: str) -> list[str]:
    """Ticker plus known aliases, preserving order and removing duplicates."""
    ticker = _normalise_symbol(ticker)
    candidates = [ticker]

    if ticker in TICKER_ALIASES:
        candidates.append(_normalise_symbol(TICKER_ALIASES[ticker]))

    # Reverse alias lookup, e.g. if ticker is XNO, also try NANO.
    for k, v in TICKER_ALIASES.items():
        if ticker == _normalise_symbol(v):
            candidates.append(_normalise_symbol(k))

    seen = set()
    out = []
    for c in candidates:
        if c and c not in seen:
            seen.add(c)
            out.append(c)
    return out


def _clean_id(x: Any) -> str | None:
    """Convert CMC ID-like values to a clean digit string."""
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return None
    s = str(x).strip()
    if not s or s.lower() in {"nan", "none", "null"}:
        return None

    # Handles values read as '1027.0'.
    if re.fullmatch(r"\d+\.0", s):
        s = s[:-2]

    m = re.search(r"\d+", s)
    return m.group(0) if m else None


def _pad_cmc_id(x: Any, width: int = 12) -> str | None:
    cid = _clean_id(x)
    return cid.zfill(width) if cid is not None else None


def _is_timestamp_column_name(col: Any) -> bool:
    c = str(col).strip().lower()
    return c in {
        "timestamp", "datetime", "date", "time", "timeopen", "time_open",
        "timeclose", "time_close", "open_time", "close_time"
    }


def _to_utc_datetime(values: Any) -> pd.Series:
    """
    Convert timestamp-like values to UTC datetimes robustly.

    Important fix for Binance/CCXT-style OHLCV data:
    numeric timestamps around 1.6e12 are milliseconds since epoch.
    Plain pd.to_datetime(...) interprets them as nanoseconds, which collapses
    an entire multi-year sample into 1970-01-01 and destroys daily grouping.

    Heuristic for numeric epoch values:
      ~1e18 -> ns, ~1e15 -> us, ~1e12 -> ms, ~1e9 -> s.
    String dates such as '2022-01-01' are parsed normally.
    """
    if isinstance(values, pd.Index):
        original_index = values
        series = pd.Series(values)
        out_index = None
    elif isinstance(values, pd.Series):
        series = values.copy()
        out_index = values.index
    else:
        series = pd.Series(values)
        out_index = None

    # Already datetime-like.
    if pd.api.types.is_datetime64_any_dtype(series):
        out = pd.to_datetime(series, utc=True, errors="coerce")
        return pd.Series(out, index=out_index, name="timestamp")

    numeric = pd.to_numeric(series, errors="coerce")
    numeric_share = numeric.notna().mean() if len(numeric) else 0.0

    # Treat mostly numeric columns as epoch timestamps, not as generic dates.
    if numeric_share >= 0.90:
        vals = numeric.dropna().abs()
        if len(vals) == 0:
            return pd.Series(pd.NaT, index=out_index, name="timestamp")

        med = float(vals.median())

        # YYYYMMDD as integer/string, e.g. 20220131.
        as_str = series.dropna().astype(str).str.replace(r"\.0$", "", regex=True).str.strip()
        yyyymmdd_share = as_str.str.fullmatch(r"\d{8}").mean() if len(as_str) else 0.0
        if 10_000_000 <= med <= 99_999_999 and yyyymmdd_share >= 0.90:
            out = pd.to_datetime(as_str.reindex(series.index), format="%Y%m%d", utc=True, errors="coerce")
            return pd.Series(out, index=out_index, name="timestamp")

        if med >= 1e17:
            unit = "ns"
        elif med >= 1e14:
            unit = "us"
        elif med >= 1e11:
            unit = "ms"
        elif med >= 1e8:
            unit = "s"
        else:
            # Small integers are unlikely to be real epoch timestamps.
            out = pd.to_datetime(series, utc=True, errors="coerce")
            return pd.Series(out, index=out_index, name="timestamp")

        out = pd.to_datetime(numeric, unit=unit, utc=True, errors="coerce")
        return pd.Series(out, index=out_index, name="timestamp")

    # Non-numeric date strings.
    out = pd.to_datetime(series, utc=True, errors="coerce")
    return pd.Series(out, index=out_index, name="timestamp")


def _asset_columns(df: pd.DataFrame) -> list[Any]:
    """Columns that are actual coin series, excluding date/time columns."""
    return [c for c in df.columns if not _is_timestamp_column_name(c)]


def _extract_timestamp_series(df: pd.DataFrame) -> pd.Series:
    """
    Robust timestamp extraction for CMC wide parquet files.

    Critical fix: Some CMC parquet files have index.name == 'unique_id' and a
    real 'date' column. The old script used the index whenever index.name was
    set, which converted 0,1,2,... into timestamps. This function prefers an
    explicit date/timestamp column and only uses the index if it is datetime-like.
    """
    # 1) Prefer explicit date/timestamp-like columns.
    for col in df.columns:
        if _is_timestamp_column_name(col):
            ts = _to_utc_datetime(df[col])
            ts.index = df.index
            if ts.notna().sum() > 0:
                return pd.Series(ts, index=df.index, name="timestamp")

    # 2) Use DatetimeIndex only if genuinely datetime-like.
    if isinstance(df.index, pd.DatetimeIndex):
        ts = _to_utc_datetime(df.index)
        ts.index = df.index
        return pd.Series(ts, index=df.index, name="timestamp")

    # 3) Try index conversion only if it yields plausible dates for most rows.
    idx_ts = _to_utc_datetime(df.index)
    idx_ts.index = df.index
    if pd.Series(idx_ts).notna().mean() > 0.8:
        return pd.Series(idx_ts, index=df.index, name="timestamp")

    # 4) Last resort: first column.
    first_col = df.columns[0]
    ts = _to_utc_datetime(df[first_col])
    ts.index = df.index
    if ts.notna().sum() > 0:
        return pd.Series(ts, index=df.index, name="timestamp")

    raise ValueError(
        "Could not identify a valid timestamp/date column in CMC dataframe. "
        f"Columns start with: {list(df.columns[:10])}; index name={df.index.name!r}"
    )


def _safe_pct_deviation(abs_dev: float, denominator: float) -> float:
    if pd.isna(denominator) or denominator == 0:
        return np.nan
    return abs_dev / abs(denominator) * 100


# ══════════════════════════════════════════════════════════════════════════════
# 3. OHLCV DATA LOADING
# ══════════════════════════════════════════════════════════════════════════════

def find_parquet(ticker: str, horizon: str) -> str | None:
    """Find parquet file for ticker + horizon, handling known aliases."""
    for sym in _symbol_candidates(ticker):
        path = os.path.join(PRICE_DIR, horizon, f"{sym}USDT_{horizon}.parquet")
        if os.path.isfile(path):
            return path
    return None


def load_ohlcv(ticker: str, horizon: str) -> pd.DataFrame | None:
    """
    Load one OHLCV parquet file. Normalises columns and ensures a UTC timestamp
    column named 'timestamp'.
    """
    path = find_parquet(ticker, horizon)
    if path is None:
        return None

    df = pd.read_parquet(path)
    df = df.copy()
    df.columns = [str(c).lower().strip() for c in df.columns]

    # Identify timestamp column.
    ts_col = None
    for candidate in ["timestamp", "datetime", "date", "time", "open_time", "close_time"]:
        if candidate in df.columns:
            ts_col = candidate
            break

    if ts_col is None and isinstance(df.index, pd.DatetimeIndex):
        df = df.reset_index()
        ts_col = df.columns[0]

    if ts_col is None:
        # Last resort: assume first column is timestamp.
        ts_col = df.columns[0]

    df["timestamp"] = _to_utc_datetime(df[ts_col])
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)

    if len(df) > 0 and df["timestamp"].dt.year.max() < 2009:
        print(
            f"  [WARN] {ticker}/{horizon}: parsed timestamps end before 2009. "
            f"Check timestamp unit. date_min={df['timestamp'].min()}, date_max={df['timestamp'].max()}"
        )

    required = ["open", "high", "low", "close", "volume"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        print(f"  [WARN] {ticker}/{horizon}: missing columns {missing}")
        return None

    for col in required:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def discover_tickers() -> list[str]:
    """Discover available tickers from the 1d directory."""
    daily_dir = os.path.join(PRICE_DIR, "1d")
    if not os.path.isdir(daily_dir):
        return []

    tickers = []
    for f in sorted(os.listdir(daily_dir)):
        m = re.match(r"([A-Za-z0-9]+)USDT_1d\.parquet$", f)
        if m:
            sym = _normalise_symbol(m.group(1))
            # Reverse-map aliases for thesis naming.
            for thesis_ticker, alias in TICKER_ALIASES.items():
                if sym == _normalise_symbol(alias):
                    sym = _normalise_symbol(thesis_ticker)
                    break
            tickers.append(sym)

    return sorted(set(tickers))


# ══════════════════════════════════════════════════════════════════════════════
# 4. CMC DATA LOADING — FIXED
# ══════════════════════════════════════════════════════════════════════════════

def load_cmc_data(ticker: str) -> pd.DataFrame | None:
    """
    Load CoinMarketCap benchmark data for a ticker.

    Supports:
      - wide parquet files: price.parquet and volume.parquet
      - per-coin CSV fallback

    Returns columns:
      timestamp, close, volume where available.
    """
    if not os.path.isdir(CMC_DIR):
        return None

    price_path = os.path.join(CMC_DIR, "price.parquet")
    volume_path = os.path.join(CMC_DIR, "volume.parquet")

    if os.path.isfile(price_path):
        return _load_cmc_wide_parquet(ticker, price_path, volume_path)

    return _load_cmc_csv(ticker)


def _get_cmc_id_map() -> dict[str, list[str]]:
    """
    Build symbol -> list[CMC IDs] from the full MetaData.csv.

    Fix vs. previous version:
      - no nrows=500 limit
      - keeps duplicate symbols as lists instead of overwriting them
    """
    for fname in ["MetaData.csv", "metadata.csv", "Metadata.csv"]:
        path = os.path.join(CMC_DIR, fname)
        if not os.path.isfile(path):
            continue

        try:
            header = pd.read_csv(path, nrows=0).columns.tolist()
            id_col = "id" if "id" in header else None
            if id_col is None:
                id_like = [c for c in header if str(c).lower().strip() == "id"]
                id_col = id_like[0] if id_like else None

            sym_col = "symbol" if "symbol" in header else None
            if sym_col is None:
                sym_like = [c for c in header if str(c).lower().strip() == "symbol"]
                sym_col = sym_like[0] if sym_like else None

            if id_col is None or sym_col is None:
                print(f"  [WARN] {fname}: could not find id/symbol columns. Header starts: {header[:10]}")
                continue

            meta = pd.read_csv(
                path,
                usecols=[id_col, sym_col],
                dtype={id_col: "string", sym_col: "string"},
                low_memory=False,
            )
            meta = meta.dropna(subset=[id_col, sym_col])

            mapping: dict[str, list[str]] = defaultdict(list)
            for _, row in meta.iterrows():
                sym = _normalise_symbol(row[sym_col])
                cid = _clean_id(row[id_col])
                if not sym or cid is None:
                    continue
                if cid not in mapping[sym]:
                    mapping[sym].append(cid)

            mapping = dict(mapping)
            _log_debug(
                f"  [DEBUG] Loaded CMC metadata from {fname}: "
                f"{sum(len(v) for v in mapping.values()):,} id-symbol rows, "
                f"{len(mapping):,} unique symbols"
            )
            return mapping

        except Exception as e:
            print(f"  [WARN] Failed to read {fname}: {e}")

    return {}


def _preferred_ids_for_symbol(sym: str, id_map: dict[str, list[str]]) -> list[str]:
    """Return canonical IDs first, then all metadata IDs not already included."""
    sym = _normalise_symbol(sym)
    ids: list[str] = []
    for cid in PREFERRED_CMC_IDS.get(sym, []):
        clean = _clean_id(cid)
        if clean and clean not in ids:
            ids.append(clean)
    for cid in id_map.get(sym, []):
        clean = _clean_id(cid)
        if clean and clean not in ids:
            ids.append(clean)
    return ids


def _build_cmc_column_candidates(df: pd.DataFrame, ticker: str, id_map: dict[str, list[str]]) -> list[tuple[Any, str, int]]:
    """
    Build possible CMC columns for a ticker.

    Returns tuples of (column, match_method, priority). Lower priority is better.
    The caller chooses among candidates using data coverage, which is crucial for
    duplicate symbols such as SOL and UNI.
    """
    columns = _asset_columns(df)
    col_by_str = {str(c): c for c in columns}
    col_by_upper = {str(c).upper(): c for c in columns}
    symbols = _symbol_candidates(ticker)

    candidates: list[tuple[Any, str, int]] = []

    def add_col(col: Any, method: str, priority: int) -> None:
        for existing, _, _ in candidates:
            if existing == col:
                return
        candidates.append((col, method, priority))

    # 1) Canonical/preferred ID + symbol candidates first.
    for sym in symbols:
        for cid in _preferred_ids_for_symbol(sym, id_map):
            padded = _pad_cmc_id(cid)
            exacts = []
            if padded:
                exacts.extend([f"{padded}_{sym}", f"{padded}_{sym.lower()}", padded])
            exacts.extend([f"{cid}_{sym}", f"{cid}_{sym.lower()}", str(cid)])
            for cand in exacts:
                if cand in col_by_str:
                    pri = 0 if cid in PREFERRED_CMC_IDS.get(sym, []) else 10
                    add_col(col_by_str[cand], f"exact_id_symbol:{cand}", pri)
                if cand.upper() in col_by_upper:
                    pri = 0 if cid in PREFERRED_CMC_IDS.get(sym, []) else 10
                    add_col(col_by_upper[cand.upper()], f"case_insensitive_id_symbol:{cand}", pri)

            # Regex ID + symbol allowing arbitrary leading zeros.
            pattern = re.compile(rf"^0*{re.escape(str(cid))}_{re.escape(sym)}$", re.IGNORECASE)
            for c in columns:
                if pattern.match(str(c)):
                    pri = 0 if cid in PREFERRED_CMC_IDS.get(sym, []) else 10
                    add_col(c, f"regex_id_symbol:{cid}_{sym}", pri)

    # 2) Direct symbol candidate.
    for sym in symbols:
        for cand in [sym, sym.lower()]:
            if cand in col_by_str:
                add_col(col_by_str[cand], f"exact_symbol:{cand}", 20)
            if cand.upper() in col_by_upper:
                add_col(col_by_upper[cand.upper()], f"case_insensitive_symbol:{cand}", 20)

    # 3) Suffix candidates, e.g. any column ending in _SOL. This catches cases
    # where metadata is stale or duplicated, but coverage scoring below prevents
    # choosing an inactive clone when the real asset is present.
    for sym in symbols:
        for c in columns:
            if str(c).upper().endswith(f"_{sym}"):
                # If the leading ID is canonical, priority remains good.
                leading = str(c).split("_", 1)[0]
                leading_id = _clean_id(leading)
                pri = 5 if leading_id in PREFERRED_CMC_IDS.get(sym, []) else 30
                add_col(c, f"suffix_match:_{sym}", pri)

    return candidates


def _score_cmc_candidate(df: pd.DataFrame, col: Any) -> tuple[int, int, int]:
    """
    Score a candidate column by actual usable data.

    Higher is better. The tuple contains:
      1) non-null, positive observations
      2) total non-null observations
      3) longest rough span proxy based on first/last valid index position
    """
    s = pd.to_numeric(df[col], errors="coerce")
    valid = s.notna()
    positive = valid & (s > 0)
    non_null_positive = int(positive.sum())
    non_null = int(valid.sum())
    if non_null > 0:
        idx = np.flatnonzero(valid.to_numpy())
        span = int(idx[-1] - idx[0]) if len(idx) else 0
    else:
        span = 0
    return non_null_positive, non_null, span


def _find_cmc_column(df: pd.DataFrame, ticker: str, id_map: dict[str, list[str]]) -> tuple[Any | None, str]:
    """
    Find the best CMC wide-parquet column for a ticker.

    Key fix vs. v2:
      - duplicate symbols are common in CMC metadata;
      - SOL and UNI can otherwise match a wrong/inactive token;
      - therefore all plausible candidates are scored by data coverage.
    """
    candidates = _build_cmc_column_candidates(df, ticker, id_map)
    if not candidates:
        return None, f"no_match; symbols={_symbol_candidates(ticker)}; ids={[id_map.get(s, []) for s in _symbol_candidates(ticker)]}"

    scored = []
    for col, method, priority in candidates:
        score = _score_cmc_candidate(df, col)
        # v4 fix: canonical CMC identity must dominate raw data coverage.
        # Some duplicate symbols have longer histories than the real asset
        # (e.g. UNI = UNICORN Token vs UNI = Uniswap). Sorting by coverage first
        # can select the wrong coin and create absurd validation errors.
        # We therefore sort by priority first, then by data coverage.
        scored.append((priority, -score[0], -score[1], -score[2], str(col), col, method, score))

    scored.sort()
    priority, neg_pos, neg_nonnull, neg_span, _, best_col, best_method, best_score = scored[0]

    _log_debug(
        f"  [DEBUG] {_normalise_symbol(ticker)} CMC candidates: "
        + "; ".join(
            f"{str(col)}[{method}, pri={pri}, pos={score[0]}, nn={score[1]}]"
            for pri, _, _, _, _, col, method, score in scored[:8]
        )
    )

    if best_score[1] == 0:
        return best_col, f"{best_method}; warning:no_non_null_values"
    return best_col, f"{best_method}; coverage_pos={best_score[0]}; coverage_non_null={best_score[1]}"

def _load_cmc_wide_parquet(ticker: str, price_path: str, volume_path: str) -> pd.DataFrame | None:
    """Load a single ticker from wide CMC price/volume parquet files."""
    if "price" not in _CMC_CACHE:
        print("  [INFO] Loading CMC price.parquet...")
        df_price = pd.read_parquet(price_path)
        _CMC_CACHE["price"] = df_price
        _CMC_CACHE["timestamp"] = _extract_timestamp_series(df_price)
        _CMC_CACHE["id_map"] = _get_cmc_id_map()
        _CMC_CACHE["matches"] = []

        if os.path.isfile(volume_path):
            print("  [INFO] Loading CMC volume.parquet...")
            _CMC_CACHE["volume"] = pd.read_parquet(volume_path)

        _log_debug(f"  [DEBUG] price.parquet shape: {df_price.shape}")
        _log_debug(f"  [DEBUG] first 15 columns: {list(df_price.columns[:15])}")
        _log_debug(f"  [DEBUG] timestamp head: {list(_CMC_CACHE['timestamp'].head(3))}")
        _log_debug(f"  [DEBUG] id_map symbols: {len(_CMC_CACHE['id_map'])}")

    df_price = _CMC_CACHE["price"]
    id_map = _CMC_CACHE.get("id_map", {})

    price_col, price_match_method = _find_cmc_column(df_price, ticker, id_map)
    if price_col is None:
        _log_debug(f"  [DEBUG] {ticker}: CMC price column not found ({price_match_method})")
        return None

    result = pd.DataFrame({
        "timestamp": _CMC_CACHE["timestamp"].values,
        "close": pd.to_numeric(df_price[price_col], errors="coerce").values,
    })

    volume_col = None
    volume_match_method = "not_available"
    if "volume" in _CMC_CACHE:
        df_vol = _CMC_CACHE["volume"]
        if price_col in df_vol.columns:
            volume_col = price_col
            volume_match_method = "same_as_price_col"
        else:
            volume_col, volume_match_method = _find_cmc_column(df_vol, ticker, id_map)

        if volume_col is not None:
            result["volume"] = pd.to_numeric(df_vol[volume_col], errors="coerce").values

    _CMC_CACHE["matches"].append({
        "ticker": _normalise_symbol(ticker),
        "price_col": str(price_col),
        "price_match_method": price_match_method,
        "volume_col": str(volume_col) if volume_col is not None else "",
        "volume_match_method": volume_match_method,
    })

    result["timestamp"] = _to_utc_datetime(result["timestamp"])
    result = result.dropna(subset=["timestamp", "close"])
    result = result.sort_values("timestamp").reset_index(drop=True)

    return result


def _load_cmc_csv(ticker: str) -> pd.DataFrame | None:
    """Fallback for per-coin CMC CSV files."""
    candidates = [
        f"{ticker}.csv", f"{ticker}_USD.csv", f"{ticker}_usd.csv", f"{ticker.lower()}.csv",
    ]

    csv_path = None
    for c in candidates:
        p = os.path.join(CMC_DIR, c)
        if os.path.isfile(p):
            csv_path = p
            break

    if csv_path is None:
        matches = glob.glob(os.path.join(CMC_DIR, f"*{ticker}*.*csv"))
        if matches:
            csv_path = matches[0]

    if csv_path is None:
        return None

    df = pd.read_csv(csv_path)
    df.columns = [str(c).lower().strip() for c in df.columns]

    ts_col = None
    for candidate in ["timestamp", "timeopen", "time_close", "timeclose", "date", "time"]:
        if candidate in df.columns:
            ts_col = candidate
            break
    if ts_col is None:
        ts_col = df.columns[0]

    df["timestamp"] = _to_utc_datetime(df[ts_col])
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)

    col_map = {}
    for orig in df.columns:
        if orig in {"close", "close**", "price", "price_usd", "close_usd"}:
            col_map[orig] = "close"
        elif orig in {"volume", "volume_24h", "volume_usd"}:
            col_map[orig] = "volume"
    df = df.rename(columns=col_map)

    if "close" in df.columns:
        df["close"] = pd.to_numeric(df["close"], errors="coerce")
    if "volume" in df.columns:
        df["volume"] = pd.to_numeric(df["volume"], errors="coerce")

    return df


# ══════════════════════════════════════════════════════════════════════════════
# 5. CROSS-HORIZON VALIDATION
# ══════════════════════════════════════════════════════════════════════════════

def validate_cross_horizon_close(ticker: str) -> list[dict[str, Any]]:
    """Compare last intraday close per day against daily close."""
    df_1d = load_ohlcv(ticker, "1d")
    if df_1d is None:
        return []

    df_1d = df_1d.copy()
    df_1d["date"] = df_1d["timestamp"].dt.date
    daily_close = df_1d.groupby("date", sort=True)["close"].last().rename("daily_close")

    records = []
    for intra_hz in ["1h", "6h"]:
        df_intra = load_ohlcv(ticker, intra_hz)
        if df_intra is None:
            continue

        df_intra = df_intra.copy()
        df_intra["date"] = df_intra["timestamp"].dt.date
        last_close_intra = df_intra.groupby("date", sort=True)["close"].last().rename("intra_close")

        merged = pd.concat([last_close_intra, daily_close], axis=1).dropna()
        for date, row in merged.iterrows():
            abs_dev = abs(row["intra_close"] - row["daily_close"])
            records.append({
                "ticker": _normalise_symbol(ticker),
                "date": str(date),
                "comparison": f"{intra_hz}_vs_1d",
                "metric": "close",
                "intra_value": row["intra_close"],
                "daily_value": row["daily_close"],
                "signed_deviation": row["intra_close"] - row["daily_close"],
                "abs_deviation": abs_dev,
                "pct_deviation": _safe_pct_deviation(abs_dev, row["daily_close"]),
            })

    return records


def validate_cross_horizon_volume(ticker: str) -> list[dict[str, Any]]:
    """Compare summed intraday volume per day against daily volume."""
    df_1d = load_ohlcv(ticker, "1d")
    if df_1d is None:
        return []

    df_1d = df_1d.copy()
    df_1d["date"] = df_1d["timestamp"].dt.date
    # A 1d bar's volume is already the daily volume. If duplicates exist, use last
    # observation after sorting rather than summing duplicates.
    daily_volume = df_1d.groupby("date", sort=True)["volume"].last().rename("daily_volume")

    records = []
    for intra_hz in ["1h", "6h"]:
        df_intra = load_ohlcv(ticker, intra_hz)
        if df_intra is None:
            continue

        df_intra = df_intra.copy()
        df_intra["date"] = df_intra["timestamp"].dt.date
        intra_volume = df_intra.groupby("date", sort=True)["volume"].sum().rename("intra_volume")

        merged = pd.concat([intra_volume, daily_volume], axis=1).dropna()
        for date, row in merged.iterrows():
            abs_dev = abs(row["intra_volume"] - row["daily_volume"])
            records.append({
                "ticker": _normalise_symbol(ticker),
                "date": str(date),
                "comparison": f"{intra_hz}_vs_1d",
                "metric": "volume",
                "intra_value": row["intra_volume"],
                "daily_value": row["daily_volume"],
                "signed_deviation": row["intra_volume"] - row["daily_volume"],
                "abs_deviation": abs_dev,
                "pct_deviation": _safe_pct_deviation(abs_dev, row["daily_volume"]),
            })

    return records


# ══════════════════════════════════════════════════════════════════════════════
# 6. CMC BENCHMARK VALIDATION
# ══════════════════════════════════════════════════════════════════════════════

def _find_quote_volume_column(df: pd.DataFrame) -> str | None:
    """Find a quote/USD notional volume column if the OHLCV parquet contains one."""
    candidates = [
        "quote_volume", "quote_asset_volume", "quoteassetvolume", "quotevolume",
        "volume_quote", "turnover", "quote_vol", "volume_usd", "usd_volume",
    ]
    cols = {str(c).lower().strip(): c for c in df.columns}
    for cand in candidates:
        if cand in cols:
            return cols[cand]
    return None


def validate_cmc_benchmark(ticker: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """
    Compare exchange 1d close/volume against CMC close/volume.

    Important volume note:
    - Binance/CCXT OHLCV 'volume' is normally base-asset volume, e.g. BTC units.
    - CMC volume is normally quote/USD 24h volume.
    - Therefore the raw base-volume comparison is kept only as a diagnostic.
    - The research-relevant CMC-volume comparison uses direct quote volume if
      the parquet has such a column; otherwise it uses base_volume × daily close
      as an approximate USD notional volume.
    """
    df_1d = load_ohlcv(ticker, "1d")
    df_cmc = load_cmc_data(ticker)

    if df_1d is None or df_cmc is None:
        return [], []

    df_1d = df_1d.copy()
    df_cmc = df_cmc.copy()
    df_1d["date"] = df_1d["timestamp"].dt.date
    df_cmc["date"] = df_cmc["timestamp"].dt.date

    close_records: list[dict[str, Any]] = []
    volume_records: list[dict[str, Any]] = []

    # Daily exchange aggregates.
    exchange_close = df_1d.groupby("date", sort=True)["close"].last().rename("exchange_close")
    exchange_base_vol = df_1d.groupby("date", sort=True)["volume"].last().rename("exchange_base_vol")

    quote_col = _find_quote_volume_column(df_1d)
    if quote_col is not None:
        exchange_quote_vol = (
            df_1d.groupby("date", sort=True)[quote_col].last().rename("exchange_quote_vol")
        )
        quote_volume_method = f"direct_column:{quote_col}"
    else:
        exchange_quote_vol = (exchange_base_vol * exchange_close).rename("exchange_quote_vol")
        quote_volume_method = "approx_base_volume_x_daily_close"

    if "close" in df_cmc.columns:
        cmc_close = df_cmc.groupby("date", sort=True)["close"].last().rename("cmc_close")
        merged = pd.concat([exchange_close, cmc_close], axis=1).dropna()

        for date, row in merged.iterrows():
            abs_dev = abs(row["exchange_close"] - row["cmc_close"])
            close_records.append({
                "ticker": _normalise_symbol(ticker),
                "date": str(date),
                "exchange_close": row["exchange_close"],
                "cmc_close": row["cmc_close"],
                "signed_deviation": row["exchange_close"] - row["cmc_close"],
                "abs_deviation": abs_dev,
                "pct_deviation": _safe_pct_deviation(abs_dev, row["cmc_close"]),
            })

    if "volume" in df_cmc.columns:
        cmc_vol = df_cmc.groupby("date", sort=True)["volume"].last().rename("cmc_vol")

        # A) Raw base-volume comparison: diagnostic only because units differ.
        merged_base = pd.concat([exchange_base_vol, cmc_vol], axis=1).dropna()
        for date, row in merged_base.iterrows():
            abs_dev = abs(row["exchange_base_vol"] - row["cmc_vol"])
            volume_records.append({
                "ticker": _normalise_symbol(ticker),
                "date": str(date),
                "comparison": "exchange_base_volume_vs_cmc_usd_volume",
                "volume_basis": "exchange_base_volume_raw_diagnostic",
                "volume_method": "raw_ohlcv_volume_base_asset_units",
                "exchange_vol": row["exchange_base_vol"],
                "cmc_vol": row["cmc_vol"],
                "signed_deviation": row["exchange_base_vol"] - row["cmc_vol"],
                "abs_deviation": abs_dev,
                "pct_deviation": _safe_pct_deviation(abs_dev, row["cmc_vol"]),
            })

        # B) Quote/USD volume comparison: the meaningful CMC volume comparison.
        merged_quote = pd.concat([exchange_quote_vol, cmc_vol], axis=1).dropna()
        for date, row in merged_quote.iterrows():
            abs_dev = abs(row["exchange_quote_vol"] - row["cmc_vol"])
            volume_records.append({
                "ticker": _normalise_symbol(ticker),
                "date": str(date),
                "comparison": "exchange_quote_volume_vs_cmc_usd_volume",
                "volume_basis": "exchange_quote_or_usd_volume",
                "volume_method": quote_volume_method,
                "exchange_vol": row["exchange_quote_vol"],
                "cmc_vol": row["cmc_vol"],
                "signed_deviation": row["exchange_quote_vol"] - row["cmc_vol"],
                "abs_deviation": abs_dev,
                "pct_deviation": _safe_pct_deviation(abs_dev, row["cmc_vol"]),
            })

    return close_records, volume_records


# ══════════════════════════════════════════════════════════════════════════════
# 7. DATA QUALITY CHECKS
# ══════════════════════════════════════════════════════════════════════════════

def check_data_quality(ticker: str) -> list[dict[str, Any]]:
    """Per-horizon quality report: gaps, duplicates, non-positive prices/volume."""
    records = []

    for hz in HORIZONS:
        df = load_ohlcv(ticker, hz)
        if df is None:
            records.append({
                "ticker": _normalise_symbol(ticker),
                "horizon": hz,
                "status": "file_missing",
            })
            continue

        n_rows = len(df)
        n_dupes = int(df["timestamp"].duplicated().sum())
        n_zero_close = int((df["close"] <= 0).sum())
        n_zero_volume = int((df["volume"] <= 0).sum())
        n_missing_close = int(df["close"].isna().sum())
        n_missing_volume = int(df["volume"].isna().sum())
        date_min = df["timestamp"].min()
        date_max = df["timestamp"].max()

        expected_freq = {
            "1h": pd.Timedelta(hours=1),
            "6h": pd.Timedelta(hours=6),
            "1d": pd.Timedelta(days=1),
        }.get(hz)

        n_gaps = 0
        max_gap_hours = 0.0
        if expected_freq is not None and n_rows > 1:
            diffs = df["timestamp"].diff().dropna()
            gaps = diffs[diffs > expected_freq * 1.5]
            n_gaps = int(len(gaps))
            if n_gaps > 0:
                max_gap_hours = float(gaps.max().total_seconds() / 3600)

        status = "ok" if (
            n_dupes == 0
            and n_zero_close == 0
            and n_missing_close == 0
            and n_gaps == 0
        ) else "issues"

        records.append({
            "ticker": _normalise_symbol(ticker),
            "horizon": hz,
            "n_bars": n_rows,
            "date_min": str(date_min),
            "date_max": str(date_max),
            "n_duplicates": n_dupes,
            "n_missing_close": n_missing_close,
            "n_missing_volume": n_missing_volume,
            "n_zero_or_negative_close": n_zero_close,
            "n_zero_or_negative_volume": n_zero_volume,
            "n_gaps": n_gaps,
            "max_gap_hours": round(max_gap_hours, 3),
            "status": status,
        })

    return records


# ══════════════════════════════════════════════════════════════════════════════
# 8. SUMMARY STATISTICS
# ══════════════════════════════════════════════════════════════════════════════

def compute_summary(records: list[dict[str, Any]], group_cols: list[str]) -> pd.DataFrame:
    """Compute raw and percentage deviation summaries per group."""
    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records)
    if "pct_deviation" not in df.columns or "abs_deviation" not in df.columns:
        return pd.DataFrame()

    summaries = []
    for name, grp in df.groupby(group_cols, dropna=False):
        if not isinstance(name, tuple):
            name = (name,)
        row = dict(zip(group_cols, name))

        abs_vals = grp["abs_deviation"].dropna()
        pct_vals = grp["pct_deviation"].dropna()

        row["n_days"] = int(len(pct_vals))
        row["mean_abs_deviation"] = float(abs_vals.mean()) if len(abs_vals) else np.nan
        row["median_abs_deviation"] = float(abs_vals.median()) if len(abs_vals) else np.nan
        row["max_abs_deviation"] = float(abs_vals.max()) if len(abs_vals) else np.nan
        row["mean_abs_pct_deviation"] = float(pct_vals.mean()) if len(pct_vals) else np.nan
        row["median_abs_pct_deviation"] = float(pct_vals.median()) if len(pct_vals) else np.nan
        row["max_abs_pct_deviation"] = float(pct_vals.max()) if len(pct_vals) else np.nan
        row["std_abs_pct_deviation"] = float(pct_vals.std()) if len(pct_vals) > 1 else np.nan

        if len(pct_vals) > 0:
            idx_max = grp["pct_deviation"].idxmax()
            row["max_dev_date"] = grp.loc[idx_max, "date"] if "date" in grp.columns else ""
        else:
            row["max_dev_date"] = ""

        # Correlation between the two compared value columns, where available.
        possible_pairs = [
            ("intra_value", "daily_value"),
            ("exchange_close", "cmc_close"),
            ("exchange_vol", "cmc_vol"),
        ]
        row["correlation"] = np.nan
        for a, b in possible_pairs:
            if a in grp.columns and b in grp.columns and grp[[a, b]].dropna().shape[0] > 1:
                row["correlation"] = float(grp[[a, b]].corr().iloc[0, 1])
                break

        summaries.append(row)

    return pd.DataFrame(summaries)


def _mean_pct(records: list[dict[str, Any]]) -> float:
    vals = [r["pct_deviation"] for r in records if pd.notna(r.get("pct_deviation"))]
    return float(np.mean(vals)) if vals else np.nan


# ══════════════════════════════════════════════════════════════════════════════
# 9. MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    global PRICE_DIR, CMC_DIR, OUTPUT_DIR, DEBUG

    parser = argparse.ArgumentParser(description="Validate price data across horizons and against CMC.")
    parser.add_argument("--price_dir", type=str, default=PRICE_DIR, help="Root folder containing 1h/6h/1d folders")
    parser.add_argument("--cmc_dir", type=str, default=None, help="Folder containing CMC price.parquet, volume.parquet, MetaData.csv")
    parser.add_argument("--output_dir", type=str, default=None, help="Folder for CSV validation outputs")
    parser.add_argument("--coin", type=str, default=None, help="Validate only this ticker, e.g. BTC")
    parser.add_argument("--debug", action="store_true", help="Print detailed CMC column-matching diagnostics")
    args = parser.parse_args()

    PRICE_DIR = args.price_dir
    CMC_DIR = args.cmc_dir if args.cmc_dir else os.path.join(PRICE_DIR, "CoinMarketCap")
    OUTPUT_DIR = args.output_dir if args.output_dir else os.path.join(PRICE_DIR, "validation")
    DEBUG = bool(args.debug)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    tickers = [_normalise_symbol(args.coin)] if args.coin else discover_tickers()
    if not tickers:
        print(f"[ERROR] No 1d price data found in: {os.path.join(PRICE_DIR, '1d')}")
        sys.exit(1)

    print(f"[INFO] PRICE_DIR: {os.path.abspath(PRICE_DIR)}")
    print(f"[INFO] CMC_DIR:   {os.path.abspath(CMC_DIR)}")
    print(f"[INFO] OUTPUT:    {os.path.abspath(OUTPUT_DIR)}")
    print(f"[INFO] Validating {len(tickers)} ticker(s): {', '.join(tickers)}")

    # Step 1: Cross-horizon close
    print("\n" + "=" * 70)
    print("STEP 1: Cross-horizon close validation")
    print("=" * 70)

    close_records: list[dict[str, Any]] = []
    for ticker in tickers:
        recs = validate_cross_horizon_close(ticker)
        close_records.extend(recs)
        if recs:
            print(f"  {ticker:8s} {len(recs):>6,} comparisons, mean abs pct dev: {_mean_pct(recs):.8f}%")
        else:
            print(f"  {ticker:8s} no cross-horizon close records")

    if close_records:
        df_close = pd.DataFrame(close_records)
        df_close.to_csv(os.path.join(OUTPUT_DIR, "cross_horizon_close.csv"), index=False)
        print(f"  Saved cross_horizon_close.csv ({len(df_close):,} rows)")

    # Step 2: Cross-horizon volume
    print("\n" + "=" * 70)
    print("STEP 2: Cross-horizon volume validation")
    print("=" * 70)

    volume_records: list[dict[str, Any]] = []
    for ticker in tickers:
        recs = validate_cross_horizon_volume(ticker)
        volume_records.extend(recs)
        if recs:
            print(f"  {ticker:8s} {len(recs):>6,} comparisons, mean abs pct dev: {_mean_pct(recs):.8f}%")
        else:
            print(f"  {ticker:8s} no cross-horizon volume records")

    if volume_records:
        df_volume = pd.DataFrame(volume_records)
        df_volume.to_csv(os.path.join(OUTPUT_DIR, "cross_horizon_volume.csv"), index=False)
        print(f"  Saved cross_horizon_volume.csv ({len(df_volume):,} rows)")

    # Step 3: CMC benchmark
    print("\n" + "=" * 70)
    print("STEP 3: CoinMarketCap benchmark validation")
    print("=" * 70)

    cmc_close_records: list[dict[str, Any]] = []
    cmc_volume_records: list[dict[str, Any]] = []

    if os.path.isdir(CMC_DIR):
        for ticker in tickers:
            c_recs, v_recs = validate_cmc_benchmark(ticker)
            cmc_close_records.extend(c_recs)
            cmc_volume_records.extend(v_recs)

            if c_recs:
                print(f"  {ticker:8s} CMC close:  {len(c_recs):>6,} days, mean abs pct dev: {_mean_pct(c_recs):.8f}%")
            else:
                print(f"  {ticker:8s} CMC close:  no records")
            if v_recs:
                v_df_print = pd.DataFrame(v_recs)
                for comp in [
                    "exchange_quote_volume_vs_cmc_usd_volume",
                    "exchange_base_volume_vs_cmc_usd_volume",
                ]:
                    sub = v_df_print[v_df_print["comparison"] == comp]
                    if not sub.empty:
                        label = "quote/USD" if "quote" in comp else "base/raw diag"
                        mean_pct = float(sub["pct_deviation"].dropna().mean())
                        print(f"  {ticker:8s} CMC volume {label:13s}: {len(sub):>6,} days, mean abs pct dev: {mean_pct:.8f}%")
            else:
                print(f"  {ticker:8s} CMC volume: no records")

        if cmc_close_records:
            pd.DataFrame(cmc_close_records).to_csv(os.path.join(OUTPUT_DIR, "cmc_benchmark_close.csv"), index=False)
            print(f"  Saved cmc_benchmark_close.csv ({len(cmc_close_records):,} rows)")
        if cmc_volume_records:
            pd.DataFrame(cmc_volume_records).to_csv(os.path.join(OUTPUT_DIR, "cmc_benchmark_volume.csv"), index=False)
            print(f"  Saved cmc_benchmark_volume.csv ({len(cmc_volume_records):,} rows)")
        if _CMC_CACHE.get("matches"):
            pd.DataFrame(_CMC_CACHE["matches"]).drop_duplicates().to_csv(
                os.path.join(OUTPUT_DIR, "cmc_column_matches.csv"), index=False
            )
            print("  Saved cmc_column_matches.csv")
    else:
        print(f"  [SKIP] CMC directory not found: {CMC_DIR}")

    # Step 4: Data quality
    print("\n" + "=" * 70)
    print("STEP 4: Data quality checks")
    print("=" * 70)

    quality_records: list[dict[str, Any]] = []
    for ticker in tickers:
        recs = check_data_quality(ticker)
        quality_records.extend(recs)
        for r in recs:
            status = "✓" if r.get("status") == "ok" else "⚠"
            print(
                f"  {status} {r['ticker']:8s}/{r['horizon']:3s} "
                f"bars={r.get('n_bars', '?'):>7} "
                f"gaps={r.get('n_gaps', 0):>3} "
                f"dupes={r.get('n_duplicates', 0):>3} "
                f"bad_close={r.get('n_zero_or_negative_close', 0):>3}"
            )

    if quality_records:
        pd.DataFrame(quality_records).to_csv(os.path.join(OUTPUT_DIR, "data_quality.csv"), index=False)
        print(f"  Saved data_quality.csv ({len(quality_records):,} rows)")

    # Step 5: Summary
    print("\n" + "=" * 70)
    print("STEP 5: Validation summary")
    print("=" * 70)

    summary_parts = []
    if close_records:
        s = compute_summary(close_records, ["ticker", "comparison"])
        s["validation_type"] = "cross_horizon_close"
        summary_parts.append(s)
    if volume_records:
        s = compute_summary(volume_records, ["ticker", "comparison"])
        s["validation_type"] = "cross_horizon_volume"
        summary_parts.append(s)
    if cmc_close_records:
        s = compute_summary(cmc_close_records, ["ticker"])
        s["validation_type"] = "cmc_close"
        summary_parts.append(s)
    if cmc_volume_records:
        s = compute_summary(cmc_volume_records, ["ticker", "comparison"])
        # Split the two CMC volume comparisons in the summary so the raw base-volume
        # diagnostic does not get averaged together with the meaningful quote/USD test.
        s["validation_type"] = s["comparison"].map({
            "exchange_base_volume_vs_cmc_usd_volume": "cmc_volume_base_raw_diagnostic",
            "exchange_quote_volume_vs_cmc_usd_volume": "cmc_volume_quote_or_usd",
        }).fillna("cmc_volume")
        summary_parts.append(s)

    if summary_parts:
        df_summary = pd.concat(summary_parts, ignore_index=True)
        df_summary.to_csv(os.path.join(OUTPUT_DIR, "validation_summary.csv"), index=False)
        print(f"  Saved validation_summary.csv ({len(df_summary):,} rows)")

        for vtype in df_summary["validation_type"].unique():
            sub = df_summary[df_summary["validation_type"] == vtype]
            print(f"\n  {vtype}:")
            print(f"    mean of group mean_abs_pct_deviation: {sub['mean_abs_pct_deviation'].mean():.8f}%")
            print(f"    worst group max_abs_pct_deviation:    {sub['max_abs_pct_deviation'].max():.8f}%")
    else:
        print("  No summary records generated.")

    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)
    print(f"Output directory: {os.path.abspath(OUTPUT_DIR)}")


if __name__ == "__main__":
    main()
