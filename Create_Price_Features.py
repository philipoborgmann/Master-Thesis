#!/usr/bin/env python3
"""
Create_Price_Features_v3.py
===========================

Creates ML-ready feature parquet files from OHLCV price data.

Version 3: fixes CMC market-cap matching for rebranded/renamed coins.
- NANO prefers XNO because it has overlap with the thesis sample.
- MIOTA prefers MIOTA because the IOTA column starts after the thesis sample.

For each horizon (1h, 6h, 1d) and each coin, the script computes:
  - target: 0 if log return in t+1 is negative, otherwise 1
  - log_return_t: log return at t
  - cum_log_return_7: cumulative log return over the last 7 observations
  - cum_log_return_14: cumulative log return over the last 14 observations
  - realized_vol_14: rolling std of log returns over 14 observations
  - volume_diff: simple first difference in volume at t
  - market_cap_t: CoinMarketCap market cap matched by calendar date

Outlier treatment:
  - log_return_t and volume_diff are winsorized per coin and horizon
  - lower tail is clipped at the 0.5% quantile
  - upper tail is clipped at the 99.5% quantile
  - cumulative log returns are computed from the winsorized log returns

Expected input structure:
  Data/Raw/Price/1h/{SYMBOL}USDT_1h.parquet
  Data/Raw/Price/6h/{SYMBOL}USDT_6h.parquet
  Data/Raw/Price/1d/{SYMBOL}USDT_1d.parquet
  Data/Raw/Price/CoinMarketCap/market_cap.parquet
  Data/Raw/Price/CoinMarketCap/MetaData.csv       optional but recommended

Output:
  Data/Features/features_1h.parquet
  Data/Features/features_6h.parquet
  Data/Features/features_1d.parquet
  Data/Features/feature_generation_report.csv
  Data/Features/winsorization_thresholds.csv
  Data/Features/cmc_marketcap_column_matches.csv

Usage:
  python Create_Price_Features.py
  python Create_Price_Features.py --coin BTC
  python Create_Price_Features.py --price_dir "Data/Raw/Price" --output_dir "Data/Features"
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

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_PRICE_DIR = BASE_DIR / "Data" / "Raw" / "Price"
DEFAULT_OUTPUT_DIR = BASE_DIR / "Data" / "Features"

HORIZONS = ["1h", "6h", "1d"]

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


def winsorize_series(s: pd.Series, p: float = 0.005) -> tuple[pd.Series, float, float]:
    """
    Clips a series at the p and 1-p quantiles.
    Returns the winsorized series and the two thresholds.
    """
    s = pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf], np.nan)
    valid = s.dropna()
    if valid.empty:
        return s, np.nan, np.nan

    lower = valid.quantile(p)
    upper = valid.quantile(1.0 - p)
    return s.clip(lower=lower, upper=upper), float(lower), float(upper)


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

        tmp = market_cap[["date", col]].rename(columns={col: "market_cap_t"}).copy()
        tmp["market_cap_t"] = pd.to_numeric(tmp["market_cap_t"], errors="coerce")
        tmp = tmp.dropna(subset=["market_cap_t"]).sort_values("date").reset_index(drop=True)
        series_by_ticker[ticker] = tmp

        cmc_id, sym = parse_cmc_column(col)
        match_records.append({
            "ticker": ticker,
            "market_cap_column": col,
            "parsed_cmc_id": cmc_id,
            "parsed_symbol": sym,
            "n_non_null_market_cap": int(tmp["market_cap_t"].notna().sum()),
            "first_market_cap_date": str(tmp["date"].min()) if len(tmp) else "",
            "last_market_cap_date": str(tmp["date"].max()) if len(tmp) else "",
            "status": "ok",
        })

    return series_by_ticker, match_records


# =============================================================================
# 5. FEATURE ENGINEERING
# =============================================================================

def create_features_for_coin_horizon(
    price_dir: Path,
    ticker: str,
    horizon: str,
    market_cap_series: Optional[pd.DataFrame],
    winsor_p: float,
    marketcap_lag_days: int = 0,
) -> tuple[pd.DataFrame, dict, list[dict]]:
    """Creates features for one ticker-horizon pair."""
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

    # Raw transformations.
    df["log_return_raw"] = np.log(df["close"] / df["close"].shift(1))
    df["log_return_raw"] = df["log_return_raw"].replace([np.inf, -np.inf], np.nan)
    df["volume_diff_raw"] = df["volume"].diff().replace([np.inf, -np.inf], np.nan)

    # Winsorize per coin and horizon.
    df["log_return_t"], lr_low, lr_high = winsorize_series(df["log_return_raw"], winsor_p)
    df["volume_diff"], vd_low, vd_high = winsorize_series(df["volume_diff_raw"], winsor_p)

    # Rolling cumulative log returns are based on winsorized log returns.
    df["cum_log_return_7"] = df["log_return_t"].rolling(window=7, min_periods=7).sum()
    df["cum_log_return_14"] = df["log_return_t"].rolling(window=14, min_periods=14).sum()

    # Realized volatility: rolling standard deviation of winsorized log returns.
    # Captures the second moment (dispersion) of returns, complementing the
    # first-moment features (level and direction). Crypto returns exhibit strong
    # volatility clustering; this feature allows the classifier to condition on
    # the current volatility regime (Sung et al., 2022; Tang et al., 2024).
    df["realized_vol_14"] = df["log_return_t"].rolling(window=14, min_periods=14).std()

    # Target: next-period log return. No winsorization needed for sign.
    next_log_return = df["log_return_raw"].shift(-1)
    df["target"] = np.where(next_log_return < 0, 0, 1).astype("float")
    df.loc[next_log_return.isna(), "target"] = np.nan

    # Market cap merge by calendar date. Optional lag can avoid same-day leakage.
    if market_cap_series is not None and not market_cap_series.empty:
        mc = market_cap_series.copy()
        if marketcap_lag_days != 0:
            mc["date"] = pd.to_datetime(mc["date"]) + pd.to_timedelta(marketcap_lag_days, unit="D")
            mc["date"] = mc["date"].dt.date
        df = df.merge(mc, on="date", how="left")
    else:
        df["market_cap_t"] = np.nan

    feature_cols = [
        "timestamp",
        "date",
        "ticker",
        "horizon",
        "target",
        "log_return_t",
        "cum_log_return_7",
        "cum_log_return_14",
        "realized_vol_14",
        "volume_diff",
        "market_cap_t",
    ]

    before_drop = len(df)
    out = df[feature_cols].copy()
    out = out.dropna(subset=[
        "target",
        "log_return_t",
        "cum_log_return_7",
        "cum_log_return_14",
        "realized_vol_14",
        "volume_diff",
        "market_cap_t",
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
        "n_missing_market_cap_before_drop": int(df["market_cap_t"].isna().sum()),
    }

    thresholds = [
        {
            "ticker": ticker,
            "horizon": horizon,
            "variable": "log_return_t",
            "winsor_lower_quantile": winsor_p,
            "winsor_upper_quantile": 1.0 - winsor_p,
            "lower_threshold": lr_low,
            "upper_threshold": lr_high,
        },
        {
            "ticker": ticker,
            "horizon": horizon,
            "variable": "volume_diff",
            "winsor_lower_quantile": winsor_p,
            "winsor_upper_quantile": 1.0 - winsor_p,
            "lower_threshold": vd_low,
            "upper_threshold": vd_high,
        },
    ]

    return out, report, thresholds


# =============================================================================
# 6. MAIN
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="Create log-return and volume-difference features from OHLCV parquet data.")
    parser.add_argument("--price_dir", type=str, default=str(DEFAULT_PRICE_DIR), help="Path to Data/Raw/Price")
    parser.add_argument("--output_dir", type=str, default=str(DEFAULT_OUTPUT_DIR), help="Path to Data/Features")
    parser.add_argument("--coin", type=str, default=None, help="Optional: process only one ticker, e.g. BTC")
    parser.add_argument("--horizons", nargs="+", default=HORIZONS, choices=HORIZONS, help="Horizons to process")
    parser.add_argument("--winsor_p", type=float, default=0.005, help="Tail probability for winsorization; default 0.005 = 0.5%%")
    parser.add_argument(
        "--marketcap_lag_days",
        type=int,
        default=0,
        help=(
            "Optional date shift applied to market cap before merging. "
            "Default 0 means same calendar date. Use 1 if you want yesterday's market cap as today's feature."
        ),
    )
    args = parser.parse_args()

    price_dir = Path(args.price_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    cmc_dir = price_dir / "CoinMarketCap"
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.winsor_p <= 0 or args.winsor_p >= 0.5:
        raise ValueError("--winsor_p must be between 0 and 0.5")

    print(f"[INFO] PRICE_DIR: {price_dir}")
    print(f"[INFO] CMC_DIR:   {cmc_dir}")
    print(f"[INFO] OUTPUT:    {output_dir}")
    print(f"[INFO] Winsorization: lower={args.winsor_p:.4f}, upper={1 - args.winsor_p:.4f}")

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

            output_path = output_dir / f"features_{horizon}.parquet"
            df_horizon.to_parquet(output_path, index=False)
            print(f"\n[INFO] Saved {output_path} ({len(df_horizon):,} rows, {df_horizon.shape[1]} columns)")
        else:
            print(f"[WARN] No features created for horizon {horizon}")

    # Audit files.
    report_df = pd.DataFrame(all_reports)
    report_path = output_dir / "feature_generation_report.csv"
    report_df.to_csv(report_path, index=False)

    thresholds_df = pd.DataFrame(all_thresholds)
    thresholds_path = output_dir / "winsorization_thresholds.csv"
    thresholds_df.to_csv(thresholds_path, index=False)

    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)
    print(f"[INFO] Feature files saved in: {output_dir}")
    print(f"[INFO] Report saved:          {report_path}")
    print(f"[INFO] Thresholds saved:      {thresholds_path}")


if __name__ == "__main__":
    main()