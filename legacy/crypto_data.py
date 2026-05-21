#!/usr/bin/env python3
"""
crypto_data_downloader.py — Sentiment Project Edition (v4)
==========================================================
Downloads OHLCV data for the coins covered by the Reddit sentiment
pipeline (leukipp dataset). Uses a per-coin candidate list with
runtime failover: each asset has an ordered list of (exchange, symbol)
sources, and if the primary fails (errors exhausted or zero bars
returned), the next source is tried automatically.

Output format: Parquet (one file per coin/timeframe), with metadata
sidecars price_sources.csv and _checkpoint.json kept as plain text
for easy human inspection.

Install:
    pip install ccxt pandas pyarrow

Usage:
    # Default run (all coins, all four timeframes, 2022 Reddit data window)
    python crypto_data_downloader.py --output /path/to/thesis/data/raw/crypto

    # Subset of timeframes
    python crypto_data_downloader.py --timeframes 15m 1h

    # Specific coins only
    python crypto_data_downloader.py --coins BTC ETH SOL ADA

    # Custom date range
    python crypto_data_downloader.py --since 2021-12-01 --until 2023-02-01

    # Resume interrupted download
    python crypto_data_downloader.py --resume
"""

import argparse
import csv
import json
import os
import sys
import time
from datetime import datetime, timezone

import pandas as pd

try:
    import ccxt
except ImportError:
    print("Installing ccxt ...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "ccxt"])
    import ccxt

# pyarrow is required by pandas for parquet I/O. Fail early with a clear message
# if it's missing rather than letting pandas raise a cryptic engine-not-found error
# on the first to_parquet call (potentially after hours of downloading).
try:
    import pyarrow  # noqa: F401
except ImportError:
    print("\n❌ Missing dependency: pyarrow")
    print("   Install with: pip install pyarrow")
    print("   (required for parquet output)")
    sys.exit(1)


# ══════════════════════════════════════════════════════════════════════════════
# COIN UNIVERSE — Matched to Reddit Sentiment Pipeline subreddit mapping
# ══════════════════════════════════════════════════════════════════════════════
#
#  These are the coins mapped from the leukipp Reddit crypto dataset.
#  Subreddit → Coin mapping:
#    r/bitcoin, r/btc, r/bitcoinbeginners, r/bitcoinmarkets, r/bitcoinmining → BTC
#    r/bitcoincash → BCH
#    r/ethereum, r/ethdev, r/ethermining, r/ethfinance, r/ethtrader → ETH
#    r/cardano → ADA           r/dogecoin → DOGE        r/solana → SOL
#    r/polkadot → DOT          r/ripple, r/xrp → XRP    r/monero, r/xmrtrader → XMR
#    r/litecoin, r/litecoinmarkets → LTC                 r/stellar → XLM
#    r/tezos → XTZ             r/vechain → VET           r/iota → IOTA
#    r/neo → NEO               r/eos → EOS               r/tronix → TRX
#    r/uniswap → UNI           r/decentraland → MANA     r/loopringorg → LRC
#    r/nanocurrency → NANO     r/batproject → BAT
#    r/binance → BNB           r/crypto_com → CRO        r/kucoin → KCS
#    r/coinbase → EXCLUDED (COIN is a stock, not a cryptocurrency)

COIN_UNIVERSE = [
    # Major caps
    "BTC", "ETH", "BNB", "XRP", "SOL", "ADA", "DOGE",
    # Mid caps
    "DOT", "LTC", "TRX", "UNI", "XLM", "XMR", "VET", "XTZ",
    # Smaller / niche
    "EOS", "NEO", "IOTA", "MANA", "LRC", "BAT", "NANO", "BCH",
    # Exchange tokens
    "CRO", "KCS",
]

# Symbol aliases — some coins are listed under different tickers per exchange.
# NANO was rebranded to XNO in 2021; some exchanges updated, others did not.
# IOTA is listed as MIOTA on a number of exchanges.
SYMBOL_ALIASES = {
    "NANO": ["NANO", "XNO"],
    "IOTA": ["IOTA", "MIOTA"],
}

# Per-coin exchange preference. The first reachable exchange that lists an
# active spot pair for the coin (against any quote in QUOTE_PRIORITY) wins.
# Coins not listed here fall through to FALLBACK_EXCHANGES.
COIN_EXCHANGE_PREFERENCE = {
    "KCS":  ["kucoin", "gate"],                  # KuCoin native token
    "CRO":  ["kucoin", "gate", "okx"],           # crypto.com token, limited listings
    "XMR":  ["kucoin", "gate", "kraken"],        # Kraken's REST DDoS-protects aggressively;
                                                  # try KuCoin/Gate first, Kraken only as last resort
    "NANO": ["binance", "kucoin", "kraken"],     # listed as XNO on several venues
    "EOS":  ["okx", "kucoin", "gate", "kraken"], # delisted on Binance/Bybit (2024); historical
                                                  # data still on the others
}

FALLBACK_EXCHANGES = ["bybit", "okx", "kucoin", "gate", "binance", "kraken"]

QUOTE_PRIORITY = ["USDT", "USD", "USDC", "BUSD", "FDUSD"]

# Default timeframes
# Default timeframes — full multi-resolution set used in the sentiment thesis.
# Note: 1m data depth varies strongly across exchanges. Binance has the deepest
# 1m history (back to 2017+); Bybit/KuCoin keep ~2 years; Kraken is limited.
# If 1m coverage looks sparse for some coins, consider --exchange binance.
DEFAULT_TIMEFRAMES = [ "6h"]

EXCHANGE_LIMITS = {
    "binance": 1000, "bybit": 1000, "okx": 300, "kucoin": 1500,
    "gate": 1000, "kraken": 720, "coinbase": 300, "bitfinex": 10000,
}

TF_MS = {
    "1m": 60_000, "5m": 300_000, "15m": 900_000, "30m": 1_800_000,
    "1h": 3_600_000, "4h": 14_400_000, "6h" : 6 * 60 * 60 * 1000, "1d": 86_400_000,
}

TF_DIR_NAMES = {
    "1m": "1min", "5m": "5min", "15m": "15min", "30m": "30min",
    "1h": "1h", "4h": "4h", "1d": "1d",
}


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def parse_date_ms(s):
    return int(datetime.strptime(s, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp() * 1000)

def ms_to_str(ms):
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M")

def ms_to_date(ms):
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).strftime("%Y-%m-%d")


# ══════════════════════════════════════════════════════════════════════════════
# CHECKPOINT (resume after interruption)
# ══════════════════════════════════════════════════════════════════════════════

def load_checkpoint(output_dir):
    path = os.path.join(output_dir, "_checkpoint.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}

def save_checkpoint(output_dir, coin, timeframe, status, n_bars=0):
    ckpt = load_checkpoint(output_dir)
    key = f"{coin}_{timeframe}"
    ckpt[key] = {"status": status, "n_bars": n_bars, "time": datetime.now().isoformat()}
    with open(os.path.join(output_dir, "_checkpoint.json"), "w") as f:
        json.dump(ckpt, f, indent=2)


# ══════════════════════════════════════════════════════════════════════════════
# PRICE SOURCES METADATA (for thesis appendix / reproducibility)
# ══════════════════════════════════════════════════════════════════════════════

PRICE_SOURCES_HEADER = [
    "coin", "timeframe", "exchange", "symbol", "quote",
    "start", "end", "n_bars", "expected_bars", "coverage_pct",
    "missing_bars", "max_gap_minutes",
    "status", "filename", "fetched_at",
]

def upsert_price_source(output_dir, row):
    """Write/replace one row in price_sources.csv keyed by (coin, timeframe).
    Reading and rewriting the whole file avoids duplicate rows when --resume
    is used multiple times. The file stays small (one row per coin/timeframe).
    """
    path = os.path.join(output_dir, "price_sources.csv")
    key = (row.get("coin"), row.get("timeframe"))

    rows = []
    if os.path.exists(path):
        with open(path, newline="") as f:
            reader = csv.DictReader(f)
            for r in reader:
                if (r.get("coin"), r.get("timeframe")) != key:
                    rows.append(r)
    rows.append({k: row.get(k, "") for k in PRICE_SOURCES_HEADER})

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=PRICE_SOURCES_HEADER)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k, "") for k in PRICE_SOURCES_HEADER})


def count_gaps(df, tf_ms):
    """Count missing bars and the largest gap inside the saved series.

    Returns (missing_bars, max_gap_minutes).
    `missing_bars` is the total number of bars expected between consecutive
    timestamps that are not present. `max_gap_minutes` is the largest single
    gap, expressed in minutes (timeframe-independent).
    """
    if df is None or len(df) < 2:
        return 0, 0
    ts = df["timestamp"].sort_values().reset_index(drop=True)
    deltas = ts.diff().dropna()
    if deltas.empty:
        return 0, 0
    # number of missing intervals between each consecutive pair
    missing = ((deltas / tf_ms) - 1).clip(lower=0).sum()
    max_gap_ms = int(deltas.max())
    return int(missing), int(round(max_gap_ms / 60_000))


# ══════════════════════════════════════════════════════════════════════════════
# EXCHANGE CONNECTION — load all reachable exchanges up front
# ══════════════════════════════════════════════════════════════════════════════

def connect_all_exchanges(preferred=None):
    """Connect to every reachable exchange and load its market list once.
    Returns an ordered dict: {ex_id: ccxt_instance}.
    """
    candidates = []
    if preferred:
        candidates.append(preferred)
    candidates += [e for e in FALLBACK_EXCHANGES if e not in candidates]

    exchanges = {}
    print("\n  Connecting to exchanges:")
    for ex_id in candidates:
        print(f"    {ex_id:10s} ...", end=" ", flush=True)
        try:
            ex = getattr(ccxt, ex_id)({
                "enableRateLimit": True,
                "timeout": 30000,
                "options": {"defaultType": "spot"},
            })
            ex.load_markets()
            exchanges[ex_id] = ex
            print(f"✓ ({len(ex.markets)} markets)")
        except Exception as e:
            print(f"✗ {type(e).__name__}")

    if not exchanges:
        print("\n❌ No exchanges reachable. Check internet connection.")
        sys.exit(1)

    return exchanges


# ══════════════════════════════════════════════════════════════════════════════
# SYMBOL RESOLUTION — per-coin fallback across all reachable exchanges
# ══════════════════════════════════════════════════════════════════════════════

def find_symbol_candidates(exchanges, coin):
    """Find ALL spot pairs for `coin` across reachable exchanges, ranked by preference.

    Returns a list of (exchange_instance, ex_id, symbol, quote, base) tuples.
    The list is sorted: coin-specific preference first, then global fallback order;
    within each exchange, base aliases in SYMBOL_ALIASES order, quotes by QUOTE_PRIORITY.
    Active markets come before inactive ones (some exchanges flag delisted-but-historic
    pairs as active=False; fetch_ohlcv often still returns historical data for them).
    """
    coin_pref = COIN_EXCHANGE_PREFERENCE.get(coin, [])
    search_order = [e for e in coin_pref if e in exchanges]
    search_order += [e for e in exchanges.keys() if e not in search_order]

    base_candidates = SYMBOL_ALIASES.get(coin, [coin])

    active_hits = []
    inactive_hits = []
    for ex_id in search_order:
        exchange = exchanges[ex_id]
        for base in base_candidates:
            for quote in QUOTE_PRIORITY:
                sym = f"{base}/{quote}"
                market = exchange.markets.get(sym)
                if market is None:
                    continue
                is_spot = market.get("spot", False) or market.get("type") == "spot"
                if not is_spot:
                    continue
                # Loosened active check: explicit False → inactive bucket; otherwise active.
                # Inactive markets are kept as fallback because fetch_ohlcv may still
                # return historical bars for delisted pairs.
                is_active = market.get("active") is not False
                bucket = active_hits if is_active else inactive_hits
                bucket.append((exchange, ex_id, sym, quote, base))

    return active_hits + inactive_hits


# Backward-compat shim: old name kept for any external caller.
def find_symbol_any_exchange(exchanges, coin):
    """Compatibility wrapper — returns the first candidate or all-Nones."""
    cands = find_symbol_candidates(exchanges, coin)
    return cands[0] if cands else (None, None, None, None, None)


# ══════════════════════════════════════════════════════════════════════════════
# DOWNLOAD ONE COIN + TIMEFRAME (symbol pre-resolved)
# ══════════════════════════════════════════════════════════════════════════════

def download_coin(candidates, coin, timeframe, since_ms, until_ms,
                  output_dir, resume=False, max_total_errors=8):
    """Download OHLCV with per-source failover.

    `candidates` is a list of (exchange, ex_id, symbol, quote, base) tuples,
    sorted by preference. Each is tried in turn; on failure (errors exhausted
    or zero bars returned) the next candidate is attempted. The price_sources.csv
    row reflects the source that ultimately succeeded.
    """
    tf_ms = TF_MS[timeframe]
    expected_bars = int((until_ms - since_ms) / tf_ms)

    if not candidates:
        return None

    # ── Resume: look for any existing partial file across candidate quotes ──
    existing_filepath = None
    existing_df = None
    if resume:
        for (_, _, _, q_check, _) in candidates:
            fp_check = os.path.join(output_dir, f"{coin}{q_check}_{timeframe}.parquet")
            if os.path.exists(fp_check):
                try:
                    df_check = pd.read_parquet(fp_check)
                    if len(df_check) > 0:
                        existing_filepath = fp_check
                        existing_df = df_check
                        break
                except Exception:
                    pass

    # If we already have a complete file, return immediately (no source switch)
    if existing_df is not None:
        last_ts = int(existing_df["timestamp"].iloc[-1])
        if last_ts >= until_ms - tf_ms:
            print(f"  ✓ {coin:5s} [{timeframe}] already complete ({len(existing_df):,} bars)")
            # Recover provenance from filename quote
            for (_, ex_id, sym, q, _) in candidates:
                if existing_filepath.endswith(f"{coin}{q}_{timeframe}.parquet"):
                    _log_price_source(output_dir, coin, timeframe, ex_id, sym, q,
                                      existing_df, expected_bars, "complete",
                                      os.path.basename(existing_filepath))
                    break
            save_checkpoint(output_dir, coin, timeframe, "complete", len(existing_df))
            return existing_filepath

    # ── Try each candidate in order ─────────────────────────────────────
    for attempt_idx, (exchange, ex_id, symbol, quote, base) in enumerate(candidates):
        filename = f"{coin}{quote}_{timeframe}.parquet"
        filepath = os.path.join(output_dir, filename)

        if attempt_idx > 0:
            print(f"  ↪ {coin:5s} [{timeframe}] failover → {ex_id} {symbol}")

        # Use existing partial file only if it matches THIS candidate's filename
        local_existing = existing_df if existing_filepath == filepath else None
        fetch_since = since_ms
        if local_existing is not None and len(local_existing) > 0:
            last_ts = int(local_existing["timestamp"].iloc[-1])
            fetch_since = last_ts + tf_ms
            print(f"  ↻ {coin:5s} [{timeframe}] resuming from {ms_to_str(fetch_since)}")

        df, status = _fetch_one_source(
            exchange=exchange, ex_id=ex_id, coin=coin, symbol=symbol,
            timeframe=timeframe, fetch_since=fetch_since, since_ms=since_ms,
            until_ms=until_ms, filepath=filepath, existing_df=local_existing,
            max_total_errors=max_total_errors,
        )

        if df is not None and len(df) > 0:
            # Cast to compact dtypes and save
            df["timestamp"] = df["timestamp"].astype("int64")
            for col in ("open", "high", "low", "close", "volume"):
                df[col] = df[col].astype("float64")
            df.to_parquet(filepath, index=False, compression="snappy")

            first = ms_to_date(int(df["timestamp"].iloc[0]))
            last = ms_to_date(int(df["timestamp"].iloc[-1]))
            print(f"  ✓ {coin:5s} [{timeframe}] {len(df):>8,} bars | {first} → {last} | "
                  f"{ex_id:8s} | {symbol}")

            _log_price_source(output_dir, coin, timeframe, ex_id, symbol, quote,
                              df, expected_bars, "complete", filename)
            save_checkpoint(output_dir, coin, timeframe, "complete", len(df))
            return filepath

        # This candidate failed — try next
        print(f"  ✗ {coin:5s} [{timeframe}] {ex_id} → {status}")

    # All candidates exhausted
    last_ex, last_ex_id, last_sym, last_quote, _ = candidates[-1]
    last_filename = f"{coin}{last_quote}_{timeframe}.parquet"
    print(f"  ✗ {coin:5s} [{timeframe}] all {len(candidates)} sources failed")
    _log_price_source(output_dir, coin, timeframe, last_ex_id, last_sym, last_quote,
                      None, expected_bars, "no_data", last_filename)
    save_checkpoint(output_dir, coin, timeframe, "no_data")
    return None


def _fetch_one_source(exchange, ex_id, coin, symbol, timeframe,
                      fetch_since, since_ms, until_ms, filepath,
                      existing_df=None, max_total_errors=8):
    """Paginated OHLCV fetch from one exchange/symbol.

    Returns (df, status) where df is the merged dataframe (or None) and status
    is a short string ("ok", "errors", "empty") for failover diagnostics.

    Retry logic: total_errors is the hard cap (does NOT reset on empty batches —
    that was the bug in v3). consecutive_errors only governs exponential backoff.
    """
    tf_ms = TF_MS[timeframe]
    limit = EXCHANGE_LIMITS.get(ex_id, 500)

    all_candles = []
    cursor = fetch_since
    last_save = time.time()
    consecutive_errors = 0
    total_errors = 0
    empty_streak = 0
    max_empty_streak = 50  # if 50 consecutive empty batches, the source has nothing

    while cursor < until_ms:
        try:
            batch = exchange.fetch_ohlcv(symbol, timeframe=timeframe,
                                         since=cursor, limit=limit)
        except Exception as e:
            total_errors += 1
            consecutive_errors += 1
            if total_errors >= max_total_errors:
                print(f"\n  ⚠ {coin:5s} [{timeframe}] {ex_id}: aborting after "
                      f"{total_errors} errors ({type(e).__name__})")
                if not all_candles:
                    return None, f"errors:{total_errors}"
                break
            wait = min(60, 2 ** consecutive_errors)
            print(f"  ⚠ {coin:5s} [{timeframe}] {ex_id} error "
                  f"({total_errors}/{max_total_errors}): {type(e).__name__} — wait {wait}s")
            time.sleep(wait)
            continue

        consecutive_errors = 0  # successful HTTP call resets backoff

        if not batch:
            empty_streak += 1
            if empty_streak >= max_empty_streak:
                if not all_candles:
                    return None, "empty"
                break
            cursor += tf_ms * limit
            continue

        # Filter to requested window
        batch = [c for c in batch if since_ms <= c[0] < until_ms]
        if not batch:
            empty_streak += 1
            if empty_streak >= max_empty_streak:
                if not all_candles:
                    return None, "empty"
                break
            cursor += tf_ms * limit
            continue

        empty_streak = 0
        all_candles.extend(batch)
        next_cursor = batch[-1][0] + tf_ms
        if next_cursor <= cursor:
            cursor += tf_ms * limit
        else:
            cursor = next_cursor

        if time.time() - last_save > 30:
            _save_progress(filepath, existing_df, all_candles, since_ms, until_ms)
            last_save = time.time()

        time.sleep(getattr(exchange, "rateLimit", 200) / 1000)

    # Merge with any existing partial file
    if not all_candles and existing_df is None:
        return None, "empty"

    new_df = pd.DataFrame(all_candles,
                          columns=["timestamp", "open", "high", "low", "close", "volume"])
    if existing_df is not None and len(existing_df) > 0:
        df = pd.concat([existing_df, new_df], ignore_index=True)
    else:
        df = new_df

    df = df.drop_duplicates(subset="timestamp", keep="first")
    df = df.sort_values("timestamp").reset_index(drop=True)
    df = df[(df["timestamp"] >= since_ms) & (df["timestamp"] < until_ms)]
    return df, "ok"


def _log_price_source(output_dir, coin, timeframe, ex_id, symbol, quote,
                      df, expected_bars, status, filename):
    """Write one provenance row to price_sources.csv (upserted by coin/timeframe)."""
    tf_ms = TF_MS[timeframe]
    if df is not None and len(df) > 0:
        n_bars = len(df)
        start = ms_to_date(int(df["timestamp"].iloc[0]))
        end = ms_to_date(int(df["timestamp"].iloc[-1]))
        coverage = round(100 * n_bars / expected_bars, 2) if expected_bars else 0
        missing_bars, max_gap_minutes = count_gaps(df, tf_ms)
    else:
        n_bars, start, end, coverage = 0, "", "", 0
        missing_bars, max_gap_minutes = 0, 0

    upsert_price_source(output_dir, {
        "coin": coin, "timeframe": timeframe,
        "exchange": ex_id, "symbol": symbol, "quote": quote,
        "start": start, "end": end,
        "n_bars": n_bars, "expected_bars": expected_bars,
        "coverage_pct": coverage,
        "missing_bars": missing_bars,
        "max_gap_minutes": max_gap_minutes,
        "status": status, "filename": filename,
        "fetched_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    })


def _save_progress(filepath, existing_df, new_candles, since_ms, until_ms):
    """Intermediate save for crash protection."""
    try:
        new_df = pd.DataFrame(new_candles,
                              columns=["timestamp", "open", "high", "low", "close", "volume"])
        if existing_df is not None and len(existing_df) > 0:
            df = pd.concat([existing_df, new_df], ignore_index=True)
        else:
            df = new_df
        df = df.drop_duplicates(subset="timestamp", keep="first").sort_values("timestamp")
        df = df[(df["timestamp"] >= since_ms) & (df["timestamp"] < until_ms)]
        df.to_parquet(filepath, index=False, compression="snappy")
    except Exception:
        pass


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Download OHLCV data with per-coin exchange fallback",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("--exchange", default=None,
                        help="Preferred lead exchange (still falls through if pair missing)")
    parser.add_argument("--coins", nargs="+", default=None,
                        help=f"Coins to download (default: all {len(COIN_UNIVERSE)})")
    parser.add_argument("--timeframes", nargs="+", default=None,
                        help="Timeframes to download (default: 15m)")
    parser.add_argument("--since", default="2021-12-01",
                        help="Start date (default: 2021-12-01, 1 month before Reddit data)")
    parser.add_argument("--until", default="2023-02-01",
                        help="End date (default: 2023-02-01, 1 month after Reddit data)")
    parser.add_argument("--output", default="data/raw/price"
                        )
    parser.add_argument("--resume", action="store_true",
                        help="Skip already-complete coins, resume partials")

    args = parser.parse_args()
    coins = args.coins or COIN_UNIVERSE
    timeframes = args.timeframes or DEFAULT_TIMEFRAMES
    since_ms = parse_date_ms(args.since)
    until_ms = parse_date_ms(args.until)

    os.makedirs(args.output, exist_ok=True)

    # ── Header ───────────────────────────────────────────────────
    print()
    print("=" * 65)
    print("  CRYPTO OHLCV DOWNLOADER — Sentiment Project Edition (v4)")
    print("=" * 65)
    print(f"  Timeframes: {', '.join(timeframes)}")
    print(f"  Period:     {args.since} → {args.until}")
    print(f"  Coins:      {len(coins)} coins")
    for row_start in range(0, len(coins), 8):
        chunk = coins[row_start:row_start+8]
        print(f"              {', '.join(chunk)}")
    print(f"  Output:     {args.output}/")
    print(f"  Provenance: {args.output}/price_sources.csv")

    # ── Connect to all reachable exchanges ───────────────────────
    exchanges = connect_all_exchanges(args.exchange)

    # ── Resolve symbol candidates per coin ───────────────────────
    print(f"\n  Resolving symbols (per-coin candidate lists):")
    resolved = {}
    unresolved = []
    for coin in coins:
        cands = find_symbol_candidates(exchanges, coin)
        if not cands:
            print(f"    {coin:5s} ✗ no spot pair found on any reachable exchange")
            unresolved.append(coin)
        else:
            primary = cands[0]
            _, ex_id, sym, _, base = primary
            alias_note = f" (alias for {coin})" if base != coin else ""
            fallback_note = (f"  [+{len(cands)-1} fallback{'s' if len(cands)>2 else ''}]"
                             if len(cands) > 1 else "")
            print(f"    {coin:5s} → {ex_id:8s} {sym}{alias_note}{fallback_note}")
            resolved[coin] = cands

    if unresolved:
        print(f"\n  ⚠ {len(unresolved)} coin(s) unavailable: {', '.join(unresolved)}")

    # ── Checkpoint info ──────────────────────────────────────────
    if args.resume:
        ckpt = load_checkpoint(args.output)
        done = [k for k, v in ckpt.items() if v.get("status") == "complete"]
        if done:
            print(f"\n  📋 Checkpoint: {len(done)} coin/timeframe combos already done")

    # ── Download loop ────────────────────────────────────────────
    t0 = time.time()
    results = {}
    failed = []

    for tf_idx, tf in enumerate(timeframes, 1):
        tf_dir = os.path.join(args.output, TF_DIR_NAMES.get(tf, tf))
        os.makedirs(tf_dir, exist_ok=True)

        bars_per_coin = (until_ms - since_ms) / TF_MS[tf]

        print(f"\n{'━' * 65}")
        print(f"  TIMEFRAME {tf_idx}/{len(timeframes)}: {tf}  "
              f"(~{bars_per_coin:,.0f} bars/coin)")
        print(f"{'━' * 65}")

        for i, coin in enumerate(coins, 1):
            if coin not in resolved:
                failed.append((coin, tf))
                continue

            # Resume check
            if args.resume:
                ckpt = load_checkpoint(args.output)
                if ckpt.get(f"{coin}_{tf}", {}).get("status") == "complete":
                    print(f"  [{i:2d}/{len(coins)}] ✓ {coin:5s} [{tf}] skipped (checkpoint)")
                    continue

            candidates = resolved[coin]
            print(f"  [{i:2d}/{len(coins)}]")
            fp = download_coin(candidates=candidates, coin=coin, timeframe=tf,
                               since_ms=since_ms, until_ms=until_ms,
                               output_dir=tf_dir, resume=args.resume)
            if fp:
                results[(coin, tf)] = fp
            else:
                failed.append((coin, tf))

    # ── Summary ──────────────────────────────────────────────────
    total_time = time.time() - t0
    print(f"\n{'='*65}")
    print(f"  DOWNLOAD COMPLETE — {total_time/60:.0f} min ({total_time/3600:.1f}h)")
    print(f"{'='*65}\n")

    for tf in timeframes:
        tf_dir = os.path.join(args.output, TF_DIR_NAMES.get(tf, tf))
        stats = []
        for coin in coins:
            for q in QUOTE_PRIORITY:
                fp = os.path.join(tf_dir, f"{coin}{q}_{tf}.parquet")
                if os.path.exists(fp):
                    df = pd.read_parquet(fp)
                    if len(df) > 0:
                        stats.append({
                            "coin": coin, "bars": len(df),
                            "start": ms_to_date(int(df["timestamp"].iloc[0])),
                            "end": ms_to_date(int(df["timestamp"].iloc[-1])),
                            "file": os.path.basename(fp),
                            "mb": os.path.getsize(fp) / 1024 / 1024,
                        })
                    break

        if stats:
            print(f"  ── {tf} ──")
            print(f"  {'Coin':5s} {'Bars':>10s}  {'Start':10s}  {'End':10s}  {'Size':>6s}  File")
            print(f"  {'─'*5} {'─'*10}  {'─'*10}  {'─'*10}  {'─'*6}  {'─'*25}")
            total_bars = 0
            total_mb = 0
            for s in stats:
                print(f"  {s['coin']:5s} {s['bars']:>10,}  {s['start']}  {s['end']}  "
                      f"{s['mb']:4.1f}MB  {s['file']}")
                total_bars += s["bars"]
                total_mb += s["mb"]
            print(f"  Total: {len(stats)} coins | {total_bars:,} bars | {total_mb:.0f}MB\n")

    if failed:
        print(f"  ✗ Issues: {', '.join(f'{c}({tf})' for c, tf in failed)}")

    print(f"\n  Provenance log written to: {args.output}/price_sources.csv")
    print(f"  → Cite this file in the thesis appendix for data source documentation.\n")


if __name__ == "__main__":
    main()