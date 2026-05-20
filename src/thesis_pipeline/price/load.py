"""Raw OHLCV loading helpers.

Thin convenience layer over the loaders defined in the root scripts.
Behavior is unchanged — this module only centralises path resolution.
"""

from __future__ import annotations

from pathlib import Path
from typing import List

import pandas as pd

from ..config import load_config, resolve_path


def raw_price_path(ticker: str, horizon: str) -> Path:
    """Return ``Data/Raw/Price/<horizon>/<TICKER>USDT_<suffix>.parquet``."""
    horizons = load_config("horizons")["raw_price_horizons"]
    if horizon not in horizons:
        raise KeyError(f"Unknown raw price horizon: {horizon}")
    suffix = horizons[horizon]["filename_suffix"]
    template = load_config("coins")["raw_price_filename_template"]
    filename = template.format(ticker=ticker.upper(), suffix=suffix)
    return resolve_path(f"raw_price_{horizon}") / filename


def discover_tickers(horizon: str = "1d") -> List[str]:
    """Discover tickers present on disk for a horizon."""
    folder = resolve_path(f"raw_price_{horizon}")
    if not folder.exists():
        return []
    horizons = load_config("horizons")["raw_price_horizons"]
    suffix = horizons[horizon]["filename_suffix"]
    out = []
    for p in folder.glob(f"*USDT_{suffix}.parquet"):
        name = p.stem.replace(f"USDT_{suffix}", "")
        out.append(name)
    return sorted(out)


def load_ohlcv(ticker: str, horizon: str) -> pd.DataFrame:
    """Load OHLCV from disk, returning a dataframe with normalised columns."""
    path = raw_price_path(ticker, horizon)
    df = pd.read_parquet(path)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=False, errors="coerce")
    return df
