"""Small IO helpers used across stages.

Most code in the original scripts still does its own pandas IO; these helpers
exist so new code can hit a single function and so that ``row_id`` can be
attached on read where it is missing.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd


def ensure_dir(path: Path | str) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def add_row_id(df: pd.DataFrame, *, ticker_col: str = "ticker",
               horizon_col: str | None = "horizon",
               horizon: str | None = None,
               timestamp_col: str = "timestamp",
               column_name: str = "row_id") -> pd.DataFrame:
    """Attach ``row_id`` = ``ticker + "_" + horizon + "_" + timestamp`` if missing.

    If the dataframe has no ``horizon`` column, ``horizon`` is taken from the
    keyword argument.
    """
    if column_name in df.columns:
        return df
    if horizon_col and horizon_col in df.columns:
        horizon_series = df[horizon_col].astype(str)
    elif horizon is not None:
        horizon_series = pd.Series([str(horizon)] * len(df), index=df.index)
    else:
        raise ValueError("Cannot build row_id: no horizon column and no `horizon` arg given.")
    if ticker_col not in df.columns:
        raise KeyError(f"Cannot build row_id: missing column {ticker_col!r}")
    if timestamp_col not in df.columns:
        raise KeyError(f"Cannot build row_id: missing column {timestamp_col!r}")
    df = df.copy()
    df[column_name] = (
        df[ticker_col].astype(str) + "_" + horizon_series + "_"
        + pd.to_datetime(df[timestamp_col]).astype("int64").astype(str)
    )
    return df


def read_parquet(path: Path | str, *, add_row_id_kwargs: dict | None = None) -> pd.DataFrame:
    df = pd.read_parquet(path)
    if add_row_id_kwargs is not None:
        df = add_row_id(df, **add_row_id_kwargs)
    return df


def write_parquet(df: pd.DataFrame, path: Path | str, *, overwrite: bool = True) -> Path:
    path = Path(path)
    if path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite (overwrite=False): {path}")
    ensure_dir(path.parent)
    df.to_parquet(path, index=False)
    return path


def write_csv(df: pd.DataFrame, path: Path | str, *, overwrite: bool = True, **kwargs) -> Path:
    path = Path(path)
    if path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite (overwrite=False): {path}")
    ensure_dir(path.parent)
    df.to_csv(path, index=False, **kwargs)
    return path


def list_existing(paths: Iterable[Path | str]) -> list[Path]:
    return [Path(p) for p in paths if Path(p).exists()]
