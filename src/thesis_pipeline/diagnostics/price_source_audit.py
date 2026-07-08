"""Price-source consistency audit (task Section 3).

Reads the per-source provenance CSV files (``price_sources*.csv``) and builds a
consolidated per-coin report:

* source exchange / symbol / quote **by horizon**;
* whether all horizons use the same source;
* first and last date;
* coverage percentage;
* internal missing bars;
* any source inconsistency across horizons.

The audit writes ``Data/Raw/Price/validation/price_source_consistency.csv`` plus
a compact summary, and **fails loudly** (or issues a prominent warning) if any
coin uses different exchanges, symbols or quote currencies across its 1H / 6H /
1D sources.

Note on volume differences (task Section 3): because the source exchange /
symbol / quote is consistent across horizons for every coin, cross-horizon
volume deviations MUST NOT be attributed to different exchanges. Any remaining
deviation is a data-quality diagnostic to be localised by coin + date via the
OHLC/volume audit in :mod:`thesis_pipeline.diagnostics.timing_invariant`, not a
source mismatch.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd

HORIZONS = ("1h", "6h", "1d")

# Flexible column-name resolution — the provenance CSVs have evolved over time.
_COLUMN_ALIASES: dict[str, tuple[str, ...]] = {
    "coin": ("coin", "ticker", "base", "base_asset", "asset", "symbol_base", "name"),
    "horizon": ("horizon", "interval", "timeframe", "tf"),
    "exchange": ("exchange", "source", "source_exchange", "venue"),
    "symbol": ("symbol", "pair", "market", "ccxt_symbol", "trading_pair"),
    "quote": ("quote", "quote_currency", "quote_asset", "quote_ccy"),
    "status": ("status", "state"),
    "missing_bars": ("missing_bars", "n_missing_bars", "gaps", "internal_gaps"),
    "first_date": ("first_date", "start", "start_date", "first", "from"),
    "last_date": ("last_date", "end", "end_date", "last", "to"),
    "coverage_pct": ("coverage_pct", "coverage", "coverage_percent", "coverage_%"),
    "n_bars": ("n_bars", "bars", "n_rows", "rows", "count"),
}


def _norm_cols(df: pd.DataFrame) -> pd.DataFrame:
    """Rename recognised columns to the canonical names; keep the rest."""
    lower = {str(c).strip().lower().replace(" ", "_"): c for c in df.columns}
    rename: dict[str, str] = {}
    for canon, aliases in _COLUMN_ALIASES.items():
        for a in aliases:
            if a in lower and lower[a] not in rename:
                rename[lower[a]] = canon
                break
    out = df.rename(columns=rename)
    return out


def load_price_sources(paths: Iterable[str | Path]) -> pd.DataFrame:
    """Load and vertically concatenate one or more ``price_sources*.csv``."""
    frames = []
    for p in paths:
        p = Path(p)
        if not p.exists():
            continue
        df = _norm_cols(pd.read_csv(p))
        df["__source_file"] = p.name
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    # Normalise horizon + coin labels.
    if "horizon" in out.columns:
        out["horizon"] = out["horizon"].astype(str).str.strip().str.lower()
    if "coin" in out.columns:
        out["coin"] = out["coin"].astype(str).str.strip().str.upper()
    # Drop exact duplicate provenance rows (the 3 files may overlap).
    key = [c for c in ("coin", "horizon", "exchange", "symbol", "quote")
           if c in out.columns]
    if key:
        out = out.drop_duplicates(subset=key + ["__source_file"]) \
                 .drop_duplicates(subset=key)
    return out.reset_index(drop=True)


def _first(series: pd.Series) -> Any:
    s = series.dropna()
    return s.iloc[0] if not s.empty else np.nan


def build_source_consistency(df: pd.DataFrame) -> pd.DataFrame:
    """One row per coin with per-horizon source columns + consistency flags."""
    if df.empty or "coin" not in df.columns:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for coin, grp in df.groupby("coin", sort=True):
        row: dict[str, Any] = {"coin": coin}
        exch_seen, sym_seen, quote_seen = set(), set(), set()
        horizons_present = []
        for hz in HORIZONS:
            sub = grp[grp.get("horizon", pd.Series(dtype=str)) == hz] \
                if "horizon" in grp.columns else grp.iloc[0:0]
            if sub.empty:
                row[f"exchange_{hz}"] = ""
                row[f"symbol_{hz}"] = ""
                row[f"quote_{hz}"] = ""
                continue
            horizons_present.append(hz)
            ex = str(_first(sub.get("exchange", pd.Series(dtype=str))))
            sy = str(_first(sub.get("symbol", pd.Series(dtype=str))))
            qu = str(_first(sub.get("quote", pd.Series(dtype=str))))
            row[f"exchange_{hz}"] = ex
            row[f"symbol_{hz}"] = sy
            row[f"quote_{hz}"] = qu
            if ex:
                exch_seen.add(ex)
            if sy:
                sym_seen.add(sy)
            if qu:
                quote_seen.add(qu.upper())
        # Coverage / dates / missing bars (min coverage, widest date span).
        if "coverage_pct" in grp.columns:
            cov = pd.to_numeric(grp["coverage_pct"], errors="coerce")
            row["min_coverage_pct"] = float(cov.min()) if cov.notna().any() else np.nan
            row["max_coverage_pct"] = float(cov.max()) if cov.notna().any() else np.nan
        if "first_date" in grp.columns:
            row["first_date"] = str(pd.to_datetime(
                grp["first_date"], errors="coerce").min())
        if "last_date" in grp.columns:
            row["last_date"] = str(pd.to_datetime(
                grp["last_date"], errors="coerce").max())
        if "missing_bars" in grp.columns:
            mb = pd.to_numeric(grp["missing_bars"], errors="coerce")
            row["total_internal_missing_bars"] = int(mb.fillna(0).sum())
        if "status" in grp.columns:
            row["status"] = ";".join(sorted(
                grp["status"].astype(str).str.strip().unique()))
        row["n_horizons"] = len(horizons_present)
        row["exchange_consistent"] = len(exch_seen) <= 1
        row["symbol_consistent"] = len(sym_seen) <= 1
        row["quote_consistent"] = len(quote_seen) <= 1
        row["same_source_all_horizons"] = (
            row["exchange_consistent"] and row["symbol_consistent"]
            and row["quote_consistent"])
        problems = []
        if not row["exchange_consistent"]:
            problems.append(f"exchange differs across horizons: {sorted(exch_seen)}")
        if not row["symbol_consistent"]:
            problems.append(f"symbol differs across horizons: {sorted(sym_seen)}")
        if not row["quote_consistent"]:
            problems.append(f"quote differs across horizons: {sorted(quote_seen)}")
        row["source_inconsistency"] = "; ".join(problems)
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_sources(df: pd.DataFrame, consistency: pd.DataFrame) -> dict[str, Any]:
    """Compact provenance summary (exchange allocation, counts, quote check)."""
    summary: dict[str, Any] = {
        "n_coins": int(consistency["coin"].nunique()) if not consistency.empty else 0,
        "n_source_rows": int(len(df)),
    }
    # Per-coin exchange allocation (use the 1d source, falling back to any).
    if not consistency.empty:
        def _coin_exchange(r: pd.Series) -> str:
            for hz in ("1d", "6h", "1h"):
                v = str(r.get(f"exchange_{hz}", "") or "")
                if v:
                    return v
            return ""
        exch = consistency.apply(_coin_exchange, axis=1)
        summary["exchange_allocation"] = {
            k: int(v) for k, v in exch[exch != ""].value_counts().items()}
        summary["n_inconsistent_coins"] = int(
            (~consistency["same_source_all_horizons"]).sum())
        summary["inconsistent_coins"] = list(
            consistency.loc[~consistency["same_source_all_horizons"], "coin"])
    # Quote-currency check across ALL rows.
    if "quote" in df.columns:
        quotes = sorted(df["quote"].dropna().astype(str).str.upper().unique())
        summary["distinct_quotes"] = quotes
        summary["all_usdt"] = quotes == ["USDT"]
    if "status" in df.columns:
        summary["status_values"] = sorted(
            df["status"].dropna().astype(str).str.strip().unique())
    if "missing_bars" in df.columns:
        mb = pd.to_numeric(df["missing_bars"], errors="coerce").fillna(0)
        summary["total_internal_missing_bars"] = int(mb.sum())
    return summary


def run_price_source_audit(paths: Sequence[str | Path],
                           *,
                           out_dir: str | Path = "Data/Raw/Price/validation",
                           strict: bool = False,
                           write: bool = True) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Full audit: load → consolidate → write CSV + summary → warn/raise.

    Returns ``(consistency_df, summary)``. When ``strict`` is True and any coin
    is source-inconsistent across horizons, raises ``ValueError``; otherwise it
    prints a prominent warning.
    """
    df = load_price_sources(paths)
    consistency = build_source_consistency(df)
    summary = summarize_sources(df, consistency)

    if write and not consistency.empty:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        consistency.to_csv(out / "price_source_consistency.csv", index=False)
        pd.DataFrame([summary]).to_csv(
            out / "price_source_consistency_summary.csv", index=False)

    _print_summary(summary, consistency)

    n_bad = summary.get("n_inconsistent_coins", 0)
    if n_bad:
        msg = (f"[FAIL] {n_bad} coin(s) use inconsistent sources across "
               f"horizons: {summary.get('inconsistent_coins')}. A coin's 1H/6H/"
               "1D bars MUST come from the same exchange + symbol + quote.")
        if strict:
            raise ValueError(msg)
        print("\n" + "!" * 72)
        print(msg)
        print("!" * 72)
    return consistency, summary


def _print_summary(summary: dict[str, Any], consistency: pd.DataFrame) -> None:
    print("\n=== Price-source consistency audit ===")
    print(f"  coins={summary.get('n_coins')} "
          f"source_rows={summary.get('n_source_rows')}")
    print(f"  exchange allocation: {summary.get('exchange_allocation')}")
    print(f"  distinct quotes: {summary.get('distinct_quotes')} "
          f"(all USDT: {summary.get('all_usdt')})")
    print(f"  status values: {summary.get('status_values')}")
    print(f"  total internal missing bars: "
          f"{summary.get('total_internal_missing_bars')}")
    print(f"  source-inconsistent coins: {summary.get('n_inconsistent_coins')}")
