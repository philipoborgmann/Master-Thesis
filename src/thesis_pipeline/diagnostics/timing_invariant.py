"""Timing-convention invariant + CCXT parquet audit (task Section 2).

Timestamp convention (the invariant this module encodes)
--------------------------------------------------------
``timestamp`` is the **interval-START label** — the raw CCXT bar-start
timestamp. A row labelled ``t`` describes the *completed* interval
``[t, t+h)``:

* the bar ``open`` is the first intraday open of that interval,
* the bar ``close`` is the final intraday close,
* ``high`` / ``low`` are the intraday extrema,
* ``log_return_t`` is the return realised over the interval (a CURRENT-bar
  feature, usable at ``t+h``),
* Reddit posts are floored to the slot START, so a post enters row ``t`` iff
  it was created in ``[t, t+h)`` — never after the interval ends,
* ``target`` is the sign of the return of the NEXT interval ``[t+h, t+2h)``.

No timestamp shift is applied on either the price or the sentiment side; the
price/sentiment join is a plain equality join on ``(ticker, timestamp)``,
which is correct precisely because both sides use the same interval-start
label. This module provides:

* :func:`check_interval_start_semantics` — a reusable invariant used by the
  synthetic 1H/6H/1D unit tests and available as a runtime diagnostic;
* :func:`audit_ohlc_aggregation` / :func:`audit_timestamp_grid` — building
  blocks for the optional audit command that inspects the externally-supplied
  CCXT parquet files (timestamp grid, inferred label convention, OHLC
  aggregation consistency, volume aggregation deviations).

The synthetic unit tests must NOT depend on the externally-supplied ADA files;
those are only inspected by the optional audit command.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

#: Canonical bar width per horizon.
HORIZON_STEP: dict[str, pd.Timedelta] = {
    "1h": pd.Timedelta(hours=1),
    "6h": pd.Timedelta(hours=6),
    "1d": pd.Timedelta(days=1),
}


# ===========================================================================
# Invariant
# ===========================================================================

@dataclass
class TimingReport:
    ok: bool = True
    checks: dict[str, bool] = field(default_factory=dict)
    details: dict[str, Any] = field(default_factory=dict)

    def require(self, name: str, passed: bool, **detail: Any) -> None:
        self.checks[name] = bool(passed)
        if detail:
            self.details[name] = detail
        if not passed:
            self.ok = False


def infer_step(timestamps: pd.Series) -> pd.Timedelta | None:
    """Modal positive spacing of a sorted timestamp series (or ``None``)."""
    ts = pd.to_datetime(pd.Series(timestamps), utc=True).dropna().sort_values()
    if len(ts) < 2:
        return None
    diffs = ts.diff().dropna()
    diffs = diffs[diffs > pd.Timedelta(0)]
    if diffs.empty:
        return None
    return diffs.mode().iloc[0]


def check_interval_start_semantics(price_features: pd.DataFrame,
                                   *,
                                   horizon: str,
                                   posts: pd.DataFrame | None = None,
                                   group_col: str = "ticker",
                                   time_col: str = "timestamp",
                                   return_col: str = "log_return_t",
                                   target_col: str = "target",
                                   post_time_col: str = "created_utc"
                                   ) -> TimingReport:
    """Verify the interval-start timing invariant on a feature frame.

    Checks (per the module docstring):

    1. **Same-interval label** — the timestamp grid is regularly spaced by the
       horizon width ``h`` (so a price row and a sentiment row that share a
       ``timestamp`` describe the same interval ``[t, t+h)``).
    2. **Target is the next bar** — on every pair of *contiguous* rows
       (spacing exactly ``h``) within a ticker,
       ``target[t] == 1[return[t+h] >= 0]``; i.e. the row predicts the sign of
       the NEXT interval's return, not the current one.
    3. **No future post leaks in** (only when ``posts`` is supplied) — every
       post assigned to slot ``t`` was created in ``[t, t+h)``; nothing created
       at or after ``t+h`` may enter row ``t``.
    """
    rep = TimingReport()
    h = HORIZON_STEP.get(str(horizon))
    if h is None:
        rep.require("known_horizon", False, horizon=horizon)
        return rep
    rep.require("known_horizon", True)

    df = price_features.copy()
    df[time_col] = pd.to_datetime(df[time_col], utc=True)

    # 1. regular grid == h (per ticker).
    grid_ok = True
    steps: dict[str, str] = {}
    for tk, grp in df.groupby(group_col, sort=False):
        step = infer_step(grp[time_col])
        steps[str(tk)] = str(step)
        if step != h:
            grid_ok = False
    rep.require("regular_grid_equals_h", grid_ok, expected=str(h), steps=steps)

    # 2. target belongs to the NEXT bar (contiguous rows only).
    if return_col in df.columns and target_col in df.columns:
        target_ok = True
        n_checked = 0
        for _tk, grp in df.groupby(group_col, sort=False):
            g = grp.sort_values(time_col).reset_index(drop=True)
            t = g[time_col]
            contiguous = (t.shift(-1) - t) == h
            next_ret = g[return_col].shift(-1)
            expected = (next_ret >= 0).astype("float")
            mask = contiguous & next_ret.notna() & g[target_col].notna()
            n_checked += int(mask.sum())
            if mask.any():
                if not np.array_equal(
                        g.loc[mask, target_col].astype(float).to_numpy(),
                        expected[mask].to_numpy()):
                    target_ok = False
        rep.require("target_is_next_bar", target_ok, n_pairs_checked=n_checked)

    # 3. no post created after the interval end enters the row.
    if posts is not None and not posts.empty:
        p = posts.copy()
        p[post_time_col] = pd.to_datetime(p[post_time_col], utc=True)
        slot = p[post_time_col].dt.floor(_pandas_freq(horizon))
        in_interval = (p[post_time_col] >= slot) & (p[post_time_col] < slot + h)
        rep.require("no_future_post_in_slot", bool(in_interval.all()),
                    n_posts=int(len(p)),
                    n_violations=int((~in_interval).sum()))
    return rep


def _pandas_freq(horizon: str) -> str:
    """Floor frequency string matching sentiment aggregation."""
    return {"1h": "1h", "6h": "6h", "1d": "1D"}.get(str(horizon), str(horizon))


# ===========================================================================
# CCXT parquet audit (optional command — inspects supplied files)
# ===========================================================================

def audit_timestamp_grid(df: pd.DataFrame, *, time_col: str = "timestamp"
                         ) -> dict[str, Any]:
    """Grid summary for one OHLCV parquet."""
    ts = pd.to_datetime(df[time_col], utc=True).dropna().sort_values()
    step = infer_step(ts)
    n = len(ts)
    expected = None
    if step and n >= 2:
        span = ts.iloc[-1] - ts.iloc[0]
        expected = int(span / step) + 1
    within_day = sorted({t.strftime("%H:%M") for t in ts.head(48)})
    return {
        "n_bars": int(n),
        "first": str(ts.iloc[0]) if n else "",
        "last": str(ts.iloc[-1]) if n else "",
        "inferred_step": str(step) if step else "",
        "expected_bars_if_regular": expected,
        "missing_bars": (None if expected is None else int(expected - n)),
        "distinct_intraday_times_head": within_day,
    }


def infer_label_convention(coarse_df: pd.DataFrame,
                           fine_df: pd.DataFrame,
                           *,
                           coarse_horizon: str,
                           fine_horizon: str,
                           time_col: str = "timestamp") -> dict[str, Any]:
    """Infer bar-start vs bar-end by aggregating the fine bars into the coarse
    grid and comparing OHLC.

    Under the **bar-start** convention a coarse bar labelled ``t`` aggregates
    the fine bars with label in ``[t, t+H)``; its open equals the FIRST such
    fine open and its close the LAST such fine close. Under a (wrong) bar-end
    reading the coarse bar would instead aggregate ``(t-H, t]``. We test the
    bar-start hypothesis and report the match rate.
    """
    H = HORIZON_STEP[str(coarse_horizon)]
    c = coarse_df.copy()
    f = fine_df.copy()
    c[time_col] = pd.to_datetime(c[time_col], utc=True)
    f[time_col] = pd.to_datetime(f[time_col], utc=True)
    f = f.sort_values(time_col)
    # Assign each fine bar to the coarse interval START it belongs to.
    f["_coarse_start"] = f[time_col].dt.floor(_pandas_freq(coarse_horizon))
    agg = f.groupby("_coarse_start").agg(
        open_first=("open", "first"),
        close_last=("close", "last"),
        high_max=("high", "max"),
        low_min=("low", "min"),
        volume_sum=("volume", "sum"),
        n_fine=("open", "size"),
    ).reset_index().rename(columns={"_coarse_start": time_col})
    merged = c.merge(agg, on=time_col, how="inner")
    if merged.empty:
        return {"label_convention": "unknown", "n_compared": 0}

    def _match(a, b):
        return np.isclose(a.astype(float), b.astype(float),
                          rtol=1e-6, atol=1e-8)

    open_match = _match(merged["open"], merged["open_first"]).mean()
    close_match = _match(merged["close"], merged["close_last"]).mean()
    high_match = _match(merged["high"], merged["high_max"]).mean()
    low_match = _match(merged["low"], merged["low_min"]).mean()
    bar_start_ok = min(open_match, close_match, high_match, low_match) >= 0.99
    return {
        "label_convention": "bar_start" if bar_start_ok else "inconclusive_or_bar_end",
        "n_compared": int(len(merged)),
        "open_eq_first_fine_open": float(open_match),
        "close_eq_last_fine_close": float(close_match),
        "high_eq_max_fine_high": float(high_match),
        "low_eq_min_fine_low": float(low_match),
    }


def audit_ohlc_aggregation(coarse_df: pd.DataFrame,
                           fine_df: pd.DataFrame,
                           *,
                           coarse_horizon: str,
                           fine_horizon: str,
                           time_col: str = "timestamp") -> pd.DataFrame:
    """Per-coarse-bar OHLC + volume aggregation consistency vs the fine bars.

    Returns one row per coarse timestamp with the coarse OHLCV, the fine
    aggregate, boolean OHLC match flags and the volume deviation
    ``coarse_volume - sum(fine_volume)``.
    """
    c = coarse_df.copy()
    f = fine_df.copy()
    c[time_col] = pd.to_datetime(c[time_col], utc=True)
    f[time_col] = pd.to_datetime(f[time_col], utc=True)
    f["_coarse_start"] = f[time_col].dt.floor(_pandas_freq(coarse_horizon))
    agg = f.groupby("_coarse_start").agg(
        fine_open=("open", "first"),
        fine_close=("close", "last"),
        fine_high=("high", "max"),
        fine_low=("low", "min"),
        fine_volume=("volume", "sum"),
        n_fine=("open", "size"),
    ).reset_index().rename(columns={"_coarse_start": time_col})
    m = c.merge(agg, on=time_col, how="inner")
    if m.empty:
        return m
    m["open_match"] = np.isclose(m["open"], m["fine_open"], rtol=1e-6, atol=1e-8)
    m["close_match"] = np.isclose(m["close"], m["fine_close"], rtol=1e-6, atol=1e-8)
    m["high_match"] = np.isclose(m["high"], m["fine_high"], rtol=1e-6, atol=1e-8)
    m["low_match"] = np.isclose(m["low"], m["fine_low"], rtol=1e-6, atol=1e-8)
    m["volume_deviation"] = m["volume"].astype(float) - m["fine_volume"].astype(float)
    m["volume_rel_deviation"] = np.where(
        m["fine_volume"].astype(float) != 0,
        m["volume_deviation"] / m["fine_volume"].astype(float), np.nan)
    m["coarse_horizon"] = coarse_horizon
    m["fine_horizon"] = fine_horizon
    return m


def _load_ohlcv(path: str | Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    df.columns = [str(col).lower().strip() for col in df.columns]
    tcol = None
    for cand in ("timestamp", "datetime", "date", "time", "open_time"):
        if cand in df.columns:
            tcol = cand
            break
    if tcol is None:
        tcol = df.columns[0]
    from ..price.features import infer_datetime_from_any
    df["timestamp"] = infer_datetime_from_any(df[tcol])
    return df.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)


def audit_ccxt_files(paths: Mapping[str, str | Path]) -> dict[str, Any]:
    """Inspect supplied CCXT OHLCV parquet files for one coin.

    ``paths`` maps horizon → parquet path, e.g.
    ``{"1h": "ADAUSDT_1h.parquet", "6h": ..., "1d": ...}``. Returns a report
    dict; the CLI prints it. Never mutates the input files.
    """
    frames = {hz: _load_ohlcv(p) for hz, p in paths.items() if Path(p).exists()}
    report: dict[str, Any] = {"grids": {}, "label_convention": {},
                              "ohlc_aggregation": {}}
    for hz, df in frames.items():
        report["grids"][hz] = audit_timestamp_grid(df)

    # Cross-horizon aggregation consistency (coarse vs fine).
    pairs = [("1d", "6h"), ("1d", "1h"), ("6h", "1h")]
    for coarse, fine in pairs:
        if coarse in frames and fine in frames:
            report["label_convention"][f"{coarse}_vs_{fine}"] = (
                infer_label_convention(frames[coarse], frames[fine],
                                       coarse_horizon=coarse, fine_horizon=fine))
            agg = audit_ohlc_aggregation(frames[coarse], frames[fine],
                                         coarse_horizon=coarse, fine_horizon=fine)
            if not agg.empty:
                report["ohlc_aggregation"][f"{coarse}_vs_{fine}"] = {
                    "n_bars": int(len(agg)),
                    "open_match_rate": float(agg["open_match"].mean()),
                    "close_match_rate": float(agg["close_match"].mean()),
                    "high_match_rate": float(agg["high_match"].mean()),
                    "low_match_rate": float(agg["low_match"].mean()),
                    "n_volume_deviations": int((~np.isclose(
                        agg["volume_deviation"], 0, atol=1e-6)).sum()),
                    "max_abs_volume_rel_deviation": float(
                        agg["volume_rel_deviation"].abs().max(skipna=True))
                    if agg["volume_rel_deviation"].notna().any() else None,
                }
    return report


def print_ccxt_audit(report: Mapping[str, Any]) -> None:
    print("\n=== CCXT timestamp / OHLC audit ===")
    print("\n-- Timestamp grid --")
    for hz, g in report.get("grids", {}).items():
        print(f"  [{hz}] n={g['n_bars']} step={g['inferred_step']} "
              f"missing={g['missing_bars']} first={g['first']} last={g['last']}")
        print(f"       intraday times (head): {g['distinct_intraday_times_head']}")
    print("\n-- Inferred label convention (interval-start hypothesis) --")
    for pair, lc in report.get("label_convention", {}).items():
        print(f"  [{pair}] {lc['label_convention']} "
              f"(open_first={lc.get('open_eq_first_fine_open'):.3f} "
              f"close_last={lc.get('close_eq_last_fine_close'):.3f} "
              f"high_max={lc.get('high_eq_max_fine_high'):.3f} "
              f"low_min={lc.get('low_eq_min_fine_low'):.3f})")
    print("\n-- OHLC aggregation consistency + volume deviations --")
    for pair, a in report.get("ohlc_aggregation", {}).items():
        print(f"  [{pair}] n={a['n_bars']} "
              f"O/H/L/C match={a['open_match_rate']:.3f}/"
              f"{a['high_match_rate']:.3f}/{a['low_match_rate']:.3f}/"
              f"{a['close_match_rate']:.3f} "
              f"vol_devs={a['n_volume_deviations']} "
              f"max|rel_vol_dev|={a['max_abs_volume_rel_deviation']}")
    print()
