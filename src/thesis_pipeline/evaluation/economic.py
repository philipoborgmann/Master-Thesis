"""Economic evaluation / backtest module.

Strictly post-hoc: consumes the existing signal parquets under
``Outputs/Signals/<horizon>/`` plus daily OHLC under
``Data/Raw/Price/1d/`` (and the matching intraday folders for 1h / 6h)
and produces thesis-grade backtest metrics. No model is re-trained.

Design — small typed helpers, no monolith:

* :func:`load_close_series`        — single-ticker close series, UTC midnight
* :func:`load_forward_returns`     — per-(ticker, timestamp) next-period log return
* :func:`build_positions`          — ±1 / 0 positions from signals (with optional threshold band)
* :func:`compute_turnover`         — half abs-diff per ticker
* :func:`apply_transaction_costs`  — net return = gross − turnover × cost
* :func:`compute_strategy_returns` — position × forward return
* :func:`compute_drawdown`         — running max / drawdown
* :func:`compute_sharpe`           — annualised Sharpe (configurable rf)
* :func:`compute_sortino`          — annualised Sortino
* :func:`summarize_backtest`       — pooled metrics per horizon × set × sentiment
* :func:`summarize_threshold_backtest` — pooled metrics per threshold
* :func:`summarize_backtest_by_ticker` — per-ticker breakdown
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from ..config import load_config, resolve_path
from ..logging_utils import get_logger
from .metrics import GROUP_KEYS, _group_meta
from .volatility import _ticker_candidates, _normalise_ohlcv


# ---------------------------------------------------------------------------
# Forward returns
# ---------------------------------------------------------------------------

def load_close_series(ticker: str, horizon: str = "1d") -> pd.DataFrame | None:
    """Return per-bar close series for one ticker, or ``None`` on miss.

    Output columns: ``timestamp`` (datetime64[ns, UTC]) and ``close``
    (float). For 1h / 6h horizons the file is loaded from the matching
    ``Data/Raw/Price/<horizon>/`` directory; OHLC normalisation reuses
    :func:`volatility._normalise_ohlcv`.
    """
    logger = get_logger()
    try:
        root = resolve_path(f"raw_price_{horizon}")
    except KeyError:
        logger.warning("economic: unknown horizon %s — close series unavailable", horizon)
        return None
    suffixes = {"1d": "1d", "1h": "1h", "6h": "6h", "15min": "15m", "4h": "4h"}
    suffix = suffixes.get(horizon, horizon)
    for cand in _ticker_candidates(ticker):
        path = Path(root) / f"{cand}USDT_{suffix}.parquet"
        if path.exists():
            try:
                raw = pd.read_parquet(path)
            except Exception as exc:  # noqa: BLE001
                logger.warning("economic: cannot read %s (%s)", path, exc)
                return None
            df = _normalise_ohlcv(raw)
            if df.empty:
                return None
            out = pd.DataFrame({"timestamp": df["date"].values,
                                 "close":     df["close"].astype(float).values})
            return out.sort_values("timestamp").reset_index(drop=True)
    return None


def load_forward_returns(tickers: Iterable[str],
                         horizon: str = "1d") -> pd.DataFrame:
    """Long-form ``(ticker, timestamp, forward_log_return)``.

    ``forward_log_return[t] = log(close[t+1] / close[t])`` — the realised
    return between bar t and bar t+1. The last bar has NaN by definition.
    """
    frames: list[pd.DataFrame] = []
    for t in sorted(set(str(x).upper() for x in tickers)):
        ser = load_close_series(t, horizon=horizon)
        if ser is None or ser.empty:
            continue
        ser = ser.copy()
        ser["forward_log_return"] = np.log(ser["close"] / ser["close"].shift(1)).shift(-1)
        ser["ticker"] = t
        frames.append(ser[["ticker", "timestamp", "forward_log_return"]])
    if not frames:
        return pd.DataFrame(columns=["ticker", "timestamp", "forward_log_return"])
    out = pd.concat(frames, ignore_index=True)
    # Drop the trailing NaN bar per ticker — it carries no realised return.
    return out.dropna(subset=["forward_log_return"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Position logic
# ---------------------------------------------------------------------------

def build_positions(df: pd.DataFrame, *,
                    threshold: float = 0.5,
                    flat_middle_zone: bool = True,
                    long_short_enabled: bool = True) -> pd.Series:
    """Return a ±1 / 0 position series aligned with ``df``.

    * ``threshold == 0.5``        →  long iff prediction == 1; short (or flat)
                                    otherwise. Every observation is traded
                                    (no neutral band).
    * ``threshold > 0.5``         →  long when prob > threshold;
                                    short when prob < (1 - threshold);
                                    flat in the middle band when
                                    ``flat_middle_zone`` is True.
    * ``long_short_enabled=False``→  shorts collapse to flat (0).
    """
    if df.empty:
        return pd.Series([], dtype=int)
    prob = df["probability"].astype(float).values
    pred = df["prediction"].astype(int).values

    pos = np.zeros(len(df), dtype=int)
    if threshold <= 0.5 + 1e-12:
        # Default rule: trade every observation.
        pos[pred == 1] = 1
        pos[pred == 0] = -1
    else:
        long_mask  = prob > threshold
        short_mask = prob < (1.0 - threshold)
        pos[long_mask]  =  1
        pos[short_mask] = -1
        if not flat_middle_zone:
            # Fall back to sign-of-prediction outside the band — coverage = 1.
            middle = ~(long_mask | short_mask)
            pos[middle & (pred == 1)] =  1
            pos[middle & (pred == 0)] = -1

    if not long_short_enabled:
        pos = np.where(pos < 0, 0, pos)
    return pd.Series(pos, index=df.index, dtype=int)


def compute_turnover(positions: pd.Series, *,
                     method: str = "half_abs_diff") -> pd.Series:
    """Turnover per step.

    ``half_abs_diff``: ``abs(pos_t - pos_{t-1}) / 2``. A +1 → -1 switch
    counts as a full round-trip (1.0); +1 → 0 or 0 → +1 counts as 0.5.
    """
    if positions.empty:
        return pd.Series([], dtype=float)
    if method != "half_abs_diff":
        raise ValueError(f"Unknown turnover method: {method}")
    diff = positions.astype(float).diff().abs() / 2.0
    diff.iloc[0] = float(abs(positions.iloc[0])) / 2.0  # initial entry
    return diff.fillna(0.0)


def apply_transaction_costs(gross_return: pd.Series,
                            turnover: pd.Series,
                            cost_bps: float) -> pd.Series:
    """Net return = gross_return − turnover × (cost_bps / 1e4)."""
    cost = float(cost_bps) / 1e4
    return gross_return - turnover * cost


def compute_strategy_returns(positions: pd.Series,
                              forward_returns: pd.Series) -> pd.Series:
    return positions.astype(float) * forward_returns.astype(float)


# ---------------------------------------------------------------------------
# Risk + return metrics
# ---------------------------------------------------------------------------

def compute_drawdown(returns: pd.Series) -> tuple[pd.Series, float]:
    """Return (drawdown series, maximum drawdown) on log-returns input."""
    if returns.empty:
        return pd.Series([], dtype=float), float("nan")
    cum = returns.fillna(0).cumsum()
    peak = cum.cummax()
    dd = cum - peak  # log-difference drawdown
    return dd, float(dd.min())


def _annualisation_factor(n_obs: int, days_spanned: float,
                          calendar_days: int) -> float:
    """Bars-per-year approximation: ``n_obs / days_spanned * calendar_days``."""
    if days_spanned <= 0 or n_obs <= 0:
        return float("nan")
    return float(n_obs) / float(days_spanned) * float(calendar_days)


def compute_sharpe(returns: pd.Series, *,
                   periods_per_year: float,
                   risk_free_rate: float = 0.0) -> float:
    if returns.empty:
        return float("nan")
    r = returns.dropna()
    if len(r) < 2 or r.std(ddof=1) == 0 or np.isnan(periods_per_year):
        return float("nan")
    excess = r - (risk_free_rate / periods_per_year)
    return float(excess.mean() / r.std(ddof=1) * np.sqrt(periods_per_year))


def compute_sortino(returns: pd.Series, *,
                    periods_per_year: float,
                    risk_free_rate: float = 0.0) -> float:
    if returns.empty:
        return float("nan")
    r = returns.dropna()
    if len(r) < 2 or np.isnan(periods_per_year):
        return float("nan")
    excess = r - (risk_free_rate / periods_per_year)
    downside = excess[excess < 0]
    if downside.empty or downside.std(ddof=1) == 0:
        return float("nan")
    return float(excess.mean() / downside.std(ddof=1) * np.sqrt(periods_per_year))


def _days_spanned(timestamps: pd.Series) -> float:
    if timestamps.empty:
        return 0.0
    ts = pd.to_datetime(timestamps, utc=True, errors="coerce").dropna()
    if len(ts) < 2:
        return 0.0
    return (ts.max() - ts.min()).total_seconds() / 86_400.0


# ---------------------------------------------------------------------------
# Pooled-row helpers
# ---------------------------------------------------------------------------

def _backtest_row(positions: pd.Series,
                  forward_returns: pd.Series,
                  timestamps: pd.Series,
                  cost_bps: float,
                  cfg: dict) -> dict:
    """Compute one row of backtest metrics for a single position series.

    All arithmetic happens on plain numpy arrays so that any index drift
    between the input Series (positions / forward_returns / timestamps)
    is irrelevant — we trust the caller to align them by row order.
    """
    calendar = float(cfg.get("crypto_calendar_days", 365))
    rf       = float(cfg.get("risk_free_rate", 0.0))
    n_obs = int(len(positions))
    if n_obs == 0:
        return {"cost_bps": cost_bps, "n_obs": 0, "n_trades": 0,
                "turnover": 0.0, "coverage": 0.0,
                "mean_return": np.nan, "median_return": np.nan,
                "cumulative_return": np.nan,
                "annualized_return": np.nan, "annualized_volatility": np.nan,
                "sharpe": np.nan, "sortino": np.nan,
                "max_drawdown": np.nan, "calmar_ratio": np.nan,
                "hit_rate": np.nan, "long_hit_rate": np.nan, "short_hit_rate": np.nan}

    pos_arr  = np.asarray(positions, dtype=float)
    fwd_arr  = np.asarray(forward_returns, dtype=float)
    if len(pos_arr) != len(fwd_arr):
        get_logger().warning(
            "economic: position/return length mismatch (%d vs %d) — skipping row",
            len(pos_arr), len(fwd_arr),
        )
        return _backtest_row(positions.iloc[:0], forward_returns.iloc[:0],
                              timestamps.iloc[:0], cost_bps, cfg)

    gross_arr = pos_arr * fwd_arr
    # Turnover: half abs-diff per step; first step is treated as initial entry.
    diff = np.abs(np.diff(pos_arr, prepend=0.0)) / 2.0
    to_arr   = diff
    cost     = float(cost_bps) / 1e4
    net_arr  = gross_arr - to_arr * cost

    net = pd.Series(net_arr)
    spanned = _days_spanned(timestamps)
    ppy     = _annualisation_factor(n_obs, spanned, int(calendar))

    cum     = float(np.nansum(net_arr))
    mean_r  = float(np.nanmean(net_arr)) if n_obs else float("nan")
    median_r = float(np.nanmedian(net_arr)) if n_obs else float("nan")
    vol_a   = float(np.nanstd(net_arr, ddof=1) * np.sqrt(ppy)) \
              if (n_obs > 1 and not np.isnan(ppy)) else float("nan")
    ann_r   = float(mean_r * ppy) if not np.isnan(ppy) else float("nan")

    sharpe   = compute_sharpe(net,  periods_per_year=ppy, risk_free_rate=rf)
    sortino  = compute_sortino(net, periods_per_year=ppy, risk_free_rate=rf)

    _, max_dd = compute_drawdown(net)
    calmar = float(ann_r / abs(max_dd)) if (max_dd is not None
                                            and not np.isnan(max_dd)
                                            and max_dd < 0) else float("nan")

    traded_mask = pos_arr != 0
    n_trades    = int(traded_mask.sum())
    coverage    = float(traded_mask.mean()) if n_obs else 0.0

    sign_match = np.sign(fwd_arr) == np.sign(pos_arr)
    hits       = int((sign_match & traded_mask).sum())
    hit_rate   = float(hits / n_trades) if n_trades else float("nan")

    long_mask  = pos_arr ==  1.0
    short_mask = pos_arr == -1.0
    long_hit   = float(((fwd_arr > 0) & long_mask).sum() / max(long_mask.sum(), 1)) \
                 if long_mask.any() else float("nan")
    short_hit  = float(((fwd_arr < 0) & short_mask).sum() / max(short_mask.sum(), 1)) \
                 if short_mask.any() else float("nan")

    return {
        "cost_bps": cost_bps,
        "n_obs":    n_obs,
        "n_trades": n_trades,
        "turnover": float(to_arr.mean()) if n_obs else 0.0,
        "coverage": coverage,
        "mean_return":          mean_r,
        "median_return":        median_r,
        "cumulative_return":    cum,
        "annualized_return":    ann_r,
        "annualized_volatility": vol_a,
        "sharpe":               sharpe,
        "sortino":              sortino,
        "max_drawdown":         max_dd,
        "calmar_ratio":         calmar,
        "hit_rate":             hit_rate,
        "long_hit_rate":        long_hit,
        "short_hit_rate":       short_hit,
    }


def _signals_with_forward_returns(signals: pd.DataFrame,
                                   forward_returns: pd.DataFrame) -> pd.DataFrame:
    """Left-merge per-ticker forward returns onto every signal row."""
    if signals.empty or forward_returns.empty:
        return pd.DataFrame()
    sig = signals.copy()
    sig["timestamp"] = pd.to_datetime(sig["timestamp"], utc=True, errors="coerce")
    sig["ticker"]    = sig["ticker"].astype(str).str.upper()

    fr = forward_returns.copy()
    fr["timestamp"] = pd.to_datetime(fr["timestamp"], utc=True, errors="coerce")
    fr["ticker"]    = fr["ticker"].astype(str).str.upper()
    # For daily horizons the signal timestamp has h/m/s set to whatever the
    # model produced; forward returns are normalised to midnight UTC. Align
    # signals onto the day so the join is robust.
    sig["__join_ts__"] = sig["timestamp"].dt.normalize()
    fr["__join_ts__"]  = fr["timestamp"].dt.normalize()
    merged = sig.merge(
        fr[["ticker", "__join_ts__", "forward_log_return"]],
        on=["ticker", "__join_ts__"], how="left",
    )
    return merged.drop(columns="__join_ts__")


# ---------------------------------------------------------------------------
# Public summarizers (consumed by evaluate_signals.py)
# ---------------------------------------------------------------------------

def summarize_backtest(signals: pd.DataFrame,
                       forward_returns: pd.DataFrame,
                       cfg: dict) -> pd.DataFrame:
    """One row per (horizon × set × sentiment × cost_bps).

    Position rule is the default (threshold = 0.5). Threshold variants live
    in :func:`summarize_threshold_backtest`.
    """
    if signals.empty:
        return pd.DataFrame()
    merged = _signals_with_forward_returns(signals, forward_returns)
    if merged.empty or merged["forward_log_return"].notna().sum() == 0:
        get_logger().warning(
            "economic: no forward-return matches for any signal — "
            "backtest will produce empty tables.")
        return pd.DataFrame()

    cost_grid = list(cfg.get("transaction_cost_bps", [0, 5, 10]))
    long_short = bool(cfg.get("long_short_enabled", True))

    rows: list[dict] = []
    for keys, grp in merged.groupby(list(GROUP_KEYS), dropna=False):
        meta = _group_meta(grp)
        valid = grp.dropna(subset=["forward_log_return", "probability"]).copy()
        if valid.empty:
            continue
        valid = valid.sort_values(["ticker", "timestamp"]).reset_index(drop=True)
        positions = build_positions(valid, threshold=0.5,
                                     long_short_enabled=long_short)
        fwd = valid["forward_log_return"].astype(float).reset_index(drop=True)
        ts  = valid["timestamp"].reset_index(drop=True)
        for cost in cost_grid:
            row = _backtest_row(positions, fwd, ts, float(cost), cfg)
            row.update({
                "horizon":          keys[0],
                "set_id":           keys[1],
                "sentiment_model":  keys[2],
                **meta,
            })
            rows.append(row)
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    front = ["horizon", "set_id", "category", "sentiment_model", "label",
             "cost_bps", "cumulative_return", "annualized_return",
             "annualized_volatility", "sharpe", "sortino", "max_drawdown",
             "calmar_ratio", "hit_rate", "long_hit_rate", "short_hit_rate",
             "turnover", "coverage", "n_trades", "n_obs",
             "mean_return", "median_return"]
    rest = [c for c in out.columns if c not in front]
    return out[front + rest].reset_index(drop=True)


def summarize_threshold_backtest(signals: pd.DataFrame,
                                  forward_returns: pd.DataFrame,
                                  cfg: dict) -> pd.DataFrame:
    """One row per (horizon × set × sentiment × threshold × cost_bps).

    Implements the conviction-band logic with ``flat_middle_zone`` from the
    config — non-traded observations contribute 0 to the realised return.
    """
    if signals.empty:
        return pd.DataFrame()
    merged = _signals_with_forward_returns(signals, forward_returns)
    if merged.empty:
        return pd.DataFrame()

    thresholds = list(cfg.get("thresholds", [0.55, 0.60, 0.65]))
    cost_grid  = list(cfg.get("transaction_cost_bps", [0, 5, 10]))
    flat_mid   = bool(cfg.get("flat_middle_zone", True))
    long_short = bool(cfg.get("long_short_enabled", True))

    rows: list[dict] = []
    for keys, grp in merged.groupby(list(GROUP_KEYS), dropna=False):
        meta = _group_meta(grp)
        valid = grp.dropna(subset=["forward_log_return", "probability"]).copy()
        if valid.empty:
            continue
        valid = valid.sort_values(["ticker", "timestamp"]).reset_index(drop=True)
        fwd = valid["forward_log_return"].astype(float).reset_index(drop=True)
        ts  = valid["timestamp"].reset_index(drop=True)
        for thr in thresholds:
            positions = build_positions(valid, threshold=float(thr),
                                         flat_middle_zone=flat_mid,
                                         long_short_enabled=long_short)
            for cost in cost_grid:
                row = _backtest_row(positions, fwd, ts, float(cost), cfg)
                row.update({
                    "horizon":          keys[0],
                    "set_id":           keys[1],
                    "sentiment_model":  keys[2],
                    "threshold":        float(thr),
                    **meta,
                })
                rows.append(row)
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    front = ["horizon", "set_id", "category", "sentiment_model", "label",
             "threshold", "cost_bps", "cumulative_return", "sharpe",
             "sortino", "max_drawdown", "turnover", "coverage", "n_trades",
             "hit_rate"]
    rest = [c for c in out.columns if c not in front]
    return out[front + rest].reset_index(drop=True)


def summarize_backtest_by_ticker(signals: pd.DataFrame,
                                 forward_returns: pd.DataFrame,
                                 cfg: dict) -> pd.DataFrame:
    """One row per (horizon × set × sentiment × ticker × cost_bps)."""
    if signals.empty:
        return pd.DataFrame()
    merged = _signals_with_forward_returns(signals, forward_returns)
    if merged.empty:
        return pd.DataFrame()

    cost_grid  = list(cfg.get("transaction_cost_bps", [0, 5, 10]))
    long_short = bool(cfg.get("long_short_enabled", True))

    rows: list[dict] = []
    for keys, grp in merged.groupby(list(GROUP_KEYS) + ["ticker"], dropna=False):
        valid = grp.dropna(subset=["forward_log_return", "probability"]).copy()
        if valid.empty:
            continue
        valid = valid.sort_values("timestamp").reset_index(drop=True)
        positions = build_positions(valid, threshold=0.5,
                                     long_short_enabled=long_short)
        fwd = valid["forward_log_return"].astype(float).reset_index(drop=True)
        ts  = valid["timestamp"].reset_index(drop=True)
        for cost in cost_grid:
            row = _backtest_row(positions, fwd, ts, float(cost), cfg)
            row.update({
                "horizon":         keys[0],
                "set_id":          keys[1],
                "sentiment_model": keys[2],
                "ticker":          keys[3],
            })
            rows.append(row)
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    front = ["horizon", "set_id", "sentiment_model", "ticker",
             "cost_bps", "cumulative_return", "sharpe", "sortino",
             "max_drawdown", "turnover", "coverage", "n_trades", "hit_rate"]
    rest = [c for c in out.columns if c not in front]
    return out[front + rest].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Buy-and-hold benchmark
# ---------------------------------------------------------------------------

def buy_and_hold_benchmark(tickers: Iterable[str],
                           forward_returns: pd.DataFrame,
                           cfg: dict,
                           horizon: str) -> dict | None:
    """Equal-weight long-only buy-and-hold reference.

    For each (timestamp), average per-ticker forward returns; that becomes
    the daily portfolio return. Returns a single dict mirroring
    :func:`_backtest_row` so it merges into the same table.
    """
    if forward_returns.empty:
        return None
    universe = sorted(set(str(t).upper() for t in tickers))
    fr = forward_returns[forward_returns["ticker"].isin(universe)]
    if fr.empty:
        return None
    daily = (fr.groupby("timestamp")["forward_log_return"]
               .mean().sort_index())
    pos   = pd.Series(1, index=daily.index, dtype=int)
    ts    = pd.Series(daily.index)
    row = _backtest_row(pos, daily.reset_index(drop=True), ts,
                        cost_bps=0.0, cfg=cfg)
    row.update({"horizon": horizon, "set_id": "BUY_HOLD",
                "sentiment_model": "-", "category": "benchmark",
                "label": "equal-weight buy & hold"})
    return row


# ---------------------------------------------------------------------------
# Config helper
# ---------------------------------------------------------------------------

def load_backtest_config(path: str | Path | None = None) -> dict:
    """Return the backtest config as a plain dict, with sensible defaults."""
    defaults = {
        "annualization_mode":  "crypto",
        "crypto_calendar_days": 365,
        "transaction_cost_bps": [0, 5, 10],
        "thresholds":          [0.55, 0.60, 0.65],
        "turnover_method":     "half_abs_diff",
        "long_short_enabled":  True,
        "flat_middle_zone":    True,
        "benchmark_ids":       ["B1", "B2"],
        "risk_free_rate":      0.0,
        "include_buy_and_hold": True,
    }
    user: dict = {}
    if path is not None:
        p = Path(path)
        if p.exists():
            import yaml
            with p.open("r", encoding="utf-8") as fh:
                user = yaml.safe_load(fh) or {}
        else:
            get_logger().warning(
                "economic: backtest config %s not found — using defaults", p,
            )
    else:
        try:
            user = load_config("backtest")
        except FileNotFoundError:
            user = {}
    defaults.update(user)
    return defaults
