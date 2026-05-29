"""Economic evaluation / backtest module — cross-sectional High-Low portfolio.

This rewrite fixes the methodological flaws of the previous observation-level
implementation:

* **Pooled annualisation no longer over-counts.** Annualisation is built
  from the number of *portfolio timestamps*, not from ``n_tickers × n_days``.
* **Cumulative return is computed on portfolio returns**, not by summing
  individual ticker observations.
* **Turnover is portfolio-level**: ``0.5 · Σ_i |w_{t,i} − w_{t-1,i}|``,
  so a long↔short flip of a single coin pays its real cost and additions
  / removals from the legs are tracked through their weight changes.
* **Intraday joins** use the *exact* timestamp (1h / 6h); only the 1d
  horizon collapses to ``normalize()``-d midnight UTC. Many-to-many merges
  are guarded against.
* **Drawdown** is the percentage drawdown on the equity curve
  ``equity = exp(cum_log_return)``, *not* a log-difference drawdown.
* **Returns are log returns throughout.** The output table exposes both
  ``annualized_log_return`` and ``annualized_simple_return = expm1(...)``.
* **Buy-and-hold** is generated for every cost level (was only cost=0).
* **Benchmark lifts** vs B1 / B2 are appended where the benchmark rows
  are available.

The portfolio is **equal-weight long-short, dollar-neutral approximated**:
``long_weights.sum() = +1``, ``short_weights.sum() = -1``, ``gross = 2``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from ..config import load_config, resolve_path
from ..logging_utils import get_logger
from .metrics import GROUP_KEYS, FRONT_META, _front_order, _group_meta, ensure_group_columns
from .volatility import _ticker_candidates, _normalise_ohlcv


# ---------------------------------------------------------------------------
# Forward-return loaders
# ---------------------------------------------------------------------------

def _ohlcv_suffix(horizon: str) -> str:
    return {"1d": "1d", "1h": "1h", "6h": "6h", "15min": "15m", "4h": "4h"}.get(horizon, horizon)


def load_close_series(ticker: str, horizon: str = "1d") -> pd.DataFrame | None:
    """Return per-bar close series for one ticker, or ``None`` on miss.

    Output columns: ``timestamp`` (datetime64[ns, UTC]) and ``close``.
    Crucially, **intraday timestamps are NOT normalised to midnight** —
    that would collapse every 1h bar of a day onto the same key and break
    the join.
    """
    logger = get_logger()
    try:
        root = resolve_path(f"raw_price_{horizon}")
    except KeyError:
        logger.warning("economic: unknown horizon %s — close series unavailable", horizon)
        return None
    suffix = _ohlcv_suffix(horizon)
    for cand in _ticker_candidates(ticker):
        path = Path(root) / f"{cand}USDT_{suffix}.parquet"
        if not path.exists():
            continue
        try:
            raw = pd.read_parquet(path)
        except Exception as exc:  # noqa: BLE001
            logger.warning("economic: cannot read %s (%s)", path, exc)
            return None
        # _normalise_ohlcv writes a ``date`` column normalised to midnight
        # (suitable for the volatility module). For the economic loader we
        # need the *original* timestamp granularity, so we re-derive it from
        # the raw frame here.
        normalised = _normalise_ohlcv(raw)
        if normalised.empty:
            return None
        # Recover a non-normalised timestamp from the raw frame.
        ts_col = _find_ts_column(raw)
        if ts_col is None:
            logger.warning("economic: %s has no timestamp column — skipping", path)
            return None
        raw_ts = _to_utc_timestamp(raw[ts_col])
        close = pd.to_numeric(_find_close_column(raw), errors="coerce")
        if raw_ts is None or close is None:
            return None
        out = pd.DataFrame({"timestamp": raw_ts.values, "close": close.values})
        out = out.dropna(subset=["timestamp", "close"])
        # Per-horizon de-duplication: 1d → on the date; intraday → on the
        # exact timestamp.
        if horizon == "1d":
            out["__key__"] = out["timestamp"].dt.normalize()
        else:
            out["__key__"] = out["timestamp"]
        before = len(out)
        out = out.drop_duplicates(subset="__key__", keep="last")
        if len(out) != before:
            logger.warning("economic: %s had %d duplicate timestamps — kept last",
                           path, before - len(out))
        return (out.drop(columns="__key__")
                   .sort_values("timestamp")
                   .reset_index(drop=True))
    return None


def _find_ts_column(df: pd.DataFrame):
    for c in df.columns:
        if str(c).lower() in ("timestamp", "open_time", "close_time",
                              "date", "datetime", "time"):
            return c
    return None


def _find_close_column(df: pd.DataFrame):
    for c in df.columns:
        if str(c).lower() in ("close", "close_price"):
            return df[c]
    return None


def _to_utc_timestamp(series: pd.Series) -> pd.Series | None:
    if pd.api.types.is_integer_dtype(series) or pd.api.types.is_float_dtype(series):
        mx = float(pd.to_numeric(series, errors="coerce").abs().max() or 0.0)
        unit = ("ns" if mx > 1e17 else "us" if mx > 1e14
                else "ms" if mx > 1e11 else "s")
        return pd.to_datetime(series, unit=unit, utc=True, errors="coerce")
    return pd.to_datetime(series, utc=True, errors="coerce")


def load_forward_returns(tickers: Iterable[str],
                         horizon: str = "1d") -> pd.DataFrame:
    """Long-form ``(ticker, timestamp, forward_log_return)`` per horizon.

    ``forward_log_return[t] = log(close[t+1] / close[t])`` per ticker.
    Trailing NaN rows (no t+1 bar) are dropped. The output is guaranteed to
    have a **unique (ticker, timestamp) primary key** — duplicate input
    rows are kept-last with a warning.
    """
    logger = get_logger()
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
    out = out.dropna(subset=["forward_log_return"])
    # Defensive: enforce uniqueness on the merge key.
    before = len(out)
    out = out.drop_duplicates(subset=["ticker", "timestamp"], keep="last")
    if len(out) != before:
        logger.warning("economic: forward_returns had %d duplicate (ticker, timestamp) "
                       "rows for horizon=%s — kept last", before - len(out), horizon)
    return out.sort_values(["ticker", "timestamp"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Signal × forward-return merge (horizon-aware)
# ---------------------------------------------------------------------------

def _signals_with_forward_returns(signals: pd.DataFrame,
                                  forward_returns: pd.DataFrame,
                                  horizon: str) -> pd.DataFrame:
    """Horizon-aware merge — exact timestamp for 1h / 6h, date for 1d."""
    if signals.empty or forward_returns.empty:
        return pd.DataFrame()

    sig = signals.copy()
    sig["timestamp"] = pd.to_datetime(sig["timestamp"], utc=True, errors="coerce")
    sig["ticker"]    = sig["ticker"].astype(str).str.upper()
    # Defensive: drop any pre-existing forward_log_return column so the merge
    # doesn't create _x / _y suffixes.
    if "forward_log_return" in sig.columns:
        sig = sig.drop(columns=["forward_log_return"])

    fr = forward_returns.copy()
    fr["timestamp"] = pd.to_datetime(fr["timestamp"], utc=True, errors="coerce")
    fr["ticker"]    = fr["ticker"].astype(str).str.upper()

    if horizon == "1d":
        sig["__join_ts__"] = sig["timestamp"].dt.normalize()
        fr["__join_ts__"]  = fr["timestamp"].dt.normalize()
    else:
        sig["__join_ts__"] = sig["timestamp"]
        fr["__join_ts__"]  = fr["timestamp"]

    # Guard against many-to-many merges.
    sig_dup_n = int(sig.duplicated(subset=["ticker", "__join_ts__"]).sum())
    fr_dup_n  = int(fr.duplicated(subset=["ticker", "__join_ts__"]).sum())
    if sig_dup_n or fr_dup_n:
        get_logger().warning(
            "economic: %d duplicate signal keys and %d duplicate forward-return keys "
            "for horizon=%s — keeping last to enforce 1:1 join",
            sig_dup_n, fr_dup_n, horizon,
        )
        sig = sig.drop_duplicates(subset=["ticker", "__join_ts__"], keep="last")
        fr  = fr.drop_duplicates(subset=["ticker", "__join_ts__"], keep="last")

    merged = sig.merge(
        fr[["ticker", "__join_ts__", "forward_log_return"]],
        on=["ticker", "__join_ts__"], how="left", validate="one_to_one",
    )
    n_matched = int(merged["forward_log_return"].notna().sum())
    if n_matched == 0:
        get_logger().warning("economic: forward-return join produced 0 matches "
                             "for horizon=%s", horizon)
    return merged.drop(columns="__join_ts__")


# ---------------------------------------------------------------------------
# High-Low portfolio construction
# ---------------------------------------------------------------------------

def _quantile_legs(probs: np.ndarray,
                   long_quantile: float,
                   short_quantile: float,
                   min_assets_per_leg: int) -> tuple[np.ndarray, np.ndarray]:
    """Return (long_mask, short_mask) for one cross-section.

    A coin is allowed in *both* masks only when the long and short cutoffs
    cross (degenerate case with too few coins); we then snap to top-/
    bottom-rank assignment to keep the legs disjoint.
    """
    n = len(probs)
    if n == 0:
        return (np.zeros(0, dtype=bool), np.zeros(0, dtype=bool))

    # Cutoffs from the empirical quantile function. Using rank-based bounds
    # so that with very few coins (e.g. n=3) the top-/bottom-leg always
    # contain at least one asset.
    long_cut  = np.quantile(probs, long_quantile,  method="linear")
    short_cut = np.quantile(probs, short_quantile, method="linear")
    long_mask  = probs >= long_cut
    short_mask = probs <= short_cut

    if (long_mask & short_mask).any():
        # Disambiguate: top assets go long, bottom assets go short, the rest
        # are flat. Use a stable rank so ties are deterministic.
        order = np.argsort(-probs, kind="stable")  # descending probability
        n_long  = max(int(long_mask.sum()),  min_assets_per_leg)
        n_short = max(int(short_mask.sum()), min_assets_per_leg)
        # Cap so that the two legs do not overlap.
        n_long  = min(n_long,  n)
        n_short = min(n_short, n - n_long) if n_long < n else 0
        long_mask  = np.zeros(n, dtype=bool)
        short_mask = np.zeros(n, dtype=bool)
        if n_long > 0:
            long_mask[order[:n_long]] = True
        if n_short > 0:
            short_mask[order[-n_short:]] = True

    return long_mask, short_mask


def _threshold_legs(probs: np.ndarray, threshold: float) -> tuple[np.ndarray, np.ndarray]:
    """Threshold rule: long if p > threshold, short if p < 1 − threshold."""
    long_mask  = probs > threshold
    short_mask = probs < (1.0 - threshold)
    return long_mask, short_mask


def build_high_low_portfolio_returns(merged: pd.DataFrame,
                                     cfg: dict,
                                     *,
                                     selector: str = "quantile",
                                     threshold: float | None = None
                                     ) -> pd.DataFrame:
    """Build per-timestamp portfolio rows from a (signals × forward_return) frame.

    Parameters
    ----------
    merged
        Output of :func:`_signals_with_forward_returns` (long form with at
        least ``ticker``, ``timestamp``, ``probability``, ``forward_log_return``).
    cfg
        Backtest config dict.
    selector
        ``"quantile"`` → top / bottom quantile cross-sectionally;
        ``"threshold"`` → ``p > threshold`` long, ``p < 1 − threshold`` short.

    Returns
    -------
    DataFrame with one row per portfolio-active timestamp:
        timestamp · gross_log_return · long_n · short_n · weights
    ``weights`` is a ``dict[ticker → float]`` for downstream turnover
    computation.
    """
    if merged.empty:
        return pd.DataFrame(columns=["timestamp", "gross_log_return",
                                      "long_n", "short_n", "weights"])

    long_q  = float(cfg.get("long_quantile",  0.8))
    short_q = float(cfg.get("short_quantile", 0.2))
    min_n   = int(cfg.get("min_assets_per_leg", 1))
    allow_single = bool(cfg.get("allow_single_leg", False))

    valid = merged.dropna(subset=["forward_log_return", "probability"]).copy()
    if valid.empty:
        return pd.DataFrame(columns=["timestamp", "gross_log_return",
                                      "long_n", "short_n", "weights"])

    rows: list[dict] = []
    for ts, snap in valid.groupby("timestamp", sort=True):
        # Deduplicate on ticker per timestamp (last wins).
        snap = snap.drop_duplicates(subset="ticker", keep="last")
        tickers = snap["ticker"].astype(str).values
        probs   = snap["probability"].astype(float).values
        fwd     = snap["forward_log_return"].astype(float).values
        n = len(snap)
        if n == 0:
            continue

        if selector == "quantile":
            long_m, short_m = _quantile_legs(probs, long_q, short_q, min_n)
        elif selector == "threshold":
            assert threshold is not None
            long_m, short_m = _threshold_legs(probs, float(threshold))
        else:
            raise ValueError(f"Unknown selector {selector}")

        n_long  = int(long_m.sum())
        n_short = int(short_m.sum())
        if n_long < min_n or n_short < min_n:
            if not allow_single:
                continue
            # Allow a one-sided portfolio only when explicitly enabled.
            if n_long < min_n and n_short < min_n:
                continue

        weights: dict[str, float] = {}
        if n_long > 0:
            w_long = 1.0 / n_long
            for tk in tickers[long_m]:
                weights[tk] = w_long
        if n_short > 0:
            w_short = -1.0 / n_short
            for tk in tickers[short_m]:
                weights[tk] = weights.get(tk, 0.0) + w_short

        long_ret  = float(np.mean(fwd[long_m]))  if n_long  > 0 else 0.0
        short_ret = float(np.mean(fwd[short_m])) if n_short > 0 else 0.0
        gross = long_ret - short_ret  # long − short

        rows.append({
            "timestamp":        ts,
            "gross_log_return": gross,
            "long_n":           n_long,
            "short_n":          n_short,
            "weights":          weights,
        })
    if not rows:
        return pd.DataFrame(columns=["timestamp", "gross_log_return",
                                      "long_n", "short_n", "weights"])
    return pd.DataFrame(rows).sort_values("timestamp").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Turnover + transaction costs
# ---------------------------------------------------------------------------

def compute_portfolio_turnover(weight_series: Sequence[dict]) -> np.ndarray:
    """Turnover per portfolio-active timestamp.

    ``turnover_t = 0.5 · Σ_i |w_{t,i} − w_{t-1,i}|``. The very first step
    is treated as an initial entry from a fully flat book (``w_{t-1}`` ≡ 0).
    """
    n = len(weight_series)
    if n == 0:
        return np.zeros(0, dtype=float)
    out = np.zeros(n, dtype=float)
    prev: dict[str, float] = {}
    for i, w in enumerate(weight_series):
        union = set(prev.keys()) | set(w.keys())
        total = 0.0
        for tk in union:
            total += abs(w.get(tk, 0.0) - prev.get(tk, 0.0))
        out[i] = 0.5 * total
        prev = dict(w)
    return out


def apply_transaction_costs(gross_log_returns: np.ndarray,
                            turnover: np.ndarray,
                            cost_bps: float) -> np.ndarray:
    """Net log return = gross − turnover × (cost_bps / 1e4).

    Treating the cost as a small additive deduction on a log return is the
    standard approximation for the bps regime we work in (≤ a few hundred
    bps). At 10 bps and below the second-order error is below 1e-7.
    """
    cost = float(cost_bps) / 1e4
    return gross_log_returns - turnover * cost


# ---------------------------------------------------------------------------
# Risk + return metrics
# ---------------------------------------------------------------------------

def compute_equity_drawdown(log_returns: np.ndarray) -> tuple[np.ndarray, float]:
    """Percentage drawdown on the equity curve.

    ``equity_t = exp(Σ_{s ≤ t} r_s)``;
    ``drawdown_t = equity_t / running_max(equity) − 1``.
    Returned ``max_dd`` is in (−1, 0].
    """
    r = np.asarray(log_returns, dtype=float)
    if r.size == 0:
        return np.array([], dtype=float), float("nan")
    r = np.where(np.isnan(r), 0.0, r)
    equity = np.exp(np.cumsum(r))
    peak = np.maximum.accumulate(equity)
    dd = equity / peak - 1.0
    return dd, float(dd.min())


def compute_sharpe(returns: np.ndarray, *,
                   periods_per_year: float,
                   risk_free_rate: float = 0.0) -> float:
    r = np.asarray(returns, dtype=float)
    r = r[~np.isnan(r)]
    if len(r) < 2 or np.isnan(periods_per_year) or periods_per_year <= 0:
        return float("nan")
    std = float(np.std(r, ddof=1))
    if std == 0:
        return float("nan")
    excess = r - (risk_free_rate / periods_per_year)
    return float(np.mean(excess) / std * np.sqrt(periods_per_year))


def compute_sortino(returns: np.ndarray, *,
                    periods_per_year: float,
                    risk_free_rate: float = 0.0) -> float:
    r = np.asarray(returns, dtype=float)
    r = r[~np.isnan(r)]
    if len(r) < 2 or np.isnan(periods_per_year) or periods_per_year <= 0:
        return float("nan")
    excess = r - (risk_free_rate / periods_per_year)
    downside = excess[excess < 0]
    if downside.size == 0:
        return float("nan")
    dstd = float(np.std(downside, ddof=1)) if downside.size >= 2 else 0.0
    if dstd == 0:
        return float("nan")
    return float(np.mean(excess) / dstd * np.sqrt(periods_per_year))


def _days_spanned(timestamps: pd.Series) -> float:
    if timestamps is None or len(timestamps) == 0:
        return 0.0
    ts = pd.to_datetime(timestamps, utc=True, errors="coerce").dropna()
    if len(ts) < 2:
        return 0.0
    return float((ts.max() - ts.min()).total_seconds() / 86_400.0)


def _periods_per_year(n_periods: int, days_spanned: float, calendar_days: int) -> float:
    if n_periods <= 0 or days_spanned <= 0:
        return float("nan")
    return float(n_periods) / float(days_spanned) * float(calendar_days)


# ---------------------------------------------------------------------------
# Pooled-row builder
# ---------------------------------------------------------------------------

def _portfolio_metrics_row(portfolio: pd.DataFrame,
                           cost_bps: float,
                           cfg: dict) -> dict:
    """Compute one row of backtest metrics for a per-timestamp portfolio."""
    calendar = int(cfg.get("crypto_calendar_days", 365))
    rf       = float(cfg.get("risk_free_rate", 0.0))
    n_periods = int(len(portfolio))
    base = {
        "cost_bps":             float(cost_bps),
        "n_periods":            n_periods,
        "n_assets_avg":         0.0,
        "long_assets_avg":      0.0,
        "short_assets_avg":     0.0,
        "turnover_avg":         0.0,
        "gross_cumulative_log_return": float("nan"),
        "net_cumulative_log_return":   float("nan"),
        "net_cumulative_simple_return": float("nan"),
        "annualized_log_return":  float("nan"),
        "annualized_simple_return": float("nan"),
        "annualized_volatility": float("nan"),
        "sharpe":                float("nan"),
        "sortino":               float("nan"),
        "max_drawdown":          float("nan"),
        "calmar_ratio":          float("nan"),
        "mean_return":           float("nan"),
        "median_return":         float("nan"),
        "hit_rate":              float("nan"),
        "long_hit_rate":         float("nan"),
        "short_hit_rate":        float("nan"),
        "coverage":              float("nan"),
    }
    if n_periods == 0:
        return base

    gross = portfolio["gross_log_return"].astype(float).values
    weights = portfolio["weights"].tolist()
    turnover = compute_portfolio_turnover(weights)
    net   = apply_transaction_costs(gross, turnover, cost_bps)

    spanned = _days_spanned(portfolio["timestamp"])
    ppy     = _periods_per_year(n_periods, spanned, calendar)

    gross_cum = float(gross.sum())
    net_cum   = float(net.sum())
    mean_r    = float(net.mean())
    median_r  = float(np.median(net))

    ann_log = float(mean_r * ppy) if not np.isnan(ppy) else float("nan")
    ann_simple = float(np.expm1(ann_log)) if not np.isnan(ann_log) else float("nan")
    vol_a   = float(np.std(net, ddof=1) * np.sqrt(ppy)) \
              if (n_periods > 1 and not np.isnan(ppy)) else float("nan")
    sharpe  = compute_sharpe(net,  periods_per_year=ppy, risk_free_rate=rf)
    sortino = compute_sortino(net, periods_per_year=ppy, risk_free_rate=rf)

    _, max_dd = compute_equity_drawdown(net)
    calmar = float(ann_log / abs(max_dd)) if (max_dd is not None
                                              and not np.isnan(max_dd)
                                              and max_dd < 0) else float("nan")

    long_n  = portfolio["long_n"].astype(int).values
    short_n = portfolio["short_n"].astype(int).values
    n_avg   = float((long_n + short_n).mean())

    # Hit-rate analysis: a portfolio-period is "hit" when its net return > 0.
    hits = int((net > 0).sum())
    base.update({
        "n_assets_avg":         n_avg,
        "long_assets_avg":      float(long_n.mean()),
        "short_assets_avg":     float(short_n.mean()),
        "turnover_avg":         float(turnover.mean()),
        "gross_cumulative_log_return": gross_cum,
        "net_cumulative_log_return":   net_cum,
        "net_cumulative_simple_return": float(np.expm1(net_cum)),
        "annualized_log_return":  ann_log,
        "annualized_simple_return": ann_simple,
        "annualized_volatility": vol_a,
        "sharpe":                sharpe,
        "sortino":               sortino,
        "max_drawdown":          max_dd,
        "calmar_ratio":          calmar,
        "mean_return":           mean_r,
        "median_return":         median_r,
        "hit_rate":              float(hits / n_periods) if n_periods else float("nan"),
        # No long-only / short-only breakdown at the portfolio level — see
        # the per-ticker table for that.
        "long_hit_rate":  float("nan"),
        "short_hit_rate": float("nan"),
        "coverage":              float("nan"),  # filled by the caller
    })
    return base


# ---------------------------------------------------------------------------
# Public summarisers
# ---------------------------------------------------------------------------

def summarize_high_low_backtest(signals: pd.DataFrame,
                                 forward_returns: pd.DataFrame,
                                 cfg: dict,
                                 *,
                                 horizon: str) -> pd.DataFrame:
    """One row per (horizon × set × sentiment × cost_bps).

    Cross-sectional High-Low portfolio with the quantile cutoffs from
    ``cfg``. The result reports portfolio-level metrics — *not* a sum over
    individual ticker observations.
    """
    if signals.empty:
        return pd.DataFrame()
    merged = _signals_with_forward_returns(signals, forward_returns, horizon=horizon)
    if merged.empty or merged["forward_log_return"].notna().sum() == 0:
        return pd.DataFrame()
    merged = ensure_group_columns(merged)

    cost_grid = list(cfg.get("transaction_cost_bps", [0, 5, 10]))
    rows: list[dict] = []
    for keys, grp in merged.groupby(list(GROUP_KEYS), dropna=False):
        meta = _group_meta(grp)
        n_signal_periods = int(grp["timestamp"].nunique())
        portfolio = build_high_low_portfolio_returns(grp, cfg, selector="quantile")
        for cost in cost_grid:
            row = _portfolio_metrics_row(portfolio, float(cost), cfg)
            row["coverage"] = (
                float(row["n_periods"] / n_signal_periods) if n_signal_periods else 0.0
            )
            row.update({
                **dict(zip(GROUP_KEYS, keys)),
                "portfolio_mode":   cfg.get("portfolio_mode", "high_low"),
                "long_quantile":    float(cfg.get("long_quantile", 0.8)),
                "short_quantile":   float(cfg.get("short_quantile", 0.2)),
                **meta,
            })
            rows.append(row)
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    front = FRONT_META + [
        "portfolio_mode", "long_quantile", "short_quantile", "cost_bps",
        "n_periods", "n_assets_avg", "long_assets_avg", "short_assets_avg",
        "turnover_avg", "gross_cumulative_log_return",
        "net_cumulative_log_return", "net_cumulative_simple_return",
        "annualized_log_return", "annualized_simple_return",
        "annualized_volatility", "sharpe", "sortino",
        "max_drawdown", "calmar_ratio", "hit_rate", "coverage",
        "mean_return", "median_return"]
    return _front_order(out, front)


def summarize_high_low_threshold_backtest(signals: pd.DataFrame,
                                           forward_returns: pd.DataFrame,
                                           cfg: dict,
                                           *,
                                           horizon: str) -> pd.DataFrame:
    """One row per (horizon × set × sentiment × threshold × cost_bps).

    Threshold rule: a coin enters the long leg iff its model probability
    exceeds the threshold; into the short leg iff p < 1 − threshold; flat
    in between. This is *not* a classification threshold — it is a
    cross-sectional **portfolio-selection** rule. Documented separately.
    """
    if signals.empty:
        return pd.DataFrame()
    merged = _signals_with_forward_returns(signals, forward_returns, horizon=horizon)
    if merged.empty or merged["forward_log_return"].notna().sum() == 0:
        return pd.DataFrame()
    merged = ensure_group_columns(merged)

    thresholds = list(cfg.get("thresholds", [0.55, 0.60, 0.65]))
    cost_grid  = list(cfg.get("transaction_cost_bps", [0, 5, 10]))
    rows: list[dict] = []
    for keys, grp in merged.groupby(list(GROUP_KEYS), dropna=False):
        meta = _group_meta(grp)
        n_signal_periods = int(grp["timestamp"].nunique())
        for thr in thresholds:
            portfolio = build_high_low_portfolio_returns(
                grp, cfg, selector="threshold", threshold=float(thr),
            )
            for cost in cost_grid:
                row = _portfolio_metrics_row(portfolio, float(cost), cfg)
                row["coverage"] = (
                    float(row["n_periods"] / n_signal_periods)
                    if n_signal_periods else 0.0
                )
                row.update({
                    **dict(zip(GROUP_KEYS, keys)),
                    "threshold":        float(thr),
                    **meta,
                })
                rows.append(row)
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    front = FRONT_META + [
        "threshold", "cost_bps", "n_periods",
        "long_assets_avg", "short_assets_avg",
        "net_cumulative_simple_return", "sharpe", "turnover_avg",
        "max_drawdown", "coverage"]
    return _front_order(out, front)


def summarize_backtest_by_ticker(signals: pd.DataFrame,
                                 forward_returns: pd.DataFrame,
                                 cfg: dict,
                                 *,
                                 horizon: str) -> pd.DataFrame:
    """Per-ticker breakdown — observation-level for diagnostic purposes only.

    This is **not** the portfolio backtest; it shows how each individual
    coin would have performed under a simple sign-of-prediction rule. The
    pooled tables above are the publishable economic results.
    """
    if signals.empty:
        return pd.DataFrame()
    merged = _signals_with_forward_returns(signals, forward_returns, horizon=horizon)
    if merged.empty or merged["forward_log_return"].notna().sum() == 0:
        return pd.DataFrame()
    merged = ensure_group_columns(merged)

    cost_grid = list(cfg.get("transaction_cost_bps", [0, 5, 10]))
    rows: list[dict] = []
    for keys, grp in merged.groupby(list(GROUP_KEYS) + ["ticker"], dropna=False):
        valid = grp.dropna(subset=["forward_log_return", "probability"]).copy()
        if valid.empty:
            continue
        valid = valid.sort_values("timestamp").reset_index(drop=True)
        pred = (valid["prediction"].astype(int).values * 2 - 1)  # 0→-1, 1→+1
        fwd  = valid["forward_log_return"].astype(float).values
        spanned = _days_spanned(valid["timestamp"])
        ppy = _periods_per_year(len(valid), spanned,
                                int(cfg.get("crypto_calendar_days", 365)))
        ident = dict(zip(GROUP_KEYS, keys[:-1]))
        for cost in cost_grid:
            # Per-ticker turnover from sign changes.
            diff = np.abs(np.diff(pred.astype(float), prepend=0.0)) / 2.0
            gross = pred.astype(float) * fwd
            net = gross - diff * (float(cost) / 1e4)
            _, max_dd = compute_equity_drawdown(net)
            sharpe = compute_sharpe(net, periods_per_year=ppy,
                                    risk_free_rate=float(cfg.get("risk_free_rate", 0.0)))
            rows.append({
                **ident,
                "ticker":          keys[-1],
                "cost_bps":        float(cost),
                "n_obs":           int(len(valid)),
                "cumulative_log_return": float(net.sum()),
                "cumulative_simple_return": float(np.expm1(net.sum())),
                "sharpe":          sharpe,
                "max_drawdown":    max_dd,
                "turnover_avg":    float(diff.mean()),
                "hit_rate":        float((net > 0).sum() / len(net)) if len(net) else float("nan"),
            })
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    front = FRONT_META + ["ticker", "cost_bps",
             "n_obs", "cumulative_log_return", "cumulative_simple_return",
             "sharpe", "max_drawdown", "turnover_avg", "hit_rate"]
    return _front_order(out, front)


# ---------------------------------------------------------------------------
# Buy-and-hold benchmark
# ---------------------------------------------------------------------------

def buy_and_hold_benchmark(tickers: Iterable[str],
                           forward_returns: pd.DataFrame,
                           cfg: dict,
                           horizon: str) -> pd.DataFrame:
    """Equal-weight long-only reference, one row per cost level.

    For each timestamp, average per-ticker forward log returns (equal-weight
    over the available cross-section). That becomes the portfolio return.
    Turnover is the change in the equal-weight vector — generally tiny but
    non-zero when a coin enters or leaves the universe.
    """
    if forward_returns.empty:
        return pd.DataFrame()
    universe = sorted(set(str(t).upper() for t in tickers))
    fr = forward_returns[forward_returns["ticker"].isin(universe)]
    if fr.empty:
        return pd.DataFrame()

    rows = []
    portfolio = (fr.groupby("timestamp")
                   .apply(lambda g: {
                       "timestamp":        g.name,
                       "gross_log_return": float(g["forward_log_return"].mean()),
                       "long_n":           int(len(g)),
                       "short_n":          0,
                       "weights":          {tk: 1.0 / len(g) for tk in g["ticker"]},
                   })
                   .tolist())
    pf_df = pd.DataFrame(portfolio).sort_values("timestamp").reset_index(drop=True)
    for cost in cfg.get("transaction_cost_bps", [0, 5, 10]):
        row = _portfolio_metrics_row(pf_df, float(cost), cfg)
        row.update({
            "horizon":         horizon,
            "set_id":          "BUY_HOLD",
            "sentiment_model": "-",
            # BUY_HOLD is a global, model-independent reference: it is tagged
            # as its own ``benchmark`` family so it never collides with a
            # per-asset / panel-logit row and may be quoted across families.
            "model_type":      "benchmark",
            "panel_mode":      "-",
            "hpo_variant":     "-",
            "category":        "benchmark",
            "label":           "equal-weight long-only buy & hold",
            "portfolio_mode":  "buy_and_hold",
            "long_quantile":   float("nan"),
            "short_quantile":  float("nan"),
        })
        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Benchmark lift helper
# ---------------------------------------------------------------------------

def attach_benchmark_lifts(economic_df: pd.DataFrame,
                           benchmark_ids: Sequence[str]) -> pd.DataFrame:
    """Append sharpe_lift_vs_<bid> and cumulative_return_lift_vs_<bid> columns.

    The benchmark Sharpe / cumulative return are matched **within the same
    model family** (horizon + cost_bps + model_type + panel_mode +
    hpo_variant). A panel-logit row is therefore lifted only against the
    panel-logit B1 / B2 rows of the same panel mode and HPO variant — fixed-C
    and HPO are never mixed, nor per-asset and panel. Rows with no same-family
    benchmark keep NaN lift columns rather than crashing or silently borrowing
    another family's number. (BUY_HOLD is a separate global reference and is
    not used here.)
    """
    if economic_df is None or economic_df.empty:
        return economic_df
    out = ensure_group_columns(economic_df).copy()
    merge_keys = ["horizon", "cost_bps", "model_type", "panel_mode", "hpo_variant"]
    for bid in benchmark_ids:
        bench = (out[out["set_id"] == bid]
                   [merge_keys + ["sharpe", "net_cumulative_simple_return"]]
                   .rename(columns={
                       "sharpe":                     f"_bench_sharpe_{bid.lower()}",
                       "net_cumulative_simple_return": f"_bench_cum_{bid.lower()}",
                   }))
        if bench.empty:
            continue
        out = out.merge(bench, on=merge_keys, how="left")
        out[f"sharpe_lift_vs_{bid.lower()}"] = (
            out["sharpe"] - out[f"_bench_sharpe_{bid.lower()}"]
        )
        out[f"cumulative_return_lift_vs_{bid.lower()}"] = (
            out["net_cumulative_simple_return"] - out[f"_bench_cum_{bid.lower()}"]
        )
        out = out.drop(columns=[f"_bench_sharpe_{bid.lower()}",
                                 f"_bench_cum_{bid.lower()}"])
    return out


# ---------------------------------------------------------------------------
# Config helper
# ---------------------------------------------------------------------------

def load_backtest_config(path: str | Path | None = None) -> dict:
    """Return the backtest config as a plain dict, with sensible defaults."""
    defaults = {
        "portfolio_mode":      "high_low",
        "long_quantile":       0.8,
        "short_quantile":      0.2,
        "min_assets_per_leg":  1,
        "allow_single_leg":    False,
        "weighting":           "equal_weight",
        "annualization_mode":  "crypto",
        "crypto_calendar_days": 365,
        "transaction_cost_bps": [0, 5, 10],
        "thresholds":          [0.55, 0.60, 0.65],
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


# ---------------------------------------------------------------------------
# Back-compat shims so external callers keep working
# ---------------------------------------------------------------------------

# Older code paths called ``summarize_backtest`` / ``summarize_threshold_backtest``.
# Map those to the new portfolio-aware summarisers.
def summarize_backtest(signals: pd.DataFrame,
                       forward_returns: pd.DataFrame,
                       cfg: dict,
                       *,
                       horizon: str | None = None) -> pd.DataFrame:
    if horizon is None:
        horizons = sorted(signals["horizon"].dropna().astype(str).unique())
        if len(horizons) != 1:
            raise ValueError(
                f"summarize_backtest needs a single horizon; got {horizons}."
            )
        horizon = horizons[0]
    return summarize_high_low_backtest(signals, forward_returns, cfg, horizon=horizon)


def summarize_threshold_backtest(signals: pd.DataFrame,
                                 forward_returns: pd.DataFrame,
                                 cfg: dict,
                                 *,
                                 horizon: str | None = None) -> pd.DataFrame:
    if horizon is None:
        horizons = sorted(signals["horizon"].dropna().astype(str).unique())
        if len(horizons) != 1:
            raise ValueError(
                f"summarize_threshold_backtest needs a single horizon; got {horizons}."
            )
        horizon = horizons[0]
    return summarize_high_low_threshold_backtest(
        signals, forward_returns, cfg, horizon=horizon,
    )
