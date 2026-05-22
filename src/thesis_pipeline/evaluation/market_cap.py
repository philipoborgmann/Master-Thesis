"""Market-capitalisation regime stratification.

Mirrors the architecture of :mod:`.volatility` (Garman-Klass regimes):

1. Load CoinMarketCap ``market_cap.parquet`` (wide format with one column
   per (CMC ID, ticker)).
2. Reshape to long form ``(date, ticker, market_cap)`` with UTC-midnight
   timestamps.
3. ``market_cap.shift(1)`` per ticker — mandatory lookahead guard. The
   regime assignment for day t uses only information up to and including
   day t-1.
4. **Daily cross-sectional** tertiles → ``small / mid / large``.
5. Look up the regime for each signal observation; for intraday horizons
   the regime is assigned at the day level (same convention as
   :mod:`.volatility`).
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from ..config import resolve_path
from ..logging_utils import get_logger
from .metrics import GROUP_KEYS, _group_meta
from .volatility import _attach_date  # reuse: signal date normalisation

REGIME_LABELS = ("small", "mid", "large")
NANO_ALIASES = ("NANO", "XNO")
IOTA_ALIASES = ("IOTA", "MIOTA")


# ---------------------------------------------------------------------------
# Loading + normalisation
# ---------------------------------------------------------------------------

def _normalise_ticker(ticker: str) -> str:
    """Canonicalise ticker, honouring NANO ↔ XNO and IOTA ↔ MIOTA."""
    t = str(ticker).upper().strip()
    if t in NANO_ALIASES:
        return "NANO"
    if t in IOTA_ALIASES:
        return "IOTA"
    return t


def _parse_cmc_column(col: str) -> tuple[str | None, str | None]:
    """Parse a CMC wide-format header like ``'000000000001_BTC'``.

    Returns ``(cmc_id, symbol)``. Headers that are not in the form
    ``<id>_<symbol>`` are returned as ``(None, None)``.
    """
    s = str(col).strip()
    if s.lower() in ("date", "timestamp", "datetime"):
        return None, None
    if "_" not in s:
        return None, None
    head, _, tail = s.partition("_")
    if not head.isdigit() or not tail:
        return None, None
    # Strip leading zeros from the CMC numeric ID for cleaner downstream use.
    return str(int(head)), tail.upper()


def load_market_cap(path: str | Path | None = None) -> pd.DataFrame:
    """Load and reshape ``market_cap.parquet`` to long form.

    Returns a frame with columns ``date`` (datetime64[ns, UTC],
    normalised to midnight), ``ticker`` (canonical), and
    ``market_cap`` (float). Empty frame on missing file.
    """
    logger = get_logger()
    if path is None:
        path = resolve_path("cmc_market_cap")
    p = Path(path)
    if not p.exists():
        logger.warning("evaluate-signals: market_cap parquet missing at %s — "
                       "market-cap stratification will be skipped", p)
        return pd.DataFrame(columns=["date", "ticker", "market_cap"])

    try:
        df = pd.read_parquet(p)
    except Exception as exc:  # noqa: BLE001
        logger.warning("evaluate-signals: cannot read %s (%s) — skipping mcap", p, exc)
        return pd.DataFrame(columns=["date", "ticker", "market_cap"])

    if df.empty:
        return pd.DataFrame(columns=["date", "ticker", "market_cap"])

    # Detect the date column (or use the index)
    date_col = None
    for candidate in ("date", "Date", "DATE", "timestamp", "Timestamp"):
        if candidate in df.columns:
            date_col = candidate
            break
    if date_col is None:
        df = df.reset_index()
        date_col = df.columns[0]

    long_rows: list[dict] = []
    dates = pd.to_datetime(df[date_col], utc=True, errors="coerce").dt.normalize()
    for col in df.columns:
        if col == date_col:
            continue
        cmc_id, symbol = _parse_cmc_column(col)
        if symbol is None:
            # Allow a plain "BTC" column too
            sym = str(col).strip().upper()
            if not sym.isalpha() or len(sym) > 6:
                continue
            symbol = sym
        ticker = _normalise_ticker(symbol)
        series = pd.to_numeric(df[col], errors="coerce")
        long_rows.append(pd.DataFrame({
            "date":       dates.values,
            "ticker":     ticker,
            "market_cap": series.values,
        }))
    if not long_rows:
        return pd.DataFrame(columns=["date", "ticker", "market_cap"])

    out = pd.concat(long_rows, ignore_index=True)
    out = out.dropna(subset=["date", "market_cap"])
    out = (out[out["market_cap"] > 0]
           .sort_values(["ticker", "date"])
           .drop_duplicates(subset=["ticker", "date"], keep="last")
           .reset_index(drop=True))
    return out


# ---------------------------------------------------------------------------
# Regime assignment
# ---------------------------------------------------------------------------

def normalize_market_cap(mcap: pd.DataFrame) -> pd.DataFrame:
    """Apply the lookahead guard: ``shift(1)`` per ticker.

    The output ``market_cap_lag`` column on day t equals the market cap on
    day t-1, so any cross-sectional comparison built from it is safe.
    """
    if mcap.empty:
        out = mcap.copy()
        out["market_cap_lag"] = np.nan
        return out
    out = mcap.copy().sort_values(["ticker", "date"]).reset_index(drop=True)
    out["market_cap_lag"] = out.groupby("ticker")["market_cap"].shift(1)
    return out


def _daily_tertiles(values: pd.Series,
                    labels: Iterable[str] = REGIME_LABELS) -> pd.Series:
    """Assign daily cross-sectional tertiles to a positive series.

    Falls back to ``pd.Series(NaN)`` when fewer than 3 distinct positive
    values are present.
    """
    s = values.dropna()
    s = s[s > 0]
    if len(s) < 3 or s.nunique() < 3:
        return pd.Series([np.nan] * len(values), index=values.index, dtype=object)
    ranked = s.rank(method="first")
    try:
        regimes = pd.qcut(ranked, q=3, labels=list(labels))
    except ValueError:
        regimes = pd.Series([np.nan] * len(s), index=s.index, dtype=object)
    full = pd.Series([np.nan] * len(values), index=values.index, dtype=object)
    full.loc[s.index] = regimes.astype(object).values
    return full


def build_market_cap_lookup(mcap_path: str | Path | None = None) -> pd.DataFrame:
    """Return ``(date, ticker, market_cap_lag, mcap_regime)`` lookup."""
    raw = load_market_cap(mcap_path)
    if raw.empty:
        return pd.DataFrame(columns=["date", "ticker", "market_cap_lag", "mcap_regime"])

    lagged = normalize_market_cap(raw)
    # Daily cross-sectional tertile assignment using lagged mcap (no leakage).
    # We re-use ``day_df["date"]`` rather than the scalar group key so the
    # tz-aware ``datetime64[ns, UTC]`` dtype survives into the output.
    regimes = []
    for _, day_df in lagged.groupby("date", sort=False):
        reg = _daily_tertiles(day_df["market_cap_lag"])
        regimes.append(pd.DataFrame({
            "date":           day_df["date"].values,
            "ticker":         day_df["ticker"].values,
            "market_cap_lag": day_df["market_cap_lag"].values,
            "mcap_regime":    reg.values,
        }))
    out = pd.concat(regimes, ignore_index=True)
    out["date"] = pd.to_datetime(out["date"], utc=True, errors="coerce").dt.normalize()
    return out.dropna(subset=["mcap_regime"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Join + stratification
# ---------------------------------------------------------------------------

def attach_market_cap_regimes(signals: pd.DataFrame,
                              mcap_lookup: pd.DataFrame) -> pd.DataFrame:
    """Left-merge ``mcap_regime`` onto signals via (ticker, date)."""
    if signals.empty:
        return signals
    if mcap_lookup is None or mcap_lookup.empty:
        out = _attach_date(signals)
        out["mcap_regime"] = np.nan
        return out
    sig = _attach_date(signals)
    sig["ticker"] = sig["ticker"].astype(str).map(_normalise_ticker)

    rl = mcap_lookup[["ticker", "date", "mcap_regime"]].copy()
    rl["ticker"] = rl["ticker"].astype(str).map(_normalise_ticker)
    rl["date"] = pd.to_datetime(rl["date"], utc=True, errors="coerce").dt.normalize()

    out = sig.merge(rl, on=["ticker", "date"], how="left")
    matched = int(out["mcap_regime"].notna().sum())
    if matched == 0 and not rl.empty:
        get_logger().warning(
            "evaluate-signals: market-cap join produced 0 matches "
            "(signals=%d, lookup=%d). Check ticker/date dtypes.",
            len(sig), len(rl),
        )
    return out


def _aggregate_correctness(group: pd.DataFrame) -> dict:
    """Per-bucket pooled metrics. Reuses sklearn for parity with metrics.py."""
    from sklearn.metrics import (
        accuracy_score, balanced_accuracy_score, f1_score,
        precision_score, recall_score, brier_score_loss,
    )
    y = group["target"].astype(int).values
    p = group["prediction"].astype(int).values
    prob = np.clip(group["probability"].astype(float).values, 1e-15, 1 - 1e-15)
    n_obs = int(len(group))
    if n_obs == 0:
        return {"accuracy": np.nan, "balanced_accuracy": np.nan,
                "precision": np.nan, "recall": np.nan, "f1": np.nan,
                "brier_score": np.nan, "n_obs": 0, "n_days": 0, "n_tickers": 0}
    n_classes = int(np.unique(y).size)
    if n_classes < 2:
        prec = rec = f1v = np.nan
    else:
        prec = precision_score(y, p, zero_division=0)
        rec  = recall_score(y, p, zero_division=0)
        f1v  = f1_score(y, p, zero_division=0)
    out = {
        "accuracy":          float(accuracy_score(y, p)),
        "balanced_accuracy": float(balanced_accuracy_score(y, p)),
        "precision":         float(prec) if not np.isnan(prec) else np.nan,
        "recall":            float(rec)  if not np.isnan(rec)  else np.nan,
        "f1":                float(f1v)  if not np.isnan(f1v)  else np.nan,
        "brier_score":       float(brier_score_loss(y, prob)) if n_classes >= 2 else np.nan,
        "n_obs":             n_obs,
        "n_days":            int(group["date"].nunique()) if "date" in group.columns else 0,
        "n_tickers":         int(group["ticker"].nunique()) if "ticker" in group.columns else 0,
    }
    return out


def market_cap_stratification_table(signals: pd.DataFrame,
                                    mcap_lookup: pd.DataFrame) -> pd.DataFrame:
    """One row per (horizon × set × sentiment_model × mcap_regime)."""
    if signals.empty:
        return pd.DataFrame()
    enriched = attach_market_cap_regimes(signals, mcap_lookup)
    if "mcap_regime" not in enriched.columns or enriched["mcap_regime"].isna().all():
        return pd.DataFrame()

    rows: list[dict] = []
    for keys, grp in enriched.groupby(list(GROUP_KEYS), dropna=False):
        meta = _group_meta(grp)
        for regime in REGIME_LABELS:
            sub = grp[grp["mcap_regime"] == regime]
            metrics_ = _aggregate_correctness(sub)
            row = {
                "horizon": keys[0], "set_id": keys[1],
                "sentiment_model": keys[2],
                "mcap_regime": regime,
                **meta,
                **metrics_,
            }
            rows.append(row)

    out = pd.DataFrame(rows)
    front = ["horizon", "set_id", "category", "sentiment_model", "label", "mcap_regime"]
    rest = [c for c in out.columns if c not in front]
    return out[front + rest].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Interaction regime (mcap × vol)
# ---------------------------------------------------------------------------

def regime_interaction_table(signals: pd.DataFrame,
                             mcap_lookup: pd.DataFrame,
                             vol_lookup: pd.DataFrame) -> pd.DataFrame:
    """One row per (horizon × set × sentiment_model × small/mid/large × low/mid/high).

    Combines :mod:`.volatility.attach_regimes` (which produces ``regime`` ∈
    low/mid/high) with :func:`attach_market_cap_regimes`.
    """
    if signals.empty:
        return pd.DataFrame()
    if mcap_lookup is None or mcap_lookup.empty or vol_lookup is None or vol_lookup.empty:
        return pd.DataFrame()

    from .volatility import attach_regimes
    enriched = attach_regimes(signals, vol_lookup)
    enriched = attach_market_cap_regimes(enriched, mcap_lookup)
    needed = {"regime", "mcap_regime"}
    if not needed.issubset(enriched.columns):
        return pd.DataFrame()
    if enriched["regime"].isna().all() or enriched["mcap_regime"].isna().all():
        return pd.DataFrame()

    rows: list[dict] = []
    for keys, grp in enriched.groupby(list(GROUP_KEYS), dropna=False):
        meta = _group_meta(grp)
        for mcap_r in REGIME_LABELS:
            for vol_r in ("low", "mid", "high"):
                sub = grp[(grp["mcap_regime"] == mcap_r) &
                          (grp["regime"]      == vol_r)]
                metrics_ = _aggregate_correctness(sub)
                # The interaction-only fields the PRD asks for.
                row = {
                    "horizon": keys[0], "set_id": keys[1],
                    "sentiment_model": keys[2],
                    "mcap_regime": mcap_r,
                    "vol_regime":  vol_r,
                    "interaction": f"{mcap_r}_{vol_r}",
                    **meta,
                    **{k: metrics_[k] for k in (
                        "accuracy", "brier_score", "n_obs", "n_days", "n_tickers",
                    )},
                }
                rows.append(row)

    out = pd.DataFrame(rows)
    front = ["horizon", "set_id", "category", "sentiment_model", "label",
             "mcap_regime", "vol_regime", "interaction"]
    rest = [c for c in out.columns if c not in front]
    return out[front + rest].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Summary helper (consumed by reporting.build_summary)
# ---------------------------------------------------------------------------

def build_market_cap_summary(mcap_strat: pd.DataFrame,
                             interaction: pd.DataFrame) -> list[dict]:
    """Return summary rows highlighting small-cap and small-high-vol effects."""
    rows: list[dict] = []
    if mcap_strat is not None and not mcap_strat.empty:
        small = (mcap_strat[mcap_strat["mcap_regime"] == "small"]
                 .dropna(subset=["accuracy"])
                 .sort_values("accuracy", ascending=False))
        if not small.empty:
            top = small.groupby("horizon", group_keys=False).head(1)
            for _, r in top.iterrows():
                rows.append({
                    "section":  "best_small_cap_model",
                    "horizon":  r["horizon"],
                    "set_id":   r["set_id"],
                    "sentiment_model": r.get("sentiment_model", "-"),
                    "metric":   "accuracy",
                    "value":    float(r["accuracy"]),
                    "n_obs":    int(r["n_obs"]),
                })
    if interaction is not None and not interaction.empty:
        sub = (interaction[(interaction["mcap_regime"] == "small") &
                           (interaction["vol_regime"]  == "high")]
               .dropna(subset=["accuracy"])
               .sort_values("accuracy", ascending=False))
        if not sub.empty:
            top = sub.groupby("horizon", group_keys=False).head(1)
            for _, r in top.iterrows():
                rows.append({
                    "section":  "best_small_high_vol_model",
                    "horizon":  r["horizon"],
                    "set_id":   r["set_id"],
                    "sentiment_model": r.get("sentiment_model", "-"),
                    "metric":   "accuracy",
                    "value":    float(r["accuracy"]),
                    "n_obs":    int(r["n_obs"]),
                })
    return rows
