"""Structural-break **diagnostics** for the final modelling feature panel.

This stage produces purely *advisory* reports: for each ticker × feature ×
horizon, we estimate the number of structural breaks in the mean using a
Bai-Perron-style sweep (binary segmentation under ``ruptures``, BIC-penalised
selection of the number of breaks), and summarise the resulting regime lengths.

**These diagnostics never feed into the modelling rolling window.**
``run-models`` cannot read these CSVs. Rolling-window length is always set
manually by the user via ``--rolling-window-timestamps`` /
``--rolling-window-days``. The optional
``regime_length_diagnostics.csv`` is clearly labelled as
``descriptive_only=True`` so downstream readers cannot mistake it for a model
configuration knob.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from ..features.final_feature_utils import (
    classify_feature_group, load_final_features,
)


logger = logging.getLogger("structural_breaks")

HORIZONS = ("1h", "6h", "1d")

# Default diagnostic basket — market_cap_t is intentionally excluded; flip
# --include-market-cap on the CLI to add it.
DEFAULT_DIAGNOSTIC_FEATURES: tuple[str, ...] = (
    "log_return_t",
    "realized_vol_14",
    "volume_diff",
    "post_count",
    "cryptobert_title_score_mean",
    "vader_title_score_mean",
    "cryptobert_bullishness_ratio",
)

# Columns that are never analysed regardless of registry inputs.
_FORBIDDEN_COLUMNS = {
    "timestamp", "date", "ticker", "horizon", "target", "row_id",
    "prediction", "probability",
}


def _setup_logging() -> None:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s",
                        datefmt="%H:%M:%S")


# ---------------------------------------------------------------------------
# Bai-Perron-style break detection
# ---------------------------------------------------------------------------

def _bic_score(residuals: np.ndarray, n_params: int, n: int) -> float:
    """Gaussian BIC for a piecewise-constant mean model.

    ``n_params`` counts breakpoints + 1 (one mean per segment); the variance is
    profiled out. Lower is better.
    """
    if n <= 0:
        return float("inf")
    rss = float(np.sum(residuals ** 2))
    if rss <= 0:
        return float("inf")
    sigma2 = rss / n
    return float(n * np.log(sigma2) + n_params * np.log(max(n, 2)))


def _segment_residuals(values: np.ndarray, breaks: list[int]) -> np.ndarray:
    """Residuals of a piecewise-constant fit on ``values`` given break indices.

    ``breaks`` follows the ``ruptures`` convention: a sorted list of break
    positions in ``[1, n)`` with the endpoint ``n`` appended (or not — we
    handle both).
    """
    n = len(values)
    if n == 0:
        return values
    bkps = sorted(b for b in breaks if 0 < b < n)
    bkps.append(n)
    out = np.empty_like(values, dtype=float)
    start = 0
    for end in bkps:
        seg = values[start:end]
        if len(seg) > 0:
            out[start:end] = seg - seg.mean()
        start = end
    return out


def detect_breaks(series: pd.Series,
                  *,
                  max_breaks: int = 5,
                  min_segment_frac: float = 0.15,
                  n_bkps: int | None = None) -> dict:
    """Bai-Perron-style multiple-break diagnostic for a single series.

    Returns a dict with ``n_breaks``, ``break_indices``, ``break_timestamps``
    (best-effort, ``[]`` for an integer index), ``segment_lengths_timestamps``,
    ``mean_segment_length``, ``median_segment_length`` and the BIC score.

    When ``ruptures`` is unavailable the function returns ``n_breaks = 0`` and
    a single segment.
    """
    clean = pd.to_numeric(series, errors="coerce").dropna()
    n = int(len(clean))
    empty = {
        "n_breaks": 0, "break_indices": [], "break_timestamps": [],
        "segment_lengths_timestamps": [n] if n else [],
        "mean_segment_length": float(n) if n else float("nan"),
        "median_segment_length": float(n) if n else float("nan"),
        "best_bic": float("nan"), "method": "none",
    }
    if n < 30:
        return empty
    try:
        import ruptures as rpt
    except ImportError:
        logger.debug("ruptures not available — returning n_breaks=0")
        return empty

    values = clean.values.astype(float)
    min_size = max(10, int(round(n * float(min_segment_frac))))
    if min_size * 2 > n:
        return empty

    # Binary segmentation with l2 cost — the standard Bai-Perron-style search.
    try:
        algo = rpt.Binseg(model="l2", min_size=min_size, jump=1).fit(values)
    except (ValueError, np.linalg.LinAlgError) as exc:  # noqa: BLE001
        logger.debug("ruptures Binseg failed (%s) — returning n_breaks=0", exc)
        return empty

    if n_bkps is not None:
        # Caller pinned a number of breaks; trust them but still report BIC.
        try:
            bkps = algo.predict(n_bkps=int(n_bkps))
        except (rpt.exceptions.BadSegmentationParameters, ValueError) as exc:
            logger.debug("predict(n_bkps=%s) failed (%s)", n_bkps, exc)
            return empty
        bkps_inner = [b for b in bkps if b < n]
        residuals = _segment_residuals(values, bkps_inner)
        bic = _bic_score(residuals, n_params=len(bkps_inner) + 1, n=n)
        chosen = bkps_inner
        chosen_bic = bic
        method = f"ruptures.Binseg(n_bkps={n_bkps})"
    else:
        best_bic = float("inf")
        chosen: list[int] = []
        for k in range(0, int(max_breaks) + 1):
            try:
                bkps = algo.predict(n_bkps=k)
            except (rpt.exceptions.BadSegmentationParameters, ValueError):
                continue
            bkps_inner = [b for b in bkps if b < n]
            residuals = _segment_residuals(values, bkps_inner)
            bic = _bic_score(residuals, n_params=len(bkps_inner) + 1, n=n)
            if bic < best_bic:
                best_bic = bic
                chosen = bkps_inner
        chosen_bic = best_bic
        method = "ruptures.Binseg(BIC over n_bkps in [0, max_breaks])"

    # Segment lengths in (closed-open) blocks ending at each break (and n).
    segment_lengths: list[int] = []
    start = 0
    for end in (*chosen, n):
        segment_lengths.append(int(end - start))
        start = end

    break_timestamps: list[str] = []
    if isinstance(clean.index, pd.DatetimeIndex):
        for idx in chosen:
            ts = clean.index[idx]
            break_timestamps.append(pd.Timestamp(ts).isoformat())

    return {
        "n_breaks": int(len(chosen)),
        "break_indices": list(map(int, chosen)),
        "break_timestamps": break_timestamps,
        "segment_lengths_timestamps": segment_lengths,
        "mean_segment_length": (float(np.mean(segment_lengths))
                                if segment_lengths else float("nan")),
        "median_segment_length": (float(np.median(segment_lengths))
                                  if segment_lengths else float("nan")),
        "best_bic": (float(chosen_bic) if chosen_bic != float("inf")
                     else float("nan")),
        "method": method,
    }


# ---------------------------------------------------------------------------
# Per-horizon driver
# ---------------------------------------------------------------------------

def _select_features(df: pd.DataFrame,
                     requested: Iterable[str] | None,
                     include_market_cap: bool) -> list[str]:
    """Pick the subset of ``requested`` (default basket) actually in ``df``.

    ``market_cap_t`` is excluded unless ``include_market_cap`` is set.
    Identifier / target columns are always excluded.
    """
    requested = list(requested) if requested else list(DEFAULT_DIAGNOSTIC_FEATURES)
    out: list[str] = []
    for col in requested:
        if col in _FORBIDDEN_COLUMNS:
            continue
        if col == "market_cap_t" and not include_market_cap:
            continue
        if col in df.columns:
            out.append(col)
    return out


def run_horizon(df: pd.DataFrame, horizon: str,
                features: list[str],
                *,
                max_breaks: int,
                min_segment_frac: float,
                n_bkps: int | None) -> list[dict]:
    records: list[dict] = []
    if not features:
        return records
    tickers = sorted(df["ticker"].dropna().unique())
    for tk in tickers:
        sub = (df[df["ticker"] == tk]
               .sort_values("timestamp")
               .set_index("timestamp"))
        for feat in features:
            if feat not in sub.columns:
                continue
            res = detect_breaks(sub[feat],
                                max_breaks=max_breaks,
                                min_segment_frac=min_segment_frac,
                                n_bkps=n_bkps)
            records.append({
                "horizon":       horizon,
                "ticker":        tk,
                "feature":       feat,
                "feature_group": classify_feature_group(feat),
                "n_obs":         int(sub[feat].dropna().size),
                "n_breaks":      res["n_breaks"],
                "break_indices":     ";".join(map(str, res["break_indices"])),
                "break_timestamps":  ";".join(res["break_timestamps"]),
                "segment_lengths_timestamps": ";".join(map(str, res["segment_lengths_timestamps"])),
                "median_segment_length":      res["median_segment_length"],
                "mean_segment_length":        res["mean_segment_length"],
                "best_bic":      res["best_bic"],
                "method":        res["method"],
                "max_breaks":    max_breaks,
                "min_segment_frac": min_segment_frac,
            })
    return records


def summary_by_feature(records_df: pd.DataFrame) -> pd.DataFrame:
    if records_df.empty:
        return pd.DataFrame(columns=[
            "horizon", "feature_group", "feature", "n_series",
            "mean_n_breaks", "median_n_breaks",
            "share_with_at_least_one_break",
            "median_segment_length_timestamps",
            "mean_segment_length_timestamps",
        ])
    g = records_df.groupby(["horizon", "feature_group", "feature"], dropna=False)
    out = g.agg(
        n_series=("n_breaks", "size"),
        mean_n_breaks=("n_breaks", "mean"),
        median_n_breaks=("n_breaks", "median"),
        share_with_at_least_one_break=("n_breaks", lambda s: float((s > 0).mean())),
        median_segment_length_timestamps=("median_segment_length", "median"),
        mean_segment_length_timestamps=("mean_segment_length", "mean"),
    ).reset_index()
    return out


def regime_length_diagnostics(records_df: pd.DataFrame) -> pd.DataFrame:
    """Descriptive-only aggregate. ``descriptive_only=True`` is stamped on
    every row so downstream readers cannot mistake the file for a configuration
    knob — ``run-models`` does not read this CSV.
    """
    if records_df.empty:
        return pd.DataFrame(columns=[
            "horizon", "feature_group",
            "median_segment_length_timestamps",
            "mean_segment_length_timestamps",
            "share_with_at_least_one_break",
            "n_series", "descriptive_only",
        ])
    g = records_df.groupby(["horizon", "feature_group"], dropna=False)
    out = g.agg(
        median_segment_length_timestamps=("median_segment_length", "median"),
        mean_segment_length_timestamps=("mean_segment_length", "mean"),
        share_with_at_least_one_break=("n_breaks", lambda s: float((s > 0).mean())),
        n_series=("n_breaks", "size"),
    ).reset_index()
    out["descriptive_only"] = True
    return out


# ---------------------------------------------------------------------------
# Output paths + CLI
# ---------------------------------------------------------------------------

def _output_root() -> Path:
    try:
        from ..config import resolve_path
        return Path(resolve_path("structural_breaks_root"))
    except Exception:  # noqa: BLE001
        return Path("Data/Features/structural_breaks")


def _write_csv(df: pd.DataFrame, path: Path, columns: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if df.empty and columns is not None:
        pd.DataFrame(columns=columns).to_csv(path, index=False, encoding="utf-8")
    else:
        df.to_csv(path, index=False, encoding="utf-8")
    logger.info("wrote %s (%d rows)", path, len(df))


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Structural-break diagnostics — advisory only. Never feeds "
                    "into the modelling rolling window."
    )
    p.add_argument("--horizon", default=None, choices=list(HORIZONS),
                   help="Restrict to a single horizon (default: all).")
    p.add_argument("--features", nargs="*", default=None,
                   help="Override the default diagnostic feature basket.")
    p.add_argument("--include-market-cap", "--include_market_cap",
                   dest="include_market_cap", action="store_true",
                   help="Include market_cap_t in the default basket "
                        "(off by default).")
    p.add_argument("--max-breaks", "--max_breaks", dest="max_breaks", type=int,
                   default=5, help="Upper bound on break count (default: 5).")
    p.add_argument("--min-segment-frac", "--min_segment_frac",
                   dest="min_segment_frac", type=float, default=0.15,
                   help="Minimum segment size as a fraction of n (default: 0.15).")
    p.add_argument("--n-bkps", "--n_bkps", dest="n_bkps", type=int, default=None,
                   help="Fixed number of breaks (skips BIC selection).")
    p.add_argument("--output-dir", "--output_dir", dest="output_dir",
                   default=None, help="Override output directory.")
    p.add_argument("--dry-run", "--dry_run", dest="dry_run", action="store_true")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    _setup_logging()
    out_dir = Path(args.output_dir) if args.output_dir else _output_root()

    if args.dry_run:
        logger.info("DRY-RUN: would write structural-break diagnostics to %s", out_dir)
        for hz in ([args.horizon] if args.horizon else HORIZONS):
            logger.info("DRY-RUN: horizon %s — features %s", hz,
                        args.features or list(DEFAULT_DIAGNOSTIC_FEATURES))
        return 0

    horizons = [args.horizon] if args.horizon else list(HORIZONS)
    all_records: list[dict] = []
    for hz in horizons:
        try:
            df = load_final_features(hz)
        except FileNotFoundError as exc:
            logger.error(str(exc)); continue
        feats = _select_features(df, args.features, args.include_market_cap)
        logger.info("horizon=%s features=%s", hz, feats)
        all_records.extend(run_horizon(df, hz, feats,
                                       max_breaks=args.max_breaks,
                                       min_segment_frac=args.min_segment_frac,
                                       n_bkps=args.n_bkps))

    records_df = pd.DataFrame(all_records)
    summary_df = summary_by_feature(records_df)
    regime_df  = regime_length_diagnostics(records_df)

    _write_csv(records_df, out_dir / "break_records.csv", columns=[
        "horizon", "ticker", "feature", "feature_group", "n_obs",
        "n_breaks", "break_indices", "break_timestamps",
        "segment_lengths_timestamps", "median_segment_length",
        "mean_segment_length", "best_bic", "method",
        "max_breaks", "min_segment_frac",
    ])
    _write_csv(summary_df, out_dir / "break_summary.csv", columns=[
        "horizon", "feature_group", "feature", "n_series",
        "mean_n_breaks", "median_n_breaks",
        "share_with_at_least_one_break",
        "median_segment_length_timestamps",
        "mean_segment_length_timestamps",
    ])
    _write_csv(regime_df, out_dir / "regime_length_diagnostics.csv", columns=[
        "horizon", "feature_group",
        "median_segment_length_timestamps",
        "mean_segment_length_timestamps",
        "share_with_at_least_one_break",
        "n_series", "descriptive_only",
    ])
    logger.info("DONE (diagnostic only — run-models does NOT consume these CSVs).")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
