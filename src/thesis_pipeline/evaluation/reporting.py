"""Excel writer, CSV writer and console summary for the evaluation stage."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Style constants for the Excel sheets.
# ---------------------------------------------------------------------------

HEADER_FILL  = "1F4E78"   # dark blue
HEADER_FONT  = "FFFFFF"   # white
CATEGORY_COLORS: dict[str, str] = {
    "benchmark": "BFBFBF",   # grey
    "economic":  "BDD7EE",   # light blue
    "sentiment": "C6E0B4",   # light green
    "combined":  "FFD966",   # amber
}


# ---------------------------------------------------------------------------
# Leaderboard + thesis summary
# ---------------------------------------------------------------------------

def build_leaderboard(pooled: pd.DataFrame, top_n: int = 20) -> pd.DataFrame:
    if pooled.empty:
        return pooled
    cols = [c for c in ("horizon", "set_id", "category", "sentiment_model", "label",
                        "accuracy", "balanced_accuracy", "f1", "brier_score",
                        "log_loss", "n_obs", "n_tickers",
                        "mcnemar_pval_vs_b1", "significant_vs_b1",
                        "mcnemar_pval_vs_b2", "significant_vs_b2",
                        "mcnemar_pval", "significant_vs_benchmark") if c in pooled.columns]
    out = (pooled[cols]
           .sort_values(["horizon", "accuracy"], ascending=[True, False])
           .groupby("horizon", group_keys=False)
           .head(top_n)
           .reset_index(drop=True))
    return out


def build_summary(pooled: pd.DataFrame,
                  threshold_df: pd.DataFrame,
                  volatility_df: pd.DataFrame,
                  *,
                  threshold_lift: pd.DataFrame | None = None) -> pd.DataFrame:
    """Thesis-ready aggregate overview as a long-form table."""
    rows: list[dict] = []
    if not pooled.empty:
        # Average accuracy per category × horizon
        cat = (pooled.dropna(subset=["accuracy"])
                     .groupby(["horizon", "category"], dropna=False)["accuracy"]
                     .mean().round(6).reset_index())
        for _, r in cat.iterrows():
            rows.append({
                "section": "avg_accuracy_per_category",
                "horizon": r["horizon"], "category": r["category"],
                "metric": "accuracy", "value": r["accuracy"],
            })
        # Best sentiment model per horizon
        sent = pooled[pooled["category"].str.lower().isin(["sentiment", "combined"])]
        if not sent.empty:
            best = (sent.dropna(subset=["accuracy"])
                        .sort_values("accuracy", ascending=False)
                        .groupby("horizon").head(1))
            for _, r in best.iterrows():
                rows.append({
                    "section": "best_sentiment_model",
                    "horizon": r["horizon"],
                    "set_id":  r["set_id"],
                    "sentiment_model": r["sentiment_model"],
                    "metric": "accuracy", "value": r["accuracy"],
                })
        # Count of significant benchmark outperformance
        if "significant_vs_benchmark" in pooled.columns:
            sig = (pooled.groupby("horizon")["significant_vs_benchmark"]
                          .apply(lambda s: int(bool(s.fillna(False).sum())))
                          .reset_index())
            sig.columns = ["horizon", "n_significant"]
            for _, r in sig.iterrows():
                rows.append({
                    "section": "n_significant_vs_benchmark",
                    "horizon": r["horizon"],
                    "metric": "count", "value": int(r["n_significant"]),
                })
    if not volatility_df.empty:
        vol = (volatility_df.dropna(subset=["accuracy"])
                            .groupby(["horizon", "vol_regime"])["accuracy"]
                            .mean().round(6).reset_index())
        for _, r in vol.iterrows():
            rows.append({
                "section": "avg_accuracy_per_vol_regime",
                "horizon": r["horizon"], "vol_regime": r["vol_regime"],
                "metric": "accuracy", "value": r["accuracy"],
            })
    if not threshold_df.empty:
        thr = (threshold_df.dropna(subset=["accuracy"])
                           .groupby(["horizon", "threshold"])
                           .agg(accuracy=("accuracy", "mean"),
                                coverage=("coverage", "mean"))
                           .round(6).reset_index())
        for _, r in thr.iterrows():
            rows.append({
                "section": "threshold_tradeoff",
                "horizon": r["horizon"], "threshold": r["threshold"],
                "metric": "accuracy", "value": r["accuracy"],
                "coverage": r["coverage"],
            })
    if threshold_lift is not None and not threshold_lift.empty:
        # Best matched-observation lift per (horizon, threshold, benchmark).
        lift = (threshold_lift.dropna(subset=["lift_accuracy"])
                              .sort_values("lift_accuracy", ascending=False)
                              .groupby(["horizon", "threshold", "benchmark"])
                              .head(1))
        for _, r in lift.iterrows():
            rows.append({
                "section": "best_threshold_lift",
                "horizon": r["horizon"], "threshold": r["threshold"],
                "benchmark": r["benchmark"],
                "set_id":    r["set_id"],
                "sentiment_model": r.get("sentiment_model", "-"),
                "metric": "lift_accuracy", "value": r["lift_accuracy"],
                "n_matched": int(r.get("n_matched", 0)),
                "significant": bool(r.get("significant_vs_benchmark", False)),
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Excel formatting helpers
# ---------------------------------------------------------------------------

def _style_header(ws) -> None:
    from openpyxl.styles import PatternFill, Font, Alignment
    fill = PatternFill(start_color=HEADER_FILL, end_color=HEADER_FILL, fill_type="solid")
    font = Font(color=HEADER_FONT, bold=True)
    align = Alignment(horizontal="left", vertical="center")
    for cell in ws[1]:
        cell.fill = fill
        cell.font = font
        cell.alignment = align


def _freeze_and_filter(ws) -> None:
    ws.freeze_panes = "A2"
    if ws.max_row >= 1 and ws.max_column >= 1:
        ws.auto_filter.ref = ws.dimensions


def _category_fills(ws, category_col_letter: str | None) -> None:
    if category_col_letter is None:
        return
    from openpyxl.styles import PatternFill
    for row in range(2, ws.max_row + 1):
        cell = ws[f"{category_col_letter}{row}"]
        val = str(cell.value).lower() if cell.value is not None else ""
        color = CATEGORY_COLORS.get(val)
        if color:
            cell.fill = PatternFill(start_color=color, end_color=color, fill_type="solid")


def _conditional_format_accuracy(ws, header_to_col: Mapping[str, str]) -> None:
    col_letter = header_to_col.get("accuracy")
    if col_letter is None or ws.max_row < 2:
        return
    from openpyxl.formatting.rule import ColorScaleRule
    rule = ColorScaleRule(
        start_type="num", start_value=0.40, start_color="F8696B",
        mid_type="num",   mid_value=0.50,   mid_color="FFEB84",
        end_type="num",   end_value=0.60,   end_color="63BE7B",
    )
    rng = f"{col_letter}2:{col_letter}{ws.max_row}"
    ws.conditional_formatting.add(rng, rule)


def _write_sheet(writer, sheet_name: str, df: pd.DataFrame) -> None:
    if df is None:
        df = pd.DataFrame()
    if df.empty:
        df = pd.DataFrame({"info": [f"no data for sheet '{sheet_name}'"]})
    df.to_excel(writer, sheet_name=sheet_name, index=False)
    ws = writer.book[sheet_name]
    _style_header(ws)
    _freeze_and_filter(ws)
    header_to_col = {cell.value: cell.column_letter for cell in ws[1]}
    _category_fills(ws, header_to_col.get("category"))
    _conditional_format_accuracy(ws, header_to_col)


# ---------------------------------------------------------------------------
# Public writers
# ---------------------------------------------------------------------------

def write_csv_outputs(out_dir: Path, *,
                      pooled: pd.DataFrame,
                      per_ticker: pd.DataFrame,
                      threshold: pd.DataFrame,
                      volatility: pd.DataFrame,
                      threshold_lift: pd.DataFrame | None = None,
                      mcnemar: pd.DataFrame | None = None) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "pooled":    out_dir / "pooled_metrics.csv",
        "per_ticker": out_dir / "per_ticker_metrics.csv",
        "threshold":  out_dir / "threshold_analysis.csv",
        "volatility": out_dir / "volatility_stratification.csv",
    }
    pooled.to_csv(paths["pooled"], index=False)
    per_ticker.to_csv(paths["per_ticker"], index=False)
    threshold.to_csv(paths["threshold"], index=False)
    volatility.to_csv(paths["volatility"], index=False)
    if threshold_lift is not None:
        paths["threshold_lift"] = out_dir / "threshold_lift.csv"
        threshold_lift.to_csv(paths["threshold_lift"], index=False)
    if mcnemar is not None:
        paths["mcnemar"] = out_dir / "mcnemar_tests.csv"
        mcnemar.to_csv(paths["mcnemar"], index=False)
    return paths


def write_excel_report(out_path: Path, *,
                       pooled: pd.DataFrame,
                       per_ticker: pd.DataFrame,
                       threshold: pd.DataFrame,
                       volatility: pd.DataFrame,
                       leaderboard: pd.DataFrame,
                       summary: pd.DataFrame,
                       threshold_lift: pd.DataFrame | None = None,
                       mcnemar: pd.DataFrame | None = None) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        _write_sheet(writer, "pooled_metrics",            pooled)
        _write_sheet(writer, "per_ticker_metrics",        per_ticker)
        _write_sheet(writer, "threshold_analysis",        threshold)
        if threshold_lift is not None:
            _write_sheet(writer, "threshold_lift",        threshold_lift)
        _write_sheet(writer, "volatility_stratification", volatility)
        if mcnemar is not None:
            _write_sheet(writer, "mcnemar_tests",         mcnemar)
        _write_sheet(writer, "leaderboard",               leaderboard)
        _write_sheet(writer, "summary",                   summary)
    return out_path


# ---------------------------------------------------------------------------
# Console summary
# ---------------------------------------------------------------------------

def print_console_summary(*,
                          pooled: pd.DataFrame,
                          threshold: pd.DataFrame,
                          mcnemar: pd.DataFrame,
                          threshold_lift: pd.DataFrame | None = None) -> None:
    print("\n" + "=" * 72)
    print(" SIGNAL EVALUATION SUMMARY")
    print("=" * 72)
    if pooled.empty:
        print("  (no pooled metrics available)")
        return

    # Benchmark accuracy per horizon
    bench = pooled[pooled["set_id"] == "B1"][["horizon", "accuracy"]].sort_values("horizon")
    if not bench.empty:
        print("\n  Benchmark (B1) accuracy by horizon:")
        for _, r in bench.iterrows():
            acc = r["accuracy"]
            acc_str = f"{acc:.4f}" if pd.notna(acc) else "n/a"
            print(f"    {r['horizon']:>3s}  acc={acc_str}")

    # Top-5 per horizon
    print("\n  Top-5 by accuracy per horizon:")
    for hz, grp in pooled.dropna(subset=["accuracy"]).groupby("horizon"):
        top = grp.sort_values("accuracy", ascending=False).head(5)
        print(f"   [{hz}]")
        for _, r in top.iterrows():
            sm = r.get("sentiment_model", "-")
            tag = f"/{sm}" if sm and sm != "-" else ""
            print(f"     {r['set_id']}{tag:14s} acc={r['accuracy']:.4f}  "
                  f"f1={r.get('f1', float('nan')):.4f}  "
                  f"brier={r.get('brier_score', float('nan')):.4f}  "
                  f"n={int(r.get('n_obs', 0))}")

    # Significant outperformers — broken out per benchmark.
    if not mcnemar.empty and "significant_vs_benchmark" in mcnemar.columns:
        for bid, grp in mcnemar.groupby("benchmark"):
            n_sig = int(grp["significant_vs_benchmark"].fillna(False).sum())
            print(f"\n  Significant (α=0.05) outperformers vs {bid}: {n_sig}")
            sig = grp[grp["significant_vs_benchmark"]].sort_values("mcnemar_pval")
            for _, r in sig.head(10).iterrows():
                sm = r.get("sentiment_model", "-")
                tag = f"/{sm}" if sm and sm != "-" else ""
                print(f"    {r['horizon']:>3s}  {r['set_id']}{tag:14s} "
                      f"stat={r['mcnemar_stat']:.3f}  p={r['mcnemar_pval']:.4f}")

    # Best sentiment model per horizon
    sent = pooled[pooled["category"].str.lower().isin(["sentiment", "combined"])]
    if not sent.empty:
        print("\n  Best sentiment/combined set per horizon:")
        best = (sent.dropna(subset=["accuracy"])
                    .sort_values("accuracy", ascending=False)
                    .groupby("horizon").head(1))
        for _, r in best.iterrows():
            print(f"    {r['horizon']:>3s}  {r['set_id']}/{r['sentiment_model']:8s} "
                  f"acc={r['accuracy']:.4f}")

    # Threshold highlights
    if not threshold.empty:
        print("\n  Threshold tradeoffs (mean across sets):")
        thr = (threshold.dropna(subset=["accuracy"])
                        .groupby(["horizon", "threshold"])
                        .agg(accuracy=("accuracy", "mean"),
                             coverage=("coverage", "mean"))
                        .reset_index())
        for hz, grp in thr.groupby("horizon"):
            print(f"   [{hz}]")
            for _, r in grp.iterrows():
                print(f"     t={r['threshold']:.2f}  acc={r['accuracy']:.4f}  "
                      f"coverage={r['coverage']:.3f}")

    # Threshold lift — does sentiment beat the benchmark on the model's
    # high-conviction observations?
    if threshold_lift is not None and not threshold_lift.empty:
        print("\n  Top sentiment lifts on matched obs (model traded → vs benchmark):")
        for hz, hz_grp in threshold_lift.dropna(subset=["lift_accuracy"]) \
                                        .groupby("horizon"):
            print(f"   [{hz}]")
            top = (hz_grp.sort_values("lift_accuracy", ascending=False)
                          .head(5))
            for _, r in top.iterrows():
                sm = r.get("sentiment_model", "-")
                tag = f"/{sm}" if sm and sm != "-" else ""
                star = "*" if bool(r.get("significant_vs_benchmark", False)) else " "
                print(f"     {star} {r['set_id']}{tag:14s} t={r['threshold']:.2f} "
                      f"vs {r['benchmark']}  lift={r['lift_accuracy']:+.4f} "
                      f"(model={r['accuracy_model']:.4f}, "
                      f"bench={r['accuracy_benchmark']:.4f}, "
                      f"n={int(r['n_matched'])})")
    print()
