"""Signal-completeness audit (final-production guard).

A partial model run — e.g. a sentiment set whose walk-forward ended a few
timestamps early — must never enter the final thesis comparisons silently. A
prior run shipped ``SENT_VAD_LD`` at ``1h`` with 81,500 rows instead of the
103,200 the matched ``ECON`` benchmark produced; this module makes that class
of defect a hard, visible failure.

For every **complete model identity** (horizon × set_id × sentiment_model ×
model_type × panel_mode × hpo_variant) the audit reports row / ticker /
timestamp counts, duplicate ``(ticker, timestamp)`` rows, the expected
ticker×timestamp grid, missing combinations, and coverage relative to (a) the
matched ``ECON`` benchmark and (b) the widest timestamp grid observed at that
horizon (the canonical forecast sample). The expected grid is taken from the
matched ``ECON`` run — never hard-coded — because ``ECON`` is the economics-only
benchmark every combined/sentiment set is compared against and therefore
defines the modelling grid for its family. The canonical forecast-sample window
(for reporting only) is read from ``configs/model_specs.yaml``.

Nothing here changes any signal value, metric, or test — it is a read-only
guard plus a diagnostic table.
"""
from __future__ import annotations

import pandas as pd

#: Full model identity that defines one comparable signal group.
GROUP_IDENTITY_COLUMNS = (
    "horizon", "set_id", "sentiment_model",
    "model_type", "panel_mode", "hpo_variant",
)

#: Family key (identity minus the feature-set) used to match a group to its
#: ECON benchmark: the ECON run in the same horizon / model family defines the
#: expected timestamp grid.
_FAMILY_KEY = ("horizon", "model_type", "panel_mode", "hpo_variant")

#: Extra identity columns folded into the family key when present (so runs with
#: different training windows / coin universes are never cross-matched).
_OPTIONAL_FAMILY_COLUMNS = (
    "train_window_mode", "rolling_window_days",
    "rolling_window_timestamps", "coin_universe_hash",
)

ECON_BENCHMARK_SET_ID = "ECON"

STATUS_COMPLETE = "complete"
STATUS_INCOMPLETE = "incomplete"
STATUS_UNKNOWN = "unknown_no_econ_reference"


def _family_columns(signals: pd.DataFrame) -> list[str]:
    cols = [c for c in _FAMILY_KEY if c in signals.columns]
    cols += [c for c in _OPTIONAL_FAMILY_COLUMNS if c in signals.columns]
    return cols


def _group_columns(signals: pd.DataFrame) -> list[str]:
    base = [c for c in GROUP_IDENTITY_COLUMNS if c in signals.columns]
    extra = [c for c in _OPTIONAL_FAMILY_COLUMNS if c in signals.columns]
    return base + extra


def audit_signal_completeness(
    signals: pd.DataFrame,
    *,
    forecast_cfg: dict | None = None,
) -> pd.DataFrame:
    """Return one completeness row per model-identity group.

    Columns: the identity columns, ``n_rows``, ``n_tickers``, ``n_timestamps``,
    ``first_timestamp``, ``last_timestamp``, ``n_duplicate_rows``,
    ``expected_timestamps``, ``missing_timestamps``, ``expected_combinations``,
    ``missing_combinations``, ``coverage_vs_econ``,
    ``coverage_vs_forecast_sample``, ``forecast_sample_start``,
    ``forecast_sample_end_exclusive``, ``status``, ``is_complete``, ``reason``.
    """
    empty_cols = [
        *GROUP_IDENTITY_COLUMNS, "n_rows", "n_tickers", "n_timestamps",
        "first_timestamp", "last_timestamp", "n_duplicate_rows",
        "expected_timestamps", "missing_timestamps", "expected_combinations",
        "missing_combinations", "coverage_vs_econ",
        "coverage_vs_forecast_sample", "forecast_sample_start",
        "forecast_sample_end_exclusive", "status", "is_complete", "reason",
    ]
    if signals is None or signals.empty:
        return pd.DataFrame(columns=empty_cols)

    df = signals.copy()
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")

    # Forecast-sample window (reporting only) from the single source of truth.
    if forecast_cfg is None:
        try:
            from ..modeling.forecast_sample import load_forecast_sample_config
            forecast_cfg = load_forecast_sample_config()
        except Exception:  # noqa: BLE001 — reporting field only
            forecast_cfg = {}
    fs_start = forecast_cfg.get("start", "")
    fs_end = forecast_cfg.get("end_exclusive", "")

    group_cols = _group_columns(df)
    fam_cols = _family_columns(df)

    # Widest timestamp grid per horizon = the canonical-forecast-sample proxy.
    ref_ts_per_h: dict = {}
    if "horizon" in df.columns:
        for hz, g in df.groupby("horizon", dropna=False):
            ref_ts_per_h[hz] = set(g["timestamp"].dropna().unique())

    # Expected timestamp grid per family key = the matched ECON run.
    econ = df[df["set_id"].astype(str) == ECON_BENCHMARK_SET_ID] if "set_id" in df.columns else df.iloc[0:0]
    econ_ts_by_fam: dict = {}
    if not econ.empty and fam_cols:
        for fkey, g in econ.groupby(list(fam_cols), dropna=False):
            econ_ts_by_fam[fkey if isinstance(fkey, tuple) else (fkey,)] = \
                set(g["timestamp"].dropna().unique())

    rows: list[dict] = []
    for gkey, g in df.groupby(list(group_cols), dropna=False):
        gkey = gkey if isinstance(gkey, tuple) else (gkey,)
        rec = dict(zip(group_cols, gkey))

        pair = g[["ticker", "timestamp"]].dropna()
        n_rows = int(len(g))
        n_unique_pairs = int(len(pair.drop_duplicates()))
        n_dup = int(n_rows - n_unique_pairs)
        gts = set(g["timestamp"].dropna().unique())
        n_ts = int(len(gts))
        n_tickers = int(g["ticker"].nunique())

        fkey = tuple(rec.get(c) for c in fam_cols)
        econ_ts = econ_ts_by_fam.get(fkey)
        is_econ = str(rec.get("set_id")) == ECON_BENCHMARK_SET_ID

        if econ_ts is None:
            exp_ts = pd.NA
            missing_ts = pd.NA
            exp_comb = pd.NA
            missing_comb = pd.NA
            cov_econ = pd.NA
        else:
            exp_ts = int(len(econ_ts))
            missing_ts = int(len(econ_ts - gts))
            exp_comb = int(n_tickers * exp_ts)
            missing_comb = int(max(exp_comb - n_unique_pairs, 0))
            cov_econ = (n_ts / exp_ts) if exp_ts else pd.NA

        ref_ts = ref_ts_per_h.get(rec.get("horizon"), set())
        cov_fs = (n_ts / len(ref_ts)) if ref_ts else pd.NA

        # Status. ECON is its own reference: judge it against the widest grid.
        if econ_ts is None and not is_econ:
            status, reason = STATUS_UNKNOWN, "no matched ECON benchmark for this family"
        elif n_dup > 0:
            status = STATUS_INCOMPLETE
            reason = f"{n_dup} duplicate (ticker, timestamp) rows"
        elif is_econ:
            # Compare ECON to the widest observed grid at its horizon.
            miss_ref = len(ref_ts - gts) if ref_ts else 0
            if miss_ref > 0:
                status = STATUS_INCOMPLETE
                reason = (f"ECON benchmark missing {miss_ref} of {len(ref_ts)} "
                          f"timestamps in the horizon grid")
            else:
                status, reason = STATUS_COMPLETE, ""
        elif missing_ts and missing_ts > 0:
            status = STATUS_INCOMPLETE
            reason = (f"missing {missing_ts} of {exp_ts} timestamps vs matched "
                      f"ECON (last={g['timestamp'].max()})")
        elif missing_comb and missing_comb > 0:
            status = STATUS_INCOMPLETE
            reason = f"ragged panel: {missing_comb} missing ticker×timestamp combinations"
        else:
            status, reason = STATUS_COMPLETE, ""

        rec.update({
            "n_rows": n_rows,
            "n_tickers": n_tickers,
            "n_timestamps": n_ts,
            "first_timestamp": g["timestamp"].min(),
            "last_timestamp": g["timestamp"].max(),
            "n_duplicate_rows": n_dup,
            "expected_timestamps": exp_ts,
            "missing_timestamps": missing_ts,
            "expected_combinations": exp_comb,
            "missing_combinations": missing_comb,
            "coverage_vs_econ": cov_econ,
            "coverage_vs_forecast_sample": cov_fs,
            "forecast_sample_start": fs_start,
            "forecast_sample_end_exclusive": fs_end,
            "status": status,
            "is_complete": status == STATUS_COMPLETE,
            "reason": reason,
        })
        rows.append(rec)

    out = pd.DataFrame(rows)
    sort_cols = [c for c in ("horizon", "set_id", "sentiment_model") if c in out.columns]
    if sort_cols:
        out = out.sort_values(sort_cols).reset_index(drop=True)
    return out


def incomplete_production_groups(
    audit_df: pd.DataFrame,
    registered_set_ids: "set[str] | frozenset[str]",
) -> pd.DataFrame:
    """Registered production feature-set groups flagged INCOMPLETE.

    Only groups whose ``set_id`` is a registered production feature set can
    block the final evaluation. ``unknown`` (no ECON reference — e.g. a
    restricted smoke run) and non-registered references never fail the run.
    """
    if audit_df is None or audit_df.empty or "status" not in audit_df.columns:
        return audit_df.iloc[0:0] if audit_df is not None else pd.DataFrame()
    reg = {str(s) for s in registered_set_ids}
    mask = (audit_df["set_id"].astype(str).isin(reg)
            & (audit_df["status"] == STATUS_INCOMPLETE))
    return audit_df[mask].copy()
