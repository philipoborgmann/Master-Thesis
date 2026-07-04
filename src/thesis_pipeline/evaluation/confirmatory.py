"""Confirmatory multiplicity orchestration (Parts 1 + 2B).

This is the single place that turns the per-(horizon, set) raw p-values
of every confirmatory test into ONE family-aware Benjamini-Hochberg
correction, then decomposes H1 into probabilistic (H1a) and directional
(H1b) support.

It consumes the already-built ``incremental_sentiment_value`` and
``diff_in_improvement`` tables plus the loaded signal frame, computes the
timestamp-level Diebold-Mariano-type effects (Part 2A), assembles the
confirmatory records for families A/B/C/D/E1/E2, runs the centralized BH
(:mod:`multiple_testing`), and distributes the corrected q-values /
significance back onto each surface. It also emits the horizon-comparison
table, the multiple-testing manifest, the metric-roles table and the
class-balance table.

Guardrails honoured here:
  * BH pools p-values ACROSS horizons within each family (never
    per-horizon); B/C/D reproduce the legacy pools byte-identically.
  * A single ``significant_bh`` at ``ALPHA_PRESPECIFIED``; no 10% flag.
  * absolute_vs_naive and regime_mcnemar are NEVER assigned a family.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import forecast_diff_tests as fdt
from . import multiple_testing as mt
from .incremental import (
    MATCHED_ECONOMIC_BENCHMARK, iter_incremental_prediction_pairs,
)
from .preregistration import (
    ALPHA_PRESPECIFIED, CLASS_IMBALANCE_DELTA, DM_INFERENCE, FAMILY_E_METRICS,
    FAMILY_A_H1_LOGLOSS, FAMILY_B_H1_DIRECTIONAL, FAMILY_C_H2_VOLATILITY,
    FAMILY_D_H3_MARKETCAP, FAMILY_E1_HORIZON_LOGLOSS, FAMILY_E2_HORIZON_ACCURACY,
    METRIC_ROLES,
)

@dataclass
class ConfirmatoryResult:
    incremental: pd.DataFrame
    diff: pd.DataFrame
    horizon_comparison: pd.DataFrame
    manifest: pd.DataFrame
    metric_roles: pd.DataFrame
    class_balance: pd.DataFrame
    dm_effects: pd.DataFrame = field(default_factory=pd.DataFrame)


# ---------------------------------------------------------------------------
# Part 2A — DM effects per (horizon, set)
# ---------------------------------------------------------------------------

def compute_dm_effects(signals: pd.DataFrame,
                       incremental_df: pd.DataFrame) -> pd.DataFrame:
    """One row per confirmatory (horizon, combined set) with the log-loss
    and accuracy timestamp effects + bootstrap inference.

    Runs on EXACTLY the paired observations the incremental table
    summarises (shared ``iter_incremental_prediction_pairs``), so the
    Nt-weighted effects reconcile with the pooled improvements.
    """
    if signals is None or getattr(signals, "empty", True):
        return pd.DataFrame()
    rows = []
    for identity, merged in iter_incremental_prediction_pairs(signals):
        set_id = str(identity["set_id"])
        if set_id not in MATCHED_ECONOMIC_BENCHMARK:
            continue
        horizon = identity["horizon"]
        sent = merged[["ticker", "timestamp", "target",
                       "prediction", "probability"]].copy()
        econ = merged[["ticker", "timestamp", "target",
                       "prediction_b", "probability_b"]].rename(
            columns={"prediction_b": "prediction",
                     "probability_b": "probability"}).copy()
        ll = fdt.dm_logloss_timestamp_test(econ, sent, DM_INFERENCE)
        acc = fdt.accuracy_timestamp_effect(econ, sent, DM_INFERENCE)
        rows.append({
            "horizon": horizon, "set_id": set_id,
            "sentiment_model": identity.get("sentiment_model"),
            "model": identity.get("sentiment_model"),
            # log-loss effect block
            "ll_effect": ll["ll_effect"],
            "ll_effect_ntweighted": ll["ll_effect_ntweighted"],
            "ll_se_hac": ll["ll_se_hac"], "ll_se_boot": ll["ll_se_boot"],
            "ll_ci95_low": ll["ll_ci95_low"], "ll_ci95_high": ll["ll_ci95_high"],
            "ll_z_hac": ll["ll_z_hac"], "ll_p_hac": ll["ll_p_hac"],
            "ll_p_boot": ll["ll_p_boot"], "ll_p_value_raw": ll["ll_p_value_raw"],
            "ll_test_valid": ll["test_valid"],
            "ll_skip_reason": ll["skip_reason"],
            # accuracy effect block
            "acc_effect": acc["acc_effect"],
            "acc_effect_ntweighted": acc["acc_effect_ntweighted"],
            "acc_se_hac": acc["acc_se_hac"], "acc_se_boot": acc["acc_se_boot"],
            "acc_ci95_low": acc["acc_ci95_low"], "acc_ci95_high": acc["acc_ci95_high"],
            "acc_p_boot": acc["acc_p_boot"],
            "acc_p_value_raw": acc["acc_p_value_raw"],
            "acc_test_valid": acc["test_valid"],
            # shared diagnostics (log-loss side is authoritative)
            "n_timestamps": ll["n_timestamps"],
            "hac_lag": ll["hac_lag"], "block_length": ll["block_length"],
            "dm_inference_primary": ll["dm_inference_primary"],
            "dm_nested_caveat": ll["dm_nested_caveat"],
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Part 1E — horizon contrasts
# ---------------------------------------------------------------------------

_HORIZON_PAIRS = (("1d", "6h"), ("1d", "1h"), ("6h", "1h"))


def _horizon_contrasts(dm_effects: pd.DataFrame, *, metric: str,
                       effect_col: str, se_col: str, family: str) -> list[dict]:
    from scipy.stats import norm
    rows = []
    for set_id, grp in dm_effects.groupby("set_id", dropna=False):
        by_h = {str(r["horizon"]): r for _, r in grp.iterrows()}
        model = grp["model"].iloc[0] if "model" in grp.columns else None
        for h1, h2 in _HORIZON_PAIRS:
            r1, r2 = by_h.get(h1), by_h.get(h2)
            e1 = float(r1[effect_col]) if r1 is not None else np.nan
            e2 = float(r2[effect_col]) if r2 is not None else np.nan
            s1 = float(r1[se_col]) if r1 is not None else np.nan
            s2 = float(r2[se_col]) if r2 is not None else np.nan
            diff = e1 - e2
            se = np.sqrt(s1 ** 2 + s2 ** 2)
            if np.isfinite(se) and se > 0 and np.isfinite(diff):
                z = diff / se
                p = float(2.0 * norm.sf(abs(z)))
            else:
                z, p = np.nan, np.nan
            rows.append({
                "family": family, "set_id": set_id, "model": model,
                "metric": metric, "horizon_pair": f"{h1}_vs_{h2}",
                "effect_h1": e1, "effect_h2": e2, "diff": diff,
                "se_diff": se, "z_stat": z, "p_value_raw": p,
                "test_id": mt.make_test_id(family, f"{h1}_vs_{h2}", set_id),
            })
    return rows


# ---------------------------------------------------------------------------
# Class balance + metric roles
# ---------------------------------------------------------------------------

def build_class_balance(signals: pd.DataFrame) -> pd.DataFrame:
    if signals is None or signals.empty or "target" not in signals.columns:
        return pd.DataFrame(columns=["horizon", "n", "base_rate_up",
                                      "delta_from_0.5", "imbalance_flag"])
    rows = []
    for hz, grp in signals.groupby("horizon", dropna=False):
        # One row per (ticker, timestamp) is enough; but base rate over the
        # full observed y is the class balance the models see.
        y = grp["target"].astype(int)
        base = float(y.mean()) if len(y) else float("nan")
        delta = abs(base - 0.5) if np.isfinite(base) else float("nan")
        rows.append({
            "horizon": hz, "n": int(len(y)),
            "base_rate_up": base, "delta_from_0.5": delta,
            "imbalance_flag": bool(np.isfinite(delta)
                                   and delta > CLASS_IMBALANCE_DELTA),
        })
    return pd.DataFrame(rows)


def build_metric_roles(class_balance: pd.DataFrame) -> pd.DataFrame:
    any_imbalance = (bool(class_balance["imbalance_flag"].any())
                     if not class_balance.empty else False)
    rows = []
    for metric, spec in METRIC_ROLES.items():
        note = spec["notes"]
        if metric == "balanced_accuracy" and any_imbalance:
            note = ("Elevated to primary-robustness: at least one horizon "
                    "is class-imbalanced. " + note)
        rows.append({
            "metric": metric, "role": spec["role"],
            "used_for_hpo": spec["used_for_hpo"], "notes": note,
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# H1 decomposition
# ---------------------------------------------------------------------------

def _h1_support_class(h1a: bool, h1b: bool) -> str:
    if h1a and h1b:
        return "strong"
    if h1a and not h1b:
        return "probabilistic"
    if h1b and not h1a:
        return "directional"
    return "none"


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def finalize_confirmatory_families(*,
                                   signals: pd.DataFrame,
                                   incremental_df: pd.DataFrame,
                                   diff_df: pd.DataFrame,
                                   alpha: float = ALPHA_PRESPECIFIED,
                                   ) -> ConfirmatoryResult:
    inc = incremental_df.copy() if incremental_df is not None else pd.DataFrame()
    dif = diff_df.copy() if diff_df is not None else pd.DataFrame()

    class_balance = build_class_balance(signals)
    metric_roles = build_metric_roles(class_balance)

    dm = compute_dm_effects(signals, inc)

    # ── Assemble confirmatory records ─────────────────────────────
    records = []
    # Family A — log-loss DM (per horizon, set).
    if not dm.empty:
        for _, r in dm.iterrows():
            records.append({
                "test_id": mt.make_test_id(FAMILY_A_H1_LOGLOSS, r["horizon"], r["set_id"]),
                "family": FAMILY_A_H1_LOGLOSS, "horizon": r["horizon"],
                "model": r.get("model"), "set_id": r["set_id"],
                "metric": "log_loss", "p_value_raw": r["ll_p_value_raw"],
            })
    # Family B — McNemar (from incremental ok rows).
    if not inc.empty:
        for _, r in inc[inc["status"].astype(str) == "ok"].iterrows():
            records.append({
                "test_id": mt.make_test_id(FAMILY_B_H1_DIRECTIONAL, r["horizon"], r["set_id"]),
                "family": FAMILY_B_H1_DIRECTIONAL, "horizon": r["horizon"],
                "model": r.get("sentiment_model"), "set_id": r["set_id"],
                "metric": "accuracy", "p_value_raw": r["mcnemar_p_value"],
            })
    # Families C / D — diff-in-improvement (valid rows only).
    if not dif.empty and "p_value" in dif.columns:
        fam_map = {"H2_volatility": FAMILY_C_H2_VOLATILITY,
                   "H3_market_cap": FAMILY_D_H3_MARKETCAP}
        valid = dif[dif["test_valid"] == True]  # noqa: E712
        for _, r in valid.iterrows():
            fam = fam_map.get(str(r.get("hypothesis_family")))
            if fam is None:
                continue
            records.append({
                "test_id": mt.make_test_id(fam, r["horizon"], r["set_id"]),
                "family": fam, "horizon": r["horizon"],
                "model": r.get("sentiment_model"), "set_id": r["set_id"],
                "metric": "accuracy", "p_value_raw": r["p_value"],
            })
    # Families E1 / E2 — horizon contrasts.
    horizon_rows = []
    if not dm.empty:
        if "log_loss" in FAMILY_E_METRICS:
            horizon_rows += _horizon_contrasts(
                dm, metric="log_loss", effect_col="ll_effect",
                se_col="ll_se_boot", family=FAMILY_E1_HORIZON_LOGLOSS)
        if "accuracy" in FAMILY_E_METRICS:
            horizon_rows += _horizon_contrasts(
                dm, metric="accuracy", effect_col="acc_effect",
                se_col="acc_se_boot", family=FAMILY_E2_HORIZON_ACCURACY)
    for hr in horizon_rows:
        records.append({
            "test_id": hr["test_id"], "family": hr["family"],
            "horizon": hr["horizon_pair"], "model": hr.get("model"),
            "set_id": hr["set_id"], "metric": hr["metric"],
            "p_value_raw": hr["p_value_raw"],
        })

    records_df = pd.DataFrame(records, columns=list(mt.RECORD_COLUMNS))
    corrected = mt.apply_family_bh(records_df, alpha=alpha)
    manifest = mt.build_manifest(corrected, alpha=alpha)

    # ── Distribute q/sig back onto each surface ───────────────────
    inc = _apply_families_to_incremental(inc, dm, corrected, alpha)
    dif = _apply_families_to_diff(dif, corrected, alpha)
    horizon_comparison = _build_horizon_comparison(horizon_rows, corrected, alpha)

    return ConfirmatoryResult(
        incremental=inc, diff=dif,
        horizon_comparison=horizon_comparison, manifest=manifest,
        metric_roles=metric_roles, class_balance=class_balance,
        dm_effects=dm,
    )


def _corrected_lookup(corrected: pd.DataFrame, family: str) -> dict:
    if corrected.empty:
        return {}
    sub = corrected[corrected["family"] == family]
    return {tid: (q, sig) for tid, q, sig in
            zip(sub["test_id"], sub["q_value_bh"], sub["significant_bh"])}


def _apply_families_to_incremental(inc, dm, corrected, alpha):
    if inc.empty:
        return inc
    inc = inc.copy()
    # Drop the legacy self-corrected columns; the centralized pass is
    # authoritative and there is no 10% flag.
    for col in ("significant_bh_5pct", "significant_bh_10pct",
                "interpretation_bh"):
        if col in inc.columns:
            inc = inc.drop(columns=col)
    inc["family"] = FAMILY_B_H1_DIRECTIONAL
    inc["alpha_prespecified"] = float(alpha)
    inc["p_value_raw"] = inc["mcnemar_p_value"]
    # Merge the log-loss DM effect columns (Family A) by (horizon, set).
    a_cols = ["ll_effect", "ll_effect_ntweighted", "ll_se_hac", "ll_se_boot",
              "ll_ci95_low", "ll_ci95_high", "ll_z_hac", "ll_p_hac",
              "ll_p_boot", "ll_p_value_raw", "dm_nested_caveat",
              "n_timestamps", "hac_lag", "block_length"]
    if not dm.empty:
        inc = inc.merge(dm[["horizon", "set_id"] + a_cols],
                        on=["horizon", "set_id"], how="left")
    else:
        for c in a_cols:
            inc[c] = np.nan

    b_look = _corrected_lookup(corrected, FAMILY_B_H1_DIRECTIONAL)
    a_look = _corrected_lookup(corrected, FAMILY_A_H1_LOGLOSS)
    q_b = np.full(len(inc), np.nan); sig_b = np.zeros(len(inc), dtype=bool)
    q_a = np.full(len(inc), np.nan); sig_a = np.zeros(len(inc), dtype=bool)
    for i, (_, r) in enumerate(inc.iterrows()):
        tid_b = mt.make_test_id(FAMILY_B_H1_DIRECTIONAL, r["horizon"], r["set_id"])
        tid_a = mt.make_test_id(FAMILY_A_H1_LOGLOSS, r["horizon"], r["set_id"])
        if tid_b in b_look:
            q_b[i], sig_b[i] = b_look[tid_b]
        if tid_a in a_look:
            q_a[i], sig_a[i] = a_look[tid_a]
    inc["q_value_bh"] = q_b                     # directional (Family B)
    inc["significant_bh"] = sig_b
    inc["ll_q_value_bh"] = q_a                  # probabilistic (Family A)
    inc["ll_significant_bh"] = sig_a

    # H1 decomposition.
    acc_lift = inc["accuracy_lift"].fillna(0.0).to_numpy(dtype=float)
    ll_eff = inc["ll_effect"].fillna(0.0).to_numpy(dtype=float)
    h1a = (ll_eff > 0) & sig_a
    h1b = (acc_lift > 0) & sig_b
    inc["h1a_supported_bh"] = h1a
    inc["h1b_supported_bh"] = h1b
    inc["h1_support_class"] = [_h1_support_class(bool(a), bool(b))
                               for a, b in zip(h1a, h1b)]
    return inc


def _apply_families_to_diff(dif, corrected, alpha):
    if dif.empty:
        return dif
    dif = dif.copy()
    for col in ("significant_bh_5pct", "significant_bh_10pct"):
        if col in dif.columns:
            dif = dif.drop(columns=col)
    fam_map = {"H2_volatility": FAMILY_C_H2_VOLATILITY,
               "H3_market_cap": FAMILY_D_H3_MARKETCAP}
    dif["family"] = dif["hypothesis_family"].map(fam_map).fillna(
        dif["hypothesis_family"])
    dif["alpha_prespecified"] = float(alpha)
    dif["p_value_raw"] = dif["p_value"]
    c_look = _corrected_lookup(corrected, FAMILY_C_H2_VOLATILITY)
    d_look = _corrected_lookup(corrected, FAMILY_D_H3_MARKETCAP)
    q = np.full(len(dif), np.nan); sig = np.zeros(len(dif), dtype=bool)
    for i, (_, r) in enumerate(dif.iterrows()):
        fam = fam_map.get(str(r.get("hypothesis_family")))
        if fam is None:
            continue
        tid = mt.make_test_id(fam, r["horizon"], r["set_id"])
        look = c_look if fam == FAMILY_C_H2_VOLATILITY else d_look
        if tid in look:
            q[i], sig[i] = look[tid]
    dif["q_value_bh"] = q
    dif["significant_bh"] = sig
    return dif


def _build_horizon_comparison(horizon_rows, corrected, alpha):
    if not horizon_rows:
        return pd.DataFrame(columns=[
            "family", "set_id", "model", "metric", "horizon_pair",
            "effect_h1", "effect_h2", "diff", "se_diff", "z_stat",
            "p_value_raw", "q_value_bh", "significant_bh", "alpha_prespecified"])
    df = pd.DataFrame(horizon_rows)
    look = {}
    for fam in (FAMILY_E1_HORIZON_LOGLOSS, FAMILY_E2_HORIZON_ACCURACY):
        look.update(_corrected_lookup(corrected, fam))
    q = np.full(len(df), np.nan); sig = np.zeros(len(df), dtype=bool)
    for i, tid in enumerate(df["test_id"]):
        if tid in look:
            q[i], sig[i] = look[tid]
    df["q_value_bh"] = q
    df["significant_bh"] = sig
    df["alpha_prespecified"] = float(alpha)
    ordered = ["family", "set_id", "model", "metric", "horizon_pair",
               "effect_h1", "effect_h2", "diff", "se_diff", "z_stat",
               "p_value_raw", "q_value_bh", "significant_bh",
               "alpha_prespecified"]
    return df[ordered].reset_index(drop=True)
