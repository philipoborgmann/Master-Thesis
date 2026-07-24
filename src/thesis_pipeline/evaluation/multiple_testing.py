"""Centralized family-aware multiple-testing correction (Part 1).

ALL confirmatory Benjamini-Hochberg correction flows through this one
module. Every confirmatory test emits a record
``(test_id, family, horizon, model, metric, p_value_raw)``; the module
groups by ``family`` and applies BH WITHIN each family at that family's
pre-specified ``alpha`` (default 0.05), pooling p-values ACROSS horizons
within the family (never per-horizon).

Design guarantees (guardrails)
------------------------------
* Log-loss (Families A, E1) and directional (Families B, E2) p-values
  NEVER share a family.
* Economic/backtest p-values (Family F) never share a family with
  forecast-quality tests.
* Descriptive surfaces (absolute_vs_naive, regime_mcnemar_*, coin-level)
  are NEVER assigned a q-value or a family here.
* Exactly one BH pass per family reproduces the legacy per-family pools
  for B, C, D byte-identically (same p-value set, same statsmodels
  ``fdr_bh``), so only the NEW families A, E1, E2 add q-values.
* A single boolean ``significant_bh`` at ``alpha`` — no 10% flag.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .preregistration import (
    ALPHA_PRESPECIFIED, CONFIRMATORY_FAMILIES, FAMILY_DESCRIPTIONS,
)

#: Canonical columns every emitted test record carries.
RECORD_COLUMNS = ("test_id", "family", "horizon", "model", "set_id",
                  "metric", "p_value_raw")

#: Columns the correction appends.
CORRECTION_COLUMNS = ("q_value_bh", "alpha_prespecified", "significant_bh")


@dataclass
class FamilyResult:
    family: str
    n_tests: int
    alpha_prespecified: float
    n_significant_bh: int
    member_test_ids: list


def _bh_qvalues(pvals: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg q-values (monotone). Uses statsmodels when
    available, else a pure-numpy fallback — both reproduce the legacy
    ``adjust_pvalues_bh_within_family`` output."""
    p = np.asarray(pvals, dtype=float)
    n = p.size
    if n == 0:
        return p
    try:
        from statsmodels.stats.multitest import multipletests
        _, q, _, _ = multipletests(p, alpha=0.05, method="fdr_bh")
        return q
    except Exception:  # noqa: BLE001 — pure-numpy fallback
        order = np.argsort(p)
        ranked = p[order]
        q = ranked * n / np.arange(1, n + 1)
        q = np.minimum.accumulate(q[::-1])[::-1]
        q = np.clip(q, 0.0, 1.0)
        out = np.empty(n, dtype=float)
        out[order] = q
        return out


def apply_family_bh(records: pd.DataFrame,
                    *,
                    alpha: float = ALPHA_PRESPECIFIED,
                    p_col: str = "p_value_raw",
                    family_col: str = "family") -> pd.DataFrame:
    """Append ``q_value_bh``, ``alpha_prespecified`` and ``significant_bh``
    to ``records``, computed WITHIN each family.

    A record whose ``p_value_raw`` is NaN does not enter its family's BH
    pool (its q-value stays NaN, ``significant_bh`` False). ``alpha`` is
    applied uniformly (one knob per family; all default 0.05).
    """
    out = records.copy()
    out["q_value_bh"]         = np.nan
    out["alpha_prespecified"] = float(alpha)
    out["significant_bh"]     = False
    if out.empty or family_col not in out.columns or p_col not in out.columns:
        return out

    for fam in out[family_col].dropna().unique():
        mask = (out[family_col] == fam) & out[p_col].notna()
        idx = out.index[mask]
        if len(idx) == 0:
            continue
        q = _bh_qvalues(out.loc[idx, p_col].to_numpy(dtype=float))
        out.loc[idx, "q_value_bh"]     = q
        out.loc[idx, "significant_bh"] = q < alpha
    return out


def build_manifest(corrected: pd.DataFrame,
                   *,
                   alpha: float = ALPHA_PRESPECIFIED,
                   families: tuple = CONFIRMATORY_FAMILIES,
                   family_role: str = "confirmatory") -> pd.DataFrame:
    """One row per family: description, n_tests, alpha_prespecified,
    n_significant_bh, correction, family_role, member test count.

    ``family_role`` tags every row emitted here (``"confirmatory"`` by
    default). Exploratory families are appended by the caller with
    ``family_role="exploratory"`` so a reviewer can tell the pre-specified
    confirmatory set apart from post-hoc exploratory families at a glance.
    Descriptive surfaces (regime_mcnemar, absolute_vs_naive) never receive a
    family and so never appear here.
    """
    rows = []
    for fam in families:
        sub = corrected[corrected["family"] == fam] if not corrected.empty \
            else corrected
        valid = sub[sub["p_value_raw"].notna()] if not sub.empty else sub
        n_tests = int(len(valid))
        n_sig = int(valid["significant_bh"].sum()) if n_tests else 0
        member_ids = (list(valid["test_id"].astype(str))
                      if n_tests else [])
        rows.append({
            "family":              fam,
            "family_role":         family_role,
            "description":         FAMILY_DESCRIPTIONS.get(fam, ""),
            "n_tests":             n_tests,
            "alpha_prespecified":  float(alpha),
            "n_significant_bh":    n_sig,
            "correction":          "BH",
            "member_test_ids":     ";".join(member_ids),
        })
    return pd.DataFrame(rows)


def make_test_id(family: str, horizon, set_id, model=None) -> str:
    """Stable test identifier ``{family}|{horizon}|{set_id}``."""
    return f"{family}|{horizon}|{set_id}"
