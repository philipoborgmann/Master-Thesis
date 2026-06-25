"""H2/H3 difference-in-improvement inference (Aufgabe 6.4).

The thesis asks **conditional** questions:

* **H2** — does the sentiment block help **more** in high-volatility regimes
  than in low-volatility regimes?
* **H3** — does the sentiment block help **more** in small-cap regimes than
  in large-cap regimes?

A regime-stratified McNemar test answers a different question: *"Did the
augmented model correct benchmark errors more often than vice versa
**within this regime**?"*. Two such tests at α = 5 % in two regimes is
**not** a test of "the effect in regime A differs from the effect in
regime B" — it's two independent location-vs-zero tests. The correct
H2/H3 inference compares the **mean improvement** between regimes
directly.

This module implements that test:

1. Compute the observation-level improvement indicator
   ``d_{i,t} = 1(augmented correct) − 1(ECON correct)`` ∈ {−1, 0, +1}.
2. Restrict to a two-bucket regime contrast (e.g. ``high`` vs ``low``
   volatility, or ``small`` vs ``large`` market-cap tercile — the
   ``mid`` bucket is dropped to keep the comparison sharp).
3. Regress ``d`` on a regime dummy: ``d = α + β · 1[regime = treatment]``
   with cluster-robust standard errors clustered **at minimum by
   ticker**.
4. ``β̂`` is the difference in mean improvement; its cluster-robust
   t-statistic + p-value answer "does the sentiment block help **more**
   under treatment than under control?"

McNemar tests **within** each regime stay in the supplementary table —
they describe the within-regime effect, not the between-regime
difference. Benjamini-Hochberg FDR control is applied **within each
hypothesis family** (H1, H2, H3 are independent families, never pooled).

Implementation notes
--------------------
* Cluster-robust variance uses the standard sandwich estimator
  ``V = (X'X)^{-1} · (Σ_g X_g'u_g u_g' X_g) · (X'X)^{-1}`` with the
  small-sample correction ``G/(G − 1) · (N − 1)/(N − k)``.
* When ``statsmodels`` is available we delegate to its OLS implementation;
  otherwise a pure-numpy fallback (``_ols_cluster_robust``) gives the
  same point estimate and SE up to floating-point precision.
* Family-aware BH is exposed via ``adjust_pvalues_bh_within_family``.
"""

from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from ..logging_utils import get_logger


# ---------------------------------------------------------------------------
# Observation-level improvement indicator
# ---------------------------------------------------------------------------

def observation_improvement_indicator(
    augmented: pd.DataFrame,
    econ: pd.DataFrame,
) -> pd.DataFrame:
    """Return a long-form frame with ``d = 1(aug correct) − 1(ECON correct)``.

    The two input frames must each carry ``timestamp``, ``ticker``,
    ``target``, ``prediction``. They are inner-joined on
    ``(timestamp, ticker)`` so ``d`` is defined only on matched
    observations.
    """
    if augmented.empty or econ.empty:
        return pd.DataFrame(columns=["timestamp", "ticker", "target",
                                      "d", "aug_correct", "econ_correct"])
    a = augmented[["timestamp", "ticker", "target", "prediction"]].copy()
    e = econ[["timestamp", "ticker", "target", "prediction"]].copy()
    a["aug_correct"]  = (a["prediction"].astype(int)
                          == a["target"].astype(int)).astype(int)
    e["econ_correct"] = (e["prediction"].astype(int)
                          == e["target"].astype(int)).astype(int)
    m = a.merge(
        e[["timestamp", "ticker", "econ_correct"]],
        on=["timestamp", "ticker"], how="inner",
    )
    m["d"] = m["aug_correct"].astype(int) - m["econ_correct"].astype(int)
    return m[["timestamp", "ticker", "target", "d",
              "aug_correct", "econ_correct"]].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Cluster-robust OLS (one-way clustering)
# ---------------------------------------------------------------------------

def _ols_cluster_robust(y: np.ndarray,
                        X: np.ndarray,
                        cluster_ids: np.ndarray,
                        small_sample: bool = True) -> dict:
    """Pure-numpy fallback for OLS with one-way cluster-robust SEs.

    Returns ``{beta, se, t, pvalue, n, k, n_clusters, dof}``. ``X`` MUST
    include the intercept column when one is desired.
    """
    from scipy.stats import t as student_t

    y = np.asarray(y, dtype=float)
    X = np.asarray(X, dtype=float)
    cluster_ids = np.asarray(cluster_ids)
    n, k = X.shape
    XtX_inv = np.linalg.pinv(X.T @ X)
    beta = XtX_inv @ X.T @ y
    resid = y - X @ beta

    # Sandwich middle: Σ_g (X_g' u_g)(X_g' u_g)'
    unique = pd.unique(cluster_ids)
    G = int(len(unique))
    mid = np.zeros((k, k), dtype=float)
    for g in unique:
        m = (cluster_ids == g)
        u_g = resid[m]
        X_g = X[m]
        s = X_g.T @ u_g
        mid += np.outer(s, s)
    correction = (G / max(G - 1, 1)) * ((n - 1) / max(n - k, 1)) if small_sample else 1.0
    vcov = correction * (XtX_inv @ mid @ XtX_inv)
    se = np.sqrt(np.clip(np.diag(vcov), 0.0, None))
    dof = max(G - 1, 1)
    t_stat = np.where(se > 0, beta / np.where(se > 0, se, 1.0), 0.0)
    pvalue = 2.0 * student_t.sf(np.abs(t_stat), df=dof)
    return {
        "beta":   beta,
        "se":     se,
        "t":      t_stat,
        "pvalue": pvalue,
        "vcov":   vcov,
        "n":      int(n),
        "k":      int(k),
        "n_clusters": G,
        "dof":    int(dof),
    }


def cluster_robust_difference_in_improvement(
    d: np.ndarray,
    treatment: np.ndarray,
    cluster_ids: np.ndarray,
    *,
    use_statsmodels: bool = True,
) -> dict:
    """Regress ``d`` on a treatment dummy with cluster-robust SE.

    Parameters
    ----------
    d
        Observation-level improvement indicator (``aug_correct - econ_correct``),
        any ∈ {−1, 0, +1}.
    treatment
        ``1`` for the *treatment* regime (e.g. high-vol), ``0`` for the
        *control* regime (e.g. low-vol). Observations with ``treatment``
        ∉ {0, 1} are dropped before the regression — the middle tercile
        therefore vanishes automatically when the caller supplies it as
        NaN.
    cluster_ids
        Cluster labels — typically the ticker. Pass anything hashable; the
        sandwich loops over the unique values.
    use_statsmodels
        Delegate to ``statsmodels.api.OLS`` with ``cov_type='cluster'``
        when available. Fall back to :func:`_ols_cluster_robust` otherwise.

    Returns
    -------
    A dict with ``alpha`` (control mean), ``beta`` (treatment − control),
    ``se_beta``, ``t_beta``, ``p_beta``, ``n_control``, ``n_treatment``,
    ``n_clusters``, ``dof``. Always non-empty; degenerate inputs collapse
    to NaN test statistics with ``test_valid=False``.
    """
    d = np.asarray(d, dtype=float)
    treatment = np.asarray(treatment, dtype=float)
    cluster_ids = np.asarray(cluster_ids)
    valid = ~np.isnan(d) & ~np.isnan(treatment) & np.isin(treatment, [0.0, 1.0])
    d = d[valid]
    t = treatment[valid]
    c = cluster_ids[valid]

    n_control = int((t == 0).sum())
    n_treat   = int((t == 1).sum())
    n_clusters = int(pd.unique(c).size) if len(c) else 0

    base = {
        "alpha":       float("nan"),
        "beta":        float("nan"),
        "se_beta":     float("nan"),
        "t_beta":      float("nan"),
        "p_beta":      float("nan"),
        "n_control":   n_control,
        "n_treatment": n_treat,
        "n_clusters":  n_clusters,
        "dof":         max(n_clusters - 1, 0),
        "test_valid":  False,
    }
    if n_control < 2 or n_treat < 2 or n_clusters < 2:
        return base

    X = np.column_stack([np.ones_like(t), t])

    if use_statsmodels:
        try:
            import statsmodels.api as sm
            model = sm.OLS(d, X).fit(
                cov_type="cluster", cov_kwds={"groups": c, "use_correction": True},
            )
            return {
                **base,
                "alpha":      float(model.params[0]),
                "beta":       float(model.params[1]),
                "se_beta":    float(model.bse[1]),
                "t_beta":     float(model.tvalues[1]),
                "p_beta":     float(model.pvalues[1]),
                "test_valid": True,
            }
        except Exception as exc:  # noqa: BLE001 — fall back to the numpy version
            get_logger().warning(
                "diff_in_improvement: statsmodels OLS cluster path failed (%s); "
                "falling back to numpy sandwich.", exc,
            )

    res = _ols_cluster_robust(d, X, c, small_sample=True)
    return {
        **base,
        "alpha":      float(res["beta"][0]),
        "beta":       float(res["beta"][1]),
        "se_beta":    float(res["se"][1]),
        "t_beta":     float(res["t"][1]),
        "p_beta":     float(res["pvalue"][1]),
        "dof":        int(res["dof"]),
        "test_valid": True,
    }


# ---------------------------------------------------------------------------
# H2/H3 table builder
# ---------------------------------------------------------------------------

H_HYPOTHESIS_FAMILIES = ("H1_incremental", "H2_volatility", "H3_market_cap")


def difference_in_improvement_table(
    signals: pd.DataFrame,
    matched_benchmark: dict[str, str],
    regime_lookup: pd.DataFrame,
    *,
    regime_col: str,
    treatment_value: str,
    control_value: str,
    hypothesis_family: str = "H2_volatility",
) -> pd.DataFrame:
    """One row per (combined-set, family) pair with the diff-in-improvement
    ``β̂`` (treatment − control mean of ``d``) plus its cluster-robust SE
    and p-value.

    Parameters
    ----------
    signals
        Long-form signal frame produced by the evaluation loader. Must
        carry ``set_id``, ``sentiment_model``, ``model_type``,
        ``panel_mode``, ``hpo_variant``, ``target``, ``prediction``,
        ``timestamp`` and ``ticker``.
    matched_benchmark
        Map ``set_id → matched-benchmark set_id`` (default for v4:
        :data:`thesis_pipeline.evaluation.incremental.MATCHED_ECONOMIC_BENCHMARK`).
    regime_lookup
        Long-form ``(ticker, date_or_timestamp, regime_col)`` frame. The
        join key on signals is ``(ticker, timestamp.normalize())`` for
        daily regimes (volatility, market-cap), so the lookup must carry
        normalised UTC midnight timestamps in ``date``.
    regime_col
        Column in ``regime_lookup`` carrying the regime label (e.g.
        ``vol_regime`` for H2, ``mcap_regime`` for H3).
    treatment_value / control_value
        Labels of the two regimes being contrasted. Any other value (e.g.
        the ``mid`` tercile) is dropped from the regression.
    hypothesis_family
        Free-text tag written to the output ``hypothesis_family`` column so
        the BH-within-family step can group rows. Conventionally
        ``"H2_volatility"`` / ``"H3_market_cap"``.
    """
    columns = [
        "hypothesis_family", "horizon", "set_id", "sentiment_model",
        "model_type", "panel_mode", "hpo_variant",
        "benchmark_set_id", "regime_col",
        "control_value", "treatment_value",
        "n_control", "n_treatment", "n_clusters",
        "mean_d_control", "mean_d_treatment",
        "diff_in_improvement", "se_diff", "t_stat", "p_value",
        "dof", "test_valid",
    ]
    if signals is None or signals.empty:
        return pd.DataFrame(columns=columns)
    if not matched_benchmark:
        return pd.DataFrame(columns=columns)

    df = signals.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df = df.dropna(subset=["timestamp"])
    df["__join_date__"] = df["timestamp"].dt.normalize()

    look = regime_lookup.copy()
    if "date" in look.columns:
        look["__join_date__"] = pd.to_datetime(look["date"], utc=True, errors="coerce")
        if look["__join_date__"].isna().any():
            # Tolerant fallback for tz-naive dates.
            naive = pd.to_datetime(look["date"], errors="coerce")
            look["__join_date__"] = naive.dt.tz_localize("UTC", nonexistent="shift_forward")
    elif "timestamp" in look.columns:
        look["__join_date__"] = pd.to_datetime(look["timestamp"], utc=True, errors="coerce").dt.normalize()
    else:
        raise ValueError(
            "difference_in_improvement_table: regime_lookup must have a "
            "'date' or 'timestamp' column."
        )
    look = look[["ticker", "__join_date__", regime_col]].copy()
    look["ticker"] = look["ticker"].astype(str).str.upper()
    df["ticker"]   = df["ticker"].astype(str).str.upper()
    look = look.dropna(subset=["__join_date__", regime_col])

    rows: list[dict] = []
    group_cols = ["horizon", "set_id", "sentiment_model", "model_type",
                  "panel_mode", "hpo_variant"]
    augmented_ids = set(matched_benchmark.keys())

    for keys, aug_grp in df[df["set_id"].astype(str).isin(augmented_ids)].groupby(
            group_cols, dropna=False):
        horizon, set_id, sm, model_type, panel_mode, hpo_variant = keys
        bench_set_id = matched_benchmark[str(set_id)]

        cand = df[
            (df["horizon"] == horizon)
            & (df["model_type"] == model_type)
            & (df["panel_mode"] == panel_mode)
            & (df["hpo_variant"].astype(str) == str(hpo_variant))
            & (df["set_id"] == bench_set_id)
        ]
        if cand.empty:
            continue

        d_frame = observation_improvement_indicator(aug_grp, cand)
        if d_frame.empty:
            continue

        d_frame = d_frame.copy()
        d_frame["__join_date__"] = pd.to_datetime(
            d_frame["timestamp"], utc=True, errors="coerce",
        ).dt.normalize()
        d_frame["ticker"] = d_frame["ticker"].astype(str).str.upper()
        joined = d_frame.merge(look, on=["ticker", "__join_date__"], how="left")
        if joined[regime_col].isna().all():
            continue

        treatment = joined[regime_col].map(
            {treatment_value: 1.0, control_value: 0.0}
        ).astype(float).to_numpy()
        res = cluster_robust_difference_in_improvement(
            d=joined["d"].to_numpy(dtype=float),
            treatment=treatment,
            cluster_ids=joined["ticker"].to_numpy(),
        )
        d_ctrl = joined.loc[joined[regime_col] == control_value,   "d"]
        d_trt  = joined.loc[joined[regime_col] == treatment_value, "d"]
        rows.append({
            "hypothesis_family":      hypothesis_family,
            "horizon":                horizon,
            "set_id":                 set_id,
            "sentiment_model":        sm,
            "model_type":             model_type,
            "panel_mode":             panel_mode,
            "hpo_variant":            hpo_variant,
            "benchmark_set_id":       bench_set_id,
            "regime_col":             regime_col,
            "control_value":          control_value,
            "treatment_value":        treatment_value,
            "n_control":              res["n_control"],
            "n_treatment":            res["n_treatment"],
            "n_clusters":             res["n_clusters"],
            "mean_d_control":         float(d_ctrl.mean()) if not d_ctrl.empty else float("nan"),
            "mean_d_treatment":       float(d_trt.mean())  if not d_trt.empty  else float("nan"),
            "diff_in_improvement":    res["beta"],
            "se_diff":                res["se_beta"],
            "t_stat":                 res["t_beta"],
            "p_value":                res["p_beta"],
            "dof":                    res["dof"],
            "test_valid":             res["test_valid"],
        })
    return pd.DataFrame(rows, columns=columns)


# ---------------------------------------------------------------------------
# Family-aware BH adjustment
# ---------------------------------------------------------------------------

def adjust_pvalues_bh_within_family(
    df: pd.DataFrame,
    *,
    family_col: str = "hypothesis_family",
    p_col: str = "p_value",
    q_col: str = "q_value_bh",
    sig5_col: str = "significant_bh_5pct",
    sig10_col: str = "significant_bh_10pct",
) -> pd.DataFrame:
    """Append BH/FDR columns ``q_value_bh`` / ``significant_bh_*`` computed
    **separately within each hypothesis family** (per the v4 H1 / H2 / H3
    convention).

    Tests with NaN p-values do not enter the FDR pool for their family.
    """
    out = df.copy()
    out[q_col]    = np.nan
    out[sig5_col]  = False
    out[sig10_col] = False
    if out.empty or family_col not in out.columns or p_col not in out.columns:
        return out

    try:
        from statsmodels.stats.multitest import multipletests
        _have_sm = True
    except Exception:  # noqa: BLE001
        _have_sm = False

    for fam in out[family_col].dropna().unique():
        mask = (out[family_col] == fam) & out[p_col].notna()
        pvals = out.loc[mask, p_col].to_numpy(dtype=float)
        if len(pvals) == 0:
            continue
        if _have_sm:
            rej5, q5, _, _ = multipletests(pvals, alpha=0.05, method="fdr_bh")
            rej10, _, _, _ = multipletests(pvals, alpha=0.10, method="fdr_bh")
        else:
            # Pure-numpy fallback.
            order = np.argsort(pvals)
            ranked = pvals[order]
            n = len(pvals)
            q = ranked * n / np.arange(1, n + 1)
            q = np.minimum.accumulate(q[::-1])[::-1]
            q = np.clip(q, 0.0, 1.0)
            q5 = np.empty(n, dtype=float)
            q5[order] = q
            rej5  = q5 < 0.05
            rej10 = q5 < 0.10
        out.loc[mask, q_col]     = q5
        out.loc[mask, sig5_col]  = rej5
        out.loc[mask, sig10_col] = rej10
    return out
