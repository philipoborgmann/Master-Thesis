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


#: Below this number of clusters the cluster-robust inference is fragile;
#: the test stays ``test_valid=True`` but the result carries
#: ``small_cluster_warning=True`` so the consumer can flag it. For the
#: thesis production setting with ~25 tickers the threshold is well clear.
SMALL_CLUSTER_THRESHOLD = 10


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
        Observation-level improvement indicator
        (``aug_correct - econ_correct``), any ∈ {−1, 0, +1}.
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
        Use ``statsmodels.api.OLS`` for the coefficient + covariance
        estimation when available. The p-value is **always** recomputed
        from ``Student-t(df = n_clusters - 1)`` so the two code paths
        agree exactly — statsmodels' default cluster path returns an
        asymptotic-normal p-value that this function intentionally
        overrides for consistency with the pure-numpy fallback.

    Returns
    -------
    A dict with ``alpha`` (control mean), ``beta`` (treatment − control),
    ``se_beta``, ``t_beta``, ``p_beta``, ``n_control``, ``n_treatment``,
    ``n_clusters``, ``dof``, ``test_valid``, ``small_cluster_warning``.
    Degenerate inputs collapse to NaN test statistics with
    ``test_valid=False``.
    """
    from scipy.stats import t as student_t

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
    small_cluster = bool(0 < n_clusters < SMALL_CLUSTER_THRESHOLD)

    base = {
        "alpha":                  float("nan"),
        "beta":                   float("nan"),
        "se_beta":                float("nan"),
        "t_beta":                 float("nan"),
        "p_beta":                 float("nan"),
        "n_control":              n_control,
        "n_treatment":            n_treat,
        "n_clusters":             n_clusters,
        "dof":                    max(n_clusters - 1, 0),
        "test_valid":             False,
        "small_cluster_warning":  small_cluster,
    }
    if n_control < 2 or n_treat < 2 or n_clusters < 2:
        return base

    X = np.column_stack([np.ones_like(t), t])
    dof = n_clusters - 1

    # ── Estimate coefficient + cluster-robust SE ─────────────────
    beta = None
    se_beta = None
    if use_statsmodels:
        try:
            import statsmodels.api as sm
            model = sm.OLS(d, X).fit(
                cov_type="cluster",
                cov_kwds={"groups": c, "use_correction": True},
            )
            beta    = float(model.params[1])
            se_beta = float(model.bse[1])
            alpha   = float(model.params[0])
        except Exception as exc:  # noqa: BLE001 — fall back to the numpy version
            get_logger().warning(
                "diff_in_improvement: statsmodels OLS cluster path failed (%s); "
                "falling back to numpy sandwich.", exc,
            )
            beta = None

    if beta is None:
        res = _ols_cluster_robust(d, X, c, small_sample=True)
        alpha   = float(res["beta"][0])
        beta    = float(res["beta"][1])
        se_beta = float(res["se"][1])

    # ── ALWAYS recompute t / p from Student-t(dof = n_clusters - 1) ──
    # Both paths share this convention so the statsmodels and numpy
    # results match bit-for-bit downstream.
    if se_beta > 0:
        t_stat = float(beta / se_beta)
    else:
        t_stat = 0.0
    p_value = float(2.0 * student_t.sf(abs(t_stat), df=dof))

    return {
        **base,
        "alpha":                 alpha,
        "beta":                  beta,
        "se_beta":               se_beta,
        "t_beta":                t_stat,
        "p_beta":                p_value,
        "dof":                   dof,
        "test_valid":            True,
        "small_cluster_warning": small_cluster,
    }


# ---------------------------------------------------------------------------
# H2/H3 table builder
# ---------------------------------------------------------------------------

H_HYPOTHESIS_FAMILIES = ("H1_incremental", "H2_volatility", "H3_market_cap")


#: Complete v4 model-family identity columns used to match augmented vs
#: ECON signals before the diff-in-improvement regression. A mismatch on
#: any one of these would silently mix incompatible runs (e.g. an
#: expanding-window ECON vs a rolling-fixed combined), so the columns
#: form a strict join key. ``hpo_objective`` is included so a model
#: tuned with brier_score is never lifted against a log_loss ECON.
H2H3_FAMILY_COLUMNS = (
    "horizon",
    "model_type",
    "panel_mode",
    "hpo_variant",
    "hpo_objective",
    "train_window_mode",
    "rolling_window_days",
    "rolling_window_timestamps",
)


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
        "model_type", "panel_mode", "hpo_variant", "hpo_objective",
        "train_window_mode", "rolling_window_days", "rolling_window_timestamps",
        "benchmark_set_id", "regime_col",
        "control_value", "treatment_value",
        "n_augmented", "n_econ", "n_matched",
        "n_unmatched_augmented", "n_unmatched_econ",
        "n_duplicate_augmented_keys", "n_duplicate_econ_keys",
        "targets_identical",
        "n_control", "n_treatment", "n_clusters",
        "mean_d_control", "mean_d_treatment",
        "diff_in_improvement", "se_diff", "t_stat", "p_value",
        "dof", "test_valid", "small_cluster_warning", "skip_reason",
    ]
    if signals is None or signals.empty:
        return pd.DataFrame(columns=columns)
    if not matched_benchmark:
        return pd.DataFrame(columns=columns)

    df = signals.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df = df.dropna(subset=["timestamp"])

    look = _prepare_regime_lookup(regime_lookup, regime_col)

    df["ticker"] = df["ticker"].astype(str).str.upper()
    look["ticker"] = look["ticker"].astype(str).str.upper()

    rows: list[dict] = []
    # Group augmented signals by COMPLETE family identity (Section D).
    # Every column in H2H3_FAMILY_COLUMNS must match between augmented
    # and ECON; partial matches were previously possible if only
    # (horizon, model_type, panel_mode, hpo_variant) lined up but the
    # training-window configurations diverged.
    group_cols = list(H2H3_FAMILY_COLUMNS) + ["set_id", "sentiment_model"]
    # Fill missing family columns on the model side with NaN so the
    # groupby never silently drops a row.
    for c in H2H3_FAMILY_COLUMNS:
        if c not in df.columns:
            df[c] = np.nan
    augmented_ids = set(matched_benchmark.keys())
    augmented = df[df["set_id"].astype(str).isin(augmented_ids)]

    for keys, aug_grp in augmented.groupby(group_cols, dropna=False):
        ident = dict(zip(group_cols, keys))
        bench_set_id = matched_benchmark[str(ident["set_id"])]

        # ECON candidate must share the complete family identity.
        cand_mask = (df["set_id"] == bench_set_id)
        for fam_col in H2H3_FAMILY_COLUMNS:
            cand_mask &= _series_eq_nan_safe(df[fam_col], ident.get(fam_col))
        cand = df[cand_mask]

        diag_row = _empty_row(columns, ident, bench_set_id, hypothesis_family,
                              regime_col, control_value, treatment_value)
        diag_row["n_augmented"] = int(len(aug_grp))
        diag_row["n_econ"]      = int(len(cand))

        if cand.empty:
            diag_row["skip_reason"] = "no_matched_econ_family"
            rows.append(diag_row)
            continue

        # ── Duplicate-key guard (Section D) ─────────────────────
        dup_aug  = int(aug_grp.duplicated(subset=["ticker", "timestamp"]).sum())
        dup_econ = int(cand.duplicated(subset=["ticker", "timestamp"]).sum())
        diag_row["n_duplicate_augmented_keys"] = dup_aug
        diag_row["n_duplicate_econ_keys"]      = dup_econ
        if dup_aug or dup_econ:
            diag_row["skip_reason"] = "duplicate_keys_within_family"
            rows.append(diag_row)
            continue

        d_frame = observation_improvement_indicator(aug_grp, cand)
        diag_row["n_matched"] = int(len(d_frame))
        diag_row["n_unmatched_augmented"] = int(len(aug_grp) - len(d_frame))
        diag_row["n_unmatched_econ"]      = int(len(cand)    - len(d_frame))

        if d_frame.empty:
            diag_row["skip_reason"] = "no_matched_observations"
            rows.append(diag_row)
            continue

        # ── Target-equality verification (Section D) ────────────
        # observation_improvement_indicator preserves the augmented-side
        # target. Compare per (ticker, ts) with the ECON-side target.
        bench_for_check = cand.merge(
            d_frame[["ticker", "timestamp"]].drop_duplicates(),
            on=["ticker", "timestamp"], how="inner",
        ).sort_values(["ticker", "timestamp"]).reset_index(drop=True)
        aug_for_check = d_frame.sort_values(["ticker", "timestamp"]).reset_index(drop=True)
        targets_identical = bool(
            (aug_for_check["target"].astype(int).values
             == bench_for_check["target"].astype(int).values).all()
        )
        diag_row["targets_identical"] = targets_identical
        if not targets_identical:
            diag_row["skip_reason"] = "target_mismatch"
            rows.append(diag_row)
            continue

        # ── Regime join ─────────────────────────────────────────
        d_frame = d_frame.copy()
        d_frame["__join_date__"] = _join_date_for_signals(
            d_frame["timestamp"], regime_col,
        )
        joined = d_frame.merge(look, on=["ticker", "__join_date__"], how="left")
        if joined[regime_col].isna().all():
            diag_row["skip_reason"] = "no_regime_match"
            rows.append(diag_row)
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
        diag_row.update({
            "n_control":             res["n_control"],
            "n_treatment":           res["n_treatment"],
            "n_clusters":            res["n_clusters"],
            "mean_d_control":        float(d_ctrl.mean()) if not d_ctrl.empty else float("nan"),
            "mean_d_treatment":      float(d_trt.mean())  if not d_trt.empty  else float("nan"),
            "diff_in_improvement":   res["beta"],
            "se_diff":               res["se_beta"],
            "t_stat":                res["t_beta"],
            "p_value":               res["p_beta"],
            "dof":                   res["dof"],
            "test_valid":            res["test_valid"],
            "small_cluster_warning": res["small_cluster_warning"],
            "skip_reason":           "" if res["test_valid"] else "too_few_clusters_or_obs",
        })
        rows.append(diag_row)
    return pd.DataFrame(rows, columns=columns)


def _series_eq_nan_safe(s: pd.Series, value) -> pd.Series:
    """Elementwise equality that treats NaN==NaN as True (so a missing
    rolling_window_timestamps on both sides counts as a match)."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return s.isna() | (s.astype(str).str.lower() == "nan")
    try:
        return s == value
    except Exception:  # noqa: BLE001
        return s.astype(str) == str(value)


def _join_date_for_signals(timestamps: pd.Series, regime_col: str) -> pd.Series:
    """Compute the regime-lookup join key from prediction timestamps.

    The default for v4 regime lookups (volatility, market-cap) is daily.
    To honour the "no future regime info enters earlier intraday
    predictions" rule (Aufgabe 6 follow-up G), every prediction at time
    t joins the regime from the **previous** calendar day's lookup —
    the daily regime is known fully by 00:00 UTC of the next day, the
    same convention used by the market-cap availability offset.
    """
    ts = pd.to_datetime(timestamps, utc=True, errors="coerce")
    # Strict-as-of-yesterday: subtract one day before normalising, so
    # an intraday 1h prediction on day D pairs with the regime from
    # day D-1.
    return (ts - pd.Timedelta(days=1)).dt.normalize()


def _prepare_regime_lookup(regime_lookup: pd.DataFrame,
                           regime_col: str) -> pd.DataFrame:
    """Standardise the regime lookup to ``(ticker, __join_date__, regime_col)``.

    The lookup is expected to carry a ``date`` (or ``timestamp``) column;
    the helper normalises to tz-aware UTC midnight so the join key matches
    :func:`_join_date_for_signals` exactly.
    """
    look = regime_lookup.copy()
    if "date" in look.columns:
        join_date = pd.to_datetime(look["date"], utc=True, errors="coerce")
        if join_date.isna().any():
            naive = pd.to_datetime(look["date"], errors="coerce")
            join_date = naive.dt.tz_localize("UTC", nonexistent="shift_forward")
        look["__join_date__"] = join_date.dt.normalize()
    elif "timestamp" in look.columns:
        look["__join_date__"] = pd.to_datetime(
            look["timestamp"], utc=True, errors="coerce"
        ).dt.normalize()
    else:
        raise ValueError(
            "difference_in_improvement_table: regime_lookup must have a "
            "'date' or 'timestamp' column."
        )
    look = look[["ticker", "__join_date__", regime_col]].copy()
    return look.dropna(subset=["__join_date__", regime_col])


def _empty_row(columns: list[str], ident: dict, bench_set_id,
               hypothesis_family: str, regime_col: str,
               control_value: str, treatment_value: str) -> dict:
    """Default-zero / NaN diagnostic row that is later overwritten."""
    row = {col: None for col in columns}
    for k in ("set_id", "sentiment_model", "horizon", "model_type",
              "panel_mode", "hpo_variant", "hpo_objective",
              "train_window_mode", "rolling_window_days",
              "rolling_window_timestamps"):
        row[k] = ident.get(k)
    row["hypothesis_family"] = hypothesis_family
    row["benchmark_set_id"]  = bench_set_id
    row["regime_col"]        = regime_col
    row["control_value"]     = control_value
    row["treatment_value"]   = treatment_value
    for k in ("n_augmented", "n_econ", "n_matched",
              "n_unmatched_augmented", "n_unmatched_econ",
              "n_duplicate_augmented_keys", "n_duplicate_econ_keys",
              "n_control", "n_treatment", "n_clusters", "dof"):
        row[k] = 0
    for k in ("mean_d_control", "mean_d_treatment", "diff_in_improvement",
              "se_diff", "t_stat", "p_value"):
        row[k] = float("nan")
    row["targets_identical"]      = False
    row["test_valid"]              = False
    row["small_cluster_warning"]  = False
    row["skip_reason"]             = ""
    return row


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
