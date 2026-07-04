"""Timestamp-level differential forecast-quality tests (Part 2A).

This module computes BOTH a log-loss differential effect and an accuracy
differential effect at the TIMESTAMP level, joining benchmark vs
sentiment predictions on the matched ``(ticker, timestamp)`` intersection
(the same coverage-intersection inner-join used by the other comparison
writers). Both use aggregation-then-timeseries logic so that
cross-sectional dependence of coins sharing a timestamp is handled by
aggregating within the timestamp and testing the resulting time series.

Methodology / naming (read before citing)
------------------------------------------
* This is a **Diebold-Mariano-TYPE** test on a *log-loss differential*.
  Diebold-Mariano (1995) compares loss differentials for a GENERAL loss
  function; it is NOT restricted to squared error, so log loss is
  admissible.
* **Harvey-Leybourne-Newbold (1997)** is a SMALL-SAMPLE correction. It is
  NOT what makes log loss usable and is applied ONLY when
  ``small_sample_correction=True`` — and then only to the HAC statistic,
  labeled as such.
* **Clark-West** is for NESTED models under MSPE / squared-error and does
  NOT transfer unchanged to log loss; it is NOT used here.
* **Nested-model caveat.** The sentiment model nests the benchmark, so
  under H0 (zero sentiment coefficients) the loss differential can be
  degenerate (``d_t ≈ 0``, near-zero variance), which invalidates the
  normal-DM asymptotics. This is why the **moving-block bootstrap is the
  PRIMARY inference**. If ``var(d_t)`` is below tolerance the test is
  marked degenerate (``test_valid=False``,
  ``skip_reason="degenerate_zero_variance"``) rather than dividing by ≈0,
  and ``dm_nested_caveat=True`` is recorded.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .coverage import coverage_intersection
from .preregistration import DM_INFERENCE

#: Fixed RNG seed so the moving-block bootstrap is reproducible across
#: runs (a thesis artefact must be deterministic).
_BOOTSTRAP_SEED = 12345


# ---------------------------------------------------------------------------
# Per-observation losses
# ---------------------------------------------------------------------------

def _log_loss_per_obs(y_true: np.ndarray, p_up: np.ndarray,
                      eps: float) -> np.ndarray:
    """``LL(i) = -[y·ln p + (1-y)·ln(1-p)]`` with ``p`` clipped to
    ``[eps, 1-eps]`` (identical clipping to the pooled metrics)."""
    p = np.clip(np.asarray(p_up, dtype=float), eps, 1.0 - eps)
    y = np.asarray(y_true, dtype=float)
    return -(y * np.log(p) + (1.0 - y) * np.log(1.0 - p))


# ---------------------------------------------------------------------------
# HAC (Newey-West) long-run variance of a mean
# ---------------------------------------------------------------------------

def _auto_hac_lag(T: int) -> int:
    """Newey-West plug-in lag ``floor(4·(T/100)^(2/9))`` (≥ 0)."""
    if T <= 1:
        return 0
    return int(np.floor(4.0 * (T / 100.0) ** (2.0 / 9.0)))


def _hac_se_of_mean(d: np.ndarray, lag: int) -> float:
    """Newey-West HAC standard error of ``mean(d)``.

    Long-run variance ``ω² = γ0 + 2·Σ_{k=1}^{L} (1-k/(L+1))·γk``; the SE of
    the sample mean is ``sqrt(ω²/T)``.
    """
    d = np.asarray(d, dtype=float)
    T = d.size
    if T < 2:
        return float("nan")
    dc = d - d.mean()
    gamma0 = float(np.dot(dc, dc) / T)
    omega = gamma0
    L = max(int(lag), 0)
    for k in range(1, L + 1):
        if k >= T:
            break
        cov_k = float(np.dot(dc[k:], dc[:-k]) / T)
        w = 1.0 - k / (L + 1.0)
        omega += 2.0 * w * cov_k
    if omega < 0:
        omega = gamma0  # guard against a negative truncated estimate
    return float(np.sqrt(max(omega, 0.0) / T))


# ---------------------------------------------------------------------------
# Moving-block bootstrap of the mean of a time series
# ---------------------------------------------------------------------------

def _auto_block_length(T: int) -> int:
    """``round(T^(1/3))`` (≥ 1)."""
    return max(int(round(T ** (1.0 / 3.0))), 1)


def _moving_block_bootstrap_mean(d: np.ndarray, *, n_boot: int,
                                 block_length: int,
                                 seed: int = _BOOTSTRAP_SEED) -> np.ndarray:
    """Return ``n_boot`` bootstrap replicates of ``mean(d)`` using an
    overlapping moving-block bootstrap (preserves short-range serial
    dependence in ``d_t``)."""
    d = np.asarray(d, dtype=float)
    T = d.size
    L = max(int(block_length), 1)
    if T == 0:
        return np.array([])
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(T / L))
    max_start = T - L
    means = np.empty(n_boot, dtype=float)
    if max_start <= 0:
        # Series shorter than one block → resample observations i.i.d.
        for b in range(n_boot):
            idx = rng.integers(0, T, size=T)
            means[b] = d[idx].mean()
        return means
    for b in range(n_boot):
        starts = rng.integers(0, max_start + 1, size=n_blocks)
        pieces = [d[s:s + L] for s in starts]
        sample = np.concatenate(pieces)[:T]
        means[b] = sample.mean()
    return means


def _two_sided_bootstrap_p(effect: float, boot_means: np.ndarray) -> float:
    """Bootstrap-null two-sided p-value: centre the bootstrap
    distribution at zero (subtract the point estimate) and measure how
    often the null replicate is at least as extreme as ``effect``."""
    if boot_means.size == 0 or not np.isfinite(effect):
        return float("nan")
    null = boot_means - effect
    p = float(np.mean(np.abs(null) >= abs(effect)))
    return min(max(p, 0.0), 1.0)


# ---------------------------------------------------------------------------
# Timestamp aggregation of a per-observation differential
# ---------------------------------------------------------------------------

def _aggregate_within_timestamp(joined: pd.DataFrame,
                                per_obs: np.ndarray) -> pd.DataFrame:
    """Return ``(timestamp, N_t, d_t)`` where ``d_t`` is the within-
    timestamp mean of ``per_obs`` and ``N_t`` the timestamp cross-section
    size."""
    tmp = pd.DataFrame({
        "timestamp": joined["timestamp"].values,
        "d": per_obs,
    })
    g = tmp.groupby("timestamp", sort=True)
    out = g["d"].agg(["mean", "size"]).reset_index()
    out.columns = ["timestamp", "d_t", "N_t"]
    return out


def _prep_pair(benchmark_pred: pd.DataFrame,
               sentiment_pred: pd.DataFrame):
    """Coverage-intersect the two prediction frames on (ticker,
    timestamp); return ``(bench_m, sent_m, targets_identical)`` aligned
    row-for-row, or ``(None, None, False)`` when there is no overlap."""
    ci = coverage_intersection(sentiment_pred, benchmark_pred,
                               key_cols=("ticker", "timestamp"))
    if ci.n_matched == 0:
        return None, None, False, ci
    sent_m = ci.candidate.reset_index(drop=True)
    bench_m = ci.reference.reset_index(drop=True)
    targets_identical = bool(
        (sent_m["target"].astype(int).values
         == bench_m["target"].astype(int).values).all()
    )
    return bench_m, sent_m, targets_identical, ci


def _resolve_cfg(cfg: dict | None) -> dict:
    c = dict(DM_INFERENCE)
    if cfg:
        c.update(cfg)
    return c


# ---------------------------------------------------------------------------
# (A) LOG-LOSS DIFFERENTIAL
# ---------------------------------------------------------------------------

def dm_logloss_timestamp_test(benchmark_pred: pd.DataFrame,
                              sentiment_pred: pd.DataFrame,
                              cfg: dict | None = None) -> dict:
    """Diebold-Mariano-type test on the timestamp-aggregated log-loss
    differential ``d_t`` (positive ⇒ sentiment better).

    Returns a dict with the effect, HAC + bootstrap SEs / CI / p-values,
    ``T``, ``hac_lag``, ``block_length``, ``dm_inference_primary``,
    ``dm_nested_caveat``, ``test_valid`` and ``skip_reason``.
    """
    c = _resolve_cfg(cfg)
    eps = float(c["clip_eps"])
    out = _empty_ll_result(c)

    bench_m, sent_m, targets_identical, ci = _prep_pair(benchmark_pred,
                                                        sentiment_pred)
    out["n_matched"] = ci.n_matched
    if bench_m is None:
        out["skip_reason"] = "no_overlap"
        return out
    if not targets_identical:
        out["skip_reason"] = "target_mismatch"
        return out

    y = sent_m["target"].astype(int).values
    ll_sent  = _log_loss_per_obs(y, sent_m["probability"].values, eps)
    ll_bench = _log_loss_per_obs(y, bench_m["probability"].values, eps)
    d_obs = ll_bench - ll_sent                     # + ⇒ sentiment better

    agg = _aggregate_within_timestamp(sent_m, d_obs)
    d_t = agg["d_t"].to_numpy(dtype=float)
    N_t = agg["N_t"].to_numpy(dtype=float)
    T = int(d_t.size)
    out["n_timestamps"] = T

    out["ll_effect"]            = float(np.mean(d_t)) if T else float("nan")
    out["ll_effect_ntweighted"] = (float(np.sum(N_t * d_t) / np.sum(N_t))
                                   if N_t.sum() > 0 else float("nan"))

    # Nested-model degeneracy guard.
    var_dt = float(np.var(d_t)) if T > 1 else 0.0
    if T < 2 or var_dt < float(c["degenerate_var_tol"]):
        out["dm_nested_caveat"] = True
        out["test_valid"] = False
        out["skip_reason"] = "degenerate_zero_variance"
        return out

    lag = (_auto_hac_lag(T) if c["hac_lag"] == "auto" else int(c["hac_lag"]))
    blk = (_auto_block_length(T) if c["block_length"] == "auto"
           else int(c["block_length"]))
    out["hac_lag"] = lag
    out["block_length"] = blk

    se_hac = _hac_se_of_mean(d_t, lag)
    out["ll_se_hac"] = se_hac
    if np.isfinite(se_hac) and se_hac > 0:
        z = out["ll_effect"] / se_hac
        out["ll_z_hac"] = float(z)
        from scipy.stats import norm
        p_hac = float(2.0 * norm.sf(abs(z)))
        if bool(c.get("small_sample_correction", False)):
            # HLN t_{T-1} correction on the HAC statistic ONLY.
            from scipy.stats import t as student_t
            p_hac = float(2.0 * student_t.sf(abs(z), df=max(T - 1, 1)))
        out["ll_p_hac"] = p_hac

    boot = _moving_block_bootstrap_mean(d_t, n_boot=int(c["n_boot"]),
                                        block_length=blk)
    out["ll_se_boot"] = float(np.std(boot, ddof=1)) if boot.size > 1 else float("nan")
    if boot.size:
        lo, hi = np.percentile(boot, [2.5, 97.5])
        out["ll_ci95_low"], out["ll_ci95_high"] = float(lo), float(hi)
    out["ll_p_boot"] = _two_sided_bootstrap_p(out["ll_effect"], boot)
    out["ll_p_value_raw"] = out["ll_p_boot"]      # primary = bootstrap
    out["test_valid"] = bool(np.isfinite(out["ll_p_boot"]))
    return out


def _empty_ll_result(c: dict) -> dict:
    return {
        "ll_effect": float("nan"), "ll_effect_ntweighted": float("nan"),
        "ll_se_hac": float("nan"), "ll_se_boot": float("nan"),
        "ll_ci95_low": float("nan"), "ll_ci95_high": float("nan"),
        "ll_z_hac": float("nan"), "ll_p_hac": float("nan"),
        "ll_p_boot": float("nan"), "ll_p_value_raw": float("nan"),
        "n_timestamps": 0, "n_matched": 0,
        "hac_lag": np.nan, "block_length": np.nan,
        "dm_inference_primary": c["primary"],
        "dm_nested_caveat": False,
        "test_valid": False, "skip_reason": "",
    }


# ---------------------------------------------------------------------------
# (B) ACCURACY DIFFERENTIAL
# ---------------------------------------------------------------------------

def accuracy_timestamp_effect(benchmark_pred: pd.DataFrame,
                              sentiment_pred: pd.DataFrame,
                              cfg: dict | None = None) -> dict:
    """Timestamp-aggregated accuracy differential ``a_t`` (mean over
    timestamps of ``1[sent correct] − 1[bench correct]``).

    NOTE: this timestamp-aggregated accuracy effect feeds ONLY the
    Family-E horizon contrasts. H1 directional (Family B) continues to
    use the pooled McNemar test — a deliberate difference in dependence
    structure (McNemar tests the pooled paired directional difference at
    one horizon; the Family-E accuracy effect uses time-series
    aggregation because horizon contrasts concern serial dependence).
    """
    c = _resolve_cfg(cfg)
    out = _empty_acc_result(c)

    bench_m, sent_m, targets_identical, ci = _prep_pair(benchmark_pred,
                                                        sentiment_pred)
    out["n_matched"] = ci.n_matched
    if bench_m is None:
        out["skip_reason"] = "no_overlap"
        return out
    if not targets_identical:
        out["skip_reason"] = "target_mismatch"
        return out

    y = sent_m["target"].astype(int).values
    sent_correct  = (sent_m["prediction"].astype(int).values == y).astype(int)
    bench_correct = (bench_m["prediction"].astype(int).values == y).astype(int)
    a_obs = sent_correct - bench_correct            # {-1, 0, 1}

    agg = _aggregate_within_timestamp(sent_m, a_obs)
    a_t = agg["d_t"].to_numpy(dtype=float)
    N_t = agg["N_t"].to_numpy(dtype=float)
    T = int(a_t.size)
    out["n_timestamps"] = T

    out["acc_effect"]            = float(np.mean(a_t)) if T else float("nan")
    out["acc_effect_ntweighted"] = (float(np.sum(N_t * a_t) / np.sum(N_t))
                                    if N_t.sum() > 0 else float("nan"))

    var_at = float(np.var(a_t)) if T > 1 else 0.0
    if T < 2 or var_at < float(c["degenerate_var_tol"]):
        out["dm_nested_caveat"] = True
        out["test_valid"] = False
        out["skip_reason"] = "degenerate_zero_variance"
        return out

    lag = (_auto_hac_lag(T) if c["hac_lag"] == "auto" else int(c["hac_lag"]))
    blk = (_auto_block_length(T) if c["block_length"] == "auto"
           else int(c["block_length"]))
    out["hac_lag"] = lag
    out["block_length"] = blk

    out["acc_se_hac"] = _hac_se_of_mean(a_t, lag)   # secondary
    boot = _moving_block_bootstrap_mean(a_t, n_boot=int(c["n_boot"]),
                                        block_length=blk)
    out["acc_se_boot"] = float(np.std(boot, ddof=1)) if boot.size > 1 else float("nan")
    if boot.size:
        lo, hi = np.percentile(boot, [2.5, 97.5])
        out["acc_ci95_low"], out["acc_ci95_high"] = float(lo), float(hi)
    out["acc_p_boot"] = _two_sided_bootstrap_p(out["acc_effect"], boot)
    out["acc_p_value_raw"] = out["acc_p_boot"]
    out["test_valid"] = bool(np.isfinite(out["acc_p_boot"]))
    return out


def _empty_acc_result(c: dict) -> dict:
    return {
        "acc_effect": float("nan"), "acc_effect_ntweighted": float("nan"),
        "acc_se_hac": float("nan"), "acc_se_boot": float("nan"),
        "acc_ci95_low": float("nan"), "acc_ci95_high": float("nan"),
        "acc_p_boot": float("nan"), "acc_p_value_raw": float("nan"),
        "n_timestamps": 0, "n_matched": 0,
        "hac_lag": np.nan, "block_length": np.nan,
        "dm_inference_primary": c["primary"],
        "dm_nested_caveat": False,
        "test_valid": False, "skip_reason": "",
    }
