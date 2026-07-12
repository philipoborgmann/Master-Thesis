"""Leakage-safe, training-window model preprocessing.

This module is the SINGLE place where outlier control (winsorisation) happens
for the modelling stage. Full-sample winsorisation was removed from feature
construction (:mod:`thesis_pipeline.price.features`,
:mod:`thesis_pipeline.sentiment.aggregate`) because clipping earlier
observations at quantiles computed over the whole series uses future data.

The replacement — :class:`TrainingWindowWinsorizer` — fits its clip thresholds
**only** on the observations currently available for training and applies those
frozen thresholds to the training frame, to inner-HPO validation rows, and to
the outer test observation. It never inspects the test point or any later
observation, so appending extreme future rows cannot change the thresholds or
the transformed values of an earlier forecast step.

Canonical preprocessing order (per walk-forward step / per HPO candidate)::

    winsorise (thresholds fit on TRAIN only)  →  StandardScaler.fit(train)  →
    LogisticRegression.fit

The StandardScaler is therefore always fit AFTER winsorisation and only on
training data.

Grouping rule
-------------
Thresholds are **ticker-specific** inside the current training window so the
previous per-coin treatment of market variables is preserved. For every
feature and ticker:

* use the ticker-specific quantiles when at least ``MIN_TICKER_OBS`` finite
  training observations are available;
* otherwise fall back to thresholds computed from the complete current training
  panel for that feature;
* if even the pooled training sample is insufficient, leave the feature
  unchanged and record the fallback (see :attr:`TrainingWindowWinsorizer.
  fallback_records`).

All of the tunables live in the auditable configuration block below — nothing
is scattered across the modelling code.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

# ===========================================================================
# AUDITABLE CONFIGURATION BLOCK
# ===========================================================================

#: Whether training-window winsorisation is applied at all. Kept as an explicit
#: switch so the preprocessing signature (cache invalidation) records it.
WINSOR_ENABLED: bool = True

#: Lower / upper clip quantiles (0.5% each tail).
WINSOR_LOWER_QUANTILE: float = 0.005
WINSOR_UPPER_QUANTILE: float = 0.995

#: Minimum number of finite training observations required to trust a
#: ticker-specific (or, at the pooled fallback, a panel-wide) quantile pair.
MIN_TICKER_OBS: int = 20

#: Grouping rule for the ticker-specific thresholds.
WINSOR_GROUP_COL: str = "ticker"

#: Default winsorisation allowlist — the unbounded / heavy-tailed continuous
#: features used by the seventeen specifications. Everything NOT on this list is
#: passed through unchanged (targets, identifiers, ticker dummies, has_posts,
#: raw / directional post counts, binary variables, bounded bullishness ratios
#: and bounded sentiment scores / probabilities). ``*_title_score_std`` is the
#: std of a bounded score and is itself bounded, so it is deliberately excluded.
DEFAULT_WINSOR_ALLOWLIST: tuple[str, ...] = (
    "log_return_t",
    "cum_log_return_7d",
    "cum_log_return_14d",
    "cum_log_return_21d",
    "realized_vol_14d",
    "volume_diff",
    "log_market_cap_lag1",
    "log1p_post_count",
)


def default_winsor_config() -> dict[str, Any]:
    """Return the default winsorisation configuration as a plain dict."""
    return {
        "enabled": WINSOR_ENABLED,
        "lower_quantile": WINSOR_LOWER_QUANTILE,
        "upper_quantile": WINSOR_UPPER_QUANTILE,
        "min_ticker_obs": MIN_TICKER_OBS,
        "group_col": WINSOR_GROUP_COL,
        "allowlist": list(DEFAULT_WINSOR_ALLOWLIST),
    }


# ===========================================================================
# Winsoriser
# ===========================================================================

@dataclass
class _FeatureThresholds:
    """Fitted clip thresholds for one feature."""
    per_ticker: dict[str, tuple[float, float]] = field(default_factory=dict)
    pooled: tuple[float, float] | None = None


class TrainingWindowWinsorizer:
    """Fit clip thresholds on training data only; apply them anywhere.

    Parameters
    ----------
    feature_cols
        The model's feature columns for this run. Only the intersection with
        ``allowlist`` is ever winsorised.
    allowlist, lower_quantile, upper_quantile, min_ticker_obs, group_col, enabled
        See the module configuration block. Defaults come from there.
    """

    def __init__(self,
                 feature_cols: Sequence[str],
                 *,
                 allowlist: Sequence[str] = DEFAULT_WINSOR_ALLOWLIST,
                 lower_quantile: float = WINSOR_LOWER_QUANTILE,
                 upper_quantile: float = WINSOR_UPPER_QUANTILE,
                 min_ticker_obs: int = MIN_TICKER_OBS,
                 group_col: str = WINSOR_GROUP_COL,
                 enabled: bool = WINSOR_ENABLED) -> None:
        self.feature_cols = list(feature_cols)
        self.allowlist = tuple(allowlist)
        self.lower_quantile = float(lower_quantile)
        self.upper_quantile = float(upper_quantile)
        self.min_ticker_obs = int(min_ticker_obs)
        self.group_col = str(group_col)
        self.enabled = bool(enabled)
        # Winsorise only allowlisted model features, in the model's order.
        self._active_features = [f for f in self.feature_cols
                                 if f in set(self.allowlist)]
        self._thresholds: dict[str, _FeatureThresholds] = {}
        self._fallbacks: list[dict[str, Any]] = []
        self._fitted = False

    # -- fit ----------------------------------------------------------------

    @staticmethod
    def _finite(values: pd.Series) -> pd.Series:
        v = pd.to_numeric(values, errors="coerce")
        return v[np.isfinite(v)]

    def _quantiles(self, values: pd.Series) -> tuple[float, float] | None:
        v = self._finite(values)
        if len(v) < self.min_ticker_obs:
            return None
        lo = float(v.quantile(self.lower_quantile))
        hi = float(v.quantile(self.upper_quantile))
        if not (np.isfinite(lo) and np.isfinite(hi)) or hi < lo:
            return None
        return lo, hi

    def fit(self, train_df: pd.DataFrame) -> "TrainingWindowWinsorizer":
        """Fit thresholds on ``train_df`` ONLY. Returns ``self``."""
        self._thresholds = {}
        self._fallbacks = []
        self._fitted = True
        if not self.enabled or train_df is None or len(train_df) == 0:
            return self

        has_group = self.group_col in train_df.columns
        for feat in self._active_features:
            if feat not in train_df.columns:
                continue
            ft = _FeatureThresholds()
            # Pooled (panel-wide) thresholds for this feature — the fallback.
            ft.pooled = self._quantiles(train_df[feat])
            # Ticker-specific thresholds where enough finite obs exist.
            if has_group:
                for tk, grp in train_df.groupby(self.group_col, sort=False):
                    q = self._quantiles(grp[feat])
                    if q is not None:
                        ft.per_ticker[str(tk)] = q
                    else:
                        # Not enough ticker obs → this ticker uses pooled (or,
                        # if pooled is also insufficient, passthrough).
                        self._fallbacks.append({
                            "feature": feat,
                            "ticker": str(tk),
                            "rule": ("pooled" if ft.pooled is not None
                                     else "passthrough_insufficient_pooled"),
                            "n_finite": int(len(self._finite(grp[feat]))),
                        })
            if ft.pooled is None and not ft.per_ticker:
                # Whole feature is unclippable on this window.
                self._fallbacks.append({
                    "feature": feat,
                    "ticker": "__all__",
                    "rule": "passthrough_insufficient_pooled",
                    "n_finite": int(len(self._finite(train_df[feat]))),
                })
            self._thresholds[feat] = ft
        return self

    # -- transform ----------------------------------------------------------

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clip ``df`` with the fitted thresholds. Non-allowlisted columns and
        rows without an applicable threshold are returned unchanged."""
        if not self._fitted:
            raise RuntimeError("TrainingWindowWinsorizer.transform called "
                               "before fit().")
        out = df.copy()
        if not self.enabled:
            return out
        has_group = self.group_col in out.columns
        for feat, ft in self._thresholds.items():
            if feat not in out.columns:
                continue
            if not ft.per_ticker and ft.pooled is None:
                continue  # passthrough feature
            n = len(out)
            lo = np.full(n, np.nan)
            hi = np.full(n, np.nan)
            if has_group and ft.per_ticker:
                tickers = out[self.group_col].astype(str).to_numpy()
                lo_map = {tk: v[0] for tk, v in ft.per_ticker.items()}
                hi_map = {tk: v[1] for tk, v in ft.per_ticker.items()}
                lo = np.array([lo_map.get(tk, np.nan) for tk in tickers])
                hi = np.array([hi_map.get(tk, np.nan) for tk in tickers])
            if ft.pooled is not None:
                lo = np.where(np.isnan(lo), ft.pooled[0], lo)
                hi = np.where(np.isnan(hi), ft.pooled[1], hi)
            mask = np.isfinite(lo) & np.isfinite(hi)
            if not mask.any():
                continue
            col = pd.to_numeric(out[feat], errors="coerce").to_numpy(dtype=float)
            clipped = np.clip(col, lo, hi)
            # Only overwrite rows that (a) have a threshold and (b) are finite.
            apply_mask = mask & np.isfinite(col)
            col_out = out[feat].to_numpy(dtype=float, copy=True)
            col_out[apply_mask] = clipped[apply_mask]
            out[feat] = col_out
        return out

    def fit_transform(self, train_df: pd.DataFrame) -> pd.DataFrame:
        return self.fit(train_df).transform(train_df)

    # -- audit --------------------------------------------------------------

    @property
    def fallback_records(self) -> list[dict[str, Any]]:
        """One record per (feature, ticker) that could not use a ticker-specific
        threshold — either it fell back to the pooled panel threshold or (when
        even the pooled sample was insufficient) was left unchanged."""
        return list(self._fallbacks)

    @property
    def active_features(self) -> list[str]:
        return list(self._active_features)


# ===========================================================================
# Convenience: winsorise a (train, test) pair with thresholds fit on train
# ===========================================================================

def winsorize_train_test(train_df: pd.DataFrame,
                         test_df: pd.DataFrame,
                         feature_cols: Sequence[str],
                         *,
                         config: Mapping[str, Any] | None = None
                         ) -> tuple[pd.DataFrame, pd.DataFrame,
                                    TrainingWindowWinsorizer]:
    """Fit the winsoriser on ``train_df`` and apply it to both frames.

    Returns ``(train_winsorised, test_winsorised, winsorizer)``. The winsoriser
    is returned so a caller can later apply the SAME frozen thresholds to a
    third frame (e.g. an inner-HPO validation block) without refitting.
    """
    cfg = dict(default_winsor_config())
    if config:
        cfg.update(config)
    w = TrainingWindowWinsorizer(
        feature_cols,
        allowlist=cfg["allowlist"],
        lower_quantile=cfg["lower_quantile"],
        upper_quantile=cfg["upper_quantile"],
        min_ticker_obs=cfg["min_ticker_obs"],
        group_col=cfg["group_col"],
        enabled=cfg["enabled"],
    ).fit(train_df)
    return w.transform(train_df), w.transform(test_df), w


# ===========================================================================
# Preprocessing signature (cache / checkpoint invalidation — Section 4)
# ===========================================================================

def preprocessing_signature(*,
                            config: Mapping[str, Any] | None = None,
                            model_window: Any = None,
                            hpo_objective: Any = None,
                            forecast_sample: Mapping[str, Any] | None = None
                            ) -> str:
    """Stable short hash of the preprocessing + output configuration.

    Covers (at least) winsorisation on/off, the quantile bounds, the grouping
    rule, the feature allowlist, the model training window, the HPO objective
    AND the forecast-origin sample contract — so a checkpoint / signal file
    produced under a different preprocessing methodology OR a different output
    sample window can be detected and rejected.
    """
    cfg = dict(default_winsor_config())
    if config:
        cfg.update(config)
    fs_sig = None
    if forecast_sample is not None:
        from .forecast_sample import forecast_sample_signature
        fs_sig = forecast_sample_signature(forecast_sample)
    payload = {
        "winsor_enabled": bool(cfg["enabled"]),
        "lower_quantile": float(cfg["lower_quantile"]),
        "upper_quantile": float(cfg["upper_quantile"]),
        "min_ticker_obs": int(cfg["min_ticker_obs"]),
        "group_col": str(cfg["group_col"]),
        "allowlist": sorted(str(f) for f in cfg["allowlist"]),
        "model_window": _norm(model_window),
        "hpo_objective": _norm(hpo_objective),
        "forecast_sample": fs_sig,
        "schema": "v5_training_window_winsor",
    }
    blob = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha1(blob.encode("utf-8")).hexdigest()[:16]


def _norm(value: Any) -> Any:
    """Normalise a manifest value into something JSON-stable."""
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        return [_norm(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _norm(v) for k, v in sorted(value.items())}
    if isinstance(value, float):
        return float(value)
    return str(value)
