"""Conservative, leakage-safe hyperparameter tuning for the modelling stage.

This module provides a single, family-agnostic grid search that is shared by
all three model families:

1. ``per_asset``                 — per-ticker expanding-window walk-forward
   (:mod:`thesis_pipeline.modeling.run_models`).
2. ``panel_logit`` / ``pooled``  — pooled panel logistic regression.
3. ``panel_logit`` / ``ticker_fixed_effects`` — panel logit with coin dummies.

Methodology — why this is leakage-safe
--------------------------------------
The search is **nested inside the walk-forward training window**. At every
walk-forward step the caller already holds a training window that contains
*only* information strictly before the test point / test timestamp. We split
**that window** chronologically into an inner-train block (the earlier
``1 - validation_fraction`` of the window) and a validation block (the most
recent ``validation_fraction``). Hyperparameters are scored on the validation
block, the best combination is chosen, and — when
``refit_on_full_train`` is true — the final model is re-fit on the *entire*
training window with the chosen parameters before predicting the test point.

The test point / test timestamp is **never** passed into this module: it is
not used to fit the scaler, to fit any candidate model, or to select
hyperparameters. The validation split is strictly chronological — there is no
shuffling, no random split, and no cross-validation over future timestamps.

Why grid search
---------------
The search space is deliberately tiny (a handful of regularisation strengths
and an optional class-weight toggle). A deterministic grid is exhaustive,
reproducible (no RNG), and cheap enough to run nested in a walk-forward loop.
Random / Bayesian search would add stochasticity for no benefit at this size.

Objectives
----------
``brier_score`` (default), ``log_loss`` and ``accuracy`` are supported. Brier
score is the default because the predicted probabilities are reused downstream
for threshold analysis and the economic backtest, so calibration matters more
than a hard-label metric. Optimisation direction:

* ``brier_score`` → minimize
* ``log_loss``    → minimize
* ``accuracy``    → maximize
"""

from __future__ import annotations

import itertools
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss

from ..config import load_config
from ..logging_utils import get_logger

# ---------------------------------------------------------------------------
# Objectives + estimator defaults
# ---------------------------------------------------------------------------

OBJECTIVE_DIRECTIONS: dict[str, str] = {
    "brier_score": "minimize",
    "log_loss":    "minimize",
    "accuracy":    "maximize",
}
ALLOWED_OBJECTIVES = tuple(OBJECTIVE_DIRECTIONS.keys())

# Default hyperparameters — identical to the pinned thesis estimator so that
# any fallback path reproduces the non-tuned behaviour exactly.
DEFAULT_PARAMS: dict[str, Any] = {"C": 1.0, "class_weight": None}

# Fixed estimator settings shared with run_models / panel_logit.
_PENALTY = "l2"
_SOLVER = "lbfgs"
_MAX_ITER = 1000
_RANDOM_STATE = 42

# Recognised model families.
PER_ASSET = "per_asset"
PANEL = "panel"


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

_DEFAULT_SEARCH_SPACE = {
    "C": [0.01, 0.1, 1.0, 10.0],
    "class_weight": [None, "balanced"],
}

_DEFAULT_HPO = {
    "enabled": False,
    "method": "grid",
    "objective": "brier_score",
    "validation_fraction": 0.2,
    "min_train_obs": 60,
    "min_validation_obs": 20,
    "refit_on_full_train": True,
}


def _coerce_class_weight(value: Any) -> Any:
    """Map CLI / YAML class-weight tokens to sklearn values.

    ``"none"`` / ``"null"`` / ``None`` → ``None``; everything else passes
    through unchanged (e.g. ``"balanced"``).
    """
    if value is None:
        return None
    if isinstance(value, str) and value.strip().lower() in ("none", "null", ""):
        return None
    return value


def load_hpo_config(model_specs: Mapping[str, Any] | None = None,
                    *,
                    enabled_override: bool | None = None,
                    objective_override: str | None = None,
                    c_grid: Sequence[float] | None = None,
                    class_weight_grid: Sequence[Any] | None = None,
                    config_path: str | Path | None = None) -> dict:
    """Return a fully-resolved hyperparameter-tuning config dict.

    The base values are read from ``configs/model_specs.yaml`` (section
    ``hyperparameter_tuning``). A legacy scalar ``hyperparameter_tuning: false``
    is honoured as "disabled". Runtime overrides (CLI flags) take precedence:

    * ``enabled_override`` — set by ``--tune-hyperparams`` (only overrides when
      not ``None``, so a bare run keeps the YAML default of ``enabled: false``).
    * ``objective_override`` — ``--hpo-objective``.
    * ``c_grid`` / ``class_weight_grid`` — optional ``--hpo-grid-C`` /
      ``--hpo-class-weight`` overrides of the search space.
    """
    if config_path is not None:
        import yaml
        p = Path(config_path)
        if p.exists():
            with p.open("r", encoding="utf-8") as fh:
                model_specs = yaml.safe_load(fh) or {}
        else:
            get_logger().warning("hpo: config %s not found — using model_specs.yaml", p)
            model_specs = None
    if model_specs is None:
        try:
            model_specs = load_config("model_specs")
        except FileNotFoundError:
            model_specs = {}

    raw_section = model_specs.get("hyperparameter_tuning")
    if isinstance(raw_section, bool):
        raw: dict[str, Any] = {"enabled": raw_section}
    elif isinstance(raw_section, Mapping):
        raw = dict(raw_section)
    else:
        raw = {}

    cfg = dict(_DEFAULT_HPO)
    for k in _DEFAULT_HPO:
        if k in raw:
            cfg[k] = raw[k]
    cfg["enabled"] = bool(cfg["enabled"])
    cfg["validation_fraction"] = float(cfg["validation_fraction"])
    cfg["min_train_obs"] = int(cfg["min_train_obs"])
    cfg["min_validation_obs"] = int(cfg["min_validation_obs"])
    cfg["refit_on_full_train"] = bool(cfg["refit_on_full_train"])

    search_space = dict(raw.get("search_space") or _DEFAULT_SEARCH_SPACE)
    # Normalise class_weight tokens coming from YAML.
    if "class_weight" in search_space:
        search_space["class_weight"] = [
            _coerce_class_weight(v) for v in search_space["class_weight"]
        ]
    cfg["search_space"] = search_space

    # ── Runtime overrides ───────────────────────────────────────
    if enabled_override is not None:
        cfg["enabled"] = bool(enabled_override)
    if objective_override:
        cfg["objective"] = objective_override
    if c_grid:
        cfg["search_space"]["C"] = [float(c) for c in c_grid]
    if class_weight_grid is not None:
        cfg["search_space"]["class_weight"] = [
            _coerce_class_weight(v) for v in class_weight_grid
        ]

    if cfg["objective"] not in OBJECTIVE_DIRECTIONS:
        raise ValueError(
            f"Unknown hpo objective {cfg['objective']!r}; "
            f"expected one of {ALLOWED_OBJECTIVES}."
        )
    return cfg


# ---------------------------------------------------------------------------
# Param grid
# ---------------------------------------------------------------------------

def make_param_grid(search_space: Mapping[str, Sequence[Any]]) -> list[dict]:
    """Build every parameter combination from a search-space mapping.

    Deterministic Cartesian product — no random or Bayesian search. Example::

        make_param_grid({"C": [0.1, 1.0], "class_weight": [None, "balanced"]})
        # → [{"C": 0.1, "class_weight": None},
        #    {"C": 0.1, "class_weight": "balanced"},
        #    {"C": 1.0, "class_weight": None},
        #    {"C": 1.0, "class_weight": "balanced"}]
    """
    if not search_space:
        return []
    keys = list(search_space.keys())
    value_lists = [list(search_space[k]) for k in keys]
    if any(len(v) == 0 for v in value_lists):
        return []
    return [dict(zip(keys, combo)) for combo in itertools.product(*value_lists)]


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def score_validation_predictions(y_true: np.ndarray,
                                  y_prob: np.ndarray,
                                  y_pred: np.ndarray,
                                  objective: str) -> float:
    """Score validation predictions under ``objective``.

    Returns the raw metric value; the caller applies
    :data:`OBJECTIVE_DIRECTIONS` to decide whether lower or higher is better.
    ``brier_score`` / ``log_loss`` are computed on clipped probabilities;
    ``log_loss`` passes ``labels=[0, 1]`` so single-class validation folds do
    not raise. Returns ``nan`` when the objective cannot be computed.
    """
    if objective not in OBJECTIVE_DIRECTIONS:
        raise ValueError(f"Unknown objective {objective!r}; expected {ALLOWED_OBJECTIVES}.")
    y_true = np.asarray(y_true).astype(int)
    if len(y_true) == 0:
        return float("nan")
    if objective == "accuracy":
        return float(accuracy_score(y_true, np.asarray(y_pred).astype(int)))
    prob = np.clip(np.asarray(y_prob, dtype=float), 1e-15, 1 - 1e-15)
    if objective == "brier_score":
        return float(brier_score_loss(y_true, prob))
    # log_loss
    try:
        return float(log_loss(y_true, prob, labels=[0, 1]))
    except ValueError:
        return float("nan")


def _is_better(candidate: float, incumbent: float | None, objective: str) -> bool:
    """Return True when ``candidate`` beats ``incumbent`` for the objective."""
    if candidate != candidate:  # NaN never wins
        return False
    if incumbent is None or incumbent != incumbent:
        return True
    if OBJECTIVE_DIRECTIONS[objective] == "minimize":
        return candidate < incumbent
    return candidate > incumbent


# ---------------------------------------------------------------------------
# Chronological inner split
# ---------------------------------------------------------------------------

def chronological_train_validation_split(train_df: pd.DataFrame,
                                         validation_fraction: float,
                                         *,
                                         family: str = PER_ASSET,
                                         time_col: str = "timestamp"
                                         ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split a walk-forward training window into (inner_train, validation).

    The most recent ``validation_fraction`` of the window becomes the
    validation block — strictly chronological, never shuffled.

    * ``per_asset`` — split by row order (a single ticker's chronological
      observations).
    * ``panel``     — split along **unique timestamps**: ``inner_train`` is
      every row with ``timestamp < validation_start`` and ``validation`` every
      row with ``timestamp >= validation_start``, where ``validation_start`` is
      the timestamp that opens the most-recent ``validation_fraction`` of the
      unique time axis. This keeps whole cross-sections together and never
      leaks a validation timestamp into the inner-train block.
    """
    empty = train_df.iloc[0:0]
    if train_df is None or train_df.empty:
        return empty, empty
    frac = min(max(float(validation_fraction), 0.0), 1.0)

    if family == PANEL:
        ts = np.sort(train_df[time_col].unique())
        n = len(ts)
        if n < 2:
            return empty, empty
        n_val = max(int(round(n * frac)), 1)
        n_val = min(n_val, n - 1)  # keep at least one timestamp for inner-train
        validation_start = ts[n - n_val]
        inner = train_df[train_df[time_col] < validation_start]
        val = train_df[train_df[time_col] >= validation_start]
        return inner.reset_index(drop=True), val.reset_index(drop=True)

    d = train_df.sort_values(time_col)
    n = len(d)
    if n < 2:
        return empty, empty
    n_val = max(int(round(n * frac)), 1)
    n_val = min(n_val, n - 1)  # keep at least one observation for inner-train
    inner = d.iloc[: n - n_val]
    val = d.iloc[n - n_val:]
    return inner.reset_index(drop=True), val.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Ticker fixed-effects helpers (leakage-safe; mirror panel_logit)
# ---------------------------------------------------------------------------

def _train_ticker_dummies(tickers: pd.Series) -> tuple[list[str], np.ndarray]:
    """Return (dummy_columns, dummy_matrix) for the training tickers.

    One ticker is dropped as the reference category (avoids collinearity with
    the intercept) — identical to
    :func:`thesis_pipeline.modeling.panel_logit.add_ticker_fixed_effects`.
    """
    tr = pd.get_dummies(pd.Series(tickers).astype(str), prefix="tkr")
    if tr.shape[1] == 0:
        return [], np.zeros((len(tickers), 0), dtype=float)
    ref = tr.columns[0]
    tr = tr.drop(columns=[ref])
    return list(tr.columns), tr.values.astype(float)


def _align_ticker_dummies(tickers: pd.Series, columns: list[str]) -> np.ndarray:
    """Reindex test-ticker dummies to the training columns.

    Tickers unseen in training collapse to the all-zero reference row.
    """
    te = pd.get_dummies(pd.Series(tickers).astype(str), prefix="tkr")
    te = te.reindex(columns=columns, fill_value=0)
    return te.values.astype(float)


# ---------------------------------------------------------------------------
# Fit + predict
# ---------------------------------------------------------------------------

def fit_logistic_model(train_df: pd.DataFrame,
                       feature_cols: list[str],
                       params: Mapping[str, Any] | None = None,
                       *,
                       family: str = PER_ASSET,
                       panel_mode: str = "pooled") -> dict:
    """Fit a ``StandardScaler`` + ``LogisticRegression`` on ``train_df``.

    The scaler is fit on the training rows only. For
    ``family='panel'`` and ``panel_mode='ticker_fixed_effects'`` the (unscaled
    0/1) ticker dummies are appended after the scaled continuous block, with
    the reference category dropped — exactly as the final panel model builds
    its design matrix. Returns an ``artifacts`` dict consumed by
    :func:`predict_proba`.
    """
    resolved = {**DEFAULT_PARAMS, **dict(params or {})}
    model = LogisticRegression(
        penalty=_PENALTY,
        C=float(resolved.get("C", 1.0)),
        class_weight=_coerce_class_weight(resolved.get("class_weight")),
        solver=_SOLVER,
        max_iter=_MAX_ITER,
        random_state=_RANDOM_STATE,
    )
    scaler = StandardScaler()
    X = scaler.fit_transform(train_df[feature_cols].values.astype(float))
    dummy_cols: list[str] | None = None
    if family == PANEL and panel_mode == "ticker_fixed_effects":
        dummy_cols, tr_dummies = _train_ticker_dummies(train_df["ticker"])
        if dummy_cols:
            X = np.hstack([X, tr_dummies])
    y = train_df["target"].values.astype(int)
    model.fit(X, y)
    return {
        "model": model,
        "scaler": scaler,
        "dummy_cols": dummy_cols,
        "family": family,
        "panel_mode": panel_mode,
        "feature_cols": list(feature_cols),
    }


def predict_proba(artifacts: Mapping[str, Any],
                  df: pd.DataFrame,
                  feature_cols: list[str] | None = None,
                  *,
                  family: str | None = None,
                  panel_mode: str | None = None) -> np.ndarray:
    """Predicted P(y=1) for ``df`` using fitted ``artifacts``.

    The ``family`` / ``panel_mode`` / ``feature_cols`` default to the values
    captured at fit time, so callers usually pass only ``artifacts`` and ``df``.
    """
    model = artifacts["model"]
    scaler = artifacts["scaler"]
    dummy_cols = artifacts.get("dummy_cols")
    family = family or artifacts.get("family", PER_ASSET)
    panel_mode = panel_mode or artifacts.get("panel_mode", "pooled")
    feature_cols = feature_cols or artifacts.get("feature_cols")

    X = scaler.transform(df[feature_cols].values.astype(float))
    if family == PANEL and panel_mode == "ticker_fixed_effects" and dummy_cols:
        te = _align_ticker_dummies(df["ticker"], dummy_cols)
        X = np.hstack([X, te])
    return model.predict_proba(X)[:, 1]


# ---------------------------------------------------------------------------
# Tuner
# ---------------------------------------------------------------------------

def tune_logistic_hyperparams(train_df: pd.DataFrame,
                              feature_cols: list[str],
                              *,
                              family: str = PER_ASSET,
                              objective: str = "brier_score",
                              search_space: Mapping[str, Sequence[Any]] | None = None,
                              hpo_cfg: Mapping[str, Any] | None = None,
                              panel_mode: str = "pooled",
                              default_params: Mapping[str, Any] | None = None) -> dict:
    """Grid-search logistic hyperparameters inside one walk-forward window.

    ``train_df`` is the **entire** current training window (per-asset rows
    ``[0, t-1]`` or panel rows with ``timestamp < tau``). The test point /
    timestamp is never supplied. The routine:

    1. Splits ``train_df`` chronologically into inner-train / validation.
    2. Falls back to default hyperparameters (``hpo_status =
       'fallback_insufficient_data'``) when either block is too small.
    3. Fits a scaler + logistic regression per grid combination on the
       inner-train block, scores the validation block under ``objective``, and
       keeps the best combination. Combinations that fail to fit are logged and
       skipped; if every combination fails the routine falls back to defaults
       (``hpo_status = 'fallback_all_failed'``).
    4. Re-fits the final model on the full ``train_df`` with the chosen
       parameters when ``refit_on_full_train`` is true (the default); otherwise
       the inner-train model is reused.

    Returns a dict with ``best_params``, ``best_score``, ``hpo_status`` and
    ``artifacts`` (a fitted model ready for :func:`predict_proba`).
    """
    logger = get_logger()
    hpo_cfg = dict(hpo_cfg or {})
    search_space = dict(search_space or hpo_cfg.get("search_space") or {})
    default_params = dict(default_params or DEFAULT_PARAMS)
    min_train = int(hpo_cfg.get("min_train_obs", 60))
    min_val = int(hpo_cfg.get("min_validation_obs", 20))
    val_frac = float(hpo_cfg.get("validation_fraction", 0.2))
    refit = bool(hpo_cfg.get("refit_on_full_train", True))

    grid = make_param_grid(search_space) or [dict(default_params)]

    inner_train, validation = chronological_train_validation_split(
        train_df, val_frac, family=PANEL if family == PANEL else PER_ASSET,
    )

    best_params = dict(default_params)
    best_score: float | None = None
    hpo_status = "ok"

    insufficient = (
        len(inner_train) < min_train
        or len(validation) < min_val
        or inner_train.empty
        or inner_train["target"].nunique() < 2
    )
    if insufficient:
        hpo_status = "fallback_insufficient_data"
    else:
        scored_any = False
        for params in grid:
            try:
                arts = fit_logistic_model(inner_train, feature_cols, params,
                                          family=family, panel_mode=panel_mode)
                proba = predict_proba(arts, validation, feature_cols,
                                      family=family, panel_mode=panel_mode)
            except Exception as exc:  # noqa: BLE001 — skip un-fittable combos
                logger.warning("hpo: params %s failed to fit (%s) — skipping",
                               params, exc)
                continue
            y_val = validation["target"].astype(int).values
            pred = (proba >= 0.5).astype(int)
            score = score_validation_predictions(y_val, proba, pred, objective)
            if score != score:  # NaN
                continue
            scored_any = True
            if _is_better(score, best_score, objective):
                best_score, best_params = score, dict(params)
        if not scored_any:
            hpo_status = "fallback_all_failed"
            best_params, best_score = dict(default_params), None

    # ── Final fit ───────────────────────────────────────────────
    final_train = train_df if refit else (inner_train if not inner_train.empty else train_df)
    try:
        artifacts = fit_logistic_model(final_train, feature_cols, best_params,
                                       family=family, panel_mode=panel_mode)
    except Exception as exc:  # noqa: BLE001 — last-resort default refit
        logger.warning("hpo: final refit with %s failed (%s) — default params",
                       best_params, exc)
        best_params = dict(default_params)
        hpo_status = "fallback_refit_failed"
        artifacts = fit_logistic_model(final_train, feature_cols, best_params,
                                       family=family, panel_mode=panel_mode)

    return {
        "best_params": best_params,
        "best_score": best_score,
        "hpo_status": hpo_status,
        "artifacts": artifacts,
    }


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

def class_weight_label(value: Any) -> str:
    """Parquet/CSV-friendly label for a class-weight value (``None`` → 'none')."""
    return "none" if _coerce_class_weight(value) is None else str(value)


def hpo_row_columns(objective: str, result: Mapping[str, Any]) -> dict:
    """Per-signal HPO provenance columns for one tuned prediction."""
    best = result.get("best_params", {})
    return {
        "hpo_enabled":       True,
        "hpo_objective":     objective,
        "best_C":            float(best.get("C", DEFAULT_PARAMS["C"])),
        "best_class_weight": class_weight_label(best.get("class_weight")),
        "hpo_score":         result.get("best_score"),
        "hpo_status":        result.get("hpo_status", "ok"),
    }


def summarize_hpo_columns(signals: pd.DataFrame) -> dict:
    """Aggregate HPO columns from a signal frame for ``metrics_summary.csv``.

    Returns an empty dict when the frame carries no enabled HPO columns, so the
    metrics schema is unchanged whenever tuning is off.
    """
    if signals is None or signals.empty or "hpo_enabled" not in signals.columns:
        return {}
    enabled = signals["hpo_enabled"].fillna(False).astype(bool)
    if not enabled.any():
        return {}
    sub = signals[enabled]
    out: dict[str, Any] = {"hpo_enabled": True}
    if "hpo_objective" in sub.columns and not sub["hpo_objective"].dropna().empty:
        out["hpo_objective"] = str(sub["hpo_objective"].dropna().iloc[0])
    if "best_C" in sub.columns:
        c = pd.to_numeric(sub["best_C"], errors="coerce").dropna()
        if not c.empty:
            out["best_C_median"] = float(c.median())
            mode = c.mode()
            out["best_C_mode"] = float(mode.iloc[0]) if not mode.empty else float("nan")
    if "best_class_weight" in sub.columns:
        cw = sub["best_class_weight"].dropna().astype(str)
        if not cw.empty:
            mode = cw.mode()
            out["best_class_weight_mode"] = mode.iloc[0] if not mode.empty else ""
    if "hpo_status" in sub.columns:
        counts = sub["hpo_status"].fillna("ok").astype(str).value_counts().to_dict()
        # Compact, CSV-friendly "status:count;status:count" string.
        out["hpo_status_counts"] = ";".join(f"{k}:{v}" for k, v in sorted(counts.items()))
    return out
