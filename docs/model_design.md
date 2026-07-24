# Model design

Every value below is pinned in `configs/model_specs.yaml` and
`configs/feature_sets.yaml` (mirror of `feature_sets.xlsx`) and implemented in
`src/thesis_pipeline/modeling/`. Nothing here is copied from memory.

## Task

Binary classification of the **direction of the next-period log return**, per
coin, per horizon (1h / 6h / 1d). Positive class (`1`) = next-period log
return ≥ 0. Threshold on the predicted probability = **0.5**.

## Feature sets (17)

Canonical in `feature_sets.xlsx`, mirrored in `configs/feature_sets.yaml`.
Economics core = `log_return_t, cum_log_return_{7d,14d,21d}, realized_vol_14d,
volume_diff, log_market_cap_lag1`. Sentiment blocks are title-based (no
engagement weighting): `L` = title-score mean, `LD` = mean + std, `DA` =
bullishness ratio + `log1p_post_count`, `F` = all four.

| family | IDs | content |
|--------|-----|---------|
| Economics benchmark | `ECON` | economics core only |
| Sentiment-only | `SENT_{VAD,CBT}_{L,LD,DA,F}` (8) | sentiment block only, per scorer |
| Combined | `ECON_{VAD,CBT}_{L,LD,DA,F}` (8) | economics core + sentiment block |

Sentiment features resolve per scorer for **vader** and **cryptobert** (FinBERT
was removed). `NAIVE` is a rolling-probability reference generated automatically
per `(horizon, family, window)` — it is **not** one of the 17 feature sets. The
legacy `B*/E*/S*/C*` IDs were retired (see `docs/refactor_log.md`).

---

## Production specification — panel logistic regression

This is the **default** model (a bare `run-models` runs exactly this):
`--model-type panel_logit --panel-mode ticker_fixed_effects
--train-window rolling_fixed --rolling-window-days 180 --tune-hyperparams
--hpo-objective log_loss`. Implementation:
`src/thesis_pipeline/modeling/panel_logit.py`.

### Estimator

```
sklearn.linear_model.LogisticRegression(
    penalty="l2",            # ridge (L2)
    C=<selected by nested HPO; 1.0 when --no-tune-hyperparams>,
    solver="lbfgs",
    max_iter=1000,
    random_state=42,
)
```

A **pooled panel** logit in a forecasting setting (not a classical inference
panel). For each unique test timestamp `τ`:

- **train** = every coin observation with `timestamp < τ`, restricted to the
  rolling window (see below);
- **test**  = every coin observation with `timestamp == τ`;

so one logit is fit over the whole panel and predicts all coins at `τ`.

- **Panel mode = `ticker_fixed_effects`**: `y_it ~ X_it + ticker dummies` —
  shared feature **slopes** with coin-specific intercepts. Dummies are fit on
  training tickers only (one reference dropped); unseen test tickers collapse to
  the reference. **No time fixed effects** — a test-timestamp dummy is
  unidentified out-of-sample. (`--panel-mode pooled` — a single shared intercept
  — is available as a robustness variant.)

### Training window — rolling, fixed 180 calendar days

`--train-window rolling_fixed --rolling-window-days 180` (config
`walk_forward.scheme = rolling_fixed`, `rolling_window_days = 180`). The window
is measured in **calendar days**, so the wall-clock training span is identical
across 1h / 6h / 1d. The initial position is the first 50 % of unique timestamps
(`init_train_frac = 0.50`), never fewer than `min_init_train_obs = 30`; a step is
skipped if the window has `< 20` valid rows or fewer than 2 classes.

### Preprocessing — training-window only (leakage-safe)

Fit on the training rows of each window **only**, then applied to `τ`:

- **Winsorisation**: a leakage-safe `TrainingWindowWinsorizer` computes its
  quantile bounds from the training window alone (full-sample winsorisation was
  removed — no `winsorization_thresholds.csv` is produced).
- **Scaling**: a fresh `StandardScaler` (`fit_scope = train_window_only`).

There is no leakage from `τ` into training by construction.

### Nested hyperparameter search (HPO)

`--tune-hyperparams` (default ON) runs a nested grid search **inside every
training window** and selects by `--hpo-objective log_loss` (config
`hyperparameter_tuning`):

- **C grid**: `[0.01, 0.1, 1.0, 10.0]`
- **class_weight grid**: `[null]` only (the production grid; `balanced` degrades
  probability calibration under a log-loss/Brier objective and is a documented
  robustness-only variant).
- validation = most-recent 20 % of the training window
  (`validation_fraction = 0.2`); falls back to fixed `C = 1.0` below
  `min_train_obs = 60` / `min_validation_obs = 20`; the best params are refit on
  the full window before predicting (`refit_on_full_train = true`).

log loss is the objective because it rewards calibrated probabilities, matching
how predictions are reused downstream (threshold analysis, economic backtest).

### Forecast-origin sample (2022)

Signal rows are restricted **at write time** to the canonical production sample
(config `forecast_sample`, the single source of truth — never hard-coded
elsewhere): inclusion is decided on the **forecast origin** (`timestamp + h`),
retaining rows with `2022-01-01T00:00:00Z ≤ forecast_origin < 2023-01-01T00:00:00Z`.
The raw interval-start `timestamp` is kept and a tz-aware `forecast_origin`
column is added.

### Numerical guards

`probability` is clipped to `[1e-15, 1 − 1e-15]` for log loss; predictions
(threshold 0.5) are unaffected by the clip.

### Complete 17-set grid + benchmarks

`run-models` runs the **full 17-set grid** when `--set-id` is omitted (the
config is filtered only when `--set-id` is given). `ECON` is the matched
economics-only benchmark every combined `ECON_*` set is compared against;
`NAIVE` (rolling probability `p̂_t = mean(y[:t])`, `ŷ_t = 1 if p̂_t ≥ 0.5`) is a
separate absolute-skill reference, generated once per `(horizon, family,
window)`, not as a feature set.

### Output

`Outputs/Signals/<horizon>/<set_id>[_<sentiment_model>]_panel_ticker_fe.parquet`
(or `_panel_pooled` for the pooled variant), plus the HPO-variant / rolling-window
suffixes. Columns include `timestamp, ticker, target, prediction, probability,
set_id, sentiment_model, model_type, panel_mode, hpo_enabled, hpo_objective,
hpo_variant, best_C, best_class_weight, hpo_score, hpo_status` (see
`configs/model_specs.yaml → output_columns`). Pooled + per-ticker metrics go to
`Outputs/Signals/metrics_summary.csv`.

### Checkpointing and resume

`--checkpoint --resume` (both default ON). The walk-forward groups its test
timestamps into checkpoint chunks (`--checkpoint-chunk-size`, default 20; the
final runs use 30). Each chunk is computed, persisted atomically
(`chunks/chunk_NNNN.parquet`) and, on resume, reloaded instead of recomputed. A
chunk is purely a storage/compute partition — every `τ` still trains on its own
window, so chunking changes nothing about the predictions. Deliberate restart:
`--restart` (ignore cached signal parquets) and/or `--clear-checkpoints` (delete
this run's checkpoint directory first). The preprocessing signature is embedded
in the checkpoint manifest so a methodology change invalidates stale checkpoints.

---

## Parallel panel-logit checkpoint chunks (`--n-jobs`)

`--n-jobs` parallelises the panel-logit **checkpoint chunks**. A chunk is purely
a storage / compute partition — it does **not** change the training window or the
forecasting methodology. Every test timestamp τ still selects its own training
window (`timestamp < τ`), fits its own preprocessing on that window only, runs
its own nested HPO when enabled, and predicts only τ. Because chunks are
independent, they can be computed in parallel worker processes.

**Checkpointing must be enabled for parallel execution.** Parallelism operates
over checkpoint chunks, so it only applies when checkpointing is on (the
default). With `--no-checkpoint`, `--n-jobs` is **ignored**: the panel
walk-forward follows its non-checkpointed **sequential** path regardless of the
requested worker count, and a warning is printed to say so.

### Option

```
--n-jobs N
```

* **default `1`** — sequential; behaviour is unchanged for existing users.
* **positive integer** — that many worker processes.
* **`-1`** — use all CPUs available to the current **process / container** (via
  `joblib.cpu_count()`), which respects CPU quotas / cgroup limits and is **not**
  necessarily every physical or host CPU.
* **`0` or `< -1`** — a clear validation error.

Each worker is capped at **one** BLAS/OpenMP thread (`threadpoolctl` +
joblib/loky `inner_max_num_threads=1`) so `N` workers cannot oversubscribe the
CPU. The sequential `--n-jobs 1` path keeps the library's normal threading.

`--n-jobs` affects execution speed only, **not** the empirical result: the final
signal frame is always re-sorted by `(timestamp, ticker)`, so worker completion
order never affects output (sequential vs parallel are equal within
`rtol=1e-10, atol=1e-12`). Only the **main process** writes checkpoints /
`manifest.json` (no cross-process race). Resume is fully preserved; a failed
worker re-raises in the main process and the command exits unsuccessfully rather
than returning partial results.

**Recommended:** start with `--n-jobs 4` on an 8-core / 32 GB machine; try
`--n-jobs 2` vs `4` and keep the faster one. Peak RAM ≈ `n_jobs × frame_size`
plus the main process.

---

## Legacy / robustness model — per-asset expanding walk-forward

**Not the production specification and not the default.** Available only through
explicit CLI flags (`--model-type per_asset`, optionally `--train-window
expanding`, `--no-tune-hyperparams`). Implementation:
`thesis_pipeline.modeling.walk_forward`. It fits an independent per-ticker logit
in an expanding window with step size 1:

```
n            = len(rows for one ticker)
init_train_n = max(int(n * 0.50), 30)
for t in range(init_train_n, n):
    train_X = X.iloc[:t].dropna()
    train_y = y.iloc[:t].loc[train_X.index]
    if len(train_X) < 20 or train_y.nunique() < 2:
        continue
    scaler  = StandardScaler().fit(train_X)                    # train window only
    model   = LogisticRegression(penalty="l2", C=1.0, solver="lbfgs",
                                 max_iter=1000, random_state=42).fit(...)
    proba_t = model.predict_proba(scaler.transform(X.iloc[[t]]))[:, 1]
    pred_t  = int(proba_t >= 0.5)
```

A fresh scaler and estimator are fit at every step from the train window alone —
no leakage by construction. This historical baseline uses a fixed `C = 1.0` and
an expanding window; it is retained for robustness comparison only.
