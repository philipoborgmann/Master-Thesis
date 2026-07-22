# Model design

All values below are extracted from `Run_Models.py` and pinned in
`configs/model_specs.yaml`.

## Task

Binary classification of the **direction of next-period log return** per
ticker per horizon. Positive class (1) = next-period log return ≥ 0.

## Feature sets

Set IDs follow the convention used in `feature_sets.xlsx` (loaded by
`Run_Models.load_feature_sets`):

| family | IDs | what it tests |
|--------|-----|----------------|
| Benchmark | `B1`, `B2` | naive rolling-probability and minimal logistic |
| Economic  | `E1`, `E2`, `E3`, `E4` | price-only feature subsets |
| Sentiment | `S1` … `S7` | sentiment features only, per scorer |
| Combined  | `C1` … `C5` | economic + sentiment, per scorer |

The exact feature membership of each set is canonical in `feature_sets.xlsx`
and mirrored in `configs/feature_sets.yaml` (verification in progress — see
`docs/refactor_log.md`). The loader resolves sentiment features as
`{model}_…` for each of `vader`, `finbert`, `cryptobert`.

## Model

```
sklearn.linear_model.LogisticRegression(
    penalty="l2",       # ridge
    C=1.0,              # fixed; no hyperparameter search
    solver="lbfgs",
    max_iter=1000,
    random_state=42,
)
```

Benchmark `B1` uses the rolling-probability rule

```
p̂_t = mean(y[:t])
ŷ_t = 1 if p̂_t ≥ 0.5 else 0
```

## Walk-forward / expanding window

Scheme: **expanding** window, step size = 1 row.

```
n            = len(rows for one ticker)
init_train_n = max(int(n * 0.50), 30)            # configs/model_specs.yaml → walk_forward
for t in range(init_train_n, n):
    train_X = X.iloc[:t].dropna()
    train_y = y.iloc[:t].loc[train_X.index]
    if len(train_X) < 20 or train_y.nunique() < 2:
        continue
    scaler  = StandardScaler().fit(train_X)
    model   = LogisticRegression(...).fit(scaler.transform(train_X), train_y)
    proba_t = model.predict_proba(scaler.transform(X.iloc[[t]]))[:, 1]
    pred_t  = int(proba_t >= 0.5)
```

A fresh `StandardScaler` and a fresh `LogisticRegression` are fit at every
step from the train-window alone — there is **no leakage** from t into the
training set by construction. `Run_Models.run_walk_forward` is the
authoritative implementation and is what the new package delegates to.

Numerical guards: `probability` is clipped to `[1e-15, 1 − 1e-15]` for log
loss; predictions are not affected by this clip.

## Output

Per ticker, per (`horizon`, `set_id`, `sentiment_model`):

`Outputs/Signals/<horizon>/<set_id>.parquet` (for benchmark / economic sets)

`Outputs/Signals/<horizon>/<set_id>_<sentiment_model>.parquet` (for sentiment
or combined sets when the script writes the model into the filename).

Pooled and per-ticker metrics are written to
`Outputs/Signals/metrics_summary.csv`.

## What this design intentionally does **not** include

- No random train/test split (walk-forward only).
- No hyperparameter tuning (`C = 1.0` is fixed).
- No feature selection beyond the set definitions.
- No nonlinear models or ensembles.

These choices are part of the thesis and are not changed by the refactor.

## Alternative model family: `panel_logit`

In addition to the canonical **per-asset** walk-forward above, a second model
family is available for comparison:
`src/thesis_pipeline/modeling/panel_logit.py`.

It estimates a **pooled panel logistic regression** in a forecasting setting
(not a classical inference panel model). For each unique test timestamp `τ`:

- **train** = every coin observation with `timestamp < τ`
- **test**  = every coin observation with `timestamp == τ`

so a single logit is fit over the whole panel and predicts all coins at `τ`.
The initial training window is the first 50 % of the *unique timestamps*
(not 50 % of observations); the `StandardScaler` is fit on the training rows
only; the estimator is the identical
`LogisticRegression(penalty="l2", C=1.0, solver="lbfgs", random_state=42)`.

Two modes:

- `pooled` — `y_it ~ X_it` (shared coefficients).
- `ticker_fixed_effects` — `y_it ~ X_it + ticker dummies`, approximating
  coin-specific intercepts. Dummies are fit on training tickers only (one
  reference dropped); unseen test tickers collapse to the reference. **No time
  fixed effects** — a test-timestamp dummy is unidentified out-of-sample.

Usage (per-asset remains the default):

```bash
python -m thesis_pipeline.cli run-models --horizon 1d --set-id C2 \
    --model-type panel_logit --panel-mode pooled --restart
python -m thesis_pipeline.cli run-models --horizon 1d --set-id C2 \
    --model-type panel_logit --panel-mode ticker_fixed_effects --restart
```

Outputs are written alongside (never overwriting) the per-asset signals:
`Outputs/Signals/<horizon>/<set_id>[_<sentiment_model>]_panel_pooled.parquet`
or `..._panel_ticker_fe.parquet`, with extra columns `model_type` and
`panel_mode`. `metrics_summary.csv` gains the same two columns. The B1
rolling-probability benchmark is not a panel-logit model and is skipped with
a warning under `--model-type panel_logit`.

## Parallel panel-logit checkpoint chunks (`--n-jobs`)

`--n-jobs` parallelises the panel-logit **checkpoint chunks**. The walk-forward
groups its test timestamps into checkpoint chunks (`--checkpoint-chunk-size`,
default 20). A chunk is purely a **storage / compute partition** — it does
**not** change the training window or the forecasting methodology. Every test
timestamp τ still selects its own training window (`timestamp < τ`), fits its
own preprocessing on that window only, runs its own nested HPO when enabled,
and predicts only τ. Because chunks are independent, they can be computed in
parallel worker processes.

**Checkpointing must be enabled for parallel execution.** Parallelism operates
over checkpoint chunks, so it only applies when checkpointing is on (the
default). With `--no-checkpoint`, `--n-jobs` is **ignored**: the panel
walk-forward follows its non-checkpointed **sequential** path regardless of the
requested worker count, and a warning is printed to say so. To run in parallel,
keep checkpointing enabled (do not pass `--no-checkpoint`).

### Option

```
--n-jobs N
```

* **default `1`** — sequential; behaviour is unchanged for existing users.
* **positive integer** — that many worker processes.
* **`-1`** — use all CPUs available to the current **process / container**
  (via `joblib.cpu_count()`), which respects CPU quotas / cgroup limits and is
  **not** necessarily every physical or host CPU.
* **`0` or `< -1`** — a clear validation error.

Each worker is capped at **one** BLAS/OpenMP thread (`threadpoolctl` +
joblib/loky `inner_max_num_threads=1`) so `N` workers cannot oversubscribe the
CPU (e.g. avoiding `4 processes × 8 BLAS threads = 32` competing threads). The
sequential `--n-jobs 1` path keeps the library's normal threading.

### Example

```bash
python -m thesis_pipeline.cli run-models \
  --horizon 1h \
  --model-type panel_logit \
  --panel-mode ticker_fixed_effects \
  --n-jobs 4 \
  --checkpoint-chunk-size 30
```

### Recommended worker count

**For an 8-core, 32 GB machine, start with `--n-jobs 4`.** More workers are not
always faster: each worker holds its own copy of the feature frame in RAM, and
process start-up plus result serialization add overhead. Past the point where
memory bandwidth or RAM saturates, throughput flattens or regresses. Try
`--n-jobs 2` and `--n-jobs 4` and keep the faster one for your hardware.

* **GitHub Codespaces:** a 4-core / 8-core machine type is recommended; use
  `--n-jobs 2` (4-core) or `--n-jobs 4` (8-core).
* **Memory:** the shared feature frame is written once to a temporary parquet
  and loaded once per worker (cached), so peak RAM ≈ `n_jobs × frame_size`
  plus the main process. On 32 GB, four workers on the 1h frame is comfortable.

### Determinism, safety and resume

* Results are **methodologically identical** to the sequential run. The final
  signal frame is always re-sorted by `(timestamp, ticker)`, so the order in
  which workers finish never affects the output. Sequential vs parallel outputs
  are equal within a strict floating-point tolerance (tests use
  `rtol=1e-10, atol=1e-12`).
* Only the **main process** writes chunk checkpoints (atomically) and updates
  `manifest.json` — workers only compute and return rows, so there is no
  cross-process write race.
* **Resume** is fully preserved: already-cached chunks are reused, only pending
  chunks are submitted, a corrupt checkpoint is recomputed, and sequential and
  parallel runs can resume each other's checkpoints (chunk IDs and paths are
  deterministic).
* A **failed worker** re-raises its original exception in the main process, does
  not mark its chunk complete, leaves completed chunks reusable, and makes the
  command exit unsuccessfully rather than returning partial results.

### Benchmark output

Parallel runs print a one-line summary per feature set:
`n_jobs`, total chunks, cached chunks, submitted chunks, elapsed wall-clock and
average completed-chunk time. Small-dataset speedups are modest (start-up and
serialization dominate); full-run scaling depends on CPU, RAM, chunk balance and
serialization overhead, and is expected to approach near-linear up to the core
count when per-chunk compute (HPO, larger windows) dwarfs the overhead. Do not
assume a fixed wall-clock target — measure on the target hardware.
