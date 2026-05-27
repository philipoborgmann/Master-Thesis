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
