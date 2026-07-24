# Data dictionary

Columns discovered from the existing scripts. Anything marked `(observed)`
was confirmed against the original code; items marked `(inferred)` follow
the dominant naming convention used by `Sentiment_feature_engineering.py`
but have not been hand-verified.

> **Timestamp convention (all tables).** `timestamp` is the **interval-START
> label**, i.e. the raw CCXT bar-start timestamp. A row labelled `t` describes
> the *completed* interval `[t, t+h)` — its `open` is the first intraday open,
> its `close` the final intraday close, and its `high`/`low` the intraday
> extrema. The market + Reddit information for `[t, t+h)` becomes usable at
> `t+h`, and the row forecasts the sign of the return of the NEXT interval
> `[t+h, t+2h)` (`target`). No timestamp shift is applied on either the price
> or the sentiment side. See
> `thesis_pipeline.diagnostics.timing_invariant`.

---

## Raw OHLCV — `Data/Raw/Price/<horizon>/<TICKER>USDT_<suffix>.parquet`

(observed; loader: `load_ohlcv` in `Price_Data_Validation.py` /
`Create_Price_Features.py`)

| column     | dtype           | notes |
|------------|-----------------|-------|
| `timestamp`| int64 / datetime| Interval-START label; UTC (raw CCXT bar-start). Row = completed interval `[t, t+h)`. Loader normalises ms → datetime. |
| `open`     | float64         | |
| `high`     | float64         | |
| `low`      | float64         | |
| `close`    | float64         | |
| `volume`   | float64         | |

Sidecar files in the same folder:

| file | purpose |
|------|---------|
| `_checkpoint.json` | `Crypto _data.py` resume state |
| `price_sources.csv` | per-source provenance: exchange, symbol, quote, coverage |

---

## CoinMarketCap — `Data/Raw/Price/CoinMarketCap/`

(observed in `Price_Data_Validation.py::load_cmc_data`)

| file | columns |
|------|---------|
| `price.parquet`      | `date` (index or column), one column per CMC ID (e.g. `1`, `1027`, …) |
| `volume.parquet`     | same shape as `price.parquet` |
| `market_cap.parquet` | same shape — used by `Create_Price_Features.py` for `market_cap_t` |
| `MetaData.csv` / `MetaData.xlsx` | per-coin metadata (symbol, name, CMC id, listing date) |

---

## `Data/Processed/Sentiment/sentiment_combined.csv`

(observed in `Sentiment_Data_Load.py`)

| column | notes |
|--------|-------|
| `id`            | Reddit submission id |
| `created_utc`   | Post timestamp (UTC) |
| `subreddit`     | Source subreddit |
| `title`         | Post title (cleaned text) |
| `selftext`      | Post body (cleaned text) |
| `score`         | Net upvotes |
| `upvote_ratio`  | Reddit upvote ratio (0–1) |
| `num_comments`  | Comment count |
| `ticker`        | Mapped ticker from `subreddit_ticker_mapping.xlsx` |
| `url`, `permalink`, … | (other Reddit fields, see `KEEP_COLUMNS`) |

---

## Scored sentiment CSVs — `Data/Transformed/Sentiment_Scored_*.csv`

Same identifier columns as `sentiment_combined.csv`, plus:

| column | producers | range | notes |
|--------|-----------|-------|-------|
| `title_score`    | all       | [-1, 1] | model-specific score on `title` |
| `selftext_score` | all       | [-1, 1] | score on `selftext` (NaN when empty) |
| `title_label`    | all       | `positive`/`neutral`/`negative` | |
| `selftext_label` | all       | `positive`/`neutral`/`negative` | |

VADER uses `compound` ∈ [-1, 1] with thresholds ±0.05; CryptoBERT uses
P(bullish) − P(bearish). (FinBERT was evaluated but removed — see
`docs/refactor_log.md`.) See `docs/feature_definitions.md`.

---

## `Data/Features/price_features_<horizon>.parquet`

(observed in `Create_Price_Features.py::create_features_for_coin_horizon`)

| column                | dtype    | notes |
|-----------------------|----------|-------|
| `timestamp`           | datetime | Interval-START label (completed interval `[t, t+h)`); the row forecasts `[t+h, t+2h)` |
| `date`                | date     | Calendar date used for CMC matching |
| `ticker`              | string   | |
| `horizon`             | string   | `1h` / `6h` / `1d` |
| `target`              | int8     | 1 if next-period log return ≥ 0, else 0 |
| `log_return_t`        | float64  | Winsorized at (0.005, 0.995) |
| `cum_log_return_7`    | float64  | Rolling sum of `log_return_t`, window 7 |
| `cum_log_return_14`   | float64  | Rolling sum of `log_return_t`, window 14 |
| `realized_vol_14`     | float64  | Rolling std of `log_return_t`, window 14 |
| `volume_diff`         | float64  | Winsorized first difference of volume |
| `market_cap_t`        | float64  | CMC market cap on `date` |

---

## `Data/Features/sentiment_features_<horizon>.parquet`

(produced by `create-sentiment-features` for each scorer — `vader`,
`cryptobert` — and for three text scopes — `title`, `selftext`, `combined`)

Key columns (per `{model}` and `{scope}`):

| column                           | notes |
|----------------------------------|-------|
| `{model}_{scope}_mean`           | Equal-weighted mean over the bar |
| `{model}_{scope}_median`         | (observed) |
| `{model}_{scope}_std`            | (observed) |
| `{model}_{scope}_weighted_mean`  | Engagement-weighted mean (see feature_definitions.md) |
| `{model}_post_count`             | Number of posts in the bar |
| `{model}_bullishness_ratio`      | (inferred) share of posts with positive label |

Identifier columns:

| column     | notes |
|------------|-------|
| `timestamp`| Interval-START label (posts floored to the slot start of `[t, t+h)`) |
| `ticker`   | Mapped via `subreddit_ticker_mapping.xlsx` |
| `horizon`  | Same as price |

---

## `Data/Final/features_<horizon>.parquet`

Left-join of price + sentiment features on `(ticker, timestamp)`. Missing
sentiment values are filled per the policy in `Merge_Features.py`:

| column kind                  | fill value |
|------------------------------|-----------:|
| `*_mean`, `*_weighted_mean`  | 0.0 (neutral) |
| `*_std`                      | NaN |
| `*_post_count`               | 0 |
| `*_bullishness_ratio`        | 0.5 |

---

## `Outputs/Signals/<horizon>/<set_id>.parquet`

(observed in `Run_Models.run_walk_forward` and `run_rolling_probability`)

| column                    | dtype        | notes |
|---------------------------|--------------|-------|
| `timestamp`               | datetime     | **Interval-START label** (raw). Completed interval `[t, t+h)`. |
| `forecast_origin`         | datetime,UTC | `timestamp + h`. Canonical output is restricted to `2022-01-01T00:00:00Z <= forecast_origin < 2023-01-01T00:00:00Z` (end-exclusive). |
| `ticker`                  | string       | |
| `target`                  | int8         | Realised direction of the NEXT interval `[t+h, t+2h)` |
| `prediction`              | int8         | Predicted direction (threshold 0.5) |
| `probability`             | float64      | Predicted P(class=1) |
| `set_id`                  | string       | E.g. `ECON`, `SENT_VAD_L`, `ECON_CBT_F`, `NAIVE` |
| `sentiment_model`         | string       | `vader` / `cryptobert` or `-` |
| `horizon`                 | string       | `1h` / `6h` / `1d` |
| `preprocessing_signature` | string       | Preprocessing + forecast-sample contract hash (cache invalidation). |

Row inclusion is decided on `forecast_origin`, **not** the raw `timestamp`, so
the last retained raw timestamp differs by horizon. The single source of truth
for the window is the `forecast_sample` block in `configs/model_specs.yaml`;
the horizon→offset map and filter live in
`thesis_pipeline.modeling.forecast_sample`. No external post-run pruning is
part of the canonical workflow.

`Outputs/Signals/metrics_summary.csv` contains pooled and per-ticker metrics
(`accuracy`, `balanced_accuracy`, `f1`, `precision`, `recall`, `log_loss`,
`brier_score`) computed from the **filtered** signal rows, plus
`first_test_timestamp` / `last_test_timestamp` (raw interval-start labels) and
`first_forecast_origin` / `last_forecast_origin`, joined with the `set_id`,
`sentiment_model`, `label`, and `category` from `feature_sets.xlsx`.

---

## `row_id` convention (new)

Where the new package generates new files, it attaches a stable identifier:

```
row_id = "{ticker}_{horizon}_{epoch_nanoseconds}"
```

Files already on disk from the old scripts do not carry `row_id`; the loader
`thesis_pipeline.io.add_row_id()` synthesises it on read when the column is
absent.
