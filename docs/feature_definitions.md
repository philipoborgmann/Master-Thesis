# Feature definitions

All formulas below match the implementations in `Create_Price_Features.py`
and `Sentiment_feature_engineering.py`. The refactor does not change any of
them; this document only re-states them in one place.

## Price features

Let `c_t` be the close price at bar t and `v_t` the volume at bar t. The
pipeline computes:

| feature | formula |
|---------|---------|
| `log_return_t`        | `log(c_t / c_{t-1})` then winsorized at (0.5%, 99.5%) per (ticker, horizon) |
| `cum_log_return_7`    | rolling sum of `log_return_t` over window = 7 |
| `cum_log_return_14`   | rolling sum of `log_return_t` over window = 14 |
| `realized_vol_14`     | rolling std of `log_return_t` over window = 14 |
| `volume_diff`         | `v_t − v_{t-1}`, then winsorized at (0.5%, 99.5%) per (ticker, horizon) |
| `market_cap_t`        | CoinMarketCap market cap on `date(timestamp_t)` |
| `target`              | `1` if `log_return_{t+1} ≥ 0` else `0` (binary direction of next return) |

Winsorisation thresholds are saved to
`Data/Features/winsorization_thresholds.csv` (one row per ticker × horizon ×
variable).

## Target construction

`target` is the **direction of the next-period log return**, not the
contemporaneous return. The feature row at timestamp t uses information up
to and including t, and the target reflects the move from t to t+1. This
makes `target` strictly forward-looking and the modelling task strictly
out-of-sample (see `docs/model_design.md`).

## Sentiment scoring (post level)

Three scorers run on the same `sentiment_combined.csv` input and write to
`Data/Transformed/Sentiment_Scored_*.csv`:

| scorer    | model                | per-text output |
|-----------|----------------------|-----------------|
| VADER     | `vaderSentiment` (lexicon-based) | `compound ∈ [-1, 1]`; thresholds ±0.05 for label |
| CryptoBERT| `ElKulako/cryptobert`| `softmax(logits)` → `P(bullish) − P(bearish)`; labels re-mapped to positive/neutral/negative |

> FinBERT (`ProsusAI/finbert`) was evaluated during development but **removed**
> from the pipeline (see `docs/refactor_log.md`); the CLI rejects
> `--model finbert` with an explanatory error.

For each post the scorers compute `title_score` and `selftext_score`
independently. Posts with empty selftext receive `NaN` for `selftext_score`.

## Sentiment features (horizon level)

Computed in `Sentiment_feature_engineering.py`. For each post the
**engagement weight** is

```
e = log1p(score) * upvote_ratio * (1 + log1p(num_comments))
```

then normalised to [0, 1] across the bar. The **combined post-level score**
mixes title and selftext:

```
post_score = 0.7 * title_score + 0.3 * selftext_score   (when selftext_score is not NaN)
post_score = title_score                                  (when selftext_score is NaN)
```

For each (ticker, bar) the script aggregates over posts in the bar to:

| variant      | aggregator           |
|--------------|----------------------|
| `*_mean`     | unweighted mean      |
| `*_median`   | unweighted median    |
| `*_std`      | unweighted std       |
| `*_weighted_mean` | `np.average(score, weights=e)` |

…and emits the three text scopes — `title`, `selftext`, `combined` — for
each of the three models. The final winsorisation step clips numerical
features at (0.5%, 99.5%).

Coverage statistics (share of bars with at least one post) are written to
`Data/Features/sentiment_coverage.csv`. Tickers below the
`COVERAGE_THRESHOLD` (currently 85% in `Merge_Features.py`) are excluded
from the modelling stage.

## Why these and only these

The thesis fixes the feature universe up-front to keep the comparison
between benchmarks, economic-only sets, and sentiment-augmented sets clean.
The refactor introduces **no new substantive features**; it only documents
what is already produced and ensures the formulas are not duplicated across
unrelated scripts.
