# Pipeline overview

The pipeline is organised as a directed acyclic graph of independent stages.
Each stage has documented inputs and outputs (see
`configs/pipeline.yaml`); each can be smoke-tested or dry-run in isolation;
each prints a clear header at start time.

```
                       ┌─────────────────────────────┐
                       │   Crypto _data.py (ccxt)    │   (one-off raw download)
                       └──────────────┬──────────────┘
                                      │ writes Data/Raw/Price/<horizon>/
                                      ▼
              ┌──────────────────────────────────────────┐
              │           validate_price                 │
              │   src/.../price/validate.py              │
              │   (wraps Price_Data_Validation.py)       │
              └──────────────┬───────────────────────────┘
                             │ Data/Raw/Price/validation/*.csv
                             ▼
              ┌──────────────────────────────────────────┐
              │        create_price_features             │
              │   src/.../price/features.py              │
              │   (wraps Create_Price_Features.py)       │
              └──────────────┬───────────────────────────┘
                             │ Data/Features/price_features_<horizon>.parquet
                             │ Data/Features/winsorization_thresholds.csv
                             │ Data/Features/feature_generation_report.csv
                             ▼
                              ─────────
                                  merge_features
                              ─────────

  Data/Raw/Sentiment/<subreddit>/submission.csv
              │
              ▼
  ┌─────────────────────────────────┐
  │       load_sentiment            │   → Data/Processed/Sentiment/sentiment_combined.csv
  │   src/.../sentiment/load.py     │   → Outputs/deskriptiv/descriptive_statistics.xlsx
  │   (wraps Sentiment_Data_Load.py)│
  └──────────────┬──────────────────┘
                 │
       ┌─────────┴────────────┬─────────────────────┐
       ▼                      ▼                     ▼
  score_vader            score_finbert          score_cryptobert
  (vaderSentiment)       (ProsusAI/finbert)     (ElKulako/cryptobert)
       │                      │                     │
       └──────────────────────┴─────────────────────┘
                                │
                                ▼
              ┌──────────────────────────────────────────┐
              │     create_sentiment_features            │
              │   src/.../sentiment/aggregate.py         │
              │   (wraps Sentiment_feature_engineering.py)│
              └──────────────┬───────────────────────────┘
                             │ Data/Features/sentiment_features_<horizon>.parquet
                             │ Data/Features/sentiment_coverage.csv
                             ▼
              ┌──────────────────────────────────────────┐
              │           stationarity                   │
              │   src/.../sentiment/stationarity.py      │
              │   (wraps Sentiment_Stationarity_Test.py) │
              └──────────────┬───────────────────────────┘
                             │ Data/Features/stationarity_*.parquet|xlsx
                             ▼
              ┌──────────────────────────────────────────┐
              │           merge_features                 │
              │   src/.../features/merge.py              │
              │   (wraps Merge_Features.py)              │
              └──────────────┬───────────────────────────┘
                             │ Data/Final/features_<horizon>.parquet
                             │ Data/Final/merge_report.csv
                             ▼
              ┌──────────────────────────────────────────┐
              │            run_models                    │
              │   src/.../modeling/run_models.py         │
              │   (wraps Run_Models.py)                  │
              └──────────────┬───────────────────────────┘
                             │ Outputs/Signals/<horizon>/<set_id>.parquet
                             │ Outputs/Signals/metrics_summary.csv
                             ▼
              ┌──────────────────────────────────────────┐
              │           diagnostics                    │
              │   src/.../diagnostics/sample_report.py   │
              └──────────────────────────────────────────┘
                Outputs/diagnostics/sample_report_<horizon>.md
```

## Horizon coverage

| Horizon | Raw OHLCV | Price features | Sentiment features | Final merged | Models |
|---------|-----------|----------------|--------------------|--------------|--------|
| 15 min  | ✓         |                | ✓                  |              |        |
| 1 h     | ✓         | ✓              | ✓                  | ✓            | ✓      |
| 4 h     | ✓         |                |                    |              |        |
| 6 h     | ✓         | ✓              | ✓                  | ✓            | ✓      |
| 1 d     | ✓         | ✓              | ✓                  | ✓            | ✓      |

Raw 4 h data exists but is not currently part of the downstream pipeline; do
not silently add it. Sentiment features exist for 15 min although the
end-to-end pipeline currently does not use that horizon.

## Stage characteristics

| Stage                       | Heavy | Supports smoke | Default smoke knob |
|-----------------------------|:-----:|:--------------:|--------------------|
| validate_price              |       | ✓              | coins=BTC,ETH      |
| create_price_features       |       | ✓              | horizon=1d, BTC,ETH |
| load_sentiment              | ✓     | ✓              | max_rows=10 000     |
| score_vader                 |       | ✓              | max_rows=5 000      |
| score_finbert               | ✓     | ✓              | max_rows=200        |
| score_cryptobert            | ✓     | ✓              | max_rows=200        |
| create_sentiment_features   |       | ✓              | horizon=1d, no_plots |
| stationarity                | ✓     | ✓              | horizon=1d, BTC,ETH |
| merge_features              |       | ✓              | horizon=1d          |
| run_models                  | ✓     | ✓              | horizon=1d, set_id=B1, BTC,ETH |
| diagnostics                 |       | ✓              | horizon=1d          |
