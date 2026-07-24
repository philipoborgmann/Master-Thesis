# Pipeline overview

The pipeline is organised as a directed acyclic graph of independent stages.
Each stage has documented inputs and outputs (see `configs/pipeline.yaml`); each
can be smoke-tested or dry-run in isolation; each prints a clear header at start
time. The package modules under `src/thesis_pipeline/` **are** the
implementation and the single source of truth — the historical root-level
scripts were retired, not wrapped.

```
  raw OHLCV (Data/Raw/Price/<h>/)                raw Reddit (Data/Raw/Sentiment/<subreddit>/submission.csv)
        │                                                    │
        ▼                                                    ▼
  validate_price                                       load_sentiment
  price/validate.py                                    sentiment/load.py
        │  Data/Raw/Price/validation/*.csv                  │  Data/Processed/Sentiment/sentiment_combined.csv
        ▼                                                    ▼
  create_price_features                          ┌───────────┴───────────┐
  price/features.py                              ▼                       ▼
        │  Data/Features/                    score_vader           score_cryptobert
        │  price_features_<h>.parquet        sentiment/            sentiment/
        │  feature_generation_report.csv     score_vader.py        score_cryptobert.py
        │                                        │  Data/Transformed/Sentiment_Scored_*.csv
        │                                        ▼
        │                                   create_sentiment_features
        │                                   sentiment/aggregate.py
        │                                        │  Data/Features/sentiment_features_<h>.parquet
        │                                        │  Data/Features/sentiment_coverage.csv
        │                                        ▼
        │                                   stationarity (diagnostic)
        │                                   sentiment/stationarity.py
        │                                        │  Data/Features/stationarity_*.parquet|xlsx
        └───────────────────┬────────────────────┘
                            ▼
                     merge_features                       features/merge.py
                            │  Data/Final/features_<h>.parquet, merge_report.csv
                            ▼
                     run_models  (SEPARATELY for 1d, 6h, 1h)      modeling/run_models.py
                            │  Outputs/Signals/<h>/<set_id>...parquet, metrics_summary.csv
                            ▼
                     evaluate_signals  (ONCE, across ALL horizons)  evaluation/evaluate_signals.py
                            │  Outputs/Evaluation/*.csv + signal_evaluation.xlsx
                            │  (incl. signal_completeness.csv, multiple_testing_manifest.csv)
                            ▼
                     final evaluation tables
```

Raw OHLCV is (re)acquired out-of-pipeline by `legacy/crypto_data.py` (ccxt); it
is a deliberate manual step, not a pipeline stage.

**Why evaluation runs once.** `run_models` is run separately per horizon, but
`evaluate_signals` is run **once with no `--horizon`** so it sees all three
horizons together: every horizon writes into the same `Outputs/Evaluation/`
directory (a per-horizon call would overwrite the previous horizon's tables),
and the family-aware Benjamini–Hochberg correction pools p-values **across
horizons** within each hypothesis family. Horizon-specific `evaluate_signals
--horizon <h>` calls are for diagnostics only and must not generate the final
thesis tables.

### Separate diagnostic / reporting stages

Run on demand, outside the modelling DAG:

- **`diagnostics`** (`diagnostics/sample_report.py`) → `Outputs/diagnostics/` —
  per-horizon sample report.
- **`descriptive-final-features`** (`diagnostics/descriptive_final_features.py`)
  → `Outputs/deskriptiv/final_feature_sets/` — descriptive statistics over the
  final feature sets (omit `--horizon` for all three horizons combined).
- **`structural-breaks`** (`diagnostics/structural_breaks.py`) — advisory
  Bai–Perron-style break diagnostics; **never** sets rolling-window sizes.

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
| score_cryptobert            | ✓     | ✓              | max_rows=200        |
| create_sentiment_features   |       | ✓              | horizon=1d, no_plots |
| stationarity                | ✓     | ✓              | horizon=1d, BTC,ETH |
| merge_features              |       | ✓              | horizon=1d          |
| run_models                  | ✓     | ✓              | horizon=1d, set_id=ECON, BTC,ETH |
| diagnostics                 |       | ✓              | horizon=1d          |
