# Reproducibility

## Environment

```bash
git clone <repo>
cd Master-Thesis
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# Transformer scoring (optional, heavy):
pip install torch transformers accelerate
```

Python ≥ 3.10 is required (see `pyproject.toml`).

## Configs as single source of truth

| file | what it pins |
|------|--------------|
| `configs/paths.yaml`        | every input/output path |
| `configs/horizons.yaml`     | which horizons each stage covers |
| `configs/coins.yaml`        | ticker universe + CMC/exchange aliases |
| `configs/feature_sets.yaml` | mirror of `feature_sets.xlsx` |
| `configs/model_specs.yaml`  | walk-forward and logistic-ridge hyperparams |
| `configs/pipeline.yaml`     | stage I/O, smoke defaults, stage order |

Anything not on this list is data-dependent and must come from the local
`Data/` tree.

## Stage-by-stage regeneration

```bash
# 1. Validate raw OHLCV against CMC reference data
python -m thesis_pipeline.cli validate-price

# 2. Build price features for each horizon
python -m thesis_pipeline.cli create-price-features --horizon 1d
python -m thesis_pipeline.cli create-price-features --horizon 6h
python -m thesis_pipeline.cli create-price-features --horizon 1h

# 3. Load + clean Reddit submissions
python -m thesis_pipeline.cli load-sentiment

# 4. Score posts with each scorer (FinBERT was removed; VADER + CryptoBERT only)
python -m thesis_pipeline.cli score-sentiment --model vader
python -m thesis_pipeline.cli score-sentiment --model cryptobert     # heavy

# 5. Aggregate to per-horizon sentiment features
python -m thesis_pipeline.cli create-sentiment-features

# 6. (optional) Stationarity tests
python -m thesis_pipeline.cli stationarity --horizon 1d

# 7. Merge price + sentiment
python -m thesis_pipeline.cli merge-features --horizon 1d

# 8. Walk-forward modelling (v4 defaults = panel_logit / ticker FE / rolling
#    180d / nested HPO / log_loss). Omit --set-id to run the full 17-set grid.
python -m thesis_pipeline.cli run-models --horizon 1d                       # full grid
python -m thesis_pipeline.cli run-models --horizon 1d --set-id ECON         # single set
python -m thesis_pipeline.cli run-models --horizon 1d --set-id ECON_CBT_F   # single set
# See README section F.2 for the fully-specified final commands.

# 9. Diagnostics report
python -m thesis_pipeline.cli diagnostics --horizon 1d
```

## Smoke mode

`--smoke` switches every heavy stage to a small, deterministic configuration
and writes outputs under `Outputs/diagnostics/smoke/` instead of overwriting
production outputs. Defaults:

| stage | smoke default |
|-------|---------------|
| validate-price             | coins = BTC, ETH |
| create-price-features      | horizon = 1d, coins = BTC, ETH |
| load-sentiment             | max_rows = 10 000 |
| score-sentiment vader      | max_rows = 5 000 |
| score-sentiment cryptobert | max_rows = 200 |
| create-sentiment-features  | horizon = 1d, no_plots |
| stationarity               | horizon = 1d, coins = BTC, ETH |
| merge-features             | horizon = 1d |
| run-models                 | horizon = 1d, set_id = ECON, coins = BTC, ETH |
| diagnostics                | horizon = 1d |

To overwrite a production output from smoke mode, pass `--force` explicitly.

## Dry-run mode

`--dry-run` loads configs, prints the planned input/output paths, and exits
without running any heavy work or writing any files. Use it before kicking
off a long stage.

## What is **not** committed

The data tree and most outputs are gitignored. See `.gitignore` and
`docs/data_layout.md`. The two files that **are** tracked for convenience:

- `feature_sets.xlsx` (sheet `feature_sets` defines the set IDs)
- `subreddit_ticker_mapping.xlsx` (sheet `Subreddit_Ticker_Mapping`)

## Determinism

Sentiment scoring is deterministic except for any non-deterministic CUDA
kernels. Modelling fixes `random_state=42`; the walk-forward loop's
ordering is determined by the timestamp index of the merged feature file.
