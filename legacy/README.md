# `legacy/` — out-of-pipeline scripts

This folder holds tools that are kept for reproducibility but are **not part of
the staged thesis pipeline** in `src/thesis_pipeline/`. They are not invoked by
`python -m thesis_pipeline.cli` and do not participate in any test suite.

## Contents

### `crypto_data.py`

One-off raw-OHLCV downloader built on top of [ccxt](https://github.com/ccxt/ccxt).
It populates `Data/Raw/Price/<horizon>/<TICKER>USDT_<suffix>.parquet` from the
configured exchanges. Originally lived at the repository root as
`Crypto _data.py` (note the space in the filename).

Run manually when you need to refresh raw price data:

```bash
python legacy/crypto_data.py --exchange binance --coins BTC ETH --timeframes 1d \
    --since 2021-12-01 --until 2023-02-01 --output Data/Raw/Price
```

This is **not** wired through the package CLI by design — it talks to live
exchange APIs and should be a deliberate, manual step rather than something
that runs accidentally as part of `run-pipeline`.

## Why this folder exists

The repository's active pipeline lives in `src/thesis_pipeline/`, with thin
entry points in `scripts/`. Any file that does not fit those roles — but is
still useful enough to keep for reproducibility — belongs here.
