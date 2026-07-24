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
python legacy/crypto_data.py --coins BTC ETH --timeframes 1d \
    --since 2021-12-01 --until 2023-02-01 --output Data/Raw/Price
```

`--output` defaults to the repository-standard `Data/Raw/Price` (note the
capitalisation). `--timeframes` defaults to the full thesis grid
(`15m 1h 4h 6h 1d`).

**Exchange selection is fallback-based.** `--exchange` only sets a *preferred
lead* exchange; the downloader still walks each coin's
`COIN_EXCHANGE_PREFERENCE` and then `FALLBACK_EXCHANGES` whenever a pair is
missing or returns no data, so the actual source can differ per coin. Every
downloaded series records its provenance in `price_sources.csv`, including the
requested `preferred_exchange`, the `exchange` actually used, `symbol`, `quote`,
`timeframe`, first/last bar, `n_bars`, coverage, `filename`, `fetched_at`, and
the runtime `ccxt_version`. (The `preferred_exchange` and `ccxt_version` columns
are new — provenance logs generated before this change do not contain them.)

This is **not** wired through the package CLI by design — it talks to live
exchange APIs and should be a deliberate, manual step rather than something
that runs accidentally as part of `run-pipeline`.

### `score_finbert.py` (retired)

A **retired** FinBERT scorer (`ProsusAI/finbert`). FinBERT was evaluated during
development and then dropped from the pipeline — it is trained on
financial-analyst language and produced no meaningful variation on Reddit/crypto
text. It is kept here only as a record of that experiment; it is **not** a
supported pipeline component. The production scorers are VADER and CryptoBERT,
and the CLI rejects `score-sentiment --model finbert` with an explanatory error.

## Why this folder exists

The repository's active pipeline lives in `src/thesis_pipeline/`, with thin
entry points in `scripts/`. Any file that does not fit those roles — but is
still useful enough to keep for reproducibility — belongs here.
