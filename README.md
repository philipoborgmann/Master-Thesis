# Master Thesis — Social-Media Sentiment and Cryptocurrency Return Direction

> **Research question:** *Does Reddit-based social-media sentiment improve the
> out-of-sample forecast of the direction of future cryptocurrency returns?*

This repository contains the full, staged pipeline behind the thesis: it
ingests OHLCV price data and Reddit submissions, scores the posts for
sentiment, engineers per-horizon price and sentiment predictors, merges them
into modelling matrices, fits walk-forward **panel logistic regressions**, and
runs the confirmatory and exploratory statistical evaluation.

The single supported entry point is the package CLI:

```bash
python -m thesis_pipeline.cli <command> [options]
```

---

## A. Empirical scope

| Dimension | Setting |
|-----------|---------|
| Cross-section | **25 cryptocurrencies** (see `configs/coins.yaml`) |
| Forecast horizons | **1 hour, 6 hours, 1 day** |
| Sentiment scorers | **VADER** (lexicon baseline) and **CryptoBERT** (`ElKulako/cryptobert`) |
| Model | **Panel logistic regression** (`sklearn.LogisticRegression`, L2 / ridge) |
| Panel mode | **Ticker fixed effects** (coin dummies, shared slopes) |
| Estimation window | **Rolling, fixed 180 calendar days** (identical wall-clock across horizons) |
| Hyperparameter tuning | **Nested grid search** inside each training window, objective = **log loss** |
| Target | Binary direction of the next-period return (`1` = next log return ≥ 0) |
| Evaluation | Directional (accuracy / McNemar) **and** probabilistic (log loss) |

**Hypotheses.** The evaluation follows a **pre-specified** testing plan — the
hypothesis families, test statistics and multiple-testing correction are fixed
in code and config (`configs/*`,
`src/thesis_pipeline/evaluation/preregistration.py`; the module name is
internal). "Pre-specified" is the accurate description: the plan is fixed in the
repository, not externally time-stamped or registered.

- **H1 — does sentiment help at all?** Combined `ECON_*` sets (economics +
  sentiment) versus the `ECON` economics-only benchmark, tested two ways:
  probabilistic (lower out-of-sample **log loss**, moving-block bootstrap) and
  directional (higher **accuracy**, pooled McNemar).
- **H2 — volatility regime.** Whether the sentiment improvement differs between
  high- and low-volatility regimes (cluster-robust difference-in-improvement).
- **H3 — market-cap regime.** Whether the sentiment improvement differs between
  small- and large-cap regimes (cluster-robust difference-in-improvement).

Sentiment-only (`SENT_*`) contrasts and cross-horizon comparisons are reported
as **exploratory** families with their own multiple-testing correction. Results
themselves live in the thesis document, not in this repository.

> FinBERT was evaluated during development but **removed** from the pipeline
> (it is trained on financial-analyst language and produced no meaningful
> variation on Reddit/crypto text). The scorer source is retained under
> `legacy/score_finbert.py` for historical traceability (see `legacy/README.md`),
> and the CLI rejects `--model finbert` with an explanatory error.

---

## B. Feature sets (17)

Canonical in `feature_sets.xlsx`, mirrored in `configs/feature_sets.yaml`. The
economics core is `log_return_t, cum_log_return_{7d,14d,21d}, realized_vol_14d,
volume_diff, log_market_cap_lag1`. Sentiment blocks are title-based (no
engagement weighting): `L` = title-score mean, `LD` = mean + std, `DA` =
bullishness ratio + `log1p_post_count`, `F` = all four.

| Family | IDs | Content |
|--------|-----|---------|
| Economics benchmark | `ECON` | economics core only |
| Sentiment-only (VADER) | `SENT_VAD_L`, `SENT_VAD_LD`, `SENT_VAD_DA`, `SENT_VAD_F` | VADER block only |
| Sentiment-only (CryptoBERT) | `SENT_CBT_L`, `SENT_CBT_LD`, `SENT_CBT_DA`, `SENT_CBT_F` | CryptoBERT block only |
| Combined (VADER) | `ECON_VAD_L`, `ECON_VAD_LD`, `ECON_VAD_DA`, `ECON_VAD_F` | economics core + VADER block |
| Combined (CryptoBERT) | `ECON_CBT_L`, `ECON_CBT_LD`, `ECON_CBT_DA`, `ECON_CBT_F` | economics core + CryptoBERT block |

`NAIVE` is a rolling-probability reference signal generated automatically per
`(horizon, family, window)` for the absolute-skill floor; it is not one of the
17 feature sets. Historical `B*/E*/S*/C*` IDs were retired and are rejected with
a helpful error (`REMOVED_SET_IDS`).

---

## C. Repository structure

```
Master-Thesis/
├── README.md                 ← this file (primary entry point)
├── pyproject.toml            ← package metadata + dependencies
├── requirements.txt          ← pip-installable runtime deps (mirror of pyproject)
├── .gitignore
├── feature_sets.xlsx         ← feature-set definitions (tracked, required)
├── subreddit_ticker_mapping.xlsx  ← subreddit → ticker map (tracked, required)
│
├── configs/                  ← all paths / constants / settings (YAML)
│   ├── paths.yaml            ← every input/output path (repo-relative)
│   ├── coins.yaml            ← the 25-coin universe + symbol aliases
│   ├── horizons.yaml         ← raw / feature / model horizons
│   ├── feature_sets.yaml     ← YAML mirror of feature_sets.xlsx
│   ├── model_specs.yaml      ← model, walk-forward, HPO, forecast-sample spec
│   ├── backtest.yaml         ← economic-backtest parameters
│   └── pipeline.yaml         ← stage list + default order
│
├── src/thesis_pipeline/      ← the package (all real logic)
│   ├── cli.py                ← `python -m thesis_pipeline.cli`
│   ├── config.py             ← config + path resolution
│   ├── price/                ← validate, features, load
│   ├── sentiment/            ← load, score_vader, score_cryptobert, aggregate, stationarity
│   ├── features/             ← merge, feature_registry, checks
│   ├── modeling/             ← panel_logit, walk_forward, checkpointing, HPO, run_models
│   ├── evaluation/           ← metrics, significance, regimes, market-cap, backtest, preregistration
│   └── diagnostics/          ← sample reports, leakage checks, structural breaks, audits
│
├── scripts/                  ← thin entry points that call a package `main()`
├── docs/                     ← detailed methodology + schema documentation
├── tests/                    ← pytest suite
└── legacy/                   ← off-pipeline raw-OHLCV downloader (ccxt); see legacy/README.md
```

All pipeline logic lives under `src/thesis_pipeline/`. Files in `scripts/` are
thin wrappers that re-export a package `main()`; the historical root-level
scripts (`Run_Models.py`, …) were retired.

### Data stages (`Data/`, `Outputs/`)

The five `Data/` stages encode genuinely different data granularities and are
**retained**; the mapping is verified against `configs/paths.yaml`:

| Stage | Meaning | Written by |
|-------|---------|-----------|
| `Data/Raw/` | Immutable external inputs: OHLCV parquet, CoinMarketCap, raw Reddit | supplied / `legacy/crypto_data.py` |
| `Data/Processed/` | Cleaned, deduplicated post-level sentiment table | `load-sentiment` |
| `Data/Transformed/` | Post-level data enriched with per-model sentiment scores | `score-sentiment` |
| `Data/Features/` | Horizon-specific aggregated price and sentiment predictors | `create-price-features`, `create-sentiment-features` |
| `Data/Final/` | Merged, modelling-ready matrices `features_{horizon}.parquet` | `merge-features` |
| `Outputs/` | Model signals, checkpoints, diagnostics, evaluation tables | `run-models`, `evaluate-signals`, diagnostics |

No large data lives in git — see section D.

---

## D. Data availability and data contract

**Large raw and derived data are not committed** (they are gitignored). To
reproduce from scratch, recreate the layout locally and place inputs where
`configs/paths.yaml` expects them.

```
Data/Raw/Price/{15min,1h,4h,6h,1d}/{TICKER}USDT_{suffix}.parquet   # OHLCV; cols: timestamp,open,high,low,close,volume (UTC)
Data/Raw/Price/CoinMarketCap/{market_cap,price,volume}.parquet     # + MetaData.{csv,xlsx}
Data/Raw/Sentiment/{subreddit}/submission.csv                      # one folder per subreddit
Data/Raw/Sentiment/st-data-full.{parquet,xlsx}                     # whole-corpus dump
```

Filename suffixes per horizon: `15min→_15m`, `1h→_1h`, `4h→_4h`, `6h→_6h`,
`1d→_1d` (only 1h/6h/1d flow to modelling). The subreddit→ticker mapping is the
tracked `subreddit_ticker_mapping.xlsx` (sheet `Subreddit_Ticker_Mapping`).

**Provenance / limits:**

- **OHLCV** was retrieved via CCXT on **2026-05-04** (see `legacy/crypto_data.py`).
  The exact installed CCXT version is only recoverable from the original
  environment and is **not** pinned here.
- **Reddit** submissions come from the public Reddit-crypto collection
  (leukipp, 2023, on Kaggle) cited in the thesis; they are not re-downloaded by
  this repository.
- **CoinMarketCap** market-cap / price / volume data were **supplied by the
  thesis supervisor**; the exact original retrieval date is **unavailable**.
- **CryptoBERT** uses the Hugging Face model `ElKulako/cryptobert` (retrieved
  **2026-04-15**).
- Reddit and CoinMarketCap inputs are **not** downloaded automatically by this
  repository and cannot be reconstructed from GitHub alone.

Reproduction that starts from committed-shaped `Data/Final/features_{horizon}.parquet`
(see path 2 below) does **not** require re-running CryptoBERT.

---

## E. Installation

Requires **Python ≥ 3.10** (developed on 3.11).

**Linux / macOS**

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[tests,transformers]"
```

**Windows PowerShell**

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[tests,transformers,acquisition]"
```

Pick the extras for what you actually run:

| Goal | Command |
|------|---------|
| Model + evaluation from `Data/Final/` (no CryptoBERT, no download) | `pip install -e ".[tests]"` |
| + CryptoBERT scoring | `pip install -e ".[tests,transformers]"` |
| Full reproduction incl. raw OHLCV acquisition | `pip install -e ".[tests,transformers,acquisition]"` |

Core dependencies always include modelling, VADER, and the stationarity /
structural-break diagnostics (`arch`, `ruptures`). The `transformers` extra adds
`torch` + `transformers` for CryptoBERT; the `acquisition` extra adds `ccxt` for
the legacy OHLCV downloader. `pip install -r requirements.txt` installs the broad
runtime environment (including `ccxt`) without the editable package; the
editable route selects `ccxt` via the `acquisition` extra instead.

---

## F. Reproducing the results

All commands below are the **canonical CLI**; nothing invokes retired scripts.
The complete order of operations is the `configs/pipeline.yaml` default order:

```
validate_price → create_price_features → load_sentiment → score_vader →
score_cryptobert → create_sentiment_features → stationarity → merge_features →
run_models → evaluate_signals → diagnostics
```

### F.1 Build features from raw inputs

```bash
python -m thesis_pipeline.cli validate-price

python -m thesis_pipeline.cli create-price-features --horizon 1d
python -m thesis_pipeline.cli create-price-features --horizon 6h
python -m thesis_pipeline.cli create-price-features --horizon 1h

python -m thesis_pipeline.cli load-sentiment
python -m thesis_pipeline.cli score-sentiment --model vader
python -m thesis_pipeline.cli score-sentiment --model cryptobert    # heavy; GPU recommended

python -m thesis_pipeline.cli create-sentiment-features

# Stationarity diagnostics (reported in the thesis appendix). This is a
# DIAGNOSTIC stage — run-models does NOT depend on it, and modelling is not
# gated on stationarity-test acceptance. The stage reads the per-horizon
# sentiment features; a bare call runs the horizons it finds.
python -m thesis_pipeline.cli stationarity

python -m thesis_pipeline.cli merge-features --horizon 1d
python -m thesis_pipeline.cli merge-features --horizon 6h
python -m thesis_pipeline.cli merge-features --horizon 1h
```

Outputs land in `Data/Features/…` and `Data/Final/features_{horizon}.parquet`.
CryptoBERT scoring is the **computationally expensive** step (many hours on the
full corpus without a GPU).

### F.2 Final model runs (the thesis specification)

The v4 CLI defaults **already are** the final specification, so a bare
`run-models` is canonical; the flags are spelled out here for transparency.
**Omitting `--set-id` runs the complete 17-set grid** (verified in
`run_models.py`: the config is filtered only when `--set-id` is given). `NAIVE`
is generated automatically.

```bash
# 1-day horizon (repeat with --horizon 6h and --horizon 1h)
python -m thesis_pipeline.cli run-models \
    --horizon 1d \
    --model-type panel_logit \
    --panel-mode ticker_fixed_effects \
    --train-window rolling_fixed \
    --rolling-window-days 180 \
    --tune-hyperparams \
    --hpo-objective log_loss \
    --checkpoint --resume \
    --checkpoint-chunk-size 30 \
    --n-jobs 4
```

```bash
python -m thesis_pipeline.cli run-models --horizon 6h \
    --model-type panel_logit --panel-mode ticker_fixed_effects \
    --train-window rolling_fixed --rolling-window-days 180 \
    --tune-hyperparams --hpo-objective log_loss \
    --checkpoint --resume --checkpoint-chunk-size 30 --n-jobs 4

python -m thesis_pipeline.cli run-models --horizon 1h \
    --model-type panel_logit --panel-mode ticker_fixed_effects \
    --train-window rolling_fixed --rolling-window-days 180 \
    --tune-hyperparams --hpo-objective log_loss \
    --checkpoint --resume --checkpoint-chunk-size 30 --n-jobs 4
```

Signals are written to `Outputs/Signals/{horizon}/{set_id}[…].parquet` (plus
`metrics_summary.csv`).

- **`--n-jobs` affects speed only, not results.** It parallelises the
  panel-logit checkpoint chunks; `4` suits an 8-core / 32 GB machine and may be
  tuned to your CPU/RAM without changing expected output. Requires
  checkpointing on (the default); with `--no-checkpoint` it is ignored and the
  run is sequential.
- **Resume:** `--checkpoint --resume` (defaults) reuse completed chunks, so an
  interrupted run continues where it stopped. **Deliberate restart:** add
  `--restart` (ignore cached signal parquets) and/or `--clear-checkpoints`
  (delete this run's checkpoint directory first).

### F.3 Evaluation, hypothesis tests and tables

Run evaluation **exactly once, with no `--horizon`**, after all three model
horizons have finished:

```bash
python -m thesis_pipeline.cli evaluate-signals --strict-feature-set-ids
```

Why once across all horizons — this matters for correctness:

- All signal files for 1d, 6h and 1h must exist **before** evaluation starts.
- A per-horizon call (`--horizon 1d`, …) writes into the same
  `Outputs/Evaluation/` directory and would **overwrite** the previous
  horizon's tables.
- The family-aware Benjamini–Hochberg correction **pools p-values across
  horizons** within each hypothesis family, so it must see all horizons at once.
- Horizon-specific `evaluate-signals --horizon <h>` is a **diagnostic** only and
  must not be used to generate the final thesis tables.

`--strict-feature-set-ids` restricts evaluation to the registered 17-set grid
(plus the `NAIVE` reference), dropping any stale/legacy signal files. Before any
metric is computed, a **signal-completeness audit** writes
`Outputs/Evaluation/signal_completeness.csv` and **aborts** (non-zero exit) if a
registered production group is incomplete — fewer test timestamps than its
matched `ECON` benchmark, or duplicate rows — so a partial run can never enter
the final tables. Pass `--allow-incomplete-signals` only to inspect a
known-partial run.

`evaluate-signals` produces the metrics, McNemar / log-loss significance tests,
volatility- and market-cap-regime tests (H2/H3), the pre-specified
Benjamini–Hochberg families, and the economic backtest, all under
`Outputs/Evaluation/`. Descriptive tables (all three horizons combined when
`--horizon` is omitted) and the sample report:

```bash
python -m thesis_pipeline.cli descriptive-final-features   # → Outputs/deskriptiv/final_feature_sets/ (1h, 6h, 1d)
python -m thesis_pipeline.cli diagnostics --horizon 1d     # → Outputs/diagnostics/
```

---

## G. Two reproduction paths

**Path 1 — full reproduction from raw inputs**

1. `pip install -e ".[tests,transformers,acquisition]"`
2. place the non-public Reddit and CoinMarketCap inputs (section D);
3. acquire or place OHLCV data (`legacy/crypto_data.py`, or supplied parquets);
4. `validate-price`;
5. `create-price-features` for 1d, 6h, 1h;
6. `load-sentiment`;
7. `score-sentiment --model vader`;
8. `score-sentiment --model cryptobert`;
9. `create-sentiment-features`;
10. `stationarity` (diagnostic; modelling does not depend on it);
11. `merge-features` for 1d, 6h, 1h;
12. `run-models` **separately** for 1d, 6h, 1h (F.2);
13. `evaluate-signals --strict-feature-set-ids` **once** (no `--horizon`);
14. `descriptive-final-features` (all horizons; omit `--horizon`);
15. other diagnostics as required (`diagnostics`, `structural-breaks`).

**Path 2 — from the final feature matrices** (no CryptoBERT, no download)

1. `pip install -e ".[tests]"`
2. place `Data/Final/features_{1h,6h,1d}.parquet`;
3. place the CoinMarketCap and daily raw-price references required by the H2/H3
   regime tests and the economic backtest (see below);
4. run all three model horizons (F.2);
5. run `evaluate-signals --strict-feature-set-ids` once across all horizons;
6. run `descriptive-final-features` once (all horizons).

This reproduces every model and evaluation result without re-running CryptoBERT.

**Non-public inputs required for parts of the evaluation.** H2 (volatility
regime) needs the daily raw-price parquets; H3 (market-cap regime) and the
economic backtest need the CoinMarketCap reference (supplied by the supervisor).
The core H1 metrics and significance tests run from the signal files alone.

Convenience wrappers run exactly path 2's CLI commands (worker count as a
parameter; strict filtering + completeness guard on; no destructive deletes):

```bash
scripts/reproduce_from_final_features.sh 4        # Linux/macOS; arg = --n-jobs
```
```powershell
scripts\reproduce_from_final_features.ps1 -NJobs 4   # Windows PowerShell
```

---

## H. Verification checks

```bash
python -m thesis_pipeline.cli --help
python -m thesis_pipeline.cli run-models --help
pytest -q
```

Every heavy stage supports `--smoke` (small deterministic inputs written to
`Outputs/diagnostics/smoke/`) and `--dry-run` (print the plan, touch nothing):

```bash
python -m thesis_pipeline.cli run-models --horizon 1d --set-id ECON --smoke --dry-run
python -m thesis_pipeline.cli merge-features --horizon 1d --smoke --dry-run
python -m thesis_pipeline.cli run-pipeline --horizon 1d --smoke
```

Expected output locations (not committed): `Data/Features/`, `Data/Final/`,
`Outputs/Signals/`, `Outputs/Evaluation/`, `Outputs/deskriptiv/`,
`Outputs/diagnostics/`.

---

## I. Reproducibility limitations

- Large raw and derived datasets are **not committed**; the documented layout
  must be recreated locally.
- **CryptoBERT** scoring is GPU-heavy and slow on the full corpus.
- The exact **CCXT** version is only known if recovered from the original
  acquisition environment; only a lower bound is declared here.
- **CoinMarketCap** data were supplied by the supervisor; the exact original
  retrieval date is unavailable.
- Some external inputs (Reddit, CoinMarketCap) cannot be reconstructed from this
  repository alone.

---

## Configuration index

| What | Where |
|------|-------|
| Paths (all I/O) | `configs/paths.yaml` |
| Coin universe | `configs/coins.yaml` |
| Horizons | `configs/horizons.yaml` |
| Feature sets | `feature_sets.xlsx` → `configs/feature_sets.yaml` |
| Model / walk-forward / HPO / forecast-sample | `configs/model_specs.yaml` |
| Pipeline stage order | `configs/pipeline.yaml` |
| Methodology & schemas | `docs/` (`model_design.md`, `data_layout.md`, `data_dictionary.md`, `feature_definitions.md`, `pipeline_overview.md`, `reproducibility.md`) |
