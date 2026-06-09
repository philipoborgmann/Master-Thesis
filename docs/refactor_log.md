# Refactor log

This log tracks what was moved where during the *organizational* refactor
from a flat set of root-level scripts to a packaged
`src/thesis_pipeline/` layout. **No empirical methodology was changed** —
the existing root scripts continue to drive the actual computation and the
new modules delegate to them via `thesis_pipeline.utils.run_script_main`.

---

## Branch

This refactor lives on `claude/social-media-sentiment-analysis-IZwgD`.
(The original task description mentioned `refactor/pipeline-structure`; the
session is pinned to the existing branch, so all changes are on that
branch.)

---

## File map (current state)

| Original location                  | Canonical package module                                    | scripts/ entry                              | Root file              |
|-----------------------------------|-------------------------------------------------------------|---------------------------------------------|------------------------|
| `Create_Price_Features.py`        | `src/thesis_pipeline/price/features.py`                     | `scripts/create_price_features.py`          | **retired**            |
| `Price_Data_Validation.py`        | `src/thesis_pipeline/price/validate.py`                     | `scripts/validate_price.py`                 | **retired**            |
| `Merge_Features.py`               | `src/thesis_pipeline/features/merge.py`                     | `scripts/merge_features.py`                 | **retired**            |
| `Run_Models.py`                   | `src/thesis_pipeline/modeling/run_models.py`                | `scripts/run_models.py`                     | **retired**            |
| `Sentiment_Data_Load.py`          | `src/thesis_pipeline/sentiment/load.py`                     | `scripts/load_sentiment.py`                 | **retired**            |
| `Sentiment_score_vader.py`        | `src/thesis_pipeline/sentiment/score_vader.py`              | `scripts/score_sentiment.py` (multi-model)  | **retired**            |
| `Sentiment_score_finbert.py`      | `src/thesis_pipeline/sentiment/score_finbert.py`            | `scripts/score_sentiment.py` (multi-model)  | **retired**            |
| `Sentiment_score_cryptobert.py`   | `src/thesis_pipeline/sentiment/score_cryptobert.py`         | `scripts/score_sentiment.py` (multi-model)  | **retired**            |
| `Sentiment_feature_engineering.py`| `src/thesis_pipeline/sentiment/aggregate.py`                | `scripts/create_sentiment_features.py`      | **retired**            |
| `Sentiment_Stationarity_Test.py`  | `src/thesis_pipeline/sentiment/stationarity.py`             | — (only via CLI `stationarity`)             | **retired**            |
| —                                  | `src/thesis_pipeline/evaluation/evaluate_signals.py`        | `scripts/evaluate_signals.py`               | — (no historical root) |
| `Crypto _data.py`                 | — (off-pipeline data acquisition)                            | —                                           | moved to `legacy/crypto_data.py` |

The direction of delegation is now consistent across the repository:

```
python scripts/run_models.py …               →  thesis_pipeline.modeling.run_models.main()
python -m thesis_pipeline.cli run-models … → cli → thesis_pipeline.modeling.run_models.main()
```

There is exactly one place each piece of logic lives. The previous
backward-compatibility redirects at the repo root were removed once the
canonical implementations had been verified — archived copies of the
historical scripts live outside the repository. The
``run_script_main`` / ``SCRIPT_MAP`` delegation layer in
``src/thesis_pipeline/utils.py`` was removed at the same time, since no
caller still routes through a root file.

## Banner comments inserted

Each root script now has a single-line banner comment right after the
shebang, e.g.

```python
# This wrapper is kept for backward compatibility.
# Preferred usage: python -m thesis_pipeline.cli create-price-features
```

No other lines in the root scripts were modified.

## Constants → configs

| Constant (origin)                              | New home |
|------------------------------------------------|----------|
| `INIT_TRAIN_FRAC = 0.50` (Run_Models.py)       | `configs/model_specs.yaml :: walk_forward.init_train_frac` |
| `DEFAULT_C = 1.0` (Run_Models.py)              | `configs/model_specs.yaml :: logistic_ridge.C` |
| `random_state = 42` (Run_Models.py)            | `configs/model_specs.yaml :: logistic_ridge.random_state` |
| `min_obs_per_window = 20` (Run_Models.py)      | `configs/model_specs.yaml :: walk_forward.min_obs_per_window` |
| `COVERAGE_THRESHOLD = 85.0` (Merge_Features.py)| TODO: not yet hoisted to YAML — leave as script default until merge code is fully migrated. |
| Winsorization quantiles 0.005 / 0.995          | TODO: hoist to `configs/feature_sets.yaml` or a new `configs/features.yaml` if needed. |
| Tickers (multiple sources)                     | `configs/coins.yaml :: tickers` |
| `NANO ↔ XNO`, `IOTA ↔ MIOTA` aliases           | `configs/coins.yaml :: exchange_symbol_aliases`, `cmc_symbol_aliases`, `preferred_cmc_symbol` |
| Preferred CMC IDs                              | `configs/coins.yaml :: cmc_ids` |
| Exchange preferences (`Crypto _data.py`)        | `configs/coins.yaml :: exchange_preferences`, `fallback_exchanges`, `quote_priority` |
| `KEEP_COLUMNS` (Sentiment_Data_Load.py)        | TODO: hoist if needed for tests |
| `MODELS=["vader","finbert","cryptobert"]`, `TITLE_WEIGHT=0.7` | Documented in `docs/feature_definitions.md`; not yet in YAML. |

## Open TODOs

1. **`Merge_Features.py` flags** — accept `--horizon`, `--feature_dir`,
   `--output_dir` so the new module can drive paths via `configs/paths.yaml`
   rather than relying on hardcoded defaults.
2. **Native `--max_rows`** on `Sentiment_Data_Load.py` and
   `Sentiment_feature_engineering.py` for a cleaner smoke mode.
3. **YAML mirror verification** — hand-check
   `configs/feature_sets.yaml :: sets` against `feature_sets.xlsx`
   (sheet `feature_sets`). Set IDs `S2`–`S7` and `C2`–`C5` are left to the
   Excel loader by default; flip `source_of_truth` only after the YAML
   matches exactly.
4. **Common-sample helper** — the new `features/checks.py` defines
   `common_sample_row_ids`, but `Run_Models.py` does not yet use it. The
   thesis question requires comparing sets on the **same** rows; before
   publishing final numbers, route the modelling stage through that helper.
5. **`stationarity` script CLI** — `Sentiment_Stationarity_Test.py` accepts
   `--ticker` only once. The wrapper iterates by passing multiple
   `--ticker` flags; verify the underlying parser allows repetition (it
   uses `--ticker` once per call; if not repeated, tighten the wrapper).
6. **`row_id`** — files written by the original scripts do not carry a
   `row_id` column; `thesis_pipeline.io.add_row_id()` synthesises it on
   read. When any output is regenerated from the new package, write
   `row_id` directly.

## Hyperparameter tuning (conservative, leakage-safe; opt-in)

Added `src/thesis_pipeline/modeling/hyperparameter_tuning.py` — a single grid
search shared by all three families (`per_asset`, `panel_logit/pooled`,
`panel_logit/ticker_fixed_effects`). It is **nested inside the walk-forward
training window**: at every step the current training window is split
*chronologically* (most-recent `validation_fraction` = validation), candidates
are scored on the validation block, and the best params are re-fit on the full
window before predicting the test point. The test point / timestamp is never
used for tuning or fitting — no lookahead leakage. Per-asset splits by row
order; panel splits along **unique timestamps** so whole cross-sections stay
together. Ticker dummies for inner-train / validation / final fit are rebuilt
leakage-safely, exactly mirroring the untuned panel design matrix.

Objectives: `brier_score` (default, since probabilities feed thresholds and the
backtest), `log_loss`, `accuracy`. Search space and toggles live in
`configs/model_specs.yaml :: hyperparameter_tuning` (default `enabled: false`).
Grid search is deliberate — the space is tiny and a deterministic, exhaustive
sweep is reproducible (no RNG).

**Backward compatibility:** with tuning disabled the modelling behaviour is
byte-for-byte identical to the pinned `C=1.0` estimator and no `hpo_*` columns
are written. When enabled (`--tune-hyperparams`), signal parquets gain
`hpo_enabled, hpo_objective, best_C, best_class_weight, hpo_score, hpo_status`
and `metrics_summary.csv` gains `hpo_enabled, hpo_objective, best_C_median,
best_C_mode, best_class_weight_mode, hpo_status_counts`. Insufficient-data and
fit-failure paths fall back to the default hyperparameters with a recorded
`hpo_status`. New CLI flags on `run-models`: `--tune-hyperparams`,
`--hpo-objective`, `--hpo-config`, `--hpo-grid-C`, `--hpo-class-weight`.

## HPO variant separation (output naming + evaluation)

Hardening pass so tuned runs never overwrite or get pooled with fixed-C runs:

* **Filenames.** A tuned per-asset run writes
  ``<set>[_<sent>]_hpo_{brier,logloss,accuracy}.parquet``; a tuned panel run
  appends the same suffix after the mode tag, e.g.
  ``C3_cryptobert_panel_pooled_hpo_brier.parquet`` /
  ``..._panel_ticker_fe_hpo_brier.parquet``. Caching/restart and the dry-run
  ``output_name`` all key on the variant-specific path, so the fixed-C parquet
  is untouched. The rolling-probability benchmark (B1) has no hyperparameters
  and is skipped under ``--tune-hyperparams`` (never suffixed/overwritten).
* **Signal metadata.** Every signal file now carries ``hpo_enabled``,
  ``hpo_objective`` and ``hpo_variant`` (fixed-C runs get
  ``False`` / ``"-"`` / ``"fixed"``); tuned files additionally carry
  ``best_C, best_class_weight, hpo_score, hpo_status``.
* **Evaluation grouping.** ``GROUP_KEYS`` gained ``hpo_variant`` (after
  ``model_type``/``panel_mode``), so fixed and each HPO variant are never
  pooled. ``loading.py`` defaults old column-less files to
  ``fixed`` / ``per_asset`` / ``-``.
* **Benchmark families.** McNemar, threshold lift and economic benchmark lift
  match benchmarks within the same horizon + model_type + panel_mode +
  hpo_variant. No fixed↔HPO or per-asset↔panel mixing — even the
  ``allow_cross_model_benchmark`` opt-in never crosses the HPO variant.
  BUY_HOLD remains a separate global economic reference.
* **Conservative default.** ``model_specs.yaml`` search space now defaults
  ``class_weight: [null]`` (``balanced`` opt-in via config/CLI) since
  ``balanced`` can hurt the Brier-calibrated probabilities.

## Model-run checkpointing (resume after crash)

New module `src/thesis_pipeline/modeling/checkpointing.py` adds optional,
resume-able intermediate checkpoints under
`Outputs/Checkpoints/Models/{horizon}/{out_name}/` (gitignored). `out_name` is
the variant-specific signal name, so fixed / HPO / panel-pooled / panel-ticker-FE
runs never share a checkpoint directory.

* **Per-asset** — one parquet per ticker (`tickers/{ticker}.parquet`). The
  ticker loop in `run_models.main` (both the logistic and the rolling-probability
  benchmark paths) saves each finished ticker and, on resume, reloads it instead
  of recomputing.
* **Panel-logit** — one parquet per *timestamp chunk*
  (`chunks/chunk_NNNN.parquet`). `run_panel_walk_forward` gained a
  `checkpoint_context` argument; the ordered test timestamps are split into
  storage chunks of `--checkpoint-chunk-size`. A chunk is a storage partition
  only — every τ still trains on all rows with `timestamp < τ`, so predictions
  are byte-identical to a non-checkpointed run (regression-tested).
* **Atomic writes** — every checkpoint and `manifest.json` is written to a
  `*.tmp` sibling then `os.replace`-d in. Corrupt/unreadable checkpoints are
  logged and recomputed, never fatal.
* **Resume / restart / clear** — defaults: `checkpoint=true`, `resume=true`,
  `checkpoint_dir=Outputs/Checkpoints/Models`, `checkpoint_chunk_size=20`,
  `clear_checkpoints=false`. The final-signal-file cache is unchanged; with
  `--restart` the final file is ignored but checkpoints are reused (rebuilding
  the final file without recompute when complete). Checkpoints are deleted only
  with `--clear-checkpoints` (which removes just this run's directory).
* CLI/parser flags `--checkpoint/--no-checkpoint`, `--resume/--no-resume`,
  `--checkpoint-dir`, `--checkpoint-chunk-size`, `--clear-checkpoints` were added
  to `cli.run-models`, `run_models.build_parser`, `panel_logit.build_parser` and
  both `run(...)` wrappers. `metrics_summary.csv` gains `checkpoint_enabled`,
  `resumed_from_checkpoint`, `n_checkpoints_loaded`, `n_checkpoints_written`.

## Stationarity & descriptive statistics on the full final feature set

New shared helper `src/thesis_pipeline/features/final_feature_utils.py`:
loads `Data/Final/features_{horizon}.parquet`, resolves the modelling feature
universe from `feature_sets.xlsx` (with `{model}` expansion to vader / finbert
/ cryptobert), classifies features into `price / volatility / volume /
market_cap / sentiment / other`, finds matching `*_post_count` columns, and
detects structurally-empty sentiment rows (neutral value with zero posts).
Falls back to numeric columns when the registry is unavailable.

`src/thesis_pipeline/sentiment/stationarity.py` gained a `--source {final,
sentiment}` switch (default `final`). Under `--source final` the full
modelling feature universe is tested for all three horizons and the results
land as CSVs under `Data/Features/stationarity_final/`:
`stationarity_final_records.csv`, `..._summary.csv`,
`..._panel_cips.csv`, `..._fold_stability.csv`,
`..._feature_resolution.csv`. The legacy sentiment-only behaviour is reachable
via `--source sentiment`.

New `src/thesis_pipeline/diagnostics/descriptive_final_features.py` (CLI:
`descriptive-final-features`) writes six long-form CSVs under
`Outputs/deskriptiv/final_feature_sets/`: per (horizon × ticker × feature),
per (horizon × feature), per (horizon × ticker), per-horizon overview,
pairwise Pearson/Spearman correlations, and a registry-resolution log. The
helper module is shared with stationarity so the two stages can never report a
different feature universe.

`configs/paths.yaml` gains `descriptive_final_root` and
`stationarity_final_root`; the CLI gains `descriptive-final-features` and
threads `--source` (plus `--no-panel`) into the stationarity subcommand.

## 2026 family-structure refactor

`feature_sets.xlsx` has been regenerated to expose a clean four-family
hierarchy aligned with the thesis research questions. The underlying model
logic is unchanged — this is purely a relabelling + restructuring layer.

### Family layout (27 rows total)

* **Benchmarks (6)** – `B1` Historical Majority, `B2` Single Lag Return,
  `B3` Momentum, `B4` Momentum + Volatility, `B5` Momentum + Volume,
  `B6` Full Economic.
* **Pure sentiment, per scorer (9)** – `SV1/SV2/SV3` (vader),
  `SF1/SF2/SF3` (finbert), `SC1/SC2/SC3` (cryptobert) at three complexity
  levels (Title Mean → Title + Bullishness → Rich Sentiment).
* **Combined (9)** – `C1/C2/C3` × {vader, finbert, cryptobert}. The
  level→benchmark pairing follows the historical thesis design:
  `C1 = SVk/SFk/SCk(1) + B4`, `C2 = …(2) + B6`, `C3 = …(3) + B6`.
  Same composition as the legacy `C1/C2/C3` sets (results unchanged).
* **Multi-source (3)** – `M1/M2/M3` combine all three scorers at the
  matching sentiment level with the same matched benchmark.

### Migration table

The old `E*`, `S4`–`S7` and `C4`–`C6` IDs no longer exist. Requesting any of
them from `run-models` raises a controlled `SystemExit` pointing at the
replacement:

| Removed | Replacement |
|---------|-------------|
| `E1` | `B3` |
| `E2` | `B4` |
| `E3` | `B5` |
| `E4` | `B6` |
| `S1` | `SV1` / `SF1` / `SC1` |
| `S2` | `SV2` / `SF2` / `SC2` |
| `S3` | `SV3` / `SF3` / `SC3` |
| `S4` / `S5` | `M1` |
| `S6` | `M2` |
| `S7` | `M3` |
| `C4` / `C5` | `M1` |
| `C6` | `M3` |

No silent aliases — the guard lives in
`thesis_pipeline.modeling.run_models.main` and is keyed off
`feature_registry.REMOVED_SET_IDS`.

### Evaluation impact

* `evaluation.incremental.MATCHED_ECONOMIC_BENCHMARK` now maps
  `{C1: B4, C2: B6, C3: B6, M1: B4, M2: B6, M3: B6}`.
* `evaluation.significance.ELIGIBLE_CATEGORIES` accepts the new
  per-scorer categories (`sentiment_vader`, `sentiment_finbert`,
  `sentiment_cryptobert`, plus `multi`) in addition to the legacy
  `sentiment` / `combined` strings.

## FinBERT removal + S*/SV*/C*/CV*/M* hierarchy

### Why FinBERT was removed
FinBERT is trained on financial-analyst and broker-research language. On
Reddit / crypto sentiment data it produced near-constant predictions with
no meaningful between-ticker variation, so it added cost without
information for the panel-logit comparison. CryptoBERT is retained as the
domain-specific transformer baseline and VADER as the transparent lexicon
baseline.

* All FinBERT references were removed from `feature_sets.xlsx`, the feature
  registry, evaluation grouping (`ELIGIBLE_CATEGORIES`,
  `MATCHED_ECONOMIC_BENCHMARK`), diagnostics
  (`structural_breaks`, `descriptive_final_features`,
  `final_feature_utils`), sentiment aggregation (`aggregate.MODELS` /
  `DEFAULT_INPUTS`), `stationarity.MODELS`, `configs/pipeline.yaml`,
  `configs/paths.yaml`, and tests.
* The `score_finbert.py` source file is **kept on disk** as a frozen legacy
  module — the spec was explicit that legacy on-disk artefacts must not be
  deleted — but the CLI dispatch was removed.
* `--sentiment-model finbert` and `score-sentiment --model finbert` raise a
  clear `argparse` error that explains why FinBERT was dropped (no silent
  alias; tests assert the rejection path).

### Final feature-family naming

| Family | Meaning | Count |
|--------|---------|-------|
| `B*`   | Benchmark models | 6 (`B1`–`B6`) |
| `S*`   | CryptoBERT-only sentiment | 3 (`S1`–`S3`) |
| `SV*`  | VADER-only sentiment | 3 (`SV1`–`SV3`) |
| `C*`   | Benchmark + CryptoBERT sentiment | 6 (`C1`–`C6`) |
| `CV*`  | Benchmark + VADER sentiment | 6 (`CV1`–`CV6`) |
| `M*`   | Benchmark + CryptoBERT + VADER | 6 (`M1`–`M6`) |

**Total:** 30 feature sets (was 27 before this revision; was 29 before the
2026 family-structure refactor).

### Migration table (post-FinBERT-removal)

| Removed | Replacement |
|---------|-------------|
| `SC1` / `SC2` / `SC3` | `S1` / `S2` / `S3` |
| `SF1` / `SF2` / `SF3` | removed (FinBERT dropped) |
| `E1` / `E2` / `E3` / `E4` | `B3` / `B4` / `B5` / `B6` |
| `S4` / `S5` | `M1` |
| `S6` | `M2` |
| `S7` | `M3` |

Every legacy ID raises a controlled `SystemExit` from
`run_models.main` pointing at the replacement (or, for FinBERT, at the
removal rationale) — see `feature_registry.REMOVED_SET_IDS`.

### Hierarchical combined logic (Variant B)

Combined and multi-source sets pair benchmark complexity with sentiment
level explicitly:

```
C1  = B1 + S1              CV1 = B1 + SV1              M1  = B1 + S1 + SV1
C2  = B2 + S2              CV2 = B2 + SV2              M2  = B2 + S2 + SV2
C3  = B3 + S3              CV3 = B3 + SV3              M3  = B3 + S3 + SV3
C4  = B4 + S3              CV4 = B4 + SV3              M4  = B4 + S3 + SV3
C5  = B5 + S3              CV5 = B5 + SV3              M5  = B5 + S3 + SV3
C6  = B6 + S3              CV6 = B6 + SV3              M6  = B6 + S3 + SV3
```

`B1` carries the rolling-probability sentinel `__majority_class__`, which
is a routing marker rather than a real predictor — when `B1` is the
benchmark component of a combined / multi set, its contribution is the
empty list, so `C1 = S1` and `M1 = S1 ∪ SV1` in feature terms.

### Incremental sentiment-value mapping

`evaluation.incremental.MATCHED_ECONOMIC_BENCHMARK` was rebuilt to mirror
the new hierarchy: `C_k`, `CV_k`, `M_k` are compared against `B_k` (18
entries). Old `E*` IDs were never reintroduced and `M_k → E_*` mappings
were removed.

## Economic backtest: cross-group merge bug fix (June 2026)

### Symptom

`Outputs/Evaluation/economic_performance.csv` carried only one
`(set_id, sentiment_model)` row per horizon (plus `BUY_HOLD`), regardless of
how many signal groups were present under `Outputs/Signals/{horizon}/`.

### Root cause

`evaluation.economic._signals_with_forward_returns` deduplicated the signal
side of the merge on `("ticker", "__join_ts__")` alone, then ran the merge
with `validate="one_to_one"`. The signal frame validly contains many rows
per `(ticker, ts)` — one per evaluated group `(set_id, sentiment_model,
model_type, panel_mode, hpo_variant)`. The dedup therefore collapsed every
group but the last-loaded one (a `keep="last"` semantics that depended on
filename order). Pooled metrics survived because they never merged with
forward returns; the bug was localised to the economic layer.

### Fix

- Forward returns dedup stays on `("ticker", "__join_ts__")` (they are
  model-independent).
- Signals dedup now includes the group identity:
  `["ticker", "__join_ts__"] + GROUP_KEYS \ {"horizon"}`.
- Merge changed to `validate="many_to_one"`.
- The horizon-wide early return
  `if merged.empty or notna.sum() == 0: return empty`
  was relaxed: each group is now gated individually in
  `summarize_high_low_backtest`, `summarize_high_low_threshold_backtest`,
  and `summarize_backtest_by_ticker`, so one failing group never masks
  every other group in the same horizon.

### New: `economic_diagnostics.csv`

`evaluation.economic.economic_group_diagnostics(...)` records exactly one
row per attempted `(horizon, set_id, sentiment_model, model_type,
panel_mode, hpo_variant)` with a `status` of one of:

```
ok
skip_missing_required_columns
skip_no_signals
skip_no_forward_returns
skip_zero_forward_matches
skip_no_valid_probability
skip_no_portfolio_periods
```

and counters: `n_signal_rows`, `n_unique_timestamps`, `n_unique_tickers`,
`n_forward_return_rows`, `n_joined_rows`,
`n_joined_non_null_forward_returns`, `n_portfolio_periods`. Written to
`Outputs/Evaluation/economic_diagnostics.csv`.

A console summary section ("Economic backtest coverage") prints
attempted / ok / skipped counts per horizon and the top-five skip reasons.

### New CLI flag: `--strict-feature-set-ids`

`evaluate-signals` now accepts `--strict-feature-set-ids`. When set, only
signal rows whose `set_id` is in the active `feature_sets.xlsx` survive
into evaluation. Default is non-strict — stale legacy IDs are still
scored, and the diagnostics CSV flags them.

### Tests

`tests/test_economic.py` gains a regression suite covering
multi-set + multi-sentiment + multi-family signal frames, plus six
`economic_group_diagnostics` unit tests. `tests/test_evaluate_signals.py`
gains four integration tests covering the strict / non-strict modes and
the on-disk diagnostics CSV.

## Things explicitly **not** changed

- Target construction (`Create_Price_Features.py`).
- Sentiment scoring formulas (per scorer).
- Engagement weighting formula
  (`log1p(score) * upvote_ratio * (1 + log1p(num_comments))`).
- Title/selftext mix (0.7 / 0.3).
- Winsorization quantiles (0.005 / 0.995).
- Walk-forward scheme, `init_train_frac`, `C`, `random_state`.
- Output file naming inside `Outputs/Signals/`.
- Coverage threshold for ticker inclusion in `Merge_Features.py` (85%).
