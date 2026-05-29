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
