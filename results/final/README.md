# `results/final/` — compact final outputs for examiners

This directory holds a **small, tracked** set of final result tables plus a run
manifest, so examiners can inspect the headline numbers without rerunning the
pipeline or downloading large data.

## Status

The final run is **complete**: all 54 signal groups (18 model identities × 3
horizons) are present and full — 103,200 rows at 1h, 17,200 at 6h, 4,300 at 1d.
The earlier `SENT_VAD_LD` 1h shortfall (81,500 rows) was **resolved** in the
final rerun; the completeness guard (`evaluate-signals`) confirms every
registered production group is complete.

The full evaluation artefacts (21 CSVs + `signal_evaluation.xlsx`) are delivered
in the **separately-submitted Outputs package**, since they are too large / not
appropriate to commit. The compact subset below may be copied here for
convenience — `.gitignore` re-includes `results/final/*.csv` and `*.json`
despite the global data-file ignore.

## Compact subset to place here

Produced by `evaluate-signals --strict-feature-set-ids` (run once, across all
horizons) under `Outputs/Evaluation/`; copy these small files here:

- `pooled_metrics.csv`
- `incremental_sentiment_value.csv`
- `diff_in_improvement.csv`
- `horizon_comparison.csv`
- `multiple_testing_manifest.csv`
- `signal_completeness.csv`  (every registered production group = `complete`)
- `run_manifest.json`

## Regenerating the compact outputs

From the final signal set (reproduction path 2 in the top-level README), run
the strict evaluation once and copy the compact CSVs here:

```bash
python -m thesis_pipeline.cli evaluate-signals --strict-feature-set-ids
# then copy the small tables named above from Outputs/Evaluation/ into results/final/
```

`--strict-feature-set-ids` restricts to the registered 17-set grid (+ the NAIVE
reference) and drops any stale IDs; on the final signal set it keeps every row
and discards none, so the tables are identical to a non-strict run — it simply
makes the strict, audited path the one on record and writes
`signal_completeness.csv`.

## `run_manifest.json` — intended fields

Generate from the **actual** final environment (never a polluted / unrelated
one). Fields:

- `git_commit_sha`
- `python_version`
- `package_versions` (pandas, numpy, scikit-learn, scipy, statsmodels, arch,
  ruptures, joblib, threadpoolctl; and torch/transformers if scoring)
- `ccxt_version` (if the raw OHLCV was re-acquired; else record "unknown /
  supplied")
- `cryptobert_model` = `ElKulako/cryptobert`
- `cryptobert_revision` (if recovered; else null — the exact 2026-04-15 snapshot
  could not be reconstructed)
- `data_file_hashes`, `feature_file_hashes`
- `horizons` = [1h, 6h, 1d]
- `model_type` = panel_logit, `panel_mode` = ticker_fixed_effects
- `rolling_window_days` = 180, `hpo_objective` = log_loss
- `feature_set_registry_hash`
- `forecast_sample` (start / end_exclusive from `configs/model_specs.yaml`)
- `generation_timestamp`
