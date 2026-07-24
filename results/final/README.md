# `results/final/` — compact final outputs for examiners

This directory is the intended home for a **small, tracked** set of final
result tables and a run manifest, so examiners can inspect the headline numbers
without rerunning the pipeline or downloading large data.

**It is currently empty of results by design.** No result files are committed
yet: they must come from a single, complete final rerun. At the time of writing,
at least one signal group was incomplete (`SENT_VAD_LD` at `1h` shipped 81,500
rows instead of the ~103,200 its matched `ECON` benchmark produced), so
committing current outputs would publish a partial comparison. Do **not** add
fabricated or partial outputs here.

## What to place here after a clean final rerun

Only compact CSV/JSON summaries — never large signal parquets or checkpoints:

- `pooled_metrics.csv`
- `incremental_sentiment_value.csv`
- `diff_in_improvement.csv`
- `horizon_comparison.csv`
- `multiple_testing_manifest.csv`
- `signal_completeness.csv`  (must show every registered production group `complete`)
- `run_manifest.json`

These are produced by `evaluate-signals --strict-feature-set-ids` (run once,
across all horizons) under `Outputs/Evaluation/`; copy the compact ones here.

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

## Precondition

`Outputs/Evaluation/signal_completeness.csv` must report **every** registered
production group as `complete` before anything is copied here — the
`evaluate-signals` completeness guard enforces this (it aborts otherwise).
