<#
.SYNOPSIS
Reproduce the thesis model + evaluation results from the merged feature
matrices in Data/Final/features_{1h,6h,1d}.parquet (reproduction path 2 in
README section G). Does NOT re-run CryptoBERT or rebuild features.

.PARAMETER NJobs
Worker processes for panel-logit checkpoint chunks (default 4). Affects speed
only, not results.

.EXAMPLE
scripts\reproduce_from_final_features.ps1 -NJobs 4

.NOTES
Runs only the public CLI. Uses checkpoint/resume so it is safe to re-run; it
never deletes checkpoints or outputs.
#>
[CmdletBinding()]
param([int]$NJobs = 4)

$ErrorActionPreference = "Stop"

foreach ($H in @("1d", "6h", "1h")) {
    Write-Host ">>> run-models --horizon $H (n_jobs=$NJobs)"
    python -m thesis_pipeline.cli run-models `
        --horizon $H `
        --model-type panel_logit `
        --panel-mode ticker_fixed_effects `
        --train-window rolling_fixed `
        --rolling-window-days 180 `
        --tune-hyperparams `
        --hpo-objective log_loss `
        --checkpoint --resume `
        --checkpoint-chunk-size 30 `
        --n-jobs $NJobs
    if ($LASTEXITCODE -ne 0) { throw "run-models --horizon $H failed ($LASTEXITCODE)" }
}

# Evaluation runs EXACTLY ONCE across all horizons (no --horizon): every horizon
# writes into the same Outputs/Evaluation/ directory and the family-aware BH
# correction pools p-values across horizons. --strict-feature-set-ids keeps only
# the registered 17-set grid (+ NAIVE); the completeness guard aborts on a
# partial run.
Write-Host ">>> evaluate-signals (once, across all horizons, strict + completeness guard)"
python -m thesis_pipeline.cli evaluate-signals --strict-feature-set-ids
if ($LASTEXITCODE -ne 0) { throw "evaluate-signals failed ($LASTEXITCODE)" }

Write-Host ">>> descriptive-final-features (all horizons)"
python -m thesis_pipeline.cli descriptive-final-features
if ($LASTEXITCODE -ne 0) { throw "descriptive-final-features failed ($LASTEXITCODE)" }

Write-Host ">>> done. Signals in Outputs/Signals/, evaluation in Outputs/Evaluation/."
