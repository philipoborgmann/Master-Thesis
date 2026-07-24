#!/usr/bin/env bash
# Reproduce the thesis model + evaluation results starting from the merged
# feature matrices in Data/Final/features_{1h,6h,1d}.parquet (reproduction
# path 2 in README section G). Does NOT re-run CryptoBERT or rebuild features.
#
# Usage:  scripts/reproduce_from_final_features.sh [N_JOBS]
#   N_JOBS  worker processes for panel-logit checkpoint chunks (default 4).
#           Affects speed only, not results.
#
# Runs only the public CLI. Uses checkpoint/resume so it is safe to re-run;
# it never deletes checkpoints or outputs.
set -euo pipefail

N_JOBS="${1:-4}"
CLI=(python -m thesis_pipeline.cli)

for H in 1d 6h 1h; do
    echo ">>> run-models --horizon ${H} (n_jobs=${N_JOBS})"
    "${CLI[@]}" run-models \
        --horizon "${H}" \
        --model-type panel_logit \
        --panel-mode ticker_fixed_effects \
        --train-window rolling_fixed \
        --rolling-window-days 180 \
        --tune-hyperparams \
        --hpo-objective log_loss \
        --checkpoint --resume \
        --checkpoint-chunk-size 30 \
        --n-jobs "${N_JOBS}"
done

# Evaluation runs EXACTLY ONCE across all horizons (no --horizon): every horizon
# writes into the same Outputs/Evaluation/ directory and the family-aware BH
# correction pools p-values across horizons. --strict-feature-set-ids keeps only
# the registered 17-set grid (+ NAIVE); the completeness guard aborts on a
# partial run.
echo ">>> evaluate-signals (once, across all horizons, strict + completeness guard)"
"${CLI[@]}" evaluate-signals --strict-feature-set-ids

echo ">>> descriptive-final-features (all horizons)"
"${CLI[@]}" descriptive-final-features

echo ">>> done. Signals in Outputs/Signals/, evaluation in Outputs/Evaluation/."
