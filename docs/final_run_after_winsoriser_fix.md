# Final production run — after the winsorisation leakage fix

This run-book is the exact sequence for the one final full model run after
full-sample winsorisation was removed from feature construction and replaced
by leakage-safe **training-window** winsorisation inside the model
(`thesis_pipeline.modeling.preprocessing`).

## Why everything must be regenerated

The preprocessing methodology changed, so **old features, signals and model
checkpoints are incompatible** and must not be reused:

* price / sentiment feature files previously contained full-sample-winsorised
  values; they must be regenerated as raw point-in-time features;
* model signal parquets and checkpoints were produced under the old
  preprocessing and now carry (or lack) a `preprocessing_signature` that no
  longer matches. Both the panel and per-asset paths reject a cached signal
  file or checkpoint whose `preprocessing_signature` differs and recompute.

## Command sequence

```bash
# 0. (once) audit the externally-supplied inputs — optional but recommended
python -m thesis_pipeline.cli audit-ccxt-timestamps \
    --1h ADAUSDT_1h.parquet --6h ADAUSDT_6h.parquet --1d ADAUSDT_1d.parquet
python -m thesis_pipeline.cli audit-price-sources \
    "price_sources(1).csv" "price_sources(2).csv" "price_sources(3).csv" \
    --out-dir Data/Raw/Price/validation

# 1. Regenerate RAW price features (no winsorisation; removes the stale
#    winsorization_thresholds.csv if present)
python -m thesis_pipeline.cli create-price-features

# 2. Regenerate RAW sentiment features (no winsorisation)
python -m thesis_pipeline.cli create-sentiment-features

# 3. Merge price + sentiment features (interval-start equality join)
python -m thesis_pipeline.cli merge-features

# 4. Clear incompatible model checkpoints (belt-and-braces; the run also
#    auto-invalidates on preprocessing_signature mismatch)
rm -rf Outputs/Checkpoints/Models

# 5. Run the final canonical model for ALL three horizons
#    (pooled panel logit, ticker fixed effects, rolling 180-calendar-day
#     window, nested HPO, log-loss objective, all 17 feature sets,
#     VADER + CryptoBERT). --restart forces a clean recompute of signals.
for HZ in 1h 6h 1d; do
  python -m thesis_pipeline.cli run-models \
      --horizon "$HZ" \
      --model-type panel_logit \
      --panel-mode ticker_fixed_effects \
      --train-window rolling_fixed --rolling-window-days 180 \
      --tune-hyperparams --hpo-objective log_loss \
      --clear-checkpoints --restart
done

# 6. Evaluate the signals (produces Results_260705-style outputs)
for HZ in 1h 6h 1d; do
  python -m thesis_pipeline.cli evaluate-signals --horizon "$HZ"
done
```

Notes:

* Step 5 uses `--clear-checkpoints` and `--restart` so no pre-winsoriser
  checkpoint or signal parquet can leak into the corrected run. Even without
  them, a mismatched `preprocessing_signature` triggers an automatic recompute.
* The canonical CLI defaults already encode panel-logit / ticker-FE /
  rolling-180d / HPO-on / log-loss; the flags above are stated explicitly for
  the record.
* Nothing here downloads data. Coins that begin at different dates are handled
  by the existing coverage logic and are **not** re-downloaded.
