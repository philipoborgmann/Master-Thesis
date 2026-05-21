"""Signal evaluation stage.

Computes thesis-ready evaluation tables, statistical significance tests,
volatility-regime analysis, threshold/conviction analysis and ticker-level
diagnostics for the walk-forward signals produced by
``thesis_pipeline.modeling.run_models``.

The implementation is intentionally split into small, separately-testable
helpers:

* :mod:`.loading`        — robust signal discovery and parquet loading
* :mod:`.metrics`        — pooled + per-ticker metrics + confusion diagnostics
* :mod:`.thresholds`     — high-conviction threshold analysis
* :mod:`.volatility`     — Garman-Klass daily variance + regime stratification
* :mod:`.significance`   — continuity-corrected McNemar vs benchmark B1
* :mod:`.reporting`      — Excel writer + CSV writer + console summary

The top-level coordinator is :func:`evaluate_signals.main`.
"""
