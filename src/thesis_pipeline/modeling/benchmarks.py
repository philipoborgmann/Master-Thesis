"""Benchmark model placeholders.

The production rolling-probability benchmark lives in Run_Models.py under
``run_rolling_probability``. This module exists so callers in the new
package can document the contract without changing behavior.
"""

from __future__ import annotations


def describe_rolling_probability() -> str:
    return (
        "Rolling probability benchmark: at step t, predict 1 if "
        "mean(y[:t]) >= 0.5 else 0; predicted probability = mean(y[:t]). "
        "No features, expanding window — same as Run_Models.run_rolling_probability."
    )
