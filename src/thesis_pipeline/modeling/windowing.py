"""Manual train-window selection for the panel-logit walk-forward.

This helper lets the caller pick **how much history** is used to train the
model at every test timestamp ``τ`` — never **which window the test point
itself lives in** (we always strictly satisfy ``timestamp < τ``).

Two modes are supported:

* ``expanding`` — train rows where ``timestamp < τ`` (the historical default
  for the panel-logit model). Returned unchanged.
* ``rolling_fixed`` — train rows whose ``timestamp`` is one of the **last N
  unique timestamps before τ**. The window length is taken from
  ``rolling_window_timestamps`` (a count of unique training timestamps) **or**
  ``rolling_window_days`` (a positive day offset measured from ``τ``).

The rolling mode has **no default size by design**. Structural-break
diagnostics produced elsewhere in the pipeline are advisory only — the
modelling rolling window is always set explicitly by the user via the CLI.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

VALID_MODES = ("expanding", "rolling_fixed")


def select_panel_train_window(
    df: pd.DataFrame,
    tau,
    *,
    train_window_mode: str = "expanding",
    rolling_window_timestamps: int | None = None,
    rolling_window_days: float | None = None,
    time_col: str = "timestamp",
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Return ``(train_df, test_df, metadata)`` for panel test timestamp ``τ``.

    Parameters
    ----------
    df
        Long-form panel sorted by ``time_col`` (sorting is enforced internally
        for safety).
    tau
        The test timestamp. ``test_df`` is the rows with ``df[time_col] == τ``
        and is identical regardless of ``train_window_mode``.
    train_window_mode
        ``"expanding"`` (default) or ``"rolling_fixed"``.
    rolling_window_timestamps
        Number of unique pre-``τ`` timestamps to keep in training. Required for
        ``rolling_fixed`` unless ``rolling_window_days`` is supplied.
    rolling_window_days
        Day-distance window: keep training timestamps with
        ``ts >= τ - rolling_window_days``. Alternative to
        ``rolling_window_timestamps``; pass only one of the two.

    Returns
    -------
    train_df, test_df
        Slices of ``df``; the train slice never contains rows with
        ``time_col >= τ``.
    metadata
        ``{train_window_mode, rolling_window_timestamps, rolling_window_days,
        train_window_timestamps, train_start_timestamp, train_end_timestamp}``.
    """
    if train_window_mode not in VALID_MODES:
        raise ValueError(
            f"Unknown train_window_mode {train_window_mode!r}; "
            f"expected one of {VALID_MODES}."
        )

    test_df = df[df[time_col] == tau]
    pre_tau = df[df[time_col] < tau]

    if train_window_mode == "expanding":
        train_df = pre_tau
    else:
        if rolling_window_timestamps is None and rolling_window_days is None:
            raise ValueError(
                "train_window_mode='rolling_fixed' requires "
                "--rolling-window-timestamps or --rolling-window-days "
                "(there is no automatic default — structural breaks are "
                "diagnostic only)."
            )
        if rolling_window_timestamps is not None and rolling_window_days is not None:
            raise ValueError(
                "Specify exactly one of --rolling-window-timestamps or "
                "--rolling-window-days, not both."
            )

        unique_ts = np.sort(pre_tau[time_col].unique())
        if len(unique_ts) == 0:
            train_df = pre_tau
        elif rolling_window_timestamps is not None:
            n = int(rolling_window_timestamps)
            if n <= 0:
                raise ValueError("rolling_window_timestamps must be positive.")
            keep_ts = unique_ts[-n:]
            train_df = pre_tau[pre_tau[time_col].isin(set(keep_ts))]
        else:
            days = float(rolling_window_days)
            if days <= 0:
                raise ValueError("rolling_window_days must be positive.")
            cutoff = pd.Timestamp(tau) - pd.Timedelta(days=days)
            train_df = pre_tau[pre_tau[time_col] >= cutoff]

    train_ts = np.sort(train_df[time_col].unique())
    metadata = {
        "train_window_mode":          train_window_mode,
        "rolling_window_timestamps":  (None if rolling_window_timestamps is None
                                       else int(rolling_window_timestamps)),
        "rolling_window_days":        (None if rolling_window_days is None
                                       else float(rolling_window_days)),
        "train_window_timestamps":    int(len(train_ts)),
        "train_start_timestamp":      (None if len(train_ts) == 0
                                       else pd.Timestamp(train_ts[0])),
        "train_end_timestamp":        (None if len(train_ts) == 0
                                       else pd.Timestamp(train_ts[-1])),
    }
    return train_df, test_df, metadata


def window_suffix(train_window_mode: str,
                  rolling_window_timestamps: int | None,
                  rolling_window_days: float | None) -> str:
    """Compact filename / checkpoint suffix for the chosen training window.

    * ``expanding`` → ``""`` (preserves historical filenames).
    * ``rolling_fixed`` with N timestamps → ``"_rw{N}"``.
    * ``rolling_fixed`` with D days → ``"_rw{D}d"`` (e.g. ``_rw30.0d``).
    """
    if train_window_mode != "rolling_fixed":
        return ""
    if rolling_window_timestamps is not None:
        return f"_rw{int(rolling_window_timestamps)}"
    if rolling_window_days is not None:
        d = float(rolling_window_days)
        # 30.0 → "30", 30.5 → "30.5" — keep filenames readable.
        return f"_rw{int(d) if d == int(d) else d}d"
    return ""
