"""Missingness reports."""

from __future__ import annotations

import pandas as pd


def missingness_table(df: pd.DataFrame) -> pd.DataFrame:
    n = len(df)
    out = pd.DataFrame({
        "column": df.columns,
        "n_missing": [df[c].isna().sum() for c in df.columns],
    })
    out["share_missing"] = out["n_missing"] / max(n, 1)
    return out.sort_values("share_missing", ascending=False).reset_index(drop=True)
