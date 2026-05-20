"""Common-sample diagnostics should produce identical row sets across compared sets."""
from __future__ import annotations

import pandas as pd

from thesis_pipeline.features.checks import common_sample_row_ids
from thesis_pipeline.io import add_row_id


def _df(ticker_ts):
    rows = []
    for t, ts in ticker_ts:
        rows.append({"ticker": t, "timestamp": pd.Timestamp(ts), "value": 1.0})
    df = pd.DataFrame(rows)
    return add_row_id(df, horizon="1d")


def test_row_id_format():
    df = _df([("BTC", "2024-01-01"), ("ETH", "2024-01-02")])
    assert df["row_id"].iloc[0].startswith("BTC_1d_")
    assert df["row_id"].iloc[1].startswith("ETH_1d_")
    # row_id is unique within the frame
    assert df["row_id"].is_unique


def test_common_sample_intersection():
    a = _df([("BTC", "2024-01-01"), ("BTC", "2024-01-02"), ("ETH", "2024-01-01")])
    b = _df([("BTC", "2024-01-02"), ("ETH", "2024-01-01"), ("ETH", "2024-01-02")])
    common = common_sample_row_ids([a, b])
    assert common == set(a["row_id"]) & set(b["row_id"])
    # BTC_2024-01-01 only in a; ETH_2024-01-02 only in b → both excluded
    assert len(common) == 2


def test_common_sample_empty_when_no_overlap():
    a = _df([("BTC", "2024-01-01")])
    b = _df([("ETH", "2024-01-02")])
    assert common_sample_row_ids([a, b]) == set()
