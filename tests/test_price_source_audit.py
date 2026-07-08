"""Price-source consistency detection (task Section 5, criterion 10)."""
from __future__ import annotations

import pandas as pd
import pytest

from thesis_pipeline.diagnostics.price_source_audit import (
    build_source_consistency, load_price_sources, run_price_source_audit,
    summarize_sources,
)

HORIZONS = ("1h", "6h", "1d")


def _write_sources(tmp_path, rows, name="price_sources.csv"):
    p = tmp_path / name
    pd.DataFrame(rows).to_csv(p, index=False)
    return p


def _consistent_rows(coin, exchange="Bybit"):
    return [{"coin": coin, "horizon": hz, "exchange": exchange,
             "symbol": f"{coin}/USDT", "quote": "USDT", "status": "complete",
             "missing_bars": 0, "first_date": "2022-01-01",
             "last_date": "2023-01-01", "coverage_pct": 100.0}
            for hz in HORIZONS]


def test_consistent_sources_pass(tmp_path):
    rows = _consistent_rows("BTC") + _consistent_rows("ETH", "KuCoin")
    p = _write_sources(tmp_path, rows)
    consistency, summary = run_price_source_audit(
        [p], out_dir=tmp_path / "validation", strict=True)
    assert summary["n_coins"] == 2
    assert summary["n_inconsistent_coins"] == 0
    assert summary["all_usdt"] is True
    assert summary["exchange_allocation"] == {"Bybit": 1, "KuCoin": 1}
    assert (tmp_path / "validation" / "price_source_consistency.csv").exists()
    assert consistency["same_source_all_horizons"].all()


def test_cross_horizon_exchange_mismatch_detected(tmp_path):
    rows = _consistent_rows("BTC")
    bad = _consistent_rows("LRC")
    bad[0]["exchange"] = "Binance"  # 1h differs from 6h/1d (Bybit)
    p = _write_sources(tmp_path, rows + bad)
    consistency, summary = run_price_source_audit(
        [p], out_dir=tmp_path / "validation", strict=False)
    assert summary["n_inconsistent_coins"] == 1
    assert "LRC" in summary["inconsistent_coins"]
    lrc = consistency[consistency["coin"] == "LRC"].iloc[0]
    assert not lrc["same_source_all_horizons"]
    assert "exchange differs" in lrc["source_inconsistency"]


def test_strict_mode_raises_on_inconsistency(tmp_path):
    bad = _consistent_rows("LRC")
    bad[1]["symbol"] = "LRC/USD"   # 6h symbol differs
    p = _write_sources(tmp_path, bad)
    with pytest.raises(ValueError, match="inconsistent sources"):
        run_price_source_audit([p], out_dir=tmp_path / "validation", strict=True)


def test_quote_mismatch_detected(tmp_path):
    bad = _consistent_rows("XRP")
    bad[2]["quote"] = "USDC"       # 1d quote differs
    p = _write_sources(tmp_path, bad)
    consistency = build_source_consistency(load_price_sources([p]))
    row = consistency.iloc[0]
    assert not row["quote_consistent"]
    assert not row["same_source_all_horizons"]


def test_multiple_files_are_joined(tmp_path):
    p1 = _write_sources(tmp_path, _consistent_rows("BTC"), "price_sources(1).csv")
    p2 = _write_sources(tmp_path, _consistent_rows("ETH"), "price_sources(2).csv")
    df = load_price_sources([p1, p2])
    summ = summarize_sources(df, build_source_consistency(df))
    assert summ["n_coins"] == 2
    assert summ["total_internal_missing_bars"] == 0
