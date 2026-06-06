"""Smoke tests: every YAML config loads and exposes the documented keys."""
from __future__ import annotations

import pytest

from thesis_pipeline.config import (
    all_configs, load_config, resolve_path, horizons, tickers,
)


def test_all_configs_load():
    cfgs = all_configs()
    assert set(cfgs) == {"paths", "horizons", "coins", "feature_sets",
                         "model_specs", "pipeline"}
    for name, cfg in cfgs.items():
        assert isinstance(cfg, dict), f"{name} should be a dict"
        assert cfg, f"{name} should be non-empty"


@pytest.mark.parametrize("key", [
    "raw_price_root", "raw_price_15min", "raw_price_1h", "raw_price_4h",
    "raw_price_6h", "raw_price_1d",
    "cmc_market_cap", "cmc_price", "cmc_volume",
    "raw_sentiment_root", "subreddit_ticker_mapping",
    "sentiment_combined",
    "sentiment_scored_vader", "sentiment_scored_cryptobert",
    "features_root", "final_root",
    "signals_root", "diagnostics_root", "smoke_root",
])
def test_required_path_keys_present(key):
    p = resolve_path(key)
    # Path objects should be absolute under the repo root
    assert p.is_absolute()


def test_path_templates_format():
    p1 = resolve_path("price_features_pattern", horizon="1d")
    assert p1.name == "price_features_1d.parquet"
    p2 = resolve_path("sentiment_features_pattern", horizon="6h")
    assert p2.name == "sentiment_features_6h.parquet"
    p3 = resolve_path("final_features_pattern", horizon="1h")
    assert p3.name == "features_1h.parquet"
    p4 = resolve_path("signals_pattern", horizon="1d", set_id="B1")
    assert p4.name == "B1.parquet"
    assert p4.parent.name == "1d"


def test_horizons_contract():
    assert horizons("raw_price") == ["15min", "1h", "4h", "6h", "1d"]
    assert horizons("price_feature") == ["1h", "6h", "1d"]
    assert horizons("sentiment_feature") == ["15min", "1h", "6h", "1d"]
    assert horizons("final_model") == ["1h", "6h", "1d"]
    assert load_config("horizons")["default_horizon"] == "1d"


def test_tickers_contract():
    ts = tickers()
    assert "BTC" in ts and "ETH" in ts
    # 25 tickers per the thesis observed set
    assert len(ts) == 25, f"expected 25 tickers, got {len(ts)}: {ts}"


def test_model_specs():
    cfg = load_config("model_specs")
    assert cfg["walk_forward"]["init_train_frac"] == 0.50
    assert cfg["logistic_ridge"]["C"] == 1.0
    assert cfg["logistic_ridge"]["random_state"] == 42
    assert cfg["threshold"] == 0.5
    assert "row_id" not in cfg["output_columns"]  # see io.add_row_id


def test_pipeline_stages_cover_known_stages():
    cfg = load_config("pipeline")
    expected = {
        "validate_price", "create_price_features", "load_sentiment",
        "score_vader", "score_cryptobert",
        "create_sentiment_features", "stationarity",
        "merge_features", "run_models", "diagnostics",
    }
    assert expected.issubset(set(cfg["stages"]))
    assert "default_order" in cfg
