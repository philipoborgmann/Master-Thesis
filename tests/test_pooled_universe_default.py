"""Tests for the v4 pooled-panel universe defaults (commit 11).

The legacy pipeline applied a hard ticker-level sentiment-coverage
filter at merge time and shrank the 1h universe to BTC / DOGE / ETH.
The v4 main specification — pooled panel logit with ticker fixed
effects and rolling-window OOS estimation — does not need that hard
filter; missing sentiment slots are handled via Variante-A neutral
fill + ``has_posts`` / ``log1p_post_count`` diagnostics. The default
is now:

* ``apply_coverage_filter = False`` (pooled main spec)
* ``neutral_fill_sentiment = True``

The filter remains available as an explicit robustness option via
``--apply-coverage-filter``.

These tests pin:

* the default keeps every low-coverage ticker that has valid price data;
* the explicit robustness flag re-enables the legacy filter;
* IOTA / MIOTA + NANO / XNO aliases collapse to one canonical ticker;
* the merge report exposes requested / available / realized counts;
* a hardcoded BTC/DOGE/ETH allowlist does NOT exist anywhere on the
  production path.
"""
from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import thesis_pipeline.features.merge as fm


# ---------------------------------------------------------------------------
# Synthetic 1h repo with low-coverage tickers
# ---------------------------------------------------------------------------

def _v4_price_row(ts, ticker):
    return {
        "timestamp": ts, "ticker": ticker, "horizon": "1h",
        "target": 1,
        "log_return_t":         0.0,
        "cum_log_return_7d":    0.0,
        "cum_log_return_14d":   0.0,
        "cum_log_return_21d":   0.0,
        "realized_vol_14d":     0.02,
        "volume_diff":          0.0,
        "log_market_cap_lag1":  20.0,
        "market_cap_available_at": ts - pd.Timedelta(days=1),
    }


@pytest.fixture
def hourly_repo(tmp_path, monkeypatch):
    feature_dir = tmp_path / "Data" / "Features"
    final_dir   = tmp_path / "Data" / "Final"
    feature_dir.mkdir(parents=True); final_dir.mkdir()

    n = 96  # 4 days of hourly bars
    ts = pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC")
    # Six tickers in the price archive — only three have any sentiment.
    price_tickers   = ("BTC", "DOGE", "ETH", "SOL", "ADA", "XRP")
    sent_tickers    = ("BTC", "DOGE", "ETH")            # high coverage only
    coverage_pct    = {"BTC": 100.0, "DOGE": 100.0, "ETH": 100.0,
                       "SOL": 0.0,   "ADA": 0.0,    "XRP": 0.0}

    pd.concat([
        pd.DataFrame([_v4_price_row(t, tk) for t in ts])
        for tk in price_tickers
    ], ignore_index=True).to_parquet(
        feature_dir / "price_features_1h.parquet", index=False,
    )

    sent_rows = []
    for tk in sent_tickers:
        sent_rows.append(pd.DataFrame({
            "timestamp": ts, "ticker": tk,
            "vader_title_score_mean":      np.linspace(-0.5, 0.5, n),
            "vader_bullishness_ratio":     np.linspace(0.4, 0.6, n),
            "vader_title_score_std":       np.linspace(0.1, 0.2, n),
            "post_count":                  np.arange(n, dtype=int),
        }))
    pd.concat(sent_rows, ignore_index=True).to_parquet(
        feature_dir / "sentiment_features_1h.parquet", index=False,
    )

    pd.DataFrame({
        "ticker":       list(price_tickers),
        "horizon":      ["1h"] * len(price_tickers),
        "coverage_pct": [coverage_pct[t] for t in price_tickers],
    }).to_csv(feature_dir / "sentiment_coverage.csv", index=False)

    monkeypatch.chdir(tmp_path)
    return tmp_path


# ---------------------------------------------------------------------------
# Default behaviour: filter OFF
# ---------------------------------------------------------------------------

def test_default_keeps_full_price_universe_for_pooled_panel(hourly_repo):
    rc = fm.main(["--horizon", "1h"])
    assert rc == 0
    merged = pd.read_parquet(hourly_repo / "Data" / "Final" / "features_1h.parquet")
    realised = set(merged["ticker"].dropna().unique())
    # The 1h universe is the FULL price universe — not just the three
    # tickers above the 85% coverage threshold.
    assert realised == {"BTC", "DOGE", "ETH", "SOL", "ADA", "XRP"}, realised


def test_low_coverage_tickers_get_neutral_filled_sentiment(hourly_repo):
    fm.main(["--horizon", "1h"])
    merged = pd.read_parquet(hourly_repo / "Data" / "Final" / "features_1h.parquet")
    low = merged[merged["ticker"] == "SOL"]
    assert (low["vader_title_score_mean"] == 0.0).all()
    assert (low["vader_bullishness_ratio"] == 0.5).all()
    assert (low["post_count"] == 0).all()
    # Diagnostics survive: has_posts / log1p_post_count are zero for
    # no-post slots, NOT NaN.
    assert (low["has_posts"] == 0).all()
    assert (low["log1p_post_count"] == 0.0).all()


def test_merge_report_records_universe_selection(hourly_repo):
    fm.main(["--horizon", "1h"])
    rep = pd.read_csv(hourly_repo / "Data" / "Final" / "merge_report.csv")
    r = rep.iloc[0]
    assert r["apply_coverage_filter"] is False or r["apply_coverage_filter"] == False  # noqa: E712
    assert int(r["n_price_tickers"]) == 6
    assert int(r["n_sentiment_tickers"]) == 3
    assert int(r["n_coverage_qualified_tickers"]) == 3
    assert int(r["n_realized_tickers"]) == 6


# ---------------------------------------------------------------------------
# Explicit robustness flag
# ---------------------------------------------------------------------------

def test_explicit_apply_coverage_filter_restricts_to_high_coverage(hourly_repo):
    rc = fm.main(["--horizon", "1h", "--apply-coverage-filter"])
    assert rc == 0
    merged = pd.read_parquet(hourly_repo / "Data" / "Final" / "features_1h.parquet")
    realised = set(merged["ticker"].dropna().unique())
    assert realised == {"BTC", "DOGE", "ETH"}, realised
    rep = pd.read_csv(hourly_repo / "Data" / "Final" / "merge_report.csv")
    excl = str(rep.iloc[0]["tickers_excluded_by_coverage"])
    for tk in ("SOL", "ADA", "XRP"):
        assert tk in excl


def test_legacy_no_sentiment_coverage_filter_flag_still_disables(hourly_repo):
    rc = fm.main(["--horizon", "1h", "--no-sentiment-coverage-filter"])
    assert rc == 0
    merged = pd.read_parquet(hourly_repo / "Data" / "Final" / "features_1h.parquet")
    assert set(merged["ticker"].unique()) == {
        "BTC", "DOGE", "ETH", "SOL", "ADA", "XRP",
    }


# ---------------------------------------------------------------------------
# Alias normalisation: IOTA / MIOTA and NANO / XNO
# ---------------------------------------------------------------------------

def test_iota_miota_alias_collapses_to_one_ticker(tmp_path, monkeypatch):
    feature_dir = tmp_path / "Data" / "Features"
    final_dir   = tmp_path / "Data" / "Final"
    feature_dir.mkdir(parents=True); final_dir.mkdir()

    n = 96
    ts = pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC")
    # Price archive uses MIOTA, sentiment uses IOTA. After alias
    # normalisation the merge must see ONE canonical ticker.
    price = pd.DataFrame([_v4_price_row(t, "MIOTA") for t in ts])
    price.to_parquet(feature_dir / "price_features_1h.parquet", index=False)
    pd.DataFrame({
        "timestamp": ts, "ticker": "IOTA",
        "vader_title_score_mean":  np.linspace(-0.5, 0.5, n),
        "vader_bullishness_ratio": np.linspace(0.4, 0.6, n),
        "vader_title_score_std":   np.linspace(0.1, 0.2, n),
        "post_count":              np.arange(n, dtype=int),
    }).to_parquet(feature_dir / "sentiment_features_1h.parquet", index=False)
    pd.DataFrame({
        "ticker": ["MIOTA"], "horizon": ["1h"], "coverage_pct": [100.0],
    }).to_csv(feature_dir / "sentiment_coverage.csv", index=False)
    monkeypatch.chdir(tmp_path)

    rc = fm.main(["--horizon", "1h"])
    assert rc == 0
    merged = pd.read_parquet(tmp_path / "Data" / "Final" / "features_1h.parquet")
    realised = set(merged["ticker"].dropna().unique())
    assert realised == {"IOTA"}, realised
    # The IOTA rows actually picked up the sentiment from the IOTA
    # source — not NaN.
    assert merged["vader_title_score_mean"].notna().all()


# ---------------------------------------------------------------------------
# No hardcoded BTC/DOGE/ETH allowlist in production code
# ---------------------------------------------------------------------------

def test_no_hardcoded_three_ticker_allowlist_in_production_code():
    """A static scan over the production modules must NOT contain a
    hardcoded BTC/DOGE/ETH list that would shrink the pooled universe."""
    root = Path(__file__).resolve().parents[1] / "src" / "thesis_pipeline"
    bad = []
    for p in root.glob("**/*.py"):
        text = p.read_text(encoding="utf-8")
        # Match the exact 3-ticker token sequence in any order.
        token = text.replace(" ", "").replace("\n", "")
        for combo in (
            '["BTC","DOGE","ETH"]', '("BTC","DOGE","ETH")',
            '["BTC","ETH","DOGE"]', '("BTC","ETH","DOGE")',
            '["DOGE","BTC","ETH"]', '("DOGE","BTC","ETH")',
            '["DOGE","ETH","BTC"]', '("DOGE","ETH","BTC")',
            '["ETH","BTC","DOGE"]', '("ETH","BTC","DOGE")',
            '["ETH","DOGE","BTC"]', '("ETH","DOGE","BTC")',
        ):
            if combo in token:
                bad.append(f"{p}: {combo}")
    assert not bad, f"hardcoded 3-ticker allowlist found: {bad}"


# ---------------------------------------------------------------------------
# CLI dry-run / preflight: the resolved universe is reported
# without estimating any model
# ---------------------------------------------------------------------------

def test_run_models_dry_run_resolves_universe_without_training(hourly_repo,
                                                                 monkeypatch):
    """``run-models --dry-run`` resolves the requested + available
    universes and emits the canonical stage header — without ever
    fitting a model or writing a parquet."""
    # First produce the merged 1h frame so run-models can load it.
    fm.main(["--horizon", "1h"])
    # Patch project_root so resolve_path() points at the synthetic repo.
    from thesis_pipeline import config as cfg
    monkeypatch.setattr(cfg, "project_root", lambda: hourly_repo)
    cfg.load_config.cache_clear()
    # Copy the canonical configs into the fixture so paths.yaml is
    # resolvable.
    cfg_src = Path(__file__).resolve().parents[1] / "configs"
    if not (hourly_repo / "configs").exists():
        shutil.copytree(cfg_src, hourly_repo / "configs")

    captured: dict = {}

    def fake_header(stage, *, mode, inputs, outputs, extra):
        captured["extra"] = dict(extra)

    monkeypatch.setattr("thesis_pipeline.logging_utils.log_stage_header",
                        fake_header)
    from thesis_pipeline import cli
    rc = cli.main(["run-models", "--horizon", "1h", "--dry-run"])
    assert rc == 0
    extra = captured.get("extra", {})
    # The dry-run header carries the horizon AND the coin universe
    # source (either "(all)" or a literal list).
    assert extra.get("horizon") == "1h"
    coins = extra.get("coins")
    assert coins is not None
