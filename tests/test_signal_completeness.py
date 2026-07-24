"""Signal-completeness audit + the evaluate-signals production guard.

Covers the concrete failure mode that motivated the audit: a sentiment set
whose walk-forward ended early (missing the final timestamp block) must be
detected and must stop the final evaluation from publishing a partial
comparison.
"""
from __future__ import annotations

import pandas as pd
import pytest

from thesis_pipeline.evaluation.completeness import (
    audit_signal_completeness,
    incomplete_production_groups,
    STATUS_COMPLETE,
    STATUS_INCOMPLETE,
    STATUS_UNKNOWN,
)

COINS = [f"C{i:02d}" for i in range(25)]
FCFG = {"start": "2022-01-01", "end_exclusive": "2023-01-01"}


def _group(set_id, sm, timestamps, tickers=COINS):
    return pd.DataFrame([
        {"timestamp": t, "ticker": c, "set_id": set_id, "sentiment_model": sm,
         "horizon": "1h", "model_type": "panel_logit",
         "panel_mode": "ticker_fixed_effects", "hpo_variant": "hpo_logloss"}
        for c in tickers for t in timestamps
    ])


@pytest.fixture
def full_ts():
    return pd.date_range("2022-01-01", periods=12, freq="h", tz="UTC")


def test_complete_run_passes(full_ts):
    sig = pd.concat([_group("ECON", "-", full_ts),
                     _group("ECON_CBT_F", "cryptobert", full_ts)],
                    ignore_index=True)
    audit = audit_signal_completeness(sig, forecast_cfg=FCFG)
    assert set(audit["status"]) == {STATUS_COMPLETE}
    assert (audit["coverage_vs_econ"] == 1.0).all()
    assert incomplete_production_groups(audit, {"ECON", "ECON_CBT_F"}).empty


def test_early_ending_sentiment_set_is_flagged(full_ts):
    # 25 coins; complete ECON benchmark; one sentiment model ends 3 ts early.
    sig = pd.concat([
        _group("ECON", "-", full_ts),
        _group("ECON_CBT_F", "cryptobert", full_ts),
        _group("SENT_VAD_LD", "vader", full_ts[:-3]),   # missing final block
    ], ignore_index=True)
    audit = audit_signal_completeness(sig, forecast_cfg=FCFG)

    row = audit[audit["set_id"] == "SENT_VAD_LD"].iloc[0]
    assert row["status"] == STATUS_INCOMPLETE
    assert row["missing_timestamps"] == 3
    assert row["n_timestamps"] == len(full_ts) - 3
    assert row["coverage_vs_econ"] < 1.0

    reg = {"ECON", "ECON_CBT_F", "SENT_VAD_LD"}
    bad = incomplete_production_groups(audit, reg)
    assert list(bad["set_id"]) == ["SENT_VAD_LD"]


def test_duplicate_rows_flagged(full_ts):
    dup = pd.concat([_group("ECON", "-", full_ts),
                     _group("ECON_CBT_F", "cryptobert", full_ts),
                     _group("ECON_CBT_F", "cryptobert", full_ts[:1])],  # dup rows
                    ignore_index=True)
    audit = audit_signal_completeness(dup, forecast_cfg=FCFG)
    row = audit[audit["set_id"] == "ECON_CBT_F"].iloc[0]
    assert row["n_duplicate_rows"] > 0
    assert row["status"] == STATUS_INCOMPLETE


def test_no_econ_reference_is_unknown_not_failure(full_ts):
    # A restricted run with no ECON benchmark cannot be judged → unknown,
    # and must NOT block evaluation.
    sig = _group("SENT_VAD_L", "vader", full_ts[:-2])
    audit = audit_signal_completeness(sig, forecast_cfg=FCFG)
    assert audit.iloc[0]["status"] == STATUS_UNKNOWN
    assert incomplete_production_groups(audit, {"SENT_VAD_L"}).empty


def test_empty_signals_returns_empty_frame():
    out = audit_signal_completeness(pd.DataFrame(), forecast_cfg=FCFG)
    assert out.empty
    assert "status" in out.columns


# --- end-to-end guard through evaluate-signals.main() ------------------------

def _write_signal_parquets(root, groups, horizon="1h"):
    d = root / horizon
    d.mkdir(parents=True, exist_ok=True)
    for name, df in groups.items():
        # Minimal columns evaluate-signals needs to load + score.
        df = df.copy()
        df["target"] = 1
        df["prediction"] = 1
        df["probability"] = 0.6
        df["hpo_objective"] = "log_loss"
        df.to_parquet(d / f"{name}.parquet", index=False)


def test_evaluate_signals_aborts_on_incomplete(tmp_path, full_ts, monkeypatch):
    from thesis_pipeline.evaluation import evaluate_signals as es

    groups = {
        "ECON": _group("ECON", "-", full_ts),
        "SENT_VAD_LD": _group("SENT_VAD_LD", "vader", full_ts[:-3]),
    }
    sig_root = tmp_path / "Signals"
    _write_signal_parquets(sig_root, groups)
    out_dir = tmp_path / "Eval"

    rc = es.main(["--horizon", "1h", "--signals-root", str(sig_root),
                  "--output-dir", str(out_dir),
                  "--no-market-cap", "--no-economic", "--no-volatility",
                  "--no-regime-mcnemar"])
    # Non-zero exit; completeness CSV written; no pooled metrics published.
    assert rc == 3
    assert (out_dir / "signal_completeness.csv").exists()
    assert not (out_dir / "pooled_metrics.csv").exists()

    # With the override it proceeds (rc 0) and publishes metrics.
    rc2 = es.main(["--horizon", "1h", "--signals-root", str(sig_root),
                   "--output-dir", str(out_dir), "--allow-incomplete-signals",
                   "--no-market-cap", "--no-economic", "--no-volatility",
                   "--no-regime-mcnemar"])
    assert rc2 == 0
    assert (out_dir / "pooled_metrics.csv").exists()


def test_top_level_cli_forwards_strict_and_allow_flags(monkeypatch):
    """The public `thesis_pipeline.cli evaluate-signals` must expose and forward
    --strict-feature-set-ids and --allow-incomplete-signals to the eval module."""
    import sys
    from pathlib import Path
    src = Path(__file__).resolve().parents[1] / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
    from thesis_pipeline import cli
    from thesis_pipeline.evaluation import evaluate_signals as es

    captured = {}

    def _fake_main(argv):
        captured["argv"] = list(argv)
        return 0

    monkeypatch.setattr(es, "main", _fake_main)

    cli.main(["evaluate-signals", "--horizon", "1d",
              "--strict-feature-set-ids", "--allow-incomplete-signals"])
    argv = captured["argv"]
    assert "--strict-feature-set-ids" in argv
    assert "--allow-incomplete-signals" in argv
