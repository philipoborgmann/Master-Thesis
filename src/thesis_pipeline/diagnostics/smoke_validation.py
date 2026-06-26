"""Smoke-run output validator (Aufgabe 8 Section F.3).

Locates the ECON model + NAIVE outputs under a signal directory and
verifies that every relevant row carries the canonical v4 metadata
expected by the smoke configuration. Mismatches are returned as a
dict of lists so the caller can either pretty-print them or raise.

Used by :mod:`tests.test_smoke_output_validation` and by the
:mod:`thesis_pipeline.diagnostics.v4_acceptance_audit` runner.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Tuple

import pandas as pd


CANONICAL_SMOKE = dict(
    horizon="1d",
    model_type="panel_logit",
    panel_mode="ticker_fixed_effects",
    train_window_mode="rolling_fixed",
    rolling_window_days=180.0,
    hpo_enabled=True,
    hpo_objective="log_loss",
    set_id_econ="ECON",
    set_id_naive="NAIVE",
    requested_tickers="BTC|ETH",
    universe_identity_source="requested_metadata",
)


def _find_one(signal_dir: Path, set_id: str) -> Path | None:
    """Pick the first parquet whose stem starts with ``set_id`` followed
    by ``_`` or ``.``. The v4 universe-hash suffix is optional."""
    for cand in sorted(signal_dir.glob(f"{set_id}*.parquet")):
        stem = cand.stem
        if stem == set_id or stem.startswith(f"{set_id}_"):
            return cand
    return None


def _check_constant(df: pd.DataFrame, col: str, expected) -> str | None:
    """Return an error message when the column is not constant-equal."""
    if col not in df.columns:
        return f"missing column {col!r}"
    vals = df[col].dropna().unique()
    if len(vals) == 0:
        return f"{col!r} is empty"
    if len(vals) > 1:
        return f"{col!r} has multiple values {sorted(map(str, vals))}"
    actual = vals[0]
    if isinstance(expected, float):
        try:
            if float(actual) != float(expected):
                return f"{col!r} = {actual} (expected {expected})"
        except (TypeError, ValueError):
            return f"{col!r} = {actual} (expected {expected}; non-numeric)"
        return None
    if str(actual) != str(expected):
        return f"{col!r} = {actual!r} (expected {expected!r})"
    return None


def validate_smoke_outputs(signal_dir: Path,
                            expected_tickers: Iterable[str] = ("BTC", "ETH"),
                            ) -> dict:
    """Return a dict of mismatches.

    A successful smoke run has ``mismatches == {}``. Empty result keys
    map to lists for symmetry with the failure path.
    """
    signal_dir = Path(signal_dir)
    if not signal_dir.exists():
        return {"signal_dir": [f"missing: {signal_dir}"]}

    horizon_dir = signal_dir / CANONICAL_SMOKE["horizon"]
    if not horizon_dir.exists():
        return {"horizon_dir": [f"missing: {horizon_dir}"]}

    requested = "|".join(sorted(set(t.upper() for t in expected_tickers)))

    out: dict[str, list[str]] = {"econ": [], "naive": []}
    econ_path = _find_one(horizon_dir, "ECON")
    if econ_path is None:
        out["econ"].append("no ECON output parquet found")
    else:
        df = pd.read_parquet(econ_path)
        out["econ"].extend(_validate_econ_rows(df, requested))

    naive_path = _find_one(horizon_dir, "NAIVE")
    if naive_path is None:
        out["naive"].append("no NAIVE output parquet found")
    else:
        nf = pd.read_parquet(naive_path)
        out["naive"].extend(_validate_naive_rows(nf, requested))

    # Cross-check: ECON and NAIVE must share the same
    # requested_coin_universe_hash (same identity → matchable in
    # absolute_vs_naive).
    if econ_path is not None and naive_path is not None:
        econ_hash = pd.read_parquet(econ_path)["requested_coin_universe_hash"].iat[0] \
            if "requested_coin_universe_hash" in pd.read_parquet(econ_path).columns else None
        naive_hash = pd.read_parquet(naive_path)["requested_coin_universe_hash"].iat[0] \
            if "requested_coin_universe_hash" in pd.read_parquet(naive_path).columns else None
        if econ_hash != naive_hash:
            out.setdefault("cross", []).append(
                f"requested_coin_universe_hash mismatch: ECON={econ_hash!r} vs "
                f"NAIVE={naive_hash!r}"
            )

    # Empty-on-success contract.
    return {k: v for k, v in out.items() if v}


def _validate_econ_rows(df: pd.DataFrame, requested: str) -> list[str]:
    if df.empty:
        return ["ECON output is empty"]
    errors: list[str] = []
    for col, exp in (
        ("model_type",         CANONICAL_SMOKE["model_type"]),
        ("panel_mode",         CANONICAL_SMOKE["panel_mode"]),
        ("train_window_mode",  CANONICAL_SMOKE["train_window_mode"]),
        ("rolling_window_days", CANONICAL_SMOKE["rolling_window_days"]),
        ("hpo_enabled",        CANONICAL_SMOKE["hpo_enabled"]),
        ("hpo_objective",      CANONICAL_SMOKE["hpo_objective"]),
        ("set_id",             CANONICAL_SMOKE["set_id_econ"]),
        ("horizon",            CANONICAL_SMOKE["horizon"]),
        ("requested_tickers",  requested),
        ("universe_identity_source",
                                CANONICAL_SMOKE["universe_identity_source"]),
    ):
        msg = _check_constant(df, col, exp)
        if msg:
            errors.append(msg)
    return errors


def _validate_naive_rows(df: pd.DataFrame, requested: str) -> list[str]:
    if df.empty:
        return ["NAIVE output is empty"]
    errors: list[str] = []
    for col, exp in (
        ("set_id",        CANONICAL_SMOKE["set_id_naive"]),
        ("hpo_enabled",   False),
        ("hpo_variant",   "naive"),
        ("model_type",    CANONICAL_SMOKE["model_type"]),
        ("panel_mode",    CANONICAL_SMOKE["panel_mode"]),
        ("train_window_mode",
                          CANONICAL_SMOKE["train_window_mode"]),
        ("rolling_window_days",
                          CANONICAL_SMOKE["rolling_window_days"]),
        ("requested_tickers", requested),
    ):
        msg = _check_constant(df, col, exp)
        if msg:
            errors.append(msg)
    return errors
