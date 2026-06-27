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


def _find_one(signal_dir: Path, set_id: str,
              expected_hash: str | None = None) -> Path | None:
    """Pick the parquet whose stem matches ``set_id`` with the smoke
    universe-hash suffix.

    When ``expected_hash`` is supplied (the canonical path for v4) the
    helper requires the filename to end in ``_u_<expected_hash>`` —
    we never select a file that came from a different requested
    universe. Without an expected hash the helper falls back to the
    legacy "any file starting with set_id" lookup so older fixtures
    keep working.
    """
    if expected_hash:
        suffix = f"_u_{expected_hash}"
        # Prefer the exact-suffix match.
        for cand in sorted(signal_dir.glob(f"{set_id}*{suffix}.parquet")):
            return cand
        return None
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

    The lookup is keyed on the EXPECTED requested-universe hash so a
    leftover full-grid parquet sitting in the same folder is never
    picked up by mistake. A successful smoke run has
    ``mismatches == {}``.
    """
    from ..modeling.naive_reference import coin_universe_hash
    signal_dir = Path(signal_dir)
    if not signal_dir.exists():
        return {"signal_dir": [f"missing: {signal_dir}"]}

    horizon_dir = signal_dir / CANONICAL_SMOKE["horizon"]
    if not horizon_dir.exists():
        return {"horizon_dir": [f"missing: {horizon_dir}"]}

    tickers_norm = sorted(set(t.upper() for t in expected_tickers))
    requested = "|".join(tickers_norm)
    expected_hash = coin_universe_hash(tickers_norm)

    out: dict[str, list[str]] = {"econ": [], "naive": []}
    econ_path = _find_one(horizon_dir, "ECON", expected_hash=expected_hash)
    if econ_path is None:
        out["econ"].append(
            f"no ECON output parquet with universe suffix _u_{expected_hash}"
        )
    else:
        df = pd.read_parquet(econ_path)
        out["econ"].extend(_validate_econ_rows(df, requested, expected_hash))

    naive_path = _find_one(horizon_dir, "NAIVE", expected_hash=expected_hash)
    if naive_path is None:
        out["naive"].append(
            f"no NAIVE output parquet with universe suffix _u_{expected_hash}"
        )
    else:
        nf = pd.read_parquet(naive_path)
        out["naive"].extend(_validate_naive_rows(nf, requested, expected_hash))

    # Cross-check: ECON and NAIVE must share the same
    # requested_coin_universe_hash AND it must equal the expected hash.
    if econ_path is not None and naive_path is not None:
        econ_df = pd.read_parquet(econ_path)
        naive_df = pd.read_parquet(naive_path)
        econ_hash = (econ_df["requested_coin_universe_hash"].iat[0]
                     if "requested_coin_universe_hash" in econ_df.columns
                     else None)
        naive_hash = (naive_df["requested_coin_universe_hash"].iat[0]
                      if "requested_coin_universe_hash" in naive_df.columns
                      else None)
        if econ_hash != naive_hash:
            out.setdefault("cross", []).append(
                f"requested_coin_universe_hash mismatch: ECON={econ_hash!r} vs "
                f"NAIVE={naive_hash!r}"
            )
        if econ_hash != expected_hash:
            out.setdefault("cross", []).append(
                f"requested_coin_universe_hash on ECON ({econ_hash!r}) does "
                f"not equal expected smoke hash {expected_hash!r}"
            )

    # Empty-on-success contract.
    return {k: v for k, v in out.items() if v}


def _validate_econ_rows(df: pd.DataFrame, requested: str,
                         expected_hash: str) -> list[str]:
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
        ("requested_coin_universe_hash", expected_hash),
    ):
        msg = _check_constant(df, col, exp)
        if msg:
            errors.append(msg)
    return errors


def _validate_naive_rows(df: pd.DataFrame, requested: str,
                          expected_hash: str) -> list[str]:
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
        ("requested_coin_universe_hash", expected_hash),
    ):
        msg = _check_constant(df, col, exp)
        if msg:
            errors.append(msg)
    return errors
