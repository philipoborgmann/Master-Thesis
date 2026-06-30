"""Shared coverage-intersection helper for directional comparisons.

All comparison-by-merge evaluation writers — the nested H1 incremental
test, the H2/H3 difference-in-improvement test, and the
absolute-vs-NAIVE comparison — must sit a *candidate* set and a
*reference* set on byte-identical observations before any McNemar /
difference-in-improvement / lift statistic is computed.

The economic backtest already does this (it restricts every set to the
common signal × forward-return sample and reports ``status=ok``); this
module mirrors that discipline for the directional side.

Contract
--------
``coverage_intersection(candidate, reference, key_cols)`` returns a
``CoverageIntersection`` carrying:

* ``candidate`` / ``reference`` — each restricted to the shared keys,
  deduplicated to one row per key, sorted identically so they align
  positionally;
* honest diagnostics: ``n_candidate``, ``n_reference``, ``n_matched``,
  ``n_unmatched_candidate``, ``n_unmatched_reference``,
  ``n_duplicate_candidate``, ``n_duplicate_reference``.

The deduplication is *defensive*: a set that genuinely carries two
predictions for one ``(ticker, timestamp, horizon)`` is a data bug that
would otherwise fan out the inner join; we keep the first occurrence and
report the count so the caller can surface it honestly. When both sides
already carry one clean row per key and share full coverage (the 6h
diff-in-improvement case), the intersection is a no-op and downstream
numbers are unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


DEFAULT_KEY_COLUMNS = ("ticker", "timestamp", "horizon")


@dataclass
class CoverageIntersection:
    candidate: pd.DataFrame
    reference: pd.DataFrame
    n_candidate: int
    n_reference: int
    n_matched: int
    n_unmatched_candidate: int
    n_unmatched_reference: int
    n_duplicate_candidate: int
    n_duplicate_reference: int

    def as_diagnostics(self, *, candidate_label: str = "candidate",
                        reference_label: str = "reference") -> dict:
        """Return the count diagnostics keyed for a writer's output row."""
        return {
            f"n_{candidate_label}":            self.n_candidate,
            f"n_{reference_label}":            self.n_reference,
            "n_matched":                       self.n_matched,
            f"n_unmatched_{candidate_label}":  self.n_unmatched_candidate,
            f"n_unmatched_{reference_label}":  self.n_unmatched_reference,
            f"n_duplicate_{candidate_label}_keys": self.n_duplicate_candidate,
            f"n_duplicate_{reference_label}_keys": self.n_duplicate_reference,
        }


def _normalise_keys(df: pd.DataFrame, key_cols: tuple[str, ...]) -> pd.DataFrame:
    """Return a copy with the join keys coerced to canonical dtypes.

    ``ticker`` → upper-cased string; ``timestamp`` → tz-aware UTC
    datetime; ``horizon`` → string. Other key columns are left as-is.
    """
    out = df.copy()
    if "ticker" in key_cols and "ticker" in out.columns:
        out["ticker"] = out["ticker"].astype(str).str.upper()
    if "timestamp" in key_cols and "timestamp" in out.columns:
        out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True,
                                           errors="coerce")
    if "horizon" in key_cols and "horizon" in out.columns:
        out["horizon"] = out["horizon"].astype(str)
    return out


def coverage_intersection(
    candidate: pd.DataFrame,
    reference: pd.DataFrame,
    *,
    key_cols: tuple[str, ...] = DEFAULT_KEY_COLUMNS,
) -> CoverageIntersection:
    """Restrict ``candidate`` and ``reference`` to their common keys.

    Both sides are deduplicated to one row per key (keeping the first
    occurrence) and sorted by the key so the returned frames align
    row-for-row. Diagnostics report the raw sizes, the matched size and
    the unmatched / duplicate counts on each side.
    """
    # Only use key columns that exist on BOTH sides.
    keys = [c for c in key_cols
            if c in candidate.columns and c in reference.columns]
    if not keys:
        # No shared key — nothing can be matched.
        return CoverageIntersection(
            candidate=candidate.iloc[0:0].copy(),
            reference=reference.iloc[0:0].copy(),
            n_candidate=int(len(candidate)),
            n_reference=int(len(reference)),
            n_matched=0,
            n_unmatched_candidate=int(len(candidate)),
            n_unmatched_reference=int(len(reference)),
            n_duplicate_candidate=0,
            n_duplicate_reference=0,
        )

    cand = _normalise_keys(candidate, tuple(keys))
    ref  = _normalise_keys(reference, tuple(keys))

    # Drop rows with a null key on either side — they can never match
    # and pandas merge refuses null join keys.
    cand = cand.dropna(subset=keys)
    ref  = ref.dropna(subset=keys)

    n_candidate = int(len(cand))
    n_reference = int(len(ref))

    n_dup_cand = int(cand.duplicated(subset=keys).sum())
    n_dup_ref  = int(ref.duplicated(subset=keys).sum())

    # Defensive dedup — keep first occurrence per key to prevent the
    # inner join from fanning out.
    cand_u = cand.drop_duplicates(subset=keys, keep="first")
    ref_u  = ref.drop_duplicates(subset=keys, keep="first")

    # Matched keys = inner-join key set.
    cand_keys = cand_u[keys].drop_duplicates()
    matched_keys = cand_keys.merge(ref_u[keys].drop_duplicates(),
                                   on=keys, how="inner")
    n_matched = int(len(matched_keys))

    if n_matched == 0:
        empty_cand = cand_u.iloc[0:0].copy()
        empty_ref  = ref_u.iloc[0:0].copy()
        return CoverageIntersection(
            candidate=empty_cand, reference=empty_ref,
            n_candidate=n_candidate, n_reference=n_reference,
            n_matched=0,
            n_unmatched_candidate=n_candidate,
            n_unmatched_reference=n_reference,
            n_duplicate_candidate=n_dup_cand,
            n_duplicate_reference=n_dup_ref,
        )

    cand_m = (cand_u.merge(matched_keys, on=keys, how="inner")
              .sort_values(keys).reset_index(drop=True))
    ref_m  = (ref_u.merge(matched_keys, on=keys, how="inner")
              .sort_values(keys).reset_index(drop=True))

    return CoverageIntersection(
        candidate=cand_m,
        reference=ref_m,
        n_candidate=n_candidate,
        n_reference=n_reference,
        n_matched=n_matched,
        n_unmatched_candidate=n_candidate - n_matched,
        n_unmatched_reference=n_reference - n_matched,
        n_duplicate_candidate=n_dup_cand,
        n_duplicate_reference=n_dup_ref,
    )
