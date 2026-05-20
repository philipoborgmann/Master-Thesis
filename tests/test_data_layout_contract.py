"""Layout contract — never fails when data is absent, only when present and malformed.

This protects developers from accidentally committing data under
unexpected names. In CI (no Data/ folder) every test simply skips.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

from thesis_pipeline.config import horizons, load_config, resolve_path


@pytest.mark.parametrize("h", ["15min", "1h", "4h", "6h", "1d"])
def test_raw_price_folder_filenames(h):
    folder = resolve_path(f"raw_price_{h}")
    if not folder.exists():
        pytest.skip(f"local data missing: {folder}")
    suffix = load_config("horizons")["raw_price_horizons"][h]["filename_suffix"]
    pattern = re.compile(rf"^[A-Z0-9]+USDT_{re.escape(suffix)}\.parquet$")
    bad = [p.name for p in folder.glob("*.parquet")
           if not pattern.match(p.name)]
    assert not bad, f"Files in {folder} violate naming convention: {bad}"


def test_cmc_files_when_present():
    root = resolve_path("cmc_root")
    if not root.exists():
        pytest.skip(f"local data missing: {root}")
    for key in ("cmc_market_cap", "cmc_price", "cmc_volume"):
        p = resolve_path(key)
        assert p.exists(), f"Missing expected CMC file: {p}"


def test_sentiment_subreddit_folders_have_submission_csv():
    root = resolve_path("raw_sentiment_root")
    if not root.exists():
        pytest.skip(f"local data missing: {root}")
    bad = []
    for d in root.iterdir():
        if not d.is_dir():
            continue
        if not (d / "submission.csv").exists():
            bad.append(d.name)
    assert not bad, f"subreddit folders missing submission.csv: {bad}"


def test_horizon_keys_cover_all_known_horizons():
    raw = horizons("raw_price")
    for h in ("15min", "1h", "4h", "6h", "1d"):
        assert h in raw
