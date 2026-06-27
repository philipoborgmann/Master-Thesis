"""Regression tests for the package-first repository layout.

These tests pin the invariants documented in ``docs/refactor_log.md``:

1. The repository root is **clean** — none of the historical pipeline-stage
   scripts (``Run_Models.py``, ``Create_Price_Features.py``, …) live there
   any more. Their archive copies live outside the repository.
2. ``scripts/`` entries are thin entry points re-exporting a single
   package ``main``.
3. The CLI dispatches every documented stage.
4. ``--dry-run`` works on every CLI sub-command.
5. ``Crypto _data.py`` is no longer in the root (it now lives in ``legacy/``).
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT  = REPO_ROOT / "src"


# Historical root-level pipeline scripts. These files MUST NOT exist any
# more — all logic lives in the corresponding package module.
RETIRED_ROOT_SCRIPTS = (
    "Run_Models.py",
    "Merge_Features.py",
    "Create_Price_Features.py",
    "Price_Data_Validation.py",
    "Sentiment_Data_Load.py",
    "Sentiment_score_vader.py",
    "Sentiment_score_finbert.py",
    "Sentiment_score_cryptobert.py",
    "Sentiment_feature_engineering.py",
    "Sentiment_Stationarity_Test.py",
)

# Map every per-stage scripts/ entry to the package module it re-exports.
SCRIPT_DIRECT_TARGETS = {
    "scripts/run_models.py":                  "thesis_pipeline.modeling.run_models",
    "scripts/merge_features.py":              "thesis_pipeline.features.merge",
    "scripts/create_price_features.py":       "thesis_pipeline.price.features",
    "scripts/validate_price.py":              "thesis_pipeline.price.validate",
    "scripts/load_sentiment.py":              "thesis_pipeline.sentiment.load",
    "scripts/create_sentiment_features.py":   "thesis_pipeline.sentiment.aggregate",
    "scripts/evaluate_signals.py":            "thesis_pipeline.evaluation.evaluate_signals",
}


# ---------------------------------------------------------------------------
# Section 1: the repo root is clean of historical pipeline scripts
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name", RETIRED_ROOT_SCRIPTS)
def test_retired_root_script_no_longer_exists(name):
    path = REPO_ROOT / name
    assert not path.exists(), (
        f"{name} should no longer live at the repo root — "
        f"its logic now lives in src/thesis_pipeline/. "
        f"If you re-created it for compatibility, archive it under legacy/ "
        f"instead."
    )


def test_crypto_data_moved_to_legacy():
    """``Crypto _data.py`` must no longer live at the repo root."""
    assert not (REPO_ROOT / "Crypto _data.py").exists(), (
        "Crypto _data.py is data-acquisition tooling and must live in legacy/"
    )
    assert (REPO_ROOT / "legacy" / "crypto_data.py").exists(), (
        "legacy/crypto_data.py is missing"
    )
    assert (REPO_ROOT / "legacy" / "README.md").exists(), (
        "legacy/README.md is missing"
    )


# ---------------------------------------------------------------------------
# Section 2: scripts/ entries are thin and re-import the package module
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("script_rel,target_module",
                         sorted(SCRIPT_DIRECT_TARGETS.items()))
def test_script_entry_is_thin_and_imports_package(script_rel, target_module):
    path = REPO_ROOT / script_rel
    assert path.exists(), f"{script_rel} should exist as a thin entry point"
    text = path.read_text(encoding="utf-8")
    n_lines = len(text.splitlines())
    assert n_lines < 30, (
        f"{script_rel} has {n_lines} lines — keep it as a thin entry point."
    )
    assert f"from {target_module} import main" in text


def test_score_sentiment_script_routes_through_cli():
    """``scripts/score_sentiment.py`` is multi-model — routes through CLI."""
    text = (REPO_ROOT / "scripts" / "score_sentiment.py").read_text(encoding="utf-8")
    assert "thesis_pipeline.cli" in text


def test_run_pipeline_script_routes_through_cli():
    """``scripts/run_pipeline.py`` orchestrates multiple stages → CLI."""
    text = (REPO_ROOT / "scripts" / "run_pipeline.py").read_text(encoding="utf-8")
    assert "thesis_pipeline.cli" in text


# ---------------------------------------------------------------------------
# Section 3: every CLI sub-command supports --dry-run
# ---------------------------------------------------------------------------

CLI_DRY_RUN_CASES = [
    (["validate-price", "--smoke", "--dry-run"]),
    (["create-price-features", "--horizon", "1d", "--coins", "BTC", "--smoke", "--dry-run"]),
    (["load-sentiment", "--smoke", "--dry-run"]),
    (["score-sentiment", "--model", "vader", "--smoke", "--dry-run"]),
    (["create-sentiment-features", "--smoke", "--dry-run"]),
    (["stationarity", "--smoke", "--dry-run"]),
    (["merge-features", "--horizon", "1d", "--smoke", "--dry-run"]),
    (["run-models", "--horizon", "1d", "--set-id", "ECON", "--smoke", "--dry-run"]),
    (["evaluate-signals", "--horizon", "1d", "--smoke", "--dry-run"]),
    (["diagnostics", "--horizon", "1d", "--dry-run"]),
]


@pytest.mark.parametrize("argv", CLI_DRY_RUN_CASES,
                         ids=[a[0] for a in CLI_DRY_RUN_CASES])
def test_cli_dry_run_returns_zero(argv):
    if str(SRC_ROOT) not in sys.path:
        sys.path.insert(0, str(SRC_ROOT))
    from thesis_pipeline import cli
    rc = cli.main(argv)
    assert rc == 0


# ---------------------------------------------------------------------------
# Section 4: package modules expose main() with optional argv
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("modname", sorted(set(SCRIPT_DIRECT_TARGETS.values()) | {
    "thesis_pipeline.modeling.run_models",
    "thesis_pipeline.features.merge",
    "thesis_pipeline.price.features",
    "thesis_pipeline.price.validate",
    "thesis_pipeline.sentiment.load",
    "thesis_pipeline.sentiment.score_vader",
    # score_finbert was retired (FinBERT removed from the pipeline); the file
    # remains on disk for historical traceability but is no longer part of
    # the import contract.
    "thesis_pipeline.sentiment.score_cryptobert",
    "thesis_pipeline.sentiment.aggregate",
    "thesis_pipeline.sentiment.stationarity",
    "thesis_pipeline.evaluation.evaluate_signals",
}))
def test_package_module_main_accepts_argv(modname):
    if str(SRC_ROOT) not in sys.path:
        sys.path.insert(0, str(SRC_ROOT))
    import importlib
    try:
        mod = importlib.import_module(modname)
    except ModuleNotFoundError as exc:
        pytest.skip(f"{modname} requires {exc.name}; not installed here")
    assert callable(getattr(mod, "main", None)), f"{modname}.main missing"
    # Inspect the signature — main must accept ``argv`` (positional or keyword)
    import inspect
    sig = inspect.signature(mod.main)
    params = sig.parameters
    assert "argv" in params, (
        f"{modname}.main() must accept an optional argv list "
        f"(current signature: {sig})"
    )


# ---------------------------------------------------------------------------
# Section 5: CLI exposes every documented sub-command
# ---------------------------------------------------------------------------

REQUIRED_CLI_COMMANDS = (
    "validate-price", "create-price-features",
    "load-sentiment", "score-sentiment", "create-sentiment-features",
    "stationarity", "merge-features", "run-models",
    "evaluate-signals", "diagnostics", "run-stage", "run-pipeline",
)


def test_cli_exposes_all_required_subcommands():
    if str(SRC_ROOT) not in sys.path:
        sys.path.insert(0, str(SRC_ROOT))
    from thesis_pipeline import cli
    parser = cli.build_parser()
    # The subparsers action is the only one with `.choices`
    actions = [a for a in parser._actions if a.__class__.__name__ == "_SubParsersAction"]
    assert actions, "CLI should have sub-parsers"
    choices = set(actions[0].choices.keys())
    for cmd in REQUIRED_CLI_COMMANDS:
        assert cmd in choices, f"CLI missing sub-command: {cmd}"
