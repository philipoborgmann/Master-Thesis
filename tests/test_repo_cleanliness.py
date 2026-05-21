"""Regression tests for the package-first repository layout.

These tests pin the invariants documented in ``docs/refactor_log.md``:

1. Root-level scripts contain no real logic — they are thin redirects to
   ``thesis_pipeline.*`` modules.
2. ``scripts/`` entries are thin entry points.
3. The CLI dispatches every documented stage.
4. ``--dry-run`` works on every CLI sub-command.
5. ``Crypto _data.py`` is no longer in the root (it now lives in ``legacy/``).
"""
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT  = REPO_ROOT / "src"


# Map every root-script redirect to the package module it must import.
ROOT_REDIRECTS = {
    "Run_Models.py":                    "thesis_pipeline.modeling.run_models",
    "Merge_Features.py":                "thesis_pipeline.features.merge",
    "Create_Price_Features.py":         "thesis_pipeline.price.features",
    "Price_Data_Validation.py":         "thesis_pipeline.price.validate",
    "Sentiment_Data_Load.py":           "thesis_pipeline.sentiment.load",
    "Sentiment_score_vader.py":         "thesis_pipeline.sentiment.score_vader",
    "Sentiment_score_finbert.py":       "thesis_pipeline.sentiment.score_finbert",
    "Sentiment_score_cryptobert.py":    "thesis_pipeline.sentiment.score_cryptobert",
    "Sentiment_feature_engineering.py": "thesis_pipeline.sentiment.aggregate",
    "Sentiment_Stationarity_Test.py":   "thesis_pipeline.sentiment.stationarity",
}

# Map every per-stage scripts/ entry to the same package module.
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
# Section 1: root-level redirects are thin
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("root_name,target_module",
                         sorted(ROOT_REDIRECTS.items()))
def test_root_redirect_is_thin_and_imports_package(root_name, target_module):
    path = REPO_ROOT / root_name
    assert path.exists(), f"{root_name} should exist as a legacy redirect"
    text = path.read_text(encoding="utf-8")

    # A redirect should be < ~40 lines: nothing else fits in that budget.
    n_lines = len(text.splitlines())
    assert n_lines < 40, (
        f"{root_name} has {n_lines} lines — it must be a thin redirect, "
        f"not a re-implementation. Move the logic into {target_module}."
    )

    # It must import the documented package module.
    assert f"from {target_module} import main" in text, (
        f"{root_name} should re-import main() from {target_module}"
    )

    # It must NOT redeclare heavy modelling-library imports — those belong to
    # the package module now.
    forbidden = (
        "from sklearn.linear_model import",
        "from sklearn.preprocessing import StandardScaler",
        "from transformers import",
        "import statsmodels",
    )
    for needle in forbidden:
        assert needle not in text, (
            f"{root_name} still contains heavy import {needle!r} — "
            "the redirect must own no real logic."
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
# Section 3: every entry-point ultimately calls the same package main()
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("root_name,target_module",
                         sorted(ROOT_REDIRECTS.items()))
def test_root_and_package_share_same_main(root_name, target_module):
    """Importing the redirect and importing the package directly must yield
    the *same* ``main`` function object — no copies, no shims.

    Skipped when the package module has heavy optional dependencies
    (matplotlib, statsmodels, arch, torch, transformers, vaderSentiment)
    that are unavailable in the current environment — the redirect contract
    is then still pinned via static text inspection in
    :func:`test_root_redirect_is_thin_and_imports_package`.
    """
    if str(SRC_ROOT) not in sys.path:
        sys.path.insert(0, str(SRC_ROOT))

    import importlib
    try:
        pkg = importlib.import_module(target_module)
    except ModuleNotFoundError as exc:
        pytest.skip(f"{target_module} requires {exc.name}; not installed here")

    # Load the redirect as a module under a unique throwaway name
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        f"__redirect_test__{root_name.replace('.', '_')}",
        REPO_ROOT / root_name,
    )
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
    except ModuleNotFoundError as exc:
        pytest.skip(f"{root_name} could not import {exc.name}")

    assert mod.main is pkg.main, (
        f"{root_name} must re-export the same main() as {target_module}"
    )


# ---------------------------------------------------------------------------
# Section 4: every CLI sub-command supports --dry-run
# ---------------------------------------------------------------------------

CLI_DRY_RUN_CASES = [
    (["validate-price", "--smoke", "--dry-run"]),
    (["create-price-features", "--horizon", "1d", "--coins", "BTC", "--smoke", "--dry-run"]),
    (["load-sentiment", "--smoke", "--dry-run"]),
    (["score-sentiment", "--model", "vader", "--smoke", "--dry-run"]),
    (["create-sentiment-features", "--smoke", "--dry-run"]),
    (["stationarity", "--smoke", "--dry-run"]),
    (["merge-features", "--horizon", "1d", "--smoke", "--dry-run"]),
    (["run-models", "--horizon", "1d", "--set-id", "B1", "--smoke", "--dry-run"]),
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
# Section 5: package modules expose main() with optional argv
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("modname", sorted(set(ROOT_REDIRECTS.values()) |
                                           set(SCRIPT_DIRECT_TARGETS.values())))
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
# Section 6: CLI exposes every documented sub-command
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
