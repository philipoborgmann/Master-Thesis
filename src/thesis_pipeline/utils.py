"""Misc utilities: script delegation and argv handling.

The refactor preserves behavior by delegating heavy work to the original
root-level scripts (whose ``main()`` functions accept argparse-driven
arguments). ``run_script_main`` imports a root script by file path and
invokes its ``main()`` with a synthetic ``sys.argv``.
"""

from __future__ import annotations

import importlib.util
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Callable, Iterable, Sequence

from .config import project_root


# Mapping from a logical module name -> root-level script filename.
# We use file paths (not import names) because ``Crypto _data.py`` contains
# a space and is not importable as a normal module.
SCRIPT_MAP: dict[str, str] = {
    "validate_price":            "Price_Data_Validation.py",
    "create_price_features":     "Create_Price_Features.py",
    "load_sentiment":            "Sentiment_Data_Load.py",
    "score_vader":               "Sentiment_score_vader.py",
    "score_finbert":             "Sentiment_score_finbert.py",
    "score_cryptobert":          "Sentiment_score_cryptobert.py",
    "create_sentiment_features": "Sentiment_feature_engineering.py",
    "stationarity":              "Sentiment_Stationarity_Test.py",
    "merge_features":            "Merge_Features.py",
    "run_models":                "Run_Models.py",
    "crypto_data":               "Crypto _data.py",
}


def _load_module_by_path(path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load script: {path}")
    mod = importlib.util.module_from_spec(spec)
    # Make submodule visible in sys.modules so dataclasses/pickling work
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod


def load_root_script(stage: str):
    """Import the root-level script associated with a logical stage name."""
    if stage not in SCRIPT_MAP:
        raise KeyError(f"Unknown stage: {stage!r}")
    filename = SCRIPT_MAP[stage]
    path = project_root() / filename
    if not path.exists():
        raise FileNotFoundError(f"Root script missing: {path}")
    safe_module_name = f"thesis_pipeline._wrapped_{stage}"
    return _load_module_by_path(path, safe_module_name)


@contextmanager
def patched_argv(argv: Sequence[str]):
    """Temporarily replace ``sys.argv`` (used to drive original scripts' argparse)."""
    saved = sys.argv
    sys.argv = list(argv)
    try:
        yield
    finally:
        sys.argv = saved


def run_script_main(stage: str, argv: Sequence[str] | None = None) -> int:
    """Import the root script for ``stage`` and call its ``main()``.

    Returns 0 on success; if the script's ``main()`` returns an int, that is
    propagated. Errors are not suppressed.
    """
    mod = load_root_script(stage)
    if not hasattr(mod, "main"):
        raise AttributeError(
            f"Root script {SCRIPT_MAP[stage]} has no main() function — "
            "cannot delegate to it programmatically."
        )
    filename = SCRIPT_MAP[stage]
    cli_argv = [filename, *(argv or [])]
    with patched_argv(cli_argv):
        result = mod.main()
    return int(result) if isinstance(result, int) else 0


def coerce_str_list(value: str | Iterable[str] | None) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [v.strip() for v in value.replace(",", " ").split() if v.strip()]
    return list(value)


def banner(msg: str, char: str = "─") -> None:
    print(char * 72)
    print(msg)
    print(char * 72)
