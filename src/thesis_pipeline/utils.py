"""Small generic utilities used across the package.

The previous delegation layer (``SCRIPT_MAP`` / ``load_root_script`` /
``run_script_main``) imported the historical root-level scripts via
``importlib.util.spec_from_file_location`` and invoked their ``main()``.
Those scripts have been retired — their logic now lives in the
corresponding ``thesis_pipeline.*`` module — so the delegation layer was
removed in the cleanup commit. Only the tiny shared helpers remain here.
"""

from __future__ import annotations

import sys
from contextlib import contextmanager
from typing import Iterable, Sequence


@contextmanager
def patched_argv(argv: Sequence[str]):
    """Temporarily replace ``sys.argv``.

    Kept available for callers that want to drive a third-party CLI whose
    ``main()`` reads ``sys.argv`` directly. The in-package modules all
    accept an explicit ``argv`` parameter and do not need this.
    """
    saved = sys.argv
    sys.argv = list(argv)
    try:
        yield
    finally:
        sys.argv = saved


def coerce_str_list(value: str | Iterable[str] | None) -> list[str]:
    """Normalise a mixed CSV / iterable / None argument to a clean list."""
    if value is None:
        return []
    if isinstance(value, str):
        return [v.strip() for v in value.replace(",", " ").split() if v.strip()]
    return list(value)


def banner(msg: str, char: str = "─") -> None:
    """Tiny console banner used by ad-hoc scripts."""
    print(char * 72)
    print(msg)
    print(char * 72)
