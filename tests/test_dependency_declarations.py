"""Dependency-declaration consistency between pyproject.toml, requirements.txt
and the README install instructions.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover
    import tomli as tomllib


def _pyproject() -> dict:
    return tomllib.loads((REPO / "pyproject.toml").read_text(encoding="utf-8"))


def test_ccxt_declared_in_acquisition_extra():
    extras = _pyproject()["project"]["optional-dependencies"]
    assert "acquisition" in extras, "missing [acquisition] optional-dependency group"
    joined = " ".join(extras["acquisition"])
    assert re.search(r"ccxt\s*>=\s*4", joined), (
        f"acquisition extra must declare ccxt>=4.0; got {extras['acquisition']}")


def test_transformers_and_tests_extras_retained():
    extras = _pyproject()["project"]["optional-dependencies"]
    assert "transformers" in extras and "tests" in extras
    assert any("torch" in d for d in extras["transformers"])
    assert any("pytest" in d for d in extras["tests"])


def test_readme_full_install_includes_acquisition():
    readme = (REPO / "README.md").read_text(encoding="utf-8")
    assert '".[tests,transformers,acquisition]"' in readme, (
        "README full-reproduction install command must include the acquisition extra")


def test_requirements_and_pyproject_do_not_contradict():
    """Shared packages must not declare contradictory lower bounds."""
    req_text = (REPO / "requirements.txt").read_text(encoding="utf-8")
    req = {}
    for line in req_text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        m = re.match(r"([A-Za-z0-9_.\-]+)\s*>=\s*([0-9][0-9A-Za-z.\-]*)", line)
        if m:
            req[m.group(1).lower()] = m.group(2)

    pp = _pyproject()["project"]
    pp_reqs = list(pp["dependencies"])
    for group in pp["optional-dependencies"].values():
        pp_reqs += list(group)
    pyproj = {}
    for spec in pp_reqs:
        m = re.match(r"([A-Za-z0-9_.\-]+)\s*>=\s*([0-9][0-9A-Za-z.\-]*)", spec)
        if m:
            pyproj[m.group(1).lower()] = m.group(2)

    # ccxt lives in requirements.txt (broad env) and the acquisition extra —
    # same lower bound either way.
    assert "ccxt" in req and "ccxt" in pyproj
    for pkg in set(req) & set(pyproj):
        assert req[pkg] == pyproj[pkg], (
            f"{pkg}: requirements.txt says >={req[pkg]} but pyproject says "
            f">={pyproj[pkg]}")
