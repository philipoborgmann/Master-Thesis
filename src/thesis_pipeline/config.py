"""Config loading and path resolution.

All YAML configs live under ``configs/``. ``load_config()`` reads them as
plain dicts; ``resolve_path()`` resolves a path key from ``paths.yaml``
against the project root and substitutes ``{horizon}`` / ``{set_id}``
placeholders when given.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Mapping

try:
    import yaml
except ImportError as exc:  # pragma: no cover
    raise RuntimeError(
        "PyYAML is required to load configs. "
        "Install with `pip install pyyaml` or `pip install -r requirements.txt`."
    ) from exc


def project_root() -> Path:
    """Return the repository root (where ``configs/`` lives)."""
    here = Path(__file__).resolve()
    # src/thesis_pipeline/config.py -> repo root is 3 levels up
    return here.parents[2]


def configs_dir() -> Path:
    return project_root() / "configs"


@lru_cache(maxsize=None)
def load_config(name: str) -> Dict[str, Any]:
    """Load ``configs/<name>.yaml`` once and cache it.

    Pass either the bare name (``"paths"``) or with the ``.yaml`` suffix.
    """
    if not name.endswith((".yaml", ".yml")):
        name = name + ".yaml"
    path = configs_dir() / name
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    return data


def all_configs() -> Dict[str, Dict[str, Any]]:
    return {
        "paths":         load_config("paths"),
        "horizons":      load_config("horizons"),
        "coins":         load_config("coins"),
        "feature_sets":  load_config("feature_sets"),
        "model_specs":   load_config("model_specs"),
        "pipeline":      load_config("pipeline"),
    }


def resolve_path(key: str, *, horizon: str | None = None,
                 set_id: str | None = None, sentiment_model: str | None = None,
                 absolute: bool = True) -> Path:
    """Resolve a path key from ``paths.yaml``.

    Templated paths (``..._pattern``) get their placeholders substituted from
    the keyword arguments.
    """
    paths_cfg = load_config("paths")
    if key not in paths_cfg:
        raise KeyError(f"Unknown path key: {key!r} (not found in configs/paths.yaml)")
    raw = paths_cfg[key]
    if not isinstance(raw, str):
        raise TypeError(f"Path key {key!r} is not a string: {raw!r}")
    fmt_kwargs = {}
    if horizon is not None:
        fmt_kwargs["horizon"] = horizon
    if set_id is not None:
        fmt_kwargs["set_id"] = set_id
    if sentiment_model is not None:
        fmt_kwargs["sentiment_model"] = sentiment_model
    if fmt_kwargs:
        raw = raw.format(**fmt_kwargs)
    p = Path(raw)
    if absolute and not p.is_absolute():
        p = project_root() / p
    return p


def horizons(kind: str = "raw_price") -> list[str]:
    """Return the list of horizons of a given kind.

    ``kind`` ∈ {raw_price, price_feature, sentiment_feature, final_model}.
    """
    cfg = load_config("horizons")
    if kind == "raw_price":
        return list(cfg.get("raw_price_horizons", {}).keys())
    key = f"{kind}_horizons"
    if key not in cfg:
        raise KeyError(f"Unknown horizon kind: {kind}")
    return list(cfg[key])


def tickers() -> list[str]:
    return list(load_config("coins")["tickers"])


def smoke_coins() -> list[str]:
    return list(load_config("coins").get("smoke_coins", ["BTC", "ETH"]))


def env_flag(name: str, default: bool = False) -> bool:
    """Return a truthy environment-variable flag, e.g. THESIS_DRY_RUN=1."""
    val = os.environ.get(name)
    if val is None:
        return default
    return val.strip().lower() in ("1", "true", "yes", "y", "on")
