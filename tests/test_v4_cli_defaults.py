"""Aufgabe 5 — pin the v4 CLI defaults.

A bare ``run-models`` invocation must produce the canonical v4 pipeline:
* model_type        = panel_logit
* panel_mode        = ticker_fixed_effects
* train_window      = rolling_fixed
* rolling_window_days = 180   (calendar days, NOT timestamps)
* tune_hyperparams  = True    (BooleanOptionalAction default)
* hpo_objective     = log_loss
* class_weight grid = [None]  (production)

And the historical opt-outs must still work via explicit flags.
"""
from __future__ import annotations

import pytest

from thesis_pipeline import cli
from thesis_pipeline.modeling import (
    run_models as rm,
    hyperparameter_tuning as ht,
)


# ---------------------------------------------------------------------------
# argparse defaults — module parser
# ---------------------------------------------------------------------------

def test_module_parser_default_model_type_is_panel_logit():
    ns = rm.build_parser().parse_args([])
    assert ns.model_type == "panel_logit"


def test_module_parser_default_panel_mode_is_ticker_fixed_effects():
    ns = rm.build_parser().parse_args([])
    assert ns.panel_mode == "ticker_fixed_effects"


def test_module_parser_default_train_window_is_rolling_fixed():
    ns = rm.build_parser().parse_args([])
    assert ns.train_window == "rolling_fixed"


def test_module_parser_default_rolling_window_is_180_calendar_days():
    ns = rm.build_parser().parse_args([])
    assert ns.rolling_window_days == 180.0
    # Crucially NOT a timestamp count — see CLI help text for why.
    assert ns.rolling_window_timestamps is None


def test_module_parser_default_tune_hyperparams_is_true():
    """v4 flips --tune-hyperparams to BooleanOptionalAction(default=True)."""
    ns = rm.build_parser().parse_args([])
    assert ns.tune_hyperparams is True


def test_module_parser_no_tune_hyperparams_disables_hpo():
    ns = rm.build_parser().parse_args(["--no-tune-hyperparams"])
    assert ns.tune_hyperparams is False


def test_module_parser_default_hpo_objective_is_log_loss():
    ns = rm.build_parser().parse_args([])
    assert ns.hpo_objective == "log_loss"


# ---------------------------------------------------------------------------
# argparse defaults — package CLI surface
# ---------------------------------------------------------------------------

def _cli_run_models_ns(extra_argv: list[str] | None = None):
    """Parse the package CLI's ``run-models`` subcommand and return its
    namespace (so we can assert against the actual defaults a user sees
    when they type ``python -m thesis_pipeline.cli run-models``)."""
    parser = cli.build_parser()
    argv = ["run-models"] + (extra_argv or [])
    return parser.parse_args(argv)


def test_cli_default_model_type_is_panel_logit():
    ns = _cli_run_models_ns([])
    assert ns.model_type == "panel_logit"


def test_cli_default_panel_mode_is_ticker_fixed_effects():
    ns = _cli_run_models_ns([])
    assert ns.panel_mode == "ticker_fixed_effects"


def test_cli_default_train_window_is_rolling_fixed_180_days():
    ns = _cli_run_models_ns([])
    assert ns.train_window == "rolling_fixed"
    assert ns.rolling_window_days == 180.0
    assert ns.rolling_window_timestamps is None


def test_cli_default_tune_hyperparams_is_true():
    ns = _cli_run_models_ns([])
    assert ns.tune_hyperparams is True


def test_cli_no_tune_hyperparams_disables_hpo():
    ns = _cli_run_models_ns(["--no-tune-hyperparams"])
    assert ns.tune_hyperparams is False


def test_cli_default_hpo_objective_is_log_loss():
    ns = _cli_run_models_ns([])
    assert ns.hpo_objective == "log_loss"


# ---------------------------------------------------------------------------
# HPO config: YAML defaults + fallback objective
# ---------------------------------------------------------------------------

def test_hpo_config_default_enabled_is_true_in_v4_yaml():
    """A bare load (no overrides) must reflect the v4 default-on stance."""
    cfg = ht.load_hpo_config(model_specs=None)
    # The model_specs.yaml shipped with the repo carries enabled: true /
    # objective: log_loss; the helper must surface those.
    assert cfg["enabled"] is True
    assert cfg["objective"] == "log_loss"


def test_hpo_config_class_weight_grid_is_null_only_in_v4_yaml():
    cfg = ht.load_hpo_config(model_specs=None)
    cw = cfg["search_space"]["class_weight"]
    assert cw == [None], f"v4 production class_weight grid must be [None], got {cw}"


def test_hpo_variant_label_fallback_is_log_loss():
    """The label fallback (when objective is None) is hpo_logloss in v4,
    matching the new default objective. v3 used hpo_brier."""
    assert ht.hpo_variant_label(enabled=True, objective=None) == "hpo_logloss"


def test_hpo_config_negative_override_actually_disables():
    """--no-tune-hyperparams must really disable HPO regardless of the YAML."""
    cfg = ht.load_hpo_config(enabled_override=False)
    assert cfg["enabled"] is False


def test_hpo_config_positive_override_keeps_enabled():
    cfg = ht.load_hpo_config(enabled_override=True)
    assert cfg["enabled"] is True


# ---------------------------------------------------------------------------
# Programmatic run() wrapper defaults match the CLI
# ---------------------------------------------------------------------------

def test_programmatic_run_defaults_match_cli(monkeypatch):
    captured: dict = {}

    def fake_main(argv):
        captured["argv"] = list(argv) if argv else []
        return 0

    monkeypatch.setattr(rm, "main", fake_main)
    rc = rm.run(horizon="1d", set_id="ECON", dry_run=True)
    assert rc == 0
    argv = captured["argv"]
    # All canonical v4 defaults must be forwarded as explicit flags.
    assert "--model-type" in argv and argv[argv.index("--model-type") + 1] == "panel_logit"
    assert "--panel-mode" in argv and argv[argv.index("--panel-mode") + 1] == "ticker_fixed_effects"
    assert "--train-window" in argv and argv[argv.index("--train-window") + 1] == "rolling_fixed"
    assert "--rolling-window-days" in argv and float(argv[argv.index("--rolling-window-days") + 1]) == 180.0
    assert "--tune-hyperparams" in argv
    assert "--no-tune-hyperparams" not in argv
    assert "--hpo-objective" in argv and argv[argv.index("--hpo-objective") + 1] == "log_loss"


def test_programmatic_run_no_tune_emits_negative_flag(monkeypatch):
    captured: dict = {}
    monkeypatch.setattr(rm, "main", lambda argv: captured.setdefault("argv", list(argv or [])) or 0)
    rm.run(horizon="1d", set_id="ECON", tune_hyperparams=False, dry_run=True)
    assert "--no-tune-hyperparams" in captured["argv"]
    assert "--tune-hyperparams" not in captured["argv"]


# ---------------------------------------------------------------------------
# rolling_fixed without either window size — clear error
# ---------------------------------------------------------------------------

def test_panel_rolling_fixed_with_no_window_size_returns_clear_error(monkeypatch, capsys):
    """If the user explicitly nulls out both rolling-window flags AND keeps
    train_window=rolling_fixed, the panel runner must fail fast with a
    message that names both flags. The v4 default chains
    rolling_window_days=180, so the only way to hit this is to ask for it.
    """
    from thesis_pipeline.modeling import panel_logit as pl
    import argparse
    ns = argparse.Namespace(
        panel_mode="ticker_fixed_effects",
        train_window="rolling_fixed",
        rolling_window_timestamps=None,
        rolling_window_days=None,
        # Plus the fields _run_panel reads down the line — we never reach
        # them because the rolling-window guard returns early.
        tune_hyperparams=False, hpo_objective=None, hpo_config=None,
        hpo_grid_C=None, hpo_class_weight=None,
        dry_run=True, horizon="1d", set_id="ECON", sentiment_model="-",
        coins=None, C=1.0,
        checkpoint=False, resume=False, checkpoint_dir="ck",
        checkpoint_chunk_size=20, clear_checkpoints=False,
    )
    rc = pl._run_panel(ns)
    err = capsys.readouterr().out
    assert rc == 2
    assert "--rolling-window-timestamps" in err
    assert "--rolling-window-days" in err


# ---------------------------------------------------------------------------
# Section 5.7 — structural-break diagnostic stays advisory
# ---------------------------------------------------------------------------

def test_structural_breaks_diagnostic_stays_advisory():
    """The windowing module documents that structural-break diagnostics
    never set the rolling-window length automatically. The v4 defaults
    don't change that — rolling_window_days=180 is a fixed knob, not a
    break-informed value."""
    from thesis_pipeline.modeling import windowing as win
    # The function's docstring is the spec. Make sure the
    # "no default size by design" phrasing is intact.
    doc = (win.select_panel_train_window.__doc__ or "") + (win.__doc__ or "")
    assert "no default size by design" in doc or "advisory only" in doc


# ---------------------------------------------------------------------------
# Profile block in model_specs.yaml
# ---------------------------------------------------------------------------

def test_model_specs_yaml_carries_documented_final_profile():
    """Aufgabe 5.6 — keep a documented `final` profile in model_specs.yaml
    so a reproducer can pin the canonical configuration in one place."""
    from thesis_pipeline.config import load_config
    cfg = load_config("model_specs")
    profile = cfg.get("profiles", {}).get("final", {})
    assert profile.get("model_type")          == "panel_logit"
    assert profile.get("panel_mode")          == "ticker_fixed_effects"
    assert profile.get("train_window")        == "rolling_fixed"
    assert profile.get("rolling_window_days") == 180
    assert profile.get("tune_hyperparams") is True
    assert profile.get("hpo_objective")       == "log_loss"
