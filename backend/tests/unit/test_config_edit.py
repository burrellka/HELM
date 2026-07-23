"""Guarded config-pipeline tests (14c §2.3).

The load-bearing cases: a broken JSON is refused, a reserved arg in a raw edit
is caught by the same parser the form uses, and flipping enable_dgpu_gtt off
raises a DANGER that the UI must make the operator acknowledge.
"""
from __future__ import annotations

import json

import pytest

from kblocallm.domain.config_edit import (
    Level,
    diff_config,
    parse_config,
    validate_global_config,
    validate_recipe_options,
)


def _codes(v, level=None):
    return {i.code for i in v.issues if level is None or i.level is level}


# -- parse gate ------------------------------------------------------------


def test_broken_json_is_refused_outright():
    """A malformed config.json is how you brick a server that will not
    restart. It must never leave the browser."""
    v = validate_global_config('{"port": 13305,}')  # trailing comma
    assert v.blocked
    assert "parse_error" in _codes(v)
    assert v.parsed is None


def test_non_object_json_is_refused():
    _, err = parse_config("[1, 2, 3]")
    assert err is not None and err.code == "not_an_object"


def test_valid_json_parses():
    parsed, err = parse_config('{"a": 1}')
    assert err is None and parsed == {"a": 1}


# -- Level A: recipe options -----------------------------------------------


def test_a_reserved_arg_in_a_raw_edit_is_caught_by_the_same_parser():
    """The whole point of routing raw edits through validation. --ctx-size in
    the raw box is exactly what the structured form blocks; the raw door must
    not be a way around it."""
    v = validate_recipe_options('{"llamacpp_args": "--ctx-size 4096"}')
    assert v.blocked and "reserved_args" in _codes(v, Level.ERROR)


@pytest.mark.parametrize("body", [
    '{"ctx_size": 0}', '{"ctx_size": -1}', '{"ctx_size": "big"}', '{"ctx_size": true}',
])
def test_bad_ctx_size_is_blocked(body):
    assert validate_recipe_options(body).blocked


def test_ctx_over_ceiling_is_blocked():
    v = validate_recipe_options(
        '{"ctx_size": 999999, "llamacpp_args": "--parallel 1"}',
        max_context_window=131072,
    )
    assert v.blocked and "ctx_over_ceiling" in _codes(v, Level.ERROR)


def test_valid_recipe_options_pass():
    v = validate_recipe_options(
        '{"ctx_size": 524288, "llamacpp_args": "--parallel 8 --temp 1.0", "pinned": true}')
    assert not v.blocked


def test_unknown_recipe_key_warns_but_does_not_block():
    """An escape hatch should allow keys HELM does not know, but a typo like
    ctx_sixe should be visible rather than silently ignored by the server."""
    v = validate_recipe_options('{"ctx_sixe": 4096}')
    assert not v.blocked
    assert "unknown_key" in _codes(v, Level.WARNING)


def test_recipe_diff_is_computed():
    v = validate_recipe_options('{"ctx_size": 262144, "pinned": true}',
                                current={"ctx_size": 524288})
    kinds = {c.key: c.kind for c in v.changes}
    assert kinds["ctx_size"] == "changed"
    assert kinds["pinned"] == "added"


# -- Level B: global config, the dangerous keys ----------------------------


def test_disabling_gtt_is_a_named_danger():
    """The 80GB saga switch. It must fight the operator on the way out, by
    name, not just as a generic dangerous key."""
    v = validate_global_config(
        json.dumps({"enable_dgpu_gtt": False}),
        current={"enable_dgpu_gtt": True},
    )
    assert not v.blocked                 # allowed...
    assert v.needs_danger_ack            # ...but only with an explicit ack
    assert "gtt_disable" in _codes(v, Level.DANGER)
    assert "enable_dgpu_gtt" in v.dangerous_keys


def test_changing_a_port_is_dangerous():
    v = validate_global_config(json.dumps({"port": 9999}), current={"port": 13305})
    assert v.needs_danger_ack and "port" in v.dangerous_keys


def test_a_secret_shaped_key_is_dangerous_and_its_value_is_never_echoed():
    v = validate_global_config(
        json.dumps({"llamacpp": {}, "admin_api_key": "s3cr3t-value"}),
        current={"llamacpp": {}},
    )
    danger = next(i for i in v.issues if i.level is Level.DANGER)
    assert "credential" in danger.message
    assert "s3cr3t-value" not in danger.message  # never leak the value


def test_an_innocuous_change_needs_no_ack():
    v = validate_global_config(
        json.dumps({"log_level": "debug"}), current={"log_level": "info"},
    )
    assert not v.blocked and not v.needs_danger_ack


def test_enabling_gtt_is_not_flagged_as_the_saga():
    """Turning it ON is the fix, not the footgun. Only OFF gets the saga
    warning."""
    v = validate_global_config(
        json.dumps({"enable_dgpu_gtt": True}), current={"enable_dgpu_gtt": False},
    )
    assert "gtt_disable" not in _codes(v)


# -- diff ------------------------------------------------------------------


def test_diff_detects_add_remove_change():
    changes = {c.key: c.kind for c in diff_config(
        {"a": 1, "b": 2, "c": 3}, {"a": 1, "b": 20, "d": 4},
    )}
    assert changes == {"b": "changed", "c": "removed", "d": "added"}


def test_the_live_params_fixture_validates_clean():
    """The real config.json off the box, unchanged, must produce no dangers."""
    from pathlib import Path
    params = (Path(__file__).parent.parent / "fixtures" / "params.json").read_text()
    current = json.loads(params)
    v = validate_global_config(params, current=current)
    assert not v.blocked and not v.needs_danger_ack and not v.changes
