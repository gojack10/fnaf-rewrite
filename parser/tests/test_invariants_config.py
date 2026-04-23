"""Tests for `fnaf_parser.invariants.config`.

All deterministic — no API key required. Covers env-var contract,
default slug fallback, missing-key error surface, and the `redacted`
mask used in log lines.
"""

from __future__ import annotations

import pytest

from fnaf_parser.invariants.config import (
    OPENROUTER_API_BASE,
    OpenRouterConfig,
    OpenRouterConfigError,
    load_config,
)


# --- Happy path ---------------------------------------------------------


def test_load_config_from_injected_env():
    """A fully-populated env produces a frozen OpenRouterConfig with
    both model slugs resolved."""
    env = {
        "OPENROUTER_API_KEY": "sk-or-v1-ABCDEFGHIJKLMNOP",
        "FNAF_LINE_COOK_MODEL": "openrouter/test/line-cook",
        "FNAF_HEAD_CHEF_MODEL": "openrouter/test/head-chef",
    }
    cfg = load_config(env=env)
    assert cfg.api_key == "sk-or-v1-ABCDEFGHIJKLMNOP"
    assert cfg.line_cook_model == "openrouter/test/line-cook"
    assert cfg.head_chef_model == "openrouter/test/head-chef"
    assert cfg.api_base == OPENROUTER_API_BASE


def test_default_models_applied_when_only_key_set():
    """If only the key is set, both model slugs fall back to the pilot
    default — single-model mode."""
    cfg = load_config(env={"OPENROUTER_API_KEY": "sk-or-v1-xxxx"})
    assert cfg.line_cook_model == cfg.head_chef_model
    assert cfg.line_cook_model.startswith("openrouter/")


def test_head_chef_defaults_to_line_cook_when_only_line_cook_overridden():
    """Overriding line cook but not head chef copies line cook to head
    chef — we don't silently pick a different default for the head
    chef when the user is intentionally pinning a model."""
    env = {
        "OPENROUTER_API_KEY": "sk-or-v1-xxxx",
        "FNAF_LINE_COOK_MODEL": "openrouter/custom/only-line-cook",
    }
    cfg = load_config(env=env)
    assert cfg.line_cook_model == "openrouter/custom/only-line-cook"
    assert cfg.head_chef_model == "openrouter/custom/only-line-cook"


def test_frozen_dataclass_rejects_mutation():
    """Frozen + slots — accidental mutation must raise to prevent stale
    config leakage across a multi-slice run."""
    cfg = load_config(env={"OPENROUTER_API_KEY": "sk-or-v1-yyyy"})
    with pytest.raises(Exception):
        cfg.api_key = "tampered"  # type: ignore[misc]


# --- Missing-key surface ------------------------------------------------


def test_missing_key_raises_openrouter_config_error():
    """Empty env → OpenRouterConfigError, NOT a generic KeyError — the
    CLI layer catches this specific type and maps it to exit 3."""
    with pytest.raises(OpenRouterConfigError) as exc_info:
        load_config(env={})
    message = str(exc_info.value)
    assert "OPENROUTER_API_KEY" in message
    assert "export" in message  # remediation hint present
    assert "openrouter.ai/keys" in message  # help link present


def test_whitespace_only_key_treated_as_missing():
    """A key of just whitespace should surface the same error — silent
    string-strip would let a copy-paste of ` ` through and produce a
    DSPy 401 hundreds of tokens later."""
    with pytest.raises(OpenRouterConfigError):
        load_config(env={"OPENROUTER_API_KEY": "   "})


# --- Redaction ----------------------------------------------------------


def test_redacted_masks_everything_except_last_four():
    """`redacted()` is what logs / CLI banners use; the full key must
    NEVER appear there."""
    cfg = OpenRouterConfig(
        api_key="sk-or-v1-SECRETKEY1234",
        line_cook_model="m",
        head_chef_model="m",
    )
    red = cfg.redacted()
    assert red["api_key"] == "****1234"
    assert "SECRETKEY" not in red["api_key"]
    assert red["line_cook_model"] == "m"
    assert red["api_base"] == OPENROUTER_API_BASE


def test_redacted_handles_short_keys_safely():
    """A mal-short key (shouldn't happen in prod but defensive) still
    produces `****` without leaking. Avoids an IndexError on [-4:]."""
    cfg = OpenRouterConfig(
        api_key="xyz",  # under 4 chars
        line_cook_model="m",
        head_chef_model="m",
    )
    red = cfg.redacted()
    assert red["api_key"] == "****"
