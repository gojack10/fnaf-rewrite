"""Tests for `fnaf_parser.invariants.literal_gate`.

Layer 1 unit tests — no LLM calls, no API key. Layer 2 is exercised by
the probe artefact at `parser/temp-probes/probe_0_7_gate.py` and is
not unit-tested here.

Primary anchors:
- Session-11 regression: missing-`claim` must fail loud (was silent-pass).
- Session-10 refinement: single-word role-word subtraction requires a
  STANDALONE vocab match, never a word-in-compound match.
"""

from __future__ import annotations

import pytest

from fnaf_parser.invariants.literal_gate import (
    filter_role_words,
    layer1_verdict,
)


# --- Session-11 regression: wrong-shape input must not silent-pass -------


def test_layer1_verdict_missing_claim_raises():
    """Wrong-shape input (citation_checker's wrapped form) used to pass
    Layer 1 silently via `record.get("claim", "")` scanning an empty
    string. Fix: direct key access with a descriptive KeyError.
    """
    wrong_shape = {
        "record": {"claim": "nested claim here"},
        "verification": {"method": "expr_str_match"},
    }
    with pytest.raises(KeyError, match="'claim'"):
        layer1_verdict(
            wrong_shape,
            effective_role_words=["power"],
            effective_forbidden_words=["power"],
            role_budget=2,
        )


def test_layer1_verdict_missing_claim_error_reports_keys():
    """The KeyError should tell the operator what keys the record DID
    have — speeds up diagnosing shape mismatches.
    """
    wrong_shape = {"id": 1, "status": "quarantined"}
    with pytest.raises(KeyError, match="id.*status"):
        layer1_verdict(
            wrong_shape,
            effective_role_words=[],
            effective_forbidden_words=[],
            role_budget=2,
        )


# --- Happy-path: literal claim passes ------------------------------------


def test_layer1_verdict_pass_on_literal_claim():
    record = {
        "claim": (
            "When Timer on freddy bear reaches 300, "
            "SetAlterableValue on freddy bear to 1."
        ),
    }
    v = layer1_verdict(
        record,
        effective_role_words=["power", "jumpscare"],
        effective_forbidden_words=["power", "jumpscare"],
        role_budget=2,
    )
    assert v["passed"] is True
    assert v["reason"] == "layer1_pass"


def test_layer1_verdict_fails_on_forbidden_word():
    record = {
        "claim": "When power drops, Freddy triggers a jumpscare.",
    }
    v = layer1_verdict(
        record,
        effective_role_words=["power", "jumpscare"],
        effective_forbidden_words=["power", "jumpscare"],
        role_budget=2,
    )
    assert v["passed"] is False
    assert "forbidden_hits" in v["reason"]


def test_layer1_verdict_fails_on_role_budget_exceeded():
    """Non-forbidden role words still fail once the budget is exceeded."""
    record = {
        "claim": (
            "Night transitions to morning at 6am, the camera feed "
            "shows ambient footage and music plays."
        ),
    }
    v = layer1_verdict(
        record,
        effective_role_words=["night", "morning", "6am", "camera feed",
                              "ambient", "music"],
        effective_forbidden_words=[],
        role_budget=2,
    )
    assert v["passed"] is False
    assert "role_hits" in v["reason"]


# --- Session-10 refinement: single-word subtraction is standalone-only ---


def test_filter_role_words_single_word_standalone_only():
    """`power` in the compound `power left` must NOT subtract the
    single-word role `power`. Probe 0.8's buggy token-in-vocab-word
    approach did this and would have blinded the gate to 31 genuine
    `power` role-inferences that session-11 proved are real.
    """
    vocab = frozenset(["power left", "right door", "office"])
    kept, subtracted = filter_role_words(
        ["power", "jumpscare", "right door"], vocab
    )
    assert "power" in kept, (
        "single-word `power` must stay when vocab only has compound "
        "`power left`"
    )
    assert "jumpscare" in kept
    assert "right door" in subtracted, (
        "multi-word `right door` matches as substring in vocab phrase "
        "and must be subtracted"
    )


def test_filter_role_words_single_word_subtracts_when_standalone():
    """If `power` IS a standalone literal name, it SHOULD be subtracted."""
    vocab = frozenset(["power", "right door", "office"])
    kept, subtracted = filter_role_words(["power", "jumpscare"], vocab)
    assert "power" in subtracted
    assert "jumpscare" in kept


def test_filter_role_words_multi_word_substring_match():
    """Multi-word role_words subtract on any substring match inside a
    vocab phrase (e.g. vocab `left door open` covers role `left door`).
    """
    vocab = frozenset(["left door open", "office"])
    kept, subtracted = filter_role_words(["left door"], vocab)
    assert "left door" in subtracted
