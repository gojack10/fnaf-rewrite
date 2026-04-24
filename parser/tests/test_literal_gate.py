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
    build_compound_vocab_mask,
    filter_role_words,
    layer1_verdict,
    mask_claim,
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


# --- Session-14 refinement: compound-vocab mask pre-pass ------------------


def test_build_compound_vocab_mask_walks_nested_structure():
    """Recursive walk must pick up multi-word strings from every level
    of the record — not just top-level `object_name` / `num_name`.
    """
    rows = [
        {
            "object_name": "power left",  # multi-word
            "num_name": "CompareCounter",  # single-word, skip
            "parameters": [
                {"name": "powerdown", "label": "power off"},  # multi in label
                {"expr_str": "power left.CounterValue <= 10"},
            ],
        },
        {
            "object_name": "freddy bear",
            "parameters": [
                {"label": "  "},  # whitespace-only after strip, skip
                {"label": "darkness music"},
            ],
        },
    ]
    vocab = build_compound_vocab_mask(rows)
    assert "power left" in vocab
    assert "power off" in vocab
    assert "power left.countervalue <= 10" in vocab
    assert "freddy bear" in vocab
    assert "darkness music" in vocab
    # Single-word strings are skipped; so is whitespace-only.
    assert "comparecounter" not in vocab
    assert "powerdown" not in vocab
    assert "" not in vocab


def test_mask_claim_empty_vocab_is_identity_lowercase():
    """Empty compound vocab => lowercase claim, no hits. Backward-compat
    path for unit tests and session-10 callers that pre-date the mask.
    """
    masked, hits = mask_claim("When POWER drops, Freddy scares.", frozenset())
    assert masked == "when power drops, freddy scares."
    assert hits == []


def test_mask_claim_absorbs_literal_compound():
    """`power` inside the literal compound `power left` must be masked,
    so a downstream `\\bpower\\b` scan sees no role-word hit.
    """
    vocab = frozenset(["power left", "power left 2", "freddy bear"])
    claim = (
        "When counter 'power left' < 0, "
        "set counter 'power left' to 0 and counter 'power left 2' to 0."
    )
    masked, hits = mask_claim(claim, vocab)
    # Original role-word is gone.
    import re as _re
    assert not _re.search(r"\bpower\b", masked)
    # Both literal compounds fired.
    assert "power left" in hits
    assert "power left 2" in hits


def test_mask_claim_longest_first_prevents_prefix_consumption():
    """If `power left` masks before `power left 2`, the longer literal
    is partially rewritten and never matches. Longest-first ordering
    guarantees the full identifier is absorbed first.
    """
    vocab = frozenset(["power left", "power left 2"])
    claim = "counter 'power left 2' was set."
    masked, hits = mask_claim(claim, vocab)
    # The longer phrase must land in hits — if prefix-consumption won,
    # we'd see only the shorter phrase followed by a residual ' 2'.
    assert "power left 2" in hits


def test_layer1_verdict_compound_vocab_unblocks_literal_power_compound():
    """Session-14 end-to-end: claim that references literal MFA object
    `power left` must pass Layer 1 when compound_vocab includes that
    phrase. Without the mask (session-10 behavior) it would fail on
    `\\bpower\\b`.
    """
    record = {
        "claim": (
            "When counter 'power left' < 0, "
            "set counter 'power left' to 0."
        ),
    }
    vocab = frozenset(["power left"])
    # Without mask: fails on power role-word.
    v_before = layer1_verdict(
        record,
        effective_role_words=["power"],
        effective_forbidden_words=["power"],
        role_budget=2,
    )
    assert v_before["passed"] is False
    # With mask: passes, and mask_hits records the absorbing phrase.
    v_after = layer1_verdict(
        record,
        effective_role_words=["power"],
        effective_forbidden_words=["power"],
        role_budget=2,
        compound_vocab=vocab,
    )
    assert v_after["passed"] is True
    assert v_after["reason"] == "layer1_pass"
    assert "power left" in v_after["scan"]["mask_hits"]


def test_layer1_verdict_compound_vocab_does_not_rescue_pure_role_claim():
    """Mask only absorbs literal compounds. A claim that uses `power`
    standalone (no matching vocab phrase) must still fail, regardless
    of how rich compound_vocab is elsewhere.
    """
    record = {
        "claim": "When power drops, Freddy triggers a jumpscare.",
    }
    vocab = frozenset(["power left", "power left 2", "freddy bear"])
    v = layer1_verdict(
        record,
        effective_role_words=["power", "jumpscare"],
        effective_forbidden_words=["power", "jumpscare"],
        role_budget=2,
        compound_vocab=vocab,
    )
    assert v["passed"] is False
    assert "forbidden_hits" in v["reason"]
