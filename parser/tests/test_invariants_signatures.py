"""Tests for `fnaf_parser.invariants.signatures` — the Pydantic schema
contract that gates every line-cook output.

No API key required. We verify shape, validator behaviour, and the
enum membership for `kind` — malformed records must raise ValidationError
so DSPy's auto-re-ask kicks in rather than silently passing garbage
downstream.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from fnaf_parser.invariants.signatures import (
    Citation,
    ExtractInvariants,
    InvariantRecord,
)


# --- Citation shape ----------------------------------------------------


def test_citation_happy_path():
    """The canonical 4-field citation constructs cleanly and keeps
    parameter_index as None when omitted."""
    c = Citation(
        frame=3, event_group_index=2, type="action", cond_or_act_index=0
    )
    assert c.frame == 3
    assert c.event_group_index == 2
    assert c.type == "action"
    assert c.cond_or_act_index == 0
    assert c.parameter_index is None


def test_citation_with_parameter_index():
    c = Citation(
        frame=0,
        event_group_index=0,
        type="condition",
        cond_or_act_index=0,
        parameter_index=2,
    )
    assert c.parameter_index == 2


@pytest.mark.parametrize("bad_type", ["conditional", "Action", "", " "])
def test_citation_rejects_bad_type_strings(bad_type: str):
    """`type` is a Literal — only the exact singular forms pass."""
    with pytest.raises(ValidationError):
        Citation(
            frame=0,
            event_group_index=0,
            type=bad_type,  # type: ignore[arg-type]
            cond_or_act_index=0,
        )


@pytest.mark.parametrize(
    "field,bad_value",
    [
        ("frame", -1),
        ("event_group_index", -5),
        ("cond_or_act_index", -1),
        ("parameter_index", -2),
    ],
)
def test_citation_rejects_negative_indices(field: str, bad_value: int):
    """Every index is `ge=0`; negatives must loud-fail."""
    payload: dict = {
        "frame": 0,
        "event_group_index": 0,
        "type": "condition",
        "cond_or_act_index": 0,
    }
    payload[field] = bad_value
    with pytest.raises(ValidationError):
        Citation(**payload)


# --- InvariantRecord shape ----------------------------------------------


def _valid_citation() -> Citation:
    return Citation(
        frame=0, event_group_index=0, type="condition", cond_or_act_index=0
    )


def test_invariant_record_happy_path():
    r = InvariantRecord(
        claim="Timer equals two seconds advances the frame.",
        citations=[_valid_citation()],
        pseudo_code="timer == 2000 -> NextFrame",
        kind="event_trigger",
    )
    assert r.claim.startswith("Timer")
    assert len(r.citations) == 1
    assert r.confidence is None


def test_invariant_record_strips_whitespace():
    """The `_strip_whitespace` validator must remove leading/trailing
    whitespace on claim + pseudo_code so prompt-echo newlines never
    cause a Citation Checker string-match miss."""
    r = InvariantRecord(
        claim="  has whitespace  \n",
        citations=[_valid_citation()],
        pseudo_code="\t50 + Random 100\n",
        kind="numeric_assignment",
    )
    assert r.claim == "has whitespace"
    assert r.pseudo_code == "50 + Random 100"


def test_invariant_record_requires_at_least_one_citation():
    """`citations: list[Citation] = Field(min_length=1)` — empty list
    must be rejected, since Citation Checker can't verify an uncited
    claim."""
    with pytest.raises(ValidationError):
        InvariantRecord(
            claim="Some claim.",
            citations=[],
            pseudo_code="x = 1",
            kind="numeric_assignment",
        )


def test_invariant_record_claim_min_length():
    """Very short claims (1-3 chars) are usually model typos; reject."""
    with pytest.raises(ValidationError):
        InvariantRecord(
            claim="ab",
            citations=[_valid_citation()],
            pseudo_code="x",
            kind="other",
        )


def test_invariant_record_rejects_unknown_kind():
    """`kind` is a closed enum; arbitrary strings must loud-fail."""
    with pytest.raises(ValidationError):
        InvariantRecord(
            claim="A valid claim.",
            citations=[_valid_citation()],
            pseudo_code="x",
            kind="arbitrary_kind",  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    "kind",
    [
        "numeric_assignment",
        "numeric_comparison",
        "state_transition",
        "event_trigger",
        "other",
    ],
)
def test_invariant_record_accepts_all_enum_kinds(kind: str):
    InvariantRecord(
        claim="A valid claim.",
        citations=[_valid_citation()],
        pseudo_code="x",
        kind=kind,  # type: ignore[arg-type]
    )


def test_invariant_record_confidence_range():
    """`confidence` clamps to [0.0, 1.0] via Field constraints."""
    # inside range
    InvariantRecord(
        claim="A valid claim.",
        citations=[_valid_citation()],
        pseudo_code="x",
        kind="other",
        confidence=0.5,
    )
    # out of range
    with pytest.raises(ValidationError):
        InvariantRecord(
            claim="A valid claim.",
            citations=[_valid_citation()],
            pseudo_code="x",
            kind="other",
            confidence=1.5,
        )
    with pytest.raises(ValidationError):
        InvariantRecord(
            claim="A valid claim.",
            citations=[_valid_citation()],
            pseudo_code="x",
            kind="other",
            confidence=-0.1,
        )


# --- Signature surface --------------------------------------------------


def test_extract_invariants_signature_exposes_tickets_and_records():
    """The DSPy Signature declares exactly `tickets` input + `records`
    output. If a future refactor renames them, every pipeline prompt
    must update in lockstep."""
    fields = ExtractInvariants.model_fields
    assert "tickets" in fields
    assert "records" in fields
