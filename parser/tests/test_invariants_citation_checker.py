"""Tests for `fnaf_parser.invariants.citation_checker`.

Exercises the verification ladder — coordinate resolution, param-index
bounds, expr_str match, SHORT label match, num_name match, quarantine
fallback — end-to-end against a small hand-built combined.json fixture.
No API key required; this is the zero-LLM-fallback layer.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from fnaf_parser.invariants.citation_checker import (
    _load_combined,
    check_records,
    verify_record,
)
from fnaf_parser.invariants.signatures import Citation, InvariantRecord


# --- Fixture builder ----------------------------------------------------


def _combined_doc() -> dict[str, Any]:
    """Minimal combined.json shape: two frames, each with one event
    group, each group with one condition and one action. Covers the
    EXPRESSION-param path (code 22) and the SHORT-label path (code 10).
    """
    return {
        "decoder_version": "test-fixture",
        "frames": [
            {
                "frame_index": 0,
                "frame_name": "Frame 0",
                "event_groups": [
                    {
                        "conditions": [
                            {
                                "num_name": "CompareCounter",
                                "parameters": [
                                    {
                                        "code": 22,
                                        "code_name": "EXP",
                                        "expr_str": "Night + 1",
                                    }
                                ],
                            }
                        ],
                        "actions": [
                            {
                                "num_name": "SetAlphaCoefficient",
                                "parameters": [
                                    {
                                        "code": 23,
                                        "code_name": "EXP2",
                                        "expr_str": "50 + Random 100",
                                    }
                                ],
                            }
                        ],
                    }
                ],
            },
            {
                "frame_index": 1,
                "frame_name": "Frame 1",
                "event_groups": [
                    {
                        "conditions": [
                            {
                                "num_name": "KeyPressed",
                                "parameters": [],
                            }
                        ],
                        "actions": [
                            {
                                "num_name": "SetAnimSequence",
                                "parameters": [
                                    {
                                        "code": 10,
                                        "code_name": "SHORT",
                                        "value": 3,
                                        "decoded": {"label": "walk"},
                                    }
                                ],
                            }
                        ],
                    }
                ],
            },
        ],
    }


@pytest.fixture
def combined_json(tmp_path: Path) -> Path:
    path = tmp_path / "combined.json"
    path.write_text(json.dumps(_combined_doc()), encoding="utf-8")
    return path


# --- Loader -------------------------------------------------------------


def test_load_combined_keyed_by_four_tuple(combined_json: Path):
    """The loader flattens to a dict keyed by (frame, eg, type, coa),
    with conditions and actions indexed independently."""
    combined = _load_combined(combined_json)
    assert (0, 0, "condition", 0) in combined
    assert (0, 0, "action", 0) in combined
    assert (1, 0, "condition", 0) in combined
    assert (1, 0, "action", 0) in combined
    assert len(combined) == 4


def test_load_combined_preserves_row_fields(combined_json: Path):
    combined = _load_combined(combined_json)
    cond_row = combined[(0, 0, "condition", 0)]
    assert cond_row["num_name"] == "CompareCounter"
    assert cond_row["parameters"][0]["expr_str"] == "Night + 1"


# --- Verification ladder -----------------------------------------------


def test_expr_str_match_accepts_verbatim(combined_json: Path):
    """Slice C happy path: pseudo_code is an exact copy of the cited
    param's expr_str — auto-accept via `expr_str_match`."""
    combined = _load_combined(combined_json)
    rec = InvariantRecord(
        claim="Alpha fades randomly between 50 and 150.",
        citations=[
            Citation(
                frame=0,
                event_group_index=0,
                type="action",
                cond_or_act_index=0,
                parameter_index=0,
            )
        ],
        pseudo_code="50 + Random 100",
        kind="numeric_assignment",
    )
    result = verify_record(combined, rec)
    assert result.accepted
    assert result.outcomes[0].reason == "expr_str_match"


def test_expr_str_match_is_case_and_whitespace_insensitive(combined_json: Path):
    """Cosmetic whitespace + case differences shouldn't tank the match
    — the normaliser collapses them."""
    combined = _load_combined(combined_json)
    rec = InvariantRecord(
        claim="Irrelevant.",
        citations=[
            Citation(
                frame=0,
                event_group_index=0,
                type="action",
                cond_or_act_index=0,
            )
        ],
        pseudo_code="50   +    random 100",
        kind="numeric_assignment",
    )
    result = verify_record(combined, rec)
    assert result.accepted
    assert result.outcomes[0].reason == "expr_str_match"


def test_short_label_match_accepts_quoted_label(combined_json: Path):
    """When a SHORT param's decoded label appears in the claim, we
    auto-accept via `short_label_match`."""
    combined = _load_combined(combined_json)
    rec = InvariantRecord(
        claim="Freddy's animation transitions to 'walk' state.",
        citations=[
            Citation(
                frame=1,
                event_group_index=0,
                type="action",
                cond_or_act_index=0,
                parameter_index=0,
            )
        ],
        pseudo_code="set anim_sequence(target=walk)",
        kind="state_transition",
    )
    result = verify_record(combined, rec)
    assert result.accepted
    assert result.outcomes[0].reason == "short_label_match"


def test_num_name_match_only_for_event_trigger_kinds(combined_json: Path):
    """A weak num_name-only signal is accepted for `event_trigger` /
    `state_transition` kinds, rejected for numeric kinds where
    expr_str is the operative gate."""
    combined = _load_combined(combined_json)
    # event_trigger kind -> accepted via num_name match
    good = InvariantRecord(
        claim="A KeyPressed event fires the group.",
        citations=[
            Citation(
                frame=1,
                event_group_index=0,
                type="condition",
                cond_or_act_index=0,
            )
        ],
        pseudo_code="KeyPressed -> trigger",
        kind="event_trigger",
    )
    assert verify_record(combined, good).outcomes[0].reason == "num_name_match"

    # numeric_assignment -> num_name alone is NOT enough
    weak = InvariantRecord(
        claim="Irrelevant KeyPressed claim.",
        citations=[
            Citation(
                frame=1,
                event_group_index=0,
                type="condition",
                cond_or_act_index=0,
            )
        ],
        pseudo_code="KeyPressed",
        kind="numeric_assignment",
    )
    assert verify_record(combined, weak).outcomes[0].reason == (
        "no_verification_strategy_matched"
    )


def test_coordinate_out_of_range_quarantined(combined_json: Path):
    combined = _load_combined(combined_json)
    rec = InvariantRecord(
        claim="Nonsense coordinate claim.",
        citations=[
            Citation(
                frame=99,
                event_group_index=99,
                type="condition",
                cond_or_act_index=99,
            )
        ],
        pseudo_code="x + 1",
        kind="numeric_assignment",
    )
    result = verify_record(combined, rec)
    assert not result.accepted
    assert result.outcomes[0].reason == "citation_coordinate_out_of_range"


def test_parameter_index_out_of_range_quarantined(combined_json: Path):
    combined = _load_combined(combined_json)
    rec = InvariantRecord(
        claim="Parameter index past end of params list.",
        citations=[
            Citation(
                frame=0,
                event_group_index=0,
                type="condition",
                cond_or_act_index=0,
                parameter_index=99,
            )
        ],
        pseudo_code="Night + 1",
        kind="numeric_assignment",
    )
    result = verify_record(combined, rec)
    assert not result.accepted
    assert result.outcomes[0].reason == "parameter_index_out_of_range"


def test_record_with_mixed_citations_rejected(combined_json: Path):
    """One good citation + one bad citation → the record goes to
    quarantine. A record is accepted iff *every* citation passes."""
    combined = _load_combined(combined_json)
    rec = InvariantRecord(
        claim="Night + 1 is the threshold.",
        citations=[
            Citation(
                frame=0,
                event_group_index=0,
                type="condition",
                cond_or_act_index=0,
            ),
            Citation(
                frame=99,
                event_group_index=99,
                type="condition",
                cond_or_act_index=99,
            ),
        ],
        pseudo_code="Night + 1",
        kind="numeric_assignment",
    )
    result = verify_record(combined, rec)
    assert not result.accepted
    reasons = [o.reason for o in result.outcomes]
    assert "expr_str_match" in reasons
    assert "citation_coordinate_out_of_range" in reasons


# --- Stream writer -----------------------------------------------------


def test_check_records_writes_both_artefacts(combined_json: Path, tmp_path: Path):
    """End-to-end: mixed-quality records stream through
    `check_records` and land in accepted/quarantine JSONL with correct
    counts."""
    accepted_path = tmp_path / "accepted.jsonl"
    quarantine_path = tmp_path / "quarantine.jsonl"
    records = [
        InvariantRecord(
            claim="Night + 1 advances the counter.",
            citations=[
                Citation(
                    frame=0,
                    event_group_index=0,
                    type="condition",
                    cond_or_act_index=0,
                )
            ],
            pseudo_code="Night + 1",
            kind="numeric_assignment",
        ),
        InvariantRecord(
            claim="Bad claim with bad coords.",
            citations=[
                Citation(
                    frame=99,
                    event_group_index=99,
                    type="condition",
                    cond_or_act_index=99,
                )
            ],
            pseudo_code="irrelevant",
            kind="numeric_assignment",
        ),
    ]
    accepted_count, quarantined_count = check_records(
        records, combined_json, accepted_path, quarantine_path
    )
    assert accepted_count == 1
    assert quarantined_count == 1
    assert accepted_path.exists()
    assert quarantine_path.exists()
    acc_lines = accepted_path.read_text().splitlines()
    q_lines = quarantine_path.read_text().splitlines()
    assert len(acc_lines) == 1
    assert len(q_lines) == 1
    # Every line is parseable JSON with the expected top-level keys
    for line in (*acc_lines, *q_lines):
        payload = json.loads(line)
        assert "record" in payload
        assert "outcomes" in payload


def test_check_records_overwrites_previous_run(
    combined_json: Path, tmp_path: Path
):
    """Output files are overwritten — the checker is idempotent per
    run, so a stale quarantine from a previous pipeline call doesn't
    silently inflate the rejection pile."""
    accepted_path = tmp_path / "accepted.jsonl"
    quarantine_path = tmp_path / "quarantine.jsonl"
    accepted_path.write_text("STALE LINE\n", encoding="utf-8")
    quarantine_path.write_text("STALE LINE\n", encoding="utf-8")
    check_records([], combined_json, accepted_path, quarantine_path)
    assert accepted_path.read_text() == ""
    assert quarantine_path.read_text() == ""
