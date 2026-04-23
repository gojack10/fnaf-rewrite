"""Tests for `fnaf_parser.invariants.slices` — DuckDB slice loaders.

Uses a small hand-built JSONL fixture so the tests stay fast,
self-contained, and decoupled from any specific `combined.jsonl`
SHA. The fixture shape mirrors the algorithm-emit contract: one row
per condition or action, flat fields matching what the Name Resolver
writes.

Slice loaders are the contract boundary between the Scout Pass SQL
(pinned on the SiftText nodes) and the DSPy.RLM pipeline; any
drift is a bug on one side or the other.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from fnaf_parser.invariants.slices import (
    load_slice_a,
    load_slice_b,
    load_slice_c,
)


# --- Fixture builder ----------------------------------------------------


def _row(
    *,
    frame: int,
    event_group_index: int,
    row_type: str,
    cond_or_act_index: int,
    num_name: str,
    object_type_name: str | None = None,
    object_name: str | None = None,
    parameters: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Build one flat row in the shape emit.py writes."""
    return {
        "frame": frame,
        "frame_name": f"Frame {frame}",
        "event_group_index": event_group_index,
        "type": row_type,
        "cond_or_act_index": cond_or_act_index,
        "num_name": num_name,
        "object_type_name": object_type_name,
        "object_name": object_name,
        "parameters": parameters or [],
    }


@pytest.fixture
def fixture_jsonl(tmp_path: Path) -> Path:
    """A 6-row JSONL covering the axes each slice needs:

    - Two `CompareCounter` conditions — Slice A bucketing check.
    - Two rows on object `("Active", "Freddy")` — Slice B bucketing.
    - Two rows with EXPRESSION params (codes 22, 23) — Slice C filter.
    - One row with no EXPRESSION params — Slice C must exclude it.
    """
    rows = [
        _row(
            frame=0,
            event_group_index=0,
            row_type="condition",
            cond_or_act_index=0,
            num_name="CompareCounter",
            object_type_name="Counter",
            object_name="Night",
            parameters=[
                {"code": 22, "code_name": "EXP", "expr_str": "Night + 1"},
            ],
        ),
        _row(
            frame=0,
            event_group_index=0,
            row_type="action",
            cond_or_act_index=0,
            num_name="SetAlphaCoefficient",
            object_type_name="Active",
            object_name="Freddy",
            parameters=[
                {"code": 23, "code_name": "EXP2", "expr_str": "50 + Random 100"},
            ],
        ),
        _row(
            frame=0,
            event_group_index=1,
            row_type="condition",
            cond_or_act_index=0,
            num_name="CompareCounter",
            object_type_name="Counter",
            object_name="Hour",
            parameters=[
                {"code": 22, "code_name": "EXP", "expr_str": "Hour > 3"},
            ],
        ),
        _row(
            frame=1,
            event_group_index=0,
            row_type="action",
            cond_or_act_index=0,
            num_name="SetAnimSequence",
            object_type_name="Active",
            object_name="Freddy",
            parameters=[
                {"code": 10, "code_name": "SHORT", "value": 3, "decoded": {"label": "walk"}},
            ],
        ),
        _row(
            frame=1,
            event_group_index=1,
            row_type="condition",
            cond_or_act_index=0,
            num_name="TimerEquals",
            object_type_name=None,  # system row
            object_name=None,
            parameters=[
                {"code": 2, "timer": 2000},
            ],
        ),
        _row(
            frame=2,
            event_group_index=0,
            row_type="action",
            cond_or_act_index=0,
            num_name="SetPosition",
            object_type_name="Active",
            object_name="Bonnie",
            parameters=[
                {"code": 45, "code_name": "EXP45", "expr_str": "X + 10"},
            ],
        ),
    ]
    path = tmp_path / "combined.jsonl"
    path.write_text(
        "\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8"
    )
    return path


# --- Slice A ------------------------------------------------------------


def test_load_slice_a_buckets_by_type_and_num_name(fixture_jsonl: Path):
    """Slice A groups rows by `(type, num_name)` with a count column."""
    tickets = load_slice_a(fixture_jsonl)
    keys = {
        (t["bucket_key"]["type"], t["bucket_key"]["num_name"]) for t in tickets
    }
    # Six rows in fixture, five distinct (type, num_name) buckets:
    assert keys == {
        ("condition", "CompareCounter"),
        ("action", "SetAlphaCoefficient"),
        ("action", "SetAnimSequence"),
        ("condition", "TimerEquals"),
        ("action", "SetPosition"),
    }


def test_load_slice_a_orders_by_count_desc(fixture_jsonl: Path):
    """The biggest bucket comes first — a downstream dispatcher relies
    on this ordering so the most-cited op gets the earliest attempt."""
    tickets = load_slice_a(fixture_jsonl)
    counts = [t["occurrence_count"] for t in tickets]
    assert counts == sorted(counts, reverse=True)
    # CompareCounter is the only 2-occurrence bucket
    assert tickets[0]["bucket_key"]["num_name"] == "CompareCounter"
    assert tickets[0]["occurrence_count"] == 2


def test_load_slice_a_rows_carry_full_schema(fixture_jsonl: Path):
    """Each row inside a Slice A ticket has the field set the DSPy
    Signature expects (frame, event_group_index, type,
    cond_or_act_index, object_*, parameters)."""
    tickets = load_slice_a(fixture_jsonl)
    cc = next(
        t for t in tickets if t["bucket_key"]["num_name"] == "CompareCounter"
    )
    row = cc["rows"][0]
    assert {
        "frame",
        "event_group_index",
        "type",
        "cond_or_act_index",
        "object_type_name",
        "object_name",
        "parameters",
    } <= set(row.keys())


# --- Slice B ------------------------------------------------------------


def test_load_slice_b_buckets_by_object(fixture_jsonl: Path):
    """Slice B groups rows by `(object_type_name, object_name)`. Nulls
    are coalesced to `(system)`."""
    tickets = load_slice_b(fixture_jsonl)
    keys = {
        (t["bucket_key"]["object_type_name"], t["bucket_key"]["object_name"])
        for t in tickets
    }
    assert ("Counter", "Night") in keys
    assert ("Counter", "Hour") in keys
    assert ("Active", "Freddy") in keys
    assert ("Active", "Bonnie") in keys
    # System row: object_type_name / object_name both null → coalesced
    assert ("(system)", "(system)") in keys


def test_load_slice_b_freddy_bucket_has_two_rows(fixture_jsonl: Path):
    """The two `Active/Freddy` rows (one condition-bearing action +
    one anim-sequence) must land in the same bucket."""
    tickets = load_slice_b(fixture_jsonl)
    freddy = next(
        t
        for t in tickets
        if (
            t["bucket_key"]["object_type_name"] == "Active"
            and t["bucket_key"]["object_name"] == "Freddy"
        )
    )
    assert freddy["occurrence_count"] == 2


# --- Slice C ------------------------------------------------------------


def test_load_slice_c_only_rows_with_expression_codes(fixture_jsonl: Path):
    """Slice C keeps only rows whose parameters include at least one
    code in {22, 23, 27, 45}. The SetAnimSequence row (code 10) and
    TimerEquals row (code 2) must be filtered out."""
    tickets = load_slice_c(fixture_jsonl)
    assert len(tickets) == 4  # CC Night, CC Hour, SetAlpha, SetPosition
    nums = [t["num_name"] for t in tickets]
    assert "SetAnimSequence" not in nums
    assert "TimerEquals" not in nums


def test_load_slice_c_exposes_expr_params_only(fixture_jsonl: Path):
    """The `expr_params` column must hold only the matching param
    subset, not the full parameters list. That's the scan the pilot
    consumes directly."""
    tickets = load_slice_c(fixture_jsonl)
    for t in tickets:
        for p in t["expr_params"]:
            assert p["code"] in (22, 23, 27, 45)


def test_load_slice_c_keys_match_citation_schema(fixture_jsonl: Path):
    """Every Slice C ticket carries (frame, event_group_index, type,
    cond_or_act_index) — matching the Citation Pydantic model exactly.
    If this drifts, the pilot's citations won't round-trip through
    Citation Checker."""
    tickets = load_slice_c(fixture_jsonl)
    for t in tickets:
        assert isinstance(t["frame"], int)
        assert isinstance(t["event_group_index"], int)
        assert t["type"] in ("condition", "action")
        assert isinstance(t["cond_or_act_index"], int)


def test_load_slice_c_deterministic_order(fixture_jsonl: Path):
    """Ordered by (frame, event_group_index, type, cond_or_act_index)
    — stable across runs so quarantine diffs are reviewable."""
    tickets = load_slice_c(fixture_jsonl)
    ordered = sorted(
        tickets,
        key=lambda t: (
            t["frame"],
            t["event_group_index"],
            t["type"],
            t["cond_or_act_index"],
        ),
    )
    assert tickets == ordered
