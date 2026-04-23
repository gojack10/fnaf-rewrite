"""Tests for `fnaf_parser.invariants.handwritten`.

Each record is pinned to a real `combined.json` coordinate and is
expected to auto-accept via Citation Checker's `expr_str_match` rung
of the verification ladder. If any of these tests start failing, one
of two things happened:

1. The Algorithm Emission changed schema — `expr_str` got renamed /
   reshaped / stopped being written. Citation Checker ladder change
   needed upstream.
2. A pack change shifted event-group indices — the handwritten
   citations now point at different rows. The records need re-pinning
   against the new pack.

Both are loud-fail conditions worth surfacing at CI. No API key
required — `citation_checker` is the zero-LLM-fallback layer.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from fnaf_parser.invariants.citation_checker import (
    _load_combined,
    check_records,
    verify_record,
)
from fnaf_parser.invariants.handwritten import HANDWRITTEN
from fnaf_parser.invariants.signatures import InvariantRecord


COMBINED_JSON = Path("out/algorithm/combined.json")


# --- Shape ---------------------------------------------------------------


def test_handwritten_list_is_nonempty_and_typed():
    """The handwritten batch exposes `list[InvariantRecord]` and ships
    at least the initial five pack-known invariants. A drop below five
    is almost certainly an accidental deletion, not an intentional
    retraction — bump the threshold consciously if you want to ship
    fewer."""
    assert isinstance(HANDWRITTEN, list)
    assert len(HANDWRITTEN) >= 5
    for rec in HANDWRITTEN:
        assert isinstance(rec, InvariantRecord)


def test_handwritten_kinds_are_numeric_only():
    """All current handwritten records are numeric (assignment or
    comparison). If we add a state_transition / event_trigger record,
    this test pins the change — the kind mix affects the Rust Test
    Emission's downstream routing, so it's worth seeing it change."""
    kinds = {rec.kind for rec in HANDWRITTEN}
    assert kinds <= {"numeric_assignment", "numeric_comparison"}


# --- End-to-end verification against the real pack -----------------------


@pytest.fixture(scope="module")
def real_combined() -> dict:
    """Load combined.json once for the module. Skips cleanly if the
    algorithm artefact hasn't been generated yet — running this test
    file requires `uv run fnaf_parser dump-algorithm` to have been run
    at least once."""
    if not COMBINED_JSON.exists():
        pytest.skip(
            "out/algorithm/combined.json missing — run "
            "`uv run fnaf_parser dump-algorithm` first"
        )
    return _load_combined(COMBINED_JSON)


def test_every_handwritten_record_verifies_against_combined(real_combined):
    """End-to-end: every handwritten record must auto-accept via
    Citation Checker. If one regresses, the specific record name +
    failing outcome reason is in the failure message — easy triage."""
    failures: list[str] = []
    for rec in HANDWRITTEN:
        result = verify_record(real_combined, rec)
        if not result.accepted:
            reasons = [o.reason for o in result.outcomes]
            failures.append(
                f"claim={rec.claim!r}  citations={rec.citations}  "
                f"outcomes={reasons}"
            )
    assert not failures, "Handwritten records failed Citation Checker:\n" + "\n".join(
        failures
    )


def test_every_handwritten_accept_reason_is_expr_str_match(real_combined):
    """Stronger contract than the previous test: every handwritten
    record must be accepted via `expr_str_match` specifically — not via
    the weaker num_name fallback. That's the whole reason we wrote the
    pseudo_code to be a verbatim expr_str substring."""
    for rec in HANDWRITTEN:
        result = verify_record(real_combined, rec)
        assert result.accepted, (
            f"Record not accepted at all: {rec.claim!r}"
        )
        for outcome in result.outcomes:
            assert outcome.reason == "expr_str_match", (
                f"Record {rec.claim!r} accepted via {outcome.reason} "
                f"instead of expr_str_match — either the pseudo_code "
                f"drifted or combined.json no longer carries the "
                f"expected expr_str at this coordinate."
            )


def test_handwritten_records_stream_through_check_records(
    real_combined, tmp_path: Path
):
    """The `check_records` stream writer must emit every handwritten
    record into `accepted.jsonl`, zero into `quarantine.jsonl`. This
    is the exact path Rust Test Emission will consume."""
    accepted_path = tmp_path / "handwritten_accepted.jsonl"
    quarantine_path = tmp_path / "handwritten_quarantine.jsonl"
    accepted, quarantined = check_records(
        HANDWRITTEN, COMBINED_JSON, accepted_path, quarantine_path
    )
    assert accepted == len(HANDWRITTEN)
    assert quarantined == 0
    lines = accepted_path.read_text().splitlines()
    assert len(lines) == len(HANDWRITTEN)
    # Every emitted line is valid JSON with the expected envelope
    for line in lines:
        payload = json.loads(line)
        assert "record" in payload
        assert "outcomes" in payload
        assert all(o["accepted"] for o in payload["outcomes"])
